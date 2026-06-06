from functools import wraps, partial
from threading import Thread, Lock, Barrier, Event, get_ident
from queue import Queue, ShutDown
from typing import Callable, Generator, Iterator, Iterable, Optional, Literal
import time
import shutil
from collections import defaultdict


class PartialGenerator[T]:
    def __init__(self, f: Callable[[], Iterator[T]]):
        self.f: Callable[[], Iterator[T]] = f

    def __iter__(self) -> Iterator[T]:
        return iter(self.f())


def partial_generator[T, **P](
    f: Callable[P, Iterator[T]],
) -> Callable[P, PartialGenerator[T]]:
    @wraps(f)
    def gen(*args: P.args, **kwargs: P.kwargs) -> PartialGenerator[T]:
        return PartialGenerator(partial(f, *args, **kwargs))

    return gen


@partial_generator
def iter_queue[T](queue: Queue[T]) -> Generator[T]:
    try:
        while True:
            yield queue.get()
    except ShutDown:
        pass
    finally:
        queue.shutdown(immediate=True)


class IterableThread[T](Thread):
    def __init__(
        self, it: Iterable[T], queue: Queue[T], barrier: Barrier, name: str
    ):
        super().__init__(name=f"{it}-{name}")
        self.it: Iterable[T] = it
        self.queue: Queue[T] = queue
        self.barrier: Barrier = barrier
        self.exception: Exception | None = None

    def run(self) -> None:
        try:
            for x in self.it:
                try:
                    self.queue.put(x)
                except ShutDown:
                    break
        except Exception as e:
            self.exception = e
            self.queue.shutdown(True)
        finally:
            self.barrier.wait()
            self.queue.shutdown()


class ParallelGenerator[T]:
    def __init__(
        self,
        it: Iterable[T],
        num_workers: int = 1,
        maxsize: int = 1,
        disable: bool = False,
        monitor: Optional["Monitor"] = None,
        name: str | None = None,
    ):
        self.it: Iterable[T] = it
        self.num_workers: int = 0 if disable else num_workers
        self.queue: Queue[T] = Queue(maxsize)
        self.maxsize = maxsize
        self.lock: Lock = Lock()
        self.refcount: int = 0
        self.workers: list[IterableThread[T]] = []
        self.name: str = name or f"{type(self).__name__}({it})"
        self.monitor = monitor
        if monitor and not disable:
            monitor.register(self)

    def __enter__(self) -> "ParallelGenerator[T]":
        with self.lock:
            self.refcount += 1
            if self.refcount == 1:
                if self.num_workers > 0:
                    barrier = Barrier(self.num_workers)
                self.queue = Queue(self.maxsize)
                self.workers = []
                for i in range(self.num_workers):
                    worker = IterableThread(
                        self.iter_stat(self.it, "p"),
                        self.queue,
                        barrier,
                        name=f"{self.name}-{i}",
                    )
                    worker.start()
                    self.workers.append(worker)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.terminate()
        else:
            self.join()

    def join(self) -> None:
        for worker in self.workers:
            worker.join()
        with self.lock:
            self.refcount -= 1
            for worker in self.workers:
                if worker.exception:
                    msg = f"Exception from {worker}"
                    raise RuntimeError(msg) from worker.exception

    def terminate(self) -> None:
        self.queue.shutdown(True)
        self.join()

    @partial_generator
    def iter_stat(self, it: Iterable[T], t: Literal["p", "c"]) -> Generator[T]:
        if self.monitor:
            t0 = time.monotonic()
            tid = get_ident()
            for x in it:
                yield x
                t1 = time.monotonic()
                delta = t1 - t0
                self.monitor.put((self, t, delta, tid))
                t0 = time.monotonic()
        else:
            yield from it

    def __iter__(self) -> Generator[T]:
        if self.num_workers == 0:
            yield from self.iter_stat(self.it, "c")
        else:
            with self:
                yield from self.iter_stat(iter_queue(self.queue), "c")


class ThreadedGenerator[T](ParallelGenerator[T]):
    def __init__(
        self,
        it: Iterable[T],
        maxsize: int = 1,
        disable: bool = False,
        monitor: Optional["Monitor"] = None,
        name: Optional[str] = None,
    ):
        super().__init__(it, 1, maxsize, disable, monitor, name)


class Monitor(Thread):
    def __init__(self, name: str = "Monitor"):
        super().__init__(name=name)
        self.stop = Event()
        self.nodes: list[ParallelGenerator] = []
        self.queue: Queue = Queue()
        self.history: defaultdict[
            ParallelGenerator, dict[str, list[float | int]]
        ] = defaultdict(lambda: {"t": [], "size": [], "p": [], "c": []})
        self.start_time = time.monotonic()

    def register(self, node: ParallelGenerator) -> None:
        self.nodes.append(node)

    def put(self, x: tuple[ParallelGenerator, Literal["p", "c"], float, int]):
        self.queue.put(x)

    def drain(self):
        node_deltas = defaultdict(list)
        while not self.queue.empty():
            node, t, delta, tid = self.queue.get()
            node_deltas[(node, t, tid)].append(delta)

        throughputs = defaultdict(float)
        for (node, t, tid), ds in node_deltas.items():
            # Average duration per item for this thread
            avg_duration = sum(ds) / len(ds)
            if avg_duration > 0:
                # Throughput for this thread is 1 / duration
                throughputs[(node, t)] += 1.0 / avg_duration

        return throughputs

    def __enter__(self) -> "Monitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop.set()
        self.join()
        if self.nodes:
            # Move cursor down past the bars so we don't overwrite them
            # (assuming the last drawn frame is still on screen)
            print(f"\033[{len(self.nodes)}B", end="")

    def run(self):
        first = True
        while not self.stop.is_set():
            throughputs = self.drain()
            now = time.monotonic() - self.start_time

            if not first and self.nodes:
                print(f"\033[{len(self.nodes)}A", end="\r")

            for node in self.nodes:
                hist = self.history[node]
                hist["t"].append(now)
                qsize = node.queue.qsize()
                hist["size"].append(qsize)

                p_tps = throughputs.get((node, "p"), 0.0)
                c_tps = throughputs.get((node, "c"), 0.0)

                hist["p"].append(p_tps)
                hist["c"].append(c_tps)

                cols = shutil.get_terminal_size().columns
                name = (node.name or "")[:10].ljust(10)
                stats = f" {qsize}/{node.maxsize} "
                stats += f"[P: {p_tps:5.1f}/s | C: {c_tps:5.1f}/s]"
                bar_len = max(0, cols - len(name) - len(stats) - 2)
                filled = (
                    int(bar_len * qsize / node.maxsize)
                    if node.maxsize > 0
                    else 0
                )
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"{name}|{bar}|{stats}")

            first = False
            time.sleep(1)

    def plot(self):
        """Display live statistics using matplotlib on the main thread."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # Set up plot
        fig, axes = plt.subplots(len(self.nodes), 1, sharex=True)
        if len(self.nodes) == 1:
            axes = [axes]

        def update(frame):
            for i, node in enumerate(self.nodes):
                hist = self.history[node]
                if not hist["t"]:
                    continue

                # Keep last 100 points for display
                limit = 100
                ts = hist["t"][-limit:]
                y_size = hist["size"][-limit:]
                y_p = hist["p"][-limit:]
                y_c = hist["c"][-limit:]

                ax = axes[i]
                ax.clear()
                ax.set_title(node.name)
                ax.plot(ts, y_size, label="Q Size")
                ax.plot(ts, y_p, label="Prod/s")
                ax.plot(ts, y_c, label="Cons/s")
                ax.legend(loc="upper left")

        _ = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
        plt.show()
