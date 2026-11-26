"""Analysis 008: breakdown of COCO AP by class frequency, object size and line class.

Reads the raw pycocotools precision arrays (``coco_eval_raw_{symbols,lines}.pkl`` written by
``evaluate_coco.py``) of several runs and regroups them. COCO's global AP is a plain mean over
classes and its "small" AP a plain mean over the classes that have small instances, so both can
be dominated by classes with a handful of instances; this script reports the groups separately.

Usage:
    PYTHONPATH=src mamba run -n pytorch python scripts/analyze_ap_breakdown.py \
        --symbol_run 023=experiments/037_029_recipe_from_023/inference_exp023_baseline ...
"""

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.coco import CocoLineAnnotation, CocoSymbolAnnotation, parse_coco

SMALL_AREA = 32.0**2
VARIABLE_SHAPE = ("slur", "tie", "brace")
FREQUENCY_BINS = ((1000, None), (100, 1000), (10, 100), (0, 10))


@dataclass(frozen=True)
class RunEval:
    """Per-class category names (in pycocotools ``K`` order) and precision [T, R, K, A] at
    maxDets=2000, with ``areas`` the area-range labels of the ``A`` axis."""

    names: list[str]
    precision: np.ndarray
    areas: list[str]


@dataclass(frozen=True)
class ClassStats:
    count: int
    small: int
    median_size: float


def load_run(run: Path, modality: str) -> RunEval:
    summary = json.loads((run / "coco_eval_summary.json").read_text())
    assert isinstance(summary, dict)
    per_category = summary[modality]["per_category"]
    assert isinstance(per_category, dict)
    with open(run / f"coco_eval_raw_{modality}.pkl", "rb") as f:
        raw = pickle.load(f)
    assert isinstance(raw, dict)
    evaluation = raw["eval"]
    assert isinstance(evaluation, dict)
    precision = evaluation["precision"]
    assert isinstance(precision, np.ndarray)
    params = evaluation["params"]
    areas = [str(a) for a in params.areaRngLbl]
    max_dets = list(params.maxDets)
    return RunEval(
        names=[str(n) for n in per_category],
        precision=precision[..., max_dets.index(2000)],
        areas=areas,
    )


def mean_ap(run: RunEval, classes: list[int], area: str = "all", iou: int | None = None) -> float:
    """Mean over (IoU thresholds,) recall points and the given classes, as pycocotools does."""
    a = run.areas.index(area)
    p = run.precision[:, :, classes, a] if iou is None else run.precision[iou, :, classes, a]
    valid = p[p > -1]
    return float(valid.mean()) if valid.size else math.nan


def class_stats(anno_path: Path, cache_dir: Path) -> dict[str, ClassStats]:
    """Instance count, small-instance count (box area < 32^2, symbols only) and median size
    (box diagonal or segment length, px) per class name."""
    dataset = parse_coco(anno_path, cache_dir)
    names = {int(c["id"]): str(c["name"]) for c in dataset.symbol_categories + dataset.line_categories}
    sizes: dict[str, list[float]] = defaultdict(list)
    small: Counter[str] = Counter()
    for anns in dataset.annotations.values():
        for a in anns:
            if isinstance(a, CocoSymbolAnnotation):
                name = names[a.category_id]
                w, h = a.bbox[2], a.bbox[3]
                sizes[name].append(math.hypot(w, h))
                small[name] += int(w * h < SMALL_AREA)
            elif isinstance(a, CocoLineAnnotation):
                name = names[a.category_id]
                k = a.keypoints
                sizes[name].append(math.hypot(k[2] - k[0], k[3] - k[1]))
    return {
        name: ClassStats(count=len(s), small=small[name], median_size=float(np.median(s)))
        for name, s in sizes.items()
    }


def cells(runs: dict[str, RunEval], classes: list[int], area: str = "all") -> str:
    return " | ".join(
        f"{mean_ap(r, classes, area):.3f} ({mean_ap(r, classes, area, 0):.3f})" for r in runs.values()
    )


def bin_label(low: int, high: int | None) -> str:
    return f">= {low}" if high is None else f"{low}-{high - 1}"


def symbol_report(runs: dict[str, RunEval], stats: dict[str, ClassStats]) -> None:
    first = next(iter(runs.values()))
    names = first.names
    for r in runs.values():
        assert r.names == names, "runs must share the category order"
    count = {n: stats[n].count if n in stats else 0 for n in names}
    print("\nSYMBOLS, AP@[.5:.95] (AP50) per run: " + " | ".join(runs))
    every = list(range(len(names)))
    variable = [names.index(v) for v in VARIABLE_SHAPE if v in names]
    fixed = [k for k in every if k not in variable]

    small_count = {n: stats[n].small if n in stats else 0 for n in names}

    def row(title: str, classes: list[int], area: str = "all") -> None:
        counter = small_count if area == "small" else count
        n = sum(counter[names[k]] for k in classes)
        print(f"{title:44s} {len(classes):3d} cls {n:7d} inst | {cells(runs, classes, area)}")

    row("all classes", every)
    for low, high in FREQUENCY_BINS:
        row(
            f"  fixed shape, {bin_label(low, high)} instances",
            [k for k in fixed if count[names[k]] >= low and (high is None or count[names[k]] < high)],
        )
    row("  variable shape (slur, tie, brace)", variable)
    for k in variable:
        row(f"    {names[k]}", [k])

    with_small = [k for k in every if not bool(np.all(first.precision[:, :, k, first.areas.index("small")] == -1))]
    print("\nsmall objects (area < 32^2), AP on the small instances only")
    row("COCO 'small' (classes with any small GT)", with_small, "small")
    for low in (10, 100):
        row(
            f"  classes with >= {low} small instances",
            [k for k in with_small if small_count[names[k]] >= low],
            "small",
        )
    for k in sorted(with_small, key=lambda k: -small_count[names[k]]):
        if small_count[names[k]] >= 100:
            row(f"    {names[k]} ({small_count[names[k]]} small)", [k], "small")


def line_report(runs: dict[str, RunEval], stats: dict[str, ClassStats]) -> None:
    first = next(iter(runs.values()))
    names = first.names
    for r in runs.values():
        assert r.names == names, "runs must share the category order"
    print("\nLINES, AP@[.5:.95] (AP50) per run: " + " | ".join(runs))
    every = list(range(len(names)))
    print(f"{'all line classes':32s} | {cells(runs, every)}")
    for k in sorted(every, key=lambda k: -stats[names[k]].count):
        s = stats[names[k]]
        title = f"  {names[k]} ({s.count}, {s.median_size:.0f} px)"
        print(f"{title:32s} | {cells(runs, [k])}")


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for v in values:
        name, _, path = v.partition("=")
        runs.append((name, Path(path)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--symbol_run", action="append", default=[], help="NAME=DIR with coco_eval_raw_symbols.pkl"
    )
    parser.add_argument(
        "--line_run", action="append", default=[], help="NAME=DIR with coco_eval_raw_lines.pkl"
    )
    args = parser.parse_args()
    stats = class_stats(Path(args.anno_path), Path(args.cache_dir))
    symbol_runs = parse_runs([str(v) for v in args.symbol_run])
    line_runs = parse_runs([str(v) for v in args.line_run])
    if symbol_runs:
        symbol_report({n: load_run(p, "symbols") for n, p in symbol_runs}, stats)
    if line_runs:
        line_report({n: load_run(p, "lines") for n, p in line_runs}, stats)


if __name__ == "__main__":
    main()
