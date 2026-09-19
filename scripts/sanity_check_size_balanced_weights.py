import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.coco import _compute_smoothed_weights


@dataclass(frozen=True)
class ClassSizeStats:
    n: int
    mean: float
    variance: float

    @property
    def cv(self) -> float:
        """Population coefficient of variation (std / mean), 0 when undefined."""
        if self.n >= 2 and self.mean > 0:
            return math.sqrt(self.variance) / self.mean
        return 0.0


def _size_stats(sizes: list[float]) -> ClassSizeStats:
    n = len(sizes)
    if n == 0:
        return ClassSizeStats(n=0, mean=0.0, variance=0.0)
    mean = sum(sizes) / n
    variance = sum((s - mean) ** 2 for s in sizes) / n
    return ClassSizeStats(n=n, mean=mean, variance=variance)


def _difficulty_cv(size_stats: dict[int, ClassSizeStats], i: int, variance_lambda: float, max_diff: float) -> float:
    cv = size_stats.get(i, ClassSizeStats(0, 0.0, 0.0)).cv
    return min(1.0 + variance_lambda * cv, max_diff)


def _new_variance_balanced_weights_pre(
    counts: dict[int, int],
    num_classes: int,
    size_stats: dict[int, ClassSizeStats],
    beta: float = 0.5,
    variance_lambda: float = 0.5,
    max_diff: float = 8.0,
    target_mean: float = 0.25,
    min_val: float = 0.05,
    max_val: float = 0.85,
) -> list[float]:
    """Difficulty factor multiplied into the raw inverse freq, THEN normalized+clamped.

    Only this standalone copy (not the module implementation) is changed.
    """
    if num_classes == 0:
        return []

    raw: list[float] = []
    for i in range(num_classes):
        freq = max(counts.get(i, 0), 1)
        inv_freq = 1.0 / math.pow(freq, beta)
        difficulty = _difficulty_cv(size_stats, i, variance_lambda, max_diff)
        raw.append(inv_freq * difficulty)

    current_mean = sum(raw) / num_classes
    scale = target_mean / current_mean if current_mean > 0 else 1.0
    normalized = [r * scale for r in raw]
    return [max(min_val, min(max_val, n)) for n in normalized]


def _new_variance_balanced_weights_post(
    current: list[float],
    size_stats: dict[int, ClassSizeStats],
    variance_lambda: float = 0.5,
    max_diff: float = 8.0,
    min_val: float = 0.05,
    max_val: float = 0.85,
) -> list[float]:
    """Difficulty factor multiplied into the FINAL (normalized+clamped) weight.

    No re-normalization: classes keep exactly their current weight when cv=0.
    Only this standalone copy (not the module implementation) is changed.
    """
    return [
        max(min_val, min(max_val, w * _difficulty_cv(size_stats, i, variance_lambda, max_diff)))
        for i, w in enumerate(current)
    ]


def _collect(anno_path: Path):
    with open(anno_path, "r") as f:
        coco_data = json.load(f)

    categories = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
    line_cat_ids = {
        c["id"]
        for c in categories
        if "keypoints" in c and "start" in c["keypoints"] and "end" in c["keypoints"]
    }
    symbol_categories = [c for c in categories if c["id"] not in line_cat_ids]
    line_categories = [c for c in categories if c["id"] in line_cat_ids]
    sym_id_to_idx = {c["id"]: i for i, c in enumerate(symbol_categories)}
    line_id_to_idx = {c["id"]: i for i, c in enumerate(line_categories)}

    sym_sizes: dict[int, list[float]] = defaultdict(list)
    line_sizes: dict[int, list[float]] = defaultdict(list)

    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in line_cat_ids and "keypoints" in ann and len(ann["keypoints"]) >= 6:
            x1, y1, v1, x2, y2, v2 = ann["keypoints"][:6]
            if v1 > 0 and v2 > 0:
                length = math.hypot(x2 - x1, y2 - y1)
            elif "bbox" in ann:
                x, y, w, h = ann["bbox"]
                length = math.hypot(w, h)
            else:
                continue
            line_sizes[line_id_to_idx[cat_id]].append(length)
        else:
            x, y, w, h = ann["bbox"]
            diagonal = math.hypot(w, h)
            sym_sizes[sym_id_to_idx[cat_id]].append(diagonal)

    return symbol_categories, line_categories, sym_sizes, line_sizes


def _print_table(
    name: str,
    categories: list[dict],
    sizes: dict[int, list[float]],
    current: list[float],
    new_pre: list[float],
    new_post: list[float],
) -> None:
    stats = {
        i: _size_stats(sizes.get(i, [])) for i in range(len(categories))
    }
    rows = [
        (
            c["name"],
            stats[i].n,
            stats[i].mean,
            stats[i].variance,
            stats[i].cv,
            current[i],
            new_pre[i],
            new_post[i],
        )
        for i, c in enumerate(categories)
    ]
    rows.sort(key=lambda r: r[4], reverse=True)

    print(f"\n=== {name} ({len(categories)} classes, sorted by size CV desc) ===")
    print(
        f"{'class':<28} {'n':>7} {'mean':>8} {'var':>10} {'cv':>7} "
        f"{'w_cur':>7} {'pre':>7} {'post':>7} {'post/cur':>7}"
    )
    for name_, n, mean, var, cv, w_cur, w_pre, w_post in rows:
        ratio = w_post / w_cur if w_cur > 0 else float("nan")
        print(
            f"{name_:<28} {n:>7} {mean:>8.1f} {var:>10.1f} {cv:>7.3f} "
            f"{w_cur:>7.3f} {w_pre:>7.3f} {w_post:>7.3f} {ratio:>7.2f}"
        )

    for label, arr in (("w_cur", current), ("pre", new_pre), ("post", new_post)):
        a = np.array(arr)
        print(
            f"summary[{label}]: mean={a.mean():.3f} med={np.median(a):.3f} "
            f"min={a.min():.3f} max={a.max():.3f}"
        )
    for label, arr in (("pre", new_pre), ("post", new_post)):
        a = np.array(arr)
        cur = np.array(current)
        ratio = a / np.where(cur > 0, cur, np.nan)
        changed = np.abs(ratio - 1.0) > 0.10
        print(f"summary[{label}]: classes with >10% change: {changed.sum()}/{len(cur)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check current vs size-balanced class weights (no source changes)."
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument("--variance_lambda", type=float, default=0.5)
    parser.add_argument("--max_diff", type=float, default=8.0)
    args = parser.parse_args()

    symbol_categories, line_categories, sym_sizes, line_sizes = _collect(
        args.anno_path
    )

    symbol_counts = {i: len(sym_sizes.get(i, [])) for i in range(len(symbol_categories))}
    line_counts = {i: len(line_sizes.get(i, [])) for i in range(len(line_categories))}

    cur_sym = _compute_smoothed_weights(symbol_counts, len(symbol_categories))
    cur_line = _compute_smoothed_weights(line_counts, len(line_categories))

    sym_stats = {
        i: _size_stats(sym_sizes.get(i, [])) for i in range(len(symbol_categories))
    }
    line_stats = {
        i: _size_stats(line_sizes.get(i, [])) for i in range(len(line_categories))
    }

    new_sym_pre = _new_variance_balanced_weights_pre(
        symbol_counts,
        len(symbol_categories),
        sym_stats,
        variance_lambda=args.variance_lambda,
        max_diff=args.max_diff,
    )
    new_line_pre = _new_variance_balanced_weights_pre(
        line_counts,
        len(line_categories),
        line_stats,
        variance_lambda=args.variance_lambda,
        max_diff=args.max_diff,
    )

    new_sym_post = _new_variance_balanced_weights_post(
        cur_sym, sym_stats, variance_lambda=args.variance_lambda, max_diff=args.max_diff
    )
    new_line_post = _new_variance_balanced_weights_post(
        cur_line, line_stats, variance_lambda=args.variance_lambda, max_diff=args.max_diff
    )

    _print_table(
        "SYMBOLS", symbol_categories, sym_sizes, cur_sym, new_sym_pre, new_sym_post
    )
    _print_table(
        "LINES", line_categories, line_sizes, cur_line, new_line_pre, new_line_post
    )


if __name__ == "__main__":
    main()