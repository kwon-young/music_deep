"""Zero-shot detection quality on synthetic pages, binned by how far each object
lies from the real Trompa-COCO geometry of its class.

* symbols: scale ratio = sqrt(w*h) / real median sqrt(w*h) of the class;
* lines: length ratio = length / real median length of the class, and angle
  deviation = degrees outside the class's real orientation range.

AP and recall per bin use COCO matching (``evaluate_coco`` fast path,
maxDets=2000), with the bin value in place of the area range: ground truths
outside a bin are ignored, as are detections matched to them, and unmatched
detections count against the bin of their own geometry. Line OKS ignores the
endpoint order (best of the two orderings) and keeps the length × thickness area.
"""

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluate_coco import fast_evaluateImg

type FloatArray = npt.NDArray[np.float64]
type Bin = tuple[str, float, float]
type DetValue = Callable[[dict[str, object]], float]

RATIO_BINS: tuple[Bin, ...] = (
    ("<1/4", 0.0, 0.25),
    ("1/4-1/2", 0.25, 0.5),
    ("1/2-0.7", 0.5, 2**-0.5),
    ("0.7-1.4", 2**-0.5, 2**0.5),
    ("1.4-2", 2**0.5, 2.0),
    ("2-4", 2.0, 4.0),
    (">4", 4.0, 1e9),
)
ANGLE_BINS: tuple[Bin, ...] = (
    ("0-5deg", 0.0, 5.0),
    ("5-20deg", 5.0, 20.0),
    ("20-45deg", 20.0, 45.0),
    ("45-90deg", 45.0, 90.0),
)
REAL_ANGLE_TOLERANCE = 5.0
OUTSIDE = -1.0


@dataclass(frozen=True)
class LineReference:
    median_length: float
    orientation: float
    half_width: float


@dataclass(frozen=True)
class References:
    symbol_size: dict[int, float]
    lines: dict[int, LineReference]


def orientation_distance(a: FloatArray, b: float) -> FloatArray:
    """Distance between undirected orientations in degrees, in [0, 90]."""
    return np.abs(np.mod(a - b + 90.0, 180.0) - 90.0)


def line_geometry(kps: FloatArray) -> tuple[FloatArray, FloatArray]:
    dx = kps[:, 3] - kps[:, 0]
    dy = kps[:, 4] - kps[:, 1]
    return np.hypot(dx, dy), np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)


def build_references(real: COCO, line_ids: set[int]) -> References:
    symbol_size: dict[int, float] = {}
    lines: dict[int, LineReference] = {}
    for cat_id in real.getCatIds():
        anns = real.loadAnns(real.getAnnIds(catIds=[cat_id]))
        if not anns:
            continue
        if cat_id in line_ids:
            kps = np.array([a["keypoints"] for a in anns], dtype=np.float64)
            length, angle = line_geometry(kps)
            doubled = np.radians(2.0 * angle)
            mean = math.degrees(
                math.atan2(float(np.sin(doubled).mean()), float(np.cos(doubled).mean()))
            )
            orientation = float(np.mod(mean / 2.0, 180.0))
            lines[cat_id] = LineReference(
                median_length=float(np.median(length)),
                orientation=orientation,
                half_width=float(
                    np.percentile(orientation_distance(angle, orientation), 99)
                ),
            )
        else:
            wh = np.array([a["bbox"][2:] for a in anns], dtype=np.float64)
            symbol_size[cat_id] = float(np.median(np.sqrt(wh[:, 0] * wh[:, 1])))
    return References(symbol_size=symbol_size, lines=lines)


def symbol_ratio(ann: dict[str, object], refs: References) -> float:
    bbox = ann["bbox"]
    cat = ann["category_id"]
    assert isinstance(bbox, list) and isinstance(cat, int)
    ref = refs.symbol_size.get(cat)
    if ref is None:
        return OUTSIDE
    return math.sqrt(max(bbox[2] * bbox[3], 0.0)) / ref


def line_values(ann: dict[str, object], refs: References) -> tuple[float, float]:
    """(length ratio, angle deviation outside the real range) of one line."""
    kps = ann["keypoints"]
    cat = ann["category_id"]
    assert isinstance(kps, list) and isinstance(cat, int)
    ref = refs.lines.get(cat)
    if ref is None:
        return OUTSIDE, OUTSIDE
    length, angle = line_geometry(np.array([kps], dtype=np.float64))
    deviation = orientation_distance(angle, ref.orientation)[0] - ref.half_width
    return float(length[0]) / ref.median_length, max(0.0, float(deviation))


def order_invariant_oks(self: COCOeval, img_id: int, cat_id: int) -> FloatArray:
    p = self.params
    gts = self._gts[img_id, cat_id]
    dts = sorted(self._dts[img_id, cat_id], key=lambda d: -d["score"])[: p.maxDets[-1]]
    if not gts or not dts:
        return np.zeros((0, 0))
    variances = (np.asarray(p.kpt_oks_sigmas) * 2) ** 2
    g = np.array([gt["keypoints"] for gt in gts], dtype=np.float64)
    d = np.array([dt["keypoints"] for dt in dts], dtype=np.float64)
    areas = np.array([gt["oks_area"] for gt in gts], dtype=np.float64)

    def oks(dx: FloatArray, dy: FloatArray) -> FloatArray:
        e = (dx**2 + dy**2) / variances / (areas[None, :, None] + np.spacing(1)) / 2
        return np.exp(-e).mean(axis=2)

    gx, gy = g[:, 0::3], g[:, 1::3]
    dx, dy = d[:, 0::3], d[:, 1::3]
    forward = oks(dx[:, None] - gx[None], dy[:, None] - gy[None])
    swapped = oks(dx[:, None, ::-1] - gx[None], dy[:, None, ::-1] - gy[None])
    return np.maximum(forward, swapped)


def binned_eval(
    gt: COCO,
    pred_path: Path,
    iou_type: str,
    cat_ids: list[int],
    value: dict[int, float],
    det_value: DetValue,
    bins: tuple[Bin, ...],
) -> list[dict[str, float | str | int]]:
    """COCO AP / AP50 / recall@0.5 per bin of ``value`` (maxDets=2000)."""
    for ann_id, ann in gt.anns.items():
        if "oks_area" not in ann:
            ann["oks_area"] = ann["area"]
        ann["area"] = value.get(ann_id, OUTSIDE)
    dt = gt.loadRes(str(pred_path))
    for ann in dt.anns.values():
        ann["area"] = det_value(ann)
    ev = COCOeval(gt, dt, iou_type)
    ev.evaluateImg = fast_evaluateImg.__get__(ev, COCOeval)
    if iou_type == "keypoints":
        ev.computeOks = order_invariant_oks.__get__(ev, COCOeval)
        ev.params.kpt_oks_sigmas = np.array([0.1, 0.1])
    ev.params.catIds = cat_ids
    ev.params.maxDets = [2000]
    ev.params.areaRng = [[lo, hi] for _, lo, hi in bins]
    ev.params.areaRngLbl = [name for name, _, _ in bins]
    ev.evaluate()
    ev.accumulate()
    precision = ev.eval["precision"]
    recall = ev.eval["recall"]

    rows: list[dict[str, float | str | int]] = []
    for a, (name, lo, hi) in enumerate(bins):
        count = sum(1 for v in value.values() if lo <= v <= hi)
        all_p = precision[:, :, :, a, 0]
        p50 = precision[0, :, :, a, 0]
        r50 = recall[0, :, a, 0]
        rows.append(
            {
                "bin": name,
                "gt": count,
                "AP": float(all_p[all_p > -1].mean()) if (all_p > -1).any() else -1.0,
                "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else -1.0,
                "recall50": float(r50[r50 > -1].mean()) if (r50 > -1).any() else -1.0,
            }
        )
    return rows



def print_rows(title: str, rows: list[dict[str, float | str | int]]) -> None:
    print(f"\n{title}")
    print(f"  {'bin':10s} {'GT':>6s} {'AP':>6s} {'AP50':>6s} {'R@0.5':>6s}")
    for r in rows:
        print(
            f"  {r['bin']!s:10s} {r['gt']:>6} {r['AP']:6.3f} {r['AP50']:6.3f} "
            f"{r['recall50']:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno_path", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_anno",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    real = COCO(str(args.reference_anno))
    line_ids = {
        c["id"] for c in real.loadCats(real.getCatIds()) if "start" in c["keypoints"]
    }
    refs = build_references(real, line_ids)
    gt = COCO(str(args.anno_path))
    symbol_ids = [c for c in gt.getCatIds() if c not in line_ids]
    line_cats = [c for c in gt.getCatIds() if c in line_ids]

    def is_line(ann: dict[str, object]) -> bool:
        return ann["category_id"] in line_ids

    sym_value = {
        i: symbol_ratio(a, refs) for i, a in gt.anns.items() if not is_line(a)
    }
    line_vals = {i: line_values(a, refs) for i, a in gt.anns.items() if is_line(a)}
    length_value = {
        i: (ratio if angle < REAL_ANGLE_TOLERANCE else OUTSIDE)
        for i, (ratio, angle) in line_vals.items()
    }
    angle_value = {i: angle for i, (_, angle) in line_vals.items()}

    def det_length(ann: dict[str, object]) -> float:
        ratio, angle = line_values(ann, refs)
        return ratio if angle < REAL_ANGLE_TOLERANCE else OUTSIDE

    results = {
        "symbols_by_scale_ratio": binned_eval(
            gt, args.pred_dir / "preds_symbols.json", "bbox", symbol_ids,
            sym_value, lambda a: symbol_ratio(a, refs), RATIO_BINS,
        ),
        "lines_by_length_ratio_real_angle": binned_eval(
            gt, args.pred_dir / "preds_lines.json", "keypoints", line_cats,
            length_value, det_length, RATIO_BINS,
        ),
        "lines_by_angle_deviation": binned_eval(
            gt, args.pred_dir / "preds_lines.json", "keypoints", line_cats,
            angle_value, lambda a: line_values(a, refs)[1], ANGLE_BINS,
        ),
    }
    for title, rows in results.items():
        print_rows(title, rows)
    if args.out_json is not None:
        args.out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
