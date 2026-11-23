"""Breaks down the COCO-OKS failures of one line class into interpretable categories.

Replays the COCO greedy matching (score order, OKS threshold) per image, then labels
every missed ground truth and every confident false positive by its geometry.
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

OKS_SIGMA = 0.1


@dataclass
class ImageLines:
    gt: FloatArray
    gt_area: FloatArray
    pred: FloatArray
    score: FloatArray


@dataclass
class MatchResult:
    pred_matched: BoolArray
    gt_matched: BoolArray
    gt_to_pred: IntArray


def keypoints_xyxy(keypoints: list[float]) -> list[float]:
    return [keypoints[0], keypoints[1], keypoints[3], keypoints[4]]


def load_class(
    annotations: str, predictions: str, class_name: str
) -> tuple[dict[int, ImageLines], dict[int, str]]:
    with open(annotations) as f:
        coco = json.load(f)
    category_id = next(
        c["id"] for c in coco["categories"] if c["name"] == class_name
    )
    file_names = {image["id"]: image["file_name"] for image in coco["images"]}

    gt: dict[int, list[list[float]]] = defaultdict(list)
    areas: dict[int, list[float]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] == category_id:
            gt[annotation["image_id"]].append(
                keypoints_xyxy(annotation["keypoints"])
            )
            areas[annotation["image_id"]].append(annotation["area"])

    with open(predictions) as f:
        preds = json.load(f)
    pred: dict[int, list[list[float]]] = defaultdict(list)
    scores: dict[int, list[float]] = defaultdict(list)
    for p in preds:
        if p["category_id"] == category_id:
            pred[p["image_id"]].append(keypoints_xyxy(p["keypoints"]))
            scores[p["image_id"]].append(p["score"])

    images: dict[int, ImageLines] = {}
    for image_id in file_names:
        score = np.array(scores[image_id], dtype=np.float64)
        order = np.argsort(-score, kind="mergesort")
        images[image_id] = ImageLines(
            gt=np.array(gt[image_id], dtype=np.float64).reshape(-1, 4),
            gt_area=np.array(areas[image_id], dtype=np.float64),
            pred=np.array(pred[image_id], dtype=np.float64).reshape(-1, 4)[order],
            score=score[order],
        )
    return images, file_names


def oks_matrix(pred: FloatArray, gt: FloatArray, area: FloatArray) -> FloatArray:
    variance = (2 * OKS_SIGMA) ** 2
    d2 = (pred[:, None, 0::2] - gt[None, :, 0::2]) ** 2 + (
        pred[:, None, 1::2] - gt[None, :, 1::2]
    ) ** 2
    e = d2 / variance / (area[None, :, None] + np.spacing(1)) / 2
    return np.asarray(np.exp(-e).mean(axis=-1), dtype=np.float64)


def coco_match(oks: FloatArray, threshold: float) -> MatchResult:
    num_pred, num_gt = oks.shape
    pred_matched = np.zeros(num_pred, dtype=np.bool_)
    gt_matched = np.zeros(num_gt, dtype=np.bool_)
    gt_to_pred = np.full(num_gt, -1, dtype=np.int64)
    for d in range(num_pred):
        candidates = np.where(gt_matched, -1.0, oks[d])
        g = int(np.argmax(candidates)) if num_gt else -1
        if g >= 0 and candidates[g] >= threshold:
            pred_matched[d] = True
            gt_matched[g] = True
            gt_to_pred[g] = d
    return MatchResult(pred_matched, gt_matched, gt_to_pred)


def staff_space(gt: FloatArray) -> float:
    ys = np.unique(np.round((gt[:, 1] + gt[:, 3]) / 2))
    gaps = np.diff(ys)
    gaps = gaps[(gaps > 5) & (gaps < 200)]
    return float(np.median(gaps)) if len(gaps) else 50.0


def row_neighbours(gt: FloatArray, space: float) -> tuple[BoolArray, BoolArray]:
    """Whether each segment is the first/last of its horizontal row of segments."""
    y = (gt[:, 1] + gt[:, 3]) / 2
    same_row = np.abs(y[:, None] - y[None, :]) < 0.25 * space
    left_of = same_row & (gt[None, :, 2] <= gt[:, None, 0] + 0.5 * space)
    right_of = same_row & (gt[None, :, 0] >= gt[:, None, 2] - 0.5 * space)
    first = np.asarray(~left_of.any(axis=1), dtype=np.bool_)
    last = np.asarray(~right_of.any(axis=1), dtype=np.bool_)
    return first, last


def classify_pred(
    pred: FloatArray, gt: FloatArray, oks_row: FloatArray, space: float
) -> str:
    """Labels a false positive by its relation to the closest ground truth."""
    if oks_row.max(initial=0.0) >= 0.5:
        return "duplicate"
    y_pred = (pred[1] + pred[3]) / 2
    y_gt = (gt[:, 1] + gt[:, 3]) / 2
    overlap = np.minimum(pred[2], gt[:, 2]) - np.maximum(pred[0], gt[:, 0])
    union = np.maximum(pred[2], gt[:, 2]) - np.minimum(pred[0], gt[:, 0])
    x_iou = np.clip(overlap, 0, None) / np.maximum(union, 1e-6)
    on_line = np.abs(y_gt - y_pred) < 0.25 * space
    if on_line.any():
        covered = (gt[:, 0] >= pred[0] - 0.5 * space) & (
            gt[:, 2] <= pred[2] + 0.5 * space
        )
        if (on_line & covered).sum() >= 2:
            return "merge (spans >=2 measures)"
        if (on_line & (x_iou > 0.0)).any():
            best = np.argmax(np.where(on_line, x_iou, -1.0))
            gt_len = gt[best, 2] - gt[best, 0]
            if pred[2] - pred[0] < 0.8 * gt_len:
                return "split (shorter than measure)"
            return "horizontal endpoint error"
    if (x_iou > 0.7).any():
        dy = np.abs(y_gt - y_pred)[x_iou > 0.7].min() / space
        if dy < 1.5:
            return "vertical: between/next line"
        return "vertical: far"
    if abs(pred[1] - pred[3]) > 0.5 * space:
        return "slanted"
    return "background"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default="data/trompa-coco/annotations/instances_trainval2017.json",
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--class_name", default="staff")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fp_rank", type=float, default=1.0)
    args = parser.parse_args()

    images, file_names = load_class(
        args.annotations, args.predictions, args.class_name
    )

    fn_kinds: Counter[str] = Counter()
    fp_kinds: Counter[str] = Counter()
    fn_first: Counter[str] = Counter()
    gt_first = Counter[str]()
    swapped = 0
    total_pred = 0
    tp_dx: list[float] = []
    tp_dy: list[float] = []
    fn_len: list[float] = []
    all_len: list[float] = []
    fn_best_err: list[tuple[float, float, float, float]] = []
    per_image: list[tuple[int, int, int, int]] = []
    pred_per_gt: list[float] = []

    for image_id, lines in images.items():
        num_gt = len(lines.gt)
        if num_gt == 0:
            continue
        space = staff_space(lines.gt)
        oks = oks_matrix(lines.pred, lines.gt, lines.gt_area)
        match = coco_match(oks, args.threshold)
        first, last = row_neighbours(lines.gt, space)
        lengths = (lines.gt[:, 2] - lines.gt[:, 0]) / space
        all_len.extend(lengths.tolist())
        total_pred += len(lines.pred)
        swapped += int((lines.pred[:, 0] > lines.pred[:, 2]).sum())
        pred_per_gt.append(len(lines.pred) / num_gt)

        for g in range(num_gt):
            position = (
                "first+last" if first[g] and last[g]
                else "first" if first[g]
                else "last" if last[g]
                else "middle"
            )
            gt_first[position] += 1
            if match.gt_matched[g]:
                d = lines.pred[match.gt_to_pred[g]]
                tp_dx.extend(np.abs(d[0::2] - lines.gt[g, 0::2]).tolist())
                tp_dy.extend(np.abs(d[1::2] - lines.gt[g, 1::2]).tolist())
                continue
            fn_first[position] += 1
            fn_len.append(lengths[g])
            best = float(oks[:, g].max(initial=0.0))
            if best >= args.threshold:
                kind = "stolen (best pred used by another GT)"
            elif best >= 0.1:
                kind = "mislocalized (best OKS 0.1-0.5)"
            else:
                kind = "missed (no pred OKS>0.1)"
            fn_kinds[kind] += 1
            if len(lines.pred):
                d = lines.pred[int(np.argmax(oks[:, g]))]
                err = (d - lines.gt[g]) / space
                fn_best_err.append(
                    (float(err[0]), float(err[1]), float(err[2]), float(err[3]))
                )

        cutoff = int(np.ceil(args.fp_rank * num_gt))
        fps = np.where(~match.pred_matched[:cutoff])[0]
        for d in fps:
            fp_kinds[classify_pred(lines.pred[d], lines.gt, oks[d], space)] += 1
        per_image.append(
            (image_id, num_gt, num_gt - int(match.gt_matched.sum()), len(fps))
        )

    total_gt = sum(gt_first.values())
    total_fn = sum(fn_kinds.values())
    print(f"class={args.class_name} OKS>={args.threshold}")
    print(f"GT {total_gt}, preds {total_pred} ({np.mean(pred_per_gt):.2f}/GT)")
    print(f"preds with swapped endpoints (x1>x2): {swapped}")
    print(f"FN {total_fn} ({total_fn / total_gt:.2%} of GT)")
    for kind, n in fn_kinds.most_common():
        print(f"  {kind:45s} {n:6d}  {n / total_fn:6.1%}")
    print("FN rate by position in row of segments:")
    for position, n in gt_first.most_common():
        print(f"  {position:12s} GT {n:6d}  FN {fn_first[position]:5d}"
              f"  rate {fn_first[position] / n:6.2%}")
    print("length (staff spaces) percentiles 10/50/90:")
    print(f"  all GT {np.percentile(all_len, [10, 50, 90]).round(1)}")
    print(f"  FN     {np.percentile(fn_len, [10, 50, 90]).round(1)}")
    errors = np.array(fn_best_err)
    print("FN best-pred endpoint error (staff spaces), median |.|:"
          f" x1 {np.median(np.abs(errors[:, 0])):.2f}"
          f" y1 {np.median(np.abs(errors[:, 1])):.2f}"
          f" x2 {np.median(np.abs(errors[:, 2])):.2f}"
          f" y2 {np.median(np.abs(errors[:, 3])):.2f}")
    print("TP endpoint |error| px percentiles 50/90/99:")
    print(f"  x {np.percentile(tp_dx, [50, 90, 99]).round(1)}")
    print(f"  y {np.percentile(tp_dy, [50, 90, 99]).round(1)}")
    total_fp = sum(fp_kinds.values())
    print(f"FP within the top {args.fp_rank:g}xGT scored preds: {total_fp}")
    for kind, n in fp_kinds.most_common():
        print(f"  {kind:45s} {n:6d}  {n / max(total_fp, 1):6.1%}")
    print("worst images (FN+FP):")
    for image_id, num_gt, fn, fp in sorted(
        per_image, key=lambda r: -(r[2] + r[3])
    )[:10]:
        print(f"  {image_id:4d} {file_names[image_id]:55s} GT {num_gt:4d}"
              f" FN {fn:4d} FP {fp:4d}")


if __name__ == "__main__":
    main()
