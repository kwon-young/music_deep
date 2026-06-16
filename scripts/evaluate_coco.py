import argparse
import numpy as np
import pickle
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_modality(coco_gt, pred_path, iou_type, cat_ids, out_dir, prefix):
    print(f"\n--- Evaluating {prefix} ({iou_type}) ---")
    if not pred_path.exists():
        print(f"Prediction file {pred_path} not found. Skipping.")
        return {}

    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.params.catIds = cat_ids
    # Lowered maxDets to 2000 to speed up evaluation
    coco_eval.params.maxDets = [1, 100, 2000]

    if iou_type == "keypoints":
        # Custom sigmas for start/end points
        coco_eval.params.kpt_oks_sigmas = np.array([0.1, 0.1])

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval["precision"]
    cats = coco_gt.loadCats(cat_ids)

    per_cat_stats = {}
    for i, cat in enumerate(cats):
        p_50 = precisions[0, :, i, 0, 2]
        p_50 = p_50[p_50 > -1]

        p_all = precisions[:, :, i, 0, 2]
        p_all = p_all[p_all > -1]

        map_50 = np.mean(p_50) if len(p_50) > 0 else None
        map_all = np.mean(p_all) if len(p_all) > 0 else None

        per_cat_stats[cat["name"]] = {
            "category_id": cat["id"],
            "mAP_0.5": float(map_50) if map_50 is not None else None,
            "mAP_0.5_0.95": float(map_all) if map_all is not None else None,
        }

        if map_50 is not None:
            print(f"{cat['name']:<30}: {map_50:.4f}")
        else:
            print(f"{cat['name']:<30}: N/A (No ground truth)")

    out_pkl = out_dir / f"coco_eval_raw_{prefix}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"eval": coco_eval.eval, "stats": coco_eval.stats}, f)

    return {
        "global_stats": coco_eval.stats.tolist()
        if coco_eval.stats is not None
        else [],
        "per_category": per_cat_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO metrics for Symbols and Lines"
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        required=True,
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing preds_symbols.json and preds_lines.json",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save detailed results. Defaults to pred_dir.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    coco_gt = COCO(str(args.anno_path))

    # Separate categories
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    line_cat_ids = []
    sym_cat_ids = []
    for cat in cats:
        if (
            "keypoints" in cat
            and "start" in cat["keypoints"]
            and "end" in cat["keypoints"]
        ):
            line_cat_ids.append(cat["id"])
        else:
            sym_cat_ids.append(cat["id"])

    summary = {}

    # Evaluate Symbols
    sym_pred_path = args.pred_dir / "preds_symbols.json"
    summary["symbols"] = evaluate_modality(
        coco_gt, sym_pred_path, "bbox", sym_cat_ids, out_dir, "symbols"
    )

    # Evaluate Lines
    line_pred_path = args.pred_dir / "preds_lines.json"
    summary["lines"] = evaluate_modality(
        coco_gt, line_pred_path, "keypoints", line_cat_ids, out_dir, "lines"
    )

    out_json = out_dir / "coco_eval_summary.json"
    print(f"\nSaving human-readable summary to {out_json}...")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
