import argparse
import numpy as np
import pickle
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO metrics with custom maxDets for OMR"
    )
    parser.add_argument(
        "--anno_path", type=Path, required=True, help="Path to ground truth JSON"
    )
    parser.add_argument(
        "--pred_path", type=Path, required=True, help="Path to predictions JSON"
    )
    parser.add_argument(
        "--out_dir", 
        type=Path, 
        default=None, 
        help="Directory to save detailed results. Defaults to pred_path's directory."
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    print("Loading ground truth...")
    coco_gt = COCO(str(args.anno_path))

    # Load predictions
    print("Loading predictions...")
    coco_dt = coco_gt.loadRes(str(args.pred_path))

    # Run evaluation
    print("Running evaluation...")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Override the default maxDets [1, 10, 100] to handle thousands of symbols per page
    coco_eval.params.maxDets = [1, 100, 5000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract detailed metrics
    precisions = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    print("\n--- Per-Category mAP@0.5 (area=all, maxDets=5000) ---")
    
    per_cat_stats = {}
    for i, cat in enumerate(cats):
        # T=0 is IoU=0.5
        # R=: is all recalls
        # K=i is the specific category
        # A=0 is area='all'
        # M=2 is maxDets=5000 (since we set maxDets = [1, 100, 5000])
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

    # Save full raw evaluation results (Pickle)
    out_pkl = out_dir / "coco_eval_raw.pkl"
    print(f"\nSaving full raw evaluation arrays to {out_pkl}...")
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "eval": coco_eval.eval,
            "stats": coco_eval.stats
        }, f)

    # Save human-readable summary (JSON)
    out_json = out_dir / "coco_eval_summary.json"
    print(f"Saving human-readable summary to {out_json}...")
    
    summary = {
        "global_stats": coco_eval.stats.tolist() if coco_eval.stats is not None else [],
        "per_category": per_cat_stats
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
