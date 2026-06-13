import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO metrics with custom maxDets for OMR")
    parser.add_argument("--anno_path", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predictions JSON")
    args = parser.parse_args()

    # Load ground truth
    print("Loading ground truth...")
    coco_gt = COCO(args.anno_path)

    # Load predictions
    print("Loading predictions...")
    coco_dt = coco_gt.loadRes(args.pred_path)

    # Run evaluation
    print("Running evaluation...")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Override the default maxDets [1, 10, 100] to handle thousands of symbols per page
    coco_eval.params.maxDets = [1, 100, 5000]
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print per-category mAP@0.5
    print("\n--- Per-Category mAP@0.5 (area=all, maxDets=5000) ---")
    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)

    for i, cat in enumerate(cats):
        # T=0 is IoU=0.5
        # R=: is all recalls
        # K=i is the specific category
        # A=0 is area='all'
        # M=2 is maxDets=5000 (since we set maxDets = [1, 100, 5000])
        p = precisions[0, :, i, 0, 2]
        
        # Filter out -1 (which means no ground truth objects for this category)
        p = p[p > -1]
        
        if len(p) > 0:
            cat_map_50 = np.mean(p)
            print(f"{cat['name']:<30}: {cat_map_50:.4f}")
        else:
            print(f"{cat['name']:<30}: N/A (No ground truth)")

if __name__ == "__main__":
    main()
