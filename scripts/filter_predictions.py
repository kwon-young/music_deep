import argparse
import json
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Filter COCO predictions to top-K per image.")
    parser.add_argument("--pred_path", type=Path, required=True, help="Path to input predictions JSON")
    parser.add_argument("--out_path", type=Path, required=True, help="Path to output filtered predictions JSON")
    parser.add_argument("--top_k", type=int, default=5000, help="Maximum number of predictions to keep per image")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Minimum confidence score to keep")
    args = parser.parse_args()

    print(f"Loading predictions from {args.pred_path}...")
    with open(args.pred_path, "r") as f:
        preds = json.load(f)

    print(f"Loaded {len(preds)} total predictions.")

    # Group by image_id
    preds_by_img = defaultdict(list)
    for p in preds:
        if p.get("score", 0) >= args.min_conf:
            preds_by_img[p["image_id"]].append(p)

    filtered_preds = []
    for img_id, img_preds in preds_by_img.items():
        # Sort by score descending
        img_preds.sort(key=lambda x: x.get("score", 0), reverse=True)
        # Keep top K
        filtered_preds.extend(img_preds[:args.top_k])

    print(f"Filtered down to {len(filtered_preds)} predictions (max {args.top_k} per image, min conf {args.min_conf}).")

    print(f"Saving to {args.out_path}...")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(filtered_preds, f)

    print("Done!")

if __name__ == "__main__":
    main()
