import argparse
import json
from pathlib import Path
import fiftyone as fo

def filter_coco_json(anno_path: Path, pred_image_ids: set, out_path: Path):
    print(f"Loading full GT JSON from {anno_path} (this takes a moment but saves RAM later)...")
    with open(anno_path, "r") as f:
        data = json.load(f)
    
    print("Filtering GT to match predicted images...")
    # Keep only the images and annotations that we actually have predictions for
    data["images"] = [img for img in data["images"] if img["id"] in pred_image_ids]
    data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] in pred_image_ids]
    
    print(f"Saving filtered GT JSON to {out_path}...")
    with open(out_path, "w") as f:
        json.dump(data, f)

def main(args):
    dataset_name = "trompa-coco-eval"
    
    # Clear the dataset if it already exists so we get a fresh load
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        
    if not args.predictions_path.exists():
        print(f"Error: Predictions file not found at {args.predictions_path}")
        return

    print(f"Loading predictions from {args.predictions_path}...")
    with open(args.predictions_path, "r") as f:
        preds = json.load(f)
        
    # Find exactly which images we ran inference on
    pred_image_ids = set(p["image_id"] for p in preds)
    print(f"Found predictions for {len(pred_image_ids)} images.")

    # Create a subset GT JSON to prevent FiftyOne from OOMing on the full dataset
    subset_anno_path = args.anno_path.parent / "instances_subset_eval.json"
    filter_coco_json(args.anno_path, pred_image_ids, subset_anno_path)

    print(f"Loading Ground Truth subset into FiftyOne...")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(args.img_dir),
        labels_path=str(subset_anno_path),
        name=dataset_name,
        label_field="ground_truth",
    )
    
    print("Adding predictions to FiftyOne...")
    dataset.add_coco_labels(
        str(args.predictions_path),
        label_field="predictions",
        coco_id_field="coco_id",
    )

    print("Launching FiftyOne App...")
    session = fo.launch_app(dataset)
    
    # Block execution until you close the FiftyOne App tab/window
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO GT and Predictions in FiftyOne")
    parser.add_argument(
        "--img_dir", 
        type=Path, 
        default=Path("data/trompa-coco/trainval2017"),
        help="Path to the directory containing the images."
    )
    parser.add_argument(
        "--anno_path", 
        type=Path, 
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
        help="Path to the ground truth COCO JSON."
    )
    parser.add_argument(
        "--predictions_path", 
        type=Path, 
        default=Path("experiments/010_full_dataset_baseline/predictions.json"),
        help="Path to the predictions COCO JSON."
    )
    
    args = parser.parse_args()
    main(args)
