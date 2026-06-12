import argparse
import json
from pathlib import Path
import fiftyone as fo
import fiftyone.utils.coco as fouc
from PIL import Image

def filter_coco_json(anno_path: Path, pred_image_ids: set, out_path: Path):
    # Check if we already have a valid subset to avoid the heavy load
    if out_path.exists():
        print(f"Checking existing subset at {out_path}...")
        try:
            with open(out_path, "r") as f:
                subset_data = json.load(f)
            subset_img_ids = set(img["id"] for img in subset_data.get("images", []))
            if subset_img_ids == pred_image_ids:
                print("Subset already matches requested images. Skipping heavy JSON load.")
                return
        except Exception:
            pass # If it fails to load or parse, just rebuild it

    print(f"Loading full GT JSON from {anno_path} (this takes a moment but saves RAM later)...")
    with open(anno_path, "r") as f:
        data = json.load(f)
    
    print("Filtering GT to match predicted images...")
    # Keep only the images and annotations that we actually have predictions for
    data["images"] = [img for img in data["images"] if img["id"] in pred_image_ids]
    
    filtered_annotations = []
    for ann in data["annotations"]:
        if ann["image_id"] in pred_image_ids:
            # Strip out heavy segmentation masks to prevent FiftyOne from OOMing
            if "segmentation" in ann:
                del ann["segmentation"]
            filtered_annotations.append(ann)
            
    # GO NUCLEAR: Keep only 10 ground truth annotations
    data["annotations"] = filtered_annotations[:10]
    print(f"Nuclear option: restricted to {len(data['annotations'])} ground truth annotations.")
    
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
    all_pred_image_ids = list(set(p["image_id"] for p in preds))
    
    # Restrict to a specific number of images to prevent OOM
    if args.num_images is not None:
        pred_image_ids = set(all_pred_image_ids[:args.num_images])
    else:
        pred_image_ids = set(all_pred_image_ids)
        
    print(f"Restricting visualization to {len(pred_image_ids)} images.")

    # Filter predictions to only these images
    preds = [p for p in preds if p["image_id"] in pred_image_ids]
    
    # GO NUCLEAR: Sort by score and keep only the top 100 predictions
    preds.sort(key=lambda x: x.get("score", 0), reverse=True)
    preds = preds[:100]
    print(f"Nuclear option: restricted to {len(preds)} predictions.")
    
    filtered_preds_path = args.predictions_path.parent / "predictions_subset_eval.json"
    with open(filtered_preds_path, "w") as f:
        json.dump(preds, f)

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
    
    # FIX: Manually add coco_id to samples so add_coco_labels can match them
    print("Mapping COCO image IDs and converting TIFFs to PNG for browser compatibility...")
    with open(subset_anno_path, "r") as f:
        subset_data = json.load(f)
    filename_to_id = {img["file_name"]: img["id"] for img in subset_data.get("images", [])}
    
    cache_dir = args.img_dir / ".fiftyone_cache"
    cache_dir.mkdir(exist_ok=True)

    for sample in dataset:
        orig_path = Path(sample.filepath)
        fname = orig_path.name
        
        if fname in filename_to_id:
            sample["coco_id"] = filename_to_id[fname]
            
            # Convert TIFF to PNG for web viewing
            if orig_path.suffix.lower() in [".tif", ".tiff"]:
                png_path = cache_dir / (orig_path.stem + ".png")
                if not png_path.exists():
                    img = Image.open(orig_path).convert("RGB")
                    img.save(png_path, "PNG")
                sample.filepath = str(png_path)
                
            sample.save()
    
    print("Extracting category mapping from Ground Truth...")
    _, classes_map, _, _, _ = fouc.load_coco_detection_annotations(str(subset_anno_path))

    print("Adding predictions to FiftyOne...")
    fouc.add_coco_labels(
        dataset,
        "predictions",
        str(filtered_preds_path),
        classes_map,
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
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=1,
        help="Number of images to load into FiftyOne to prevent OOM."
    )
    
    args = parser.parse_args()
    main(args)
