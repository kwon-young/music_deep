import argparse
from pathlib import Path
import fiftyone as fo

def main(args):
    dataset_name = "trompa-coco-eval"
    
    # Clear the dataset if it already exists so we get a fresh load
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        
    print(f"Loading Ground Truth dataset from {args.anno_path}...")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(args.img_dir),
        labels_path=str(args.anno_path),
        name=dataset_name,
        label_field="ground_truth",
    )
    
    if args.predictions_path and args.predictions_path.exists():
        print(f"Loading predictions from {args.predictions_path}...")
        # add_coco_labels matches the 'image_id' in your predictions.json 
        # to the 'coco_id' automatically populated by from_dir()
        dataset.add_coco_labels(
            str(args.predictions_path),
            label_field="predictions",
            coco_id_field="coco_id",
        )
    else:
        print(f"Warning: Predictions file not found at {args.predictions_path}")

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
