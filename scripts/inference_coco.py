import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append("src")

from model.vit import vit_nano
from model.detector import OMRDetector
from dataset.coco import parse_coco, load_coco_sample
import transform.det as det_tf


def run_inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load dataset to get metadata and category mappings
    dataset = parse_coco(args.anno_path, cache_dir=args.cache_dir)
    # Create reverse mapping from 0-indexed contiguous IDs back to original COCO category IDs
    idx_to_cat_id = {v: k for k, v in dataset.cat_id_to_idx.items()}

    # 2. Setup model
    backbone = vit_nano(
        patch_size=args.patch_size, channels=args.channels, use_sdpa=args.use_sdpa
    )
    model = OMRDetector(
        backbone,
        num_classes=dataset.num_classes,
        num_shapes=args.num_shapes,
        base_anchor_size=args.base_anchor_size,
    ).to(device)

    # 3. Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    coco_results = []

    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset.images))):
            img_meta = dataset.images[i]

            # Load sample
            item = load_coco_sample(dataset, args.img_dir, i)

            # Decode and transform
            if device.type == "cuda":
                try:
                    item_decoded = det_tf.decode_nvimgcodec(item, device=device)
                except Exception:
                    # Fallback for unsupported formats
                    item_decoded = det_tf.decode_pyvips(item, device=device)
            else:
                item_decoded = det_tf.decode_pyvips(item, device=device)

            item_tf = det_tf.to_float1(item_decoded)
            item_padded = det_tf.pad_to_patch_size(
                item_tf, patch_size=(args.patch_size, args.patch_size)
            )

            # Get padded dimensions for un-normalizing boxes later
            _, padded_h, padded_w = item_padded.sample.image.data.shape

            # Collate to add batch dimension and extract patches
            batched_item = det_tf.collate((item_padded,))
            patched_item = det_tf.extract_patches(
                batched_item, patch_size=(args.patch_size, args.patch_size)
            )

            # Forward pass
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=args.use_amp
            ):
                outputs = model(patched_item.sample.image)

            # Post-process outputs
            pred_logits = outputs.pred_logits.data[0]  # (P*K, C)
            pred_boxes = outputs.pred_boxes.data[0]  # (P*K, 4)

            probs = torch.sigmoid(pred_logits)
            max_probs, pred_labels = probs.max(dim=-1)

            # Filter by confidence threshold
            keep = max_probs > args.conf_thresh
            pred_boxes_kept = pred_boxes[keep]
            pred_probs_kept = max_probs[keep]
            pred_labels_kept = pred_labels[keep]

            # Convert boxes from [0, 1] (relative to padded image) to absolute pixels
            pred_boxes_kept[:, [0, 2]] *= padded_w
            pred_boxes_kept[:, [1, 3]] *= padded_h

            # Convert to COCO format [x, y, width, height] and append
            for box, prob, label in zip(
                pred_boxes_kept, pred_probs_kept, pred_labels_kept
            ):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                # Clip to original image dimensions
                x1 = max(0.0, min(x1, float(img_meta.width)))
                y1 = max(0.0, min(y1, float(img_meta.height)))
                w = max(0.0, min(w, float(img_meta.width - x1)))
                h = max(0.0, min(h, float(img_meta.height - y1)))

                coco_results.append(
                    {
                        "image_id": img_meta.img_id,
                        "category_id": idx_to_cat_id[label.item()],
                        "bbox": [x1, y1, w, h],
                        "score": prob.item(),
                    }
                )

    # Save results
    out_path = args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(coco_results)} predictions to {out_path}")
    with open(out_path, "w") as f:
        json.dump(coco_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and save COCO results")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (e.g., latest_model.pt)",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("predictions.json"),
        help="Path to save the COCO format predictions JSON",
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument("--conf_thresh", type=float, default=0.01)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    run_inference(args)
