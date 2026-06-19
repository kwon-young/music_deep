import argparse
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from model.detector import OMRDetector, create_detector
from dataset.coco import (
    parse_coco,
    load_coco_sample,
    CocoSymbolAnnotation,
    CocoLineAnnotation,
)
import transform.det as det_tf


def process_single_image(
    i,
    dataset,
    args,
    model,
    device,
    sym_idx_to_cat_id,
    line_idx_to_cat_id,
    max_symbols,
    max_lines,
):
    img_meta = dataset.images[i]

    # Load sample with tensors on the target device
    item = load_coco_sample(dataset, args.img_dir, i, device)

    # Decode and transform
    # Note: decode_nvimgcodec handles the image, boxes/keypoints are already on device
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

    # Collate to add batch dimension and extract patches
    batched_item = det_tf.collate((item_padded,))
    patched_item = det_tf.extract_patches(
        batched_item, patch_size=(args.patch_size, args.patch_size)
    )

    # Apply variance patch dropping (same as training)
    dropped_item = det_tf.variance_patch_drop(
        patched_item,
        var_threshold=args.var_threshold,
        drop_rate=args.drop_rate,
    )

    # Forward pass
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=args.use_amp
    ):
        outputs = model(dropped_item.sample.image)

    # --- Process Symbols ---
    sym_logits = outputs.symbols.pred_logits.data[0]
    sym_boxes = outputs.symbols.pred_boxes.data[0]

    sym_probs = torch.sigmoid(sym_logits)
    sym_max_probs, sym_labels = sym_probs.max(dim=-1)

    k_sym = min(max_symbols, sym_max_probs.shape[0])
    sym_probs_kept, sym_keep = torch.topk(sym_max_probs, k_sym)
    sym_boxes_kept = sym_boxes[sym_keep]
    sym_labels_kept = sym_labels[sym_keep]

    sym_boxes_kept[:, [0, 1, 2, 3]] *= args.patch_size

    sym_results = []
    for box, prob, label in zip(
        sym_boxes_kept, sym_probs_kept, sym_labels_kept
    ):
        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        x1 = max(0.0, min(x1, float(img_meta.width)))
        y1 = max(0.0, min(y1, float(img_meta.height)))
        w = max(0.0, min(w, float(img_meta.width - x1)))
        h = max(0.0, min(h, float(img_meta.height - y1)))

        sym_results.append(
            {
                "image_id": img_meta.img_id,
                "category_id": sym_idx_to_cat_id[label.item()],
                "bbox": [x1, y1, w, h],
                "score": prob.item(),
            }
        )

    # --- Process Lines ---
    line_logits = outputs.lines.pred_logits.data[0]
    line_kps = outputs.lines.pred_keypoints.data[0]

    line_probs = torch.sigmoid(line_logits)
    line_max_probs, line_labels = line_probs.max(dim=-1)

    k_line = min(max_lines, line_max_probs.shape[0])
    line_probs_kept, line_keep = torch.topk(line_max_probs, k_line)
    line_kps_kept = line_kps[line_keep]
    line_labels_kept = line_labels[line_keep]

    line_kps_kept[:, [0, 1, 2, 3]] *= args.patch_size

    line_results = []
    for kp, prob, label in zip(
        line_kps_kept, line_probs_kept, line_labels_kept
    ):
        x1, y1, x2, y2 = kp.tolist()

        x1 = max(0.0, min(x1, float(img_meta.width)))
        y1 = max(0.0, min(y1, float(img_meta.height)))
        x2 = max(0.0, min(x2, float(img_meta.width)))
        y2 = max(0.0, min(y2, float(img_meta.height)))

        line_results.append(
            {
                "image_id": img_meta.img_id,
                "category_id": line_idx_to_cat_id[label.item()],
                "keypoints": [x1, y1, 1, x2, y2, 1],
                "score": prob.item(),
            }
        )

    return sym_results, line_results


def run_inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load dataset to get metadata and category mappings
    dataset = parse_coco(args.anno_path, cache_dir=args.cache_dir)
    sym_idx_to_cat_id = {v: k for k, v in dataset.symbol_cat_id_to_idx.items()}
    line_idx_to_cat_id = {v: k for k, v in dataset.line_cat_id_to_idx.items()}

    # Dynamically compute the maximum objects per image
    max_symbols = 0
    max_lines = 0
    for anns in dataset.annotations.values():
        sym_count = sum(1 for a in anns if isinstance(a, CocoSymbolAnnotation))
        line_count = sum(1 for a in anns if isinstance(a, CocoLineAnnotation))
        max_symbols = max(max_symbols, sym_count)
        max_lines = max(max_lines, line_count)

    # Add a small safety buffer
    max_symbols += 100
    max_lines += 100
    print(f"Dynamic Top-K limits -> Symbols: {max_symbols}, Lines: {max_lines}")

    # 2. Setup model
    model = create_detector(
        backbone_size=args.backbone_size,
        patch_size=args.patch_size,
        channels=args.channels,
        use_sdpa=args.use_sdpa,
        num_symbol_classes=dataset.num_symbol_classes,
        num_line_classes=dataset.num_line_classes,
        num_shapes=args.num_shapes,
        base_anchor_size=args.base_anchor_size,
    ).to(device)

    # 3. Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_sym_results = []
    all_line_results = []

    # Determine indices to process
    num_images = len(dataset.images)
    if args.num_samples is not None and args.num_samples < num_images:
        indices = random.sample(range(num_images), args.num_samples)
        print(
            f"Randomly subsampled {args.num_samples} images out of {num_images}."
        )
    else:
        indices = range(num_images)

    print("Running inference...")
    with torch.no_grad():
        for i in tqdm(indices):
            sym_res, line_res = process_single_image(
                i,
                dataset,
                args,
                model,
                device,
                sym_idx_to_cat_id,
                line_idx_to_cat_id,
                max_symbols,
                max_lines,
            )
            all_sym_results.extend(sym_res)
            all_line_results.extend(line_res)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save results
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sym_out = out_dir / "preds_symbols.json"
    print(f"Saving {len(all_sym_results)} symbol predictions to {sym_out}")
    with open(sym_out, "w") as f:
        json.dump(all_sym_results, f)

    line_out = out_dir / "preds_lines.json"
    print(f"Saving {len(all_line_results)} line predictions to {line_out}")
    with open(line_out, "w") as f:
        json.dump(all_line_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and save COCO results"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (e.g., latest_model.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("predictions"),
        help="Directory to save the COCO format predictions JSONs",
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path(
            "data/trompa-coco/annotations/instances_trainval2017.json"
        ),
    )
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument(
        "--backbone_size",
        type=str,
        choices=["nano", "small", "base"],
        default="nano",
        help="Size of the ViT backbone to use.",
    )
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument("--var_threshold", type=float, default=None)
    parser.add_argument("--drop_rate", type=float, default=None)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of random images to process. If not set, processes the whole dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    
    if args.var_threshold is None and args.drop_rate is None:
        args.var_threshold = 0.001
    elif args.var_threshold is not None and args.drop_rate is not None:
        raise ValueError("Cannot specify both --var_threshold and --drop_rate")
        
    run_inference(args)
