import argparse
import json
import random
import threading
import torch
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
from model.detector import OMRDetector, create_detector
from dataset.coco import (
    parse_coco,
    load_coco_sample,
    CocoSymbolAnnotation,
    CocoLineAnnotation,
)
import transform.det as det_tf
from threaded_generator import (
    ParallelGenerator,
    partial_generator,
    ThreadedGenerator,
)


def _greedy_match(pred_classes, gt_classes, pred_geom, gt_geom):
    """Class-aware greedy nearest-neighbor matching (CPU).

    Returns two tensors (gt_indices, pred_indices) of matched pairs, or
    (None, None) if no matches. All computation is done on CPU to avoid
    per-element GPU sync overhead.
    """
    num_gt = gt_classes.shape[0]
    num_pred = pred_classes.shape[0]
    if num_gt == 0 or num_pred == 0:
        return None, None

    pred_classes = pred_classes.cpu()
    gt_classes = gt_classes.cpu()
    pred_geom = pred_geom.cpu()
    gt_geom = gt_geom.cpu()

    all_dists = []
    all_gi = []
    all_pi = []

    for cls in gt_classes.unique():
        gt_idx = torch.nonzero(gt_classes == cls, as_tuple=True)[0]
        pred_idx = torch.nonzero(pred_classes == cls, as_tuple=True)[0]
        if len(gt_idx) == 0 or len(pred_idx) == 0:
            continue
        diff = gt_geom[gt_idx].unsqueeze(1) - pred_geom[pred_idx].unsqueeze(0)
        dists = (diff ** 2).sum(-1)
        gi_grid, pi_grid = torch.meshgrid(gt_idx, pred_idx, indexing="ij")
        all_dists.append(dists.flatten())
        all_gi.append(gi_grid.flatten())
        all_pi.append(pi_grid.flatten())

    if not all_dists:
        return None, None

    all_dists = torch.cat(all_dists)
    all_gi = torch.cat(all_gi)
    all_pi = torch.cat(all_pi)
    order = torch.argsort(all_dists)

    matched_gt = []
    matched_pred = []
    used_preds = set()
    used_gts = set()
    order_list = order.tolist()
    all_gi_list = all_gi.tolist()
    all_pi_list = all_pi.tolist()
    for idx in order_list:
        gi = all_gi_list[idx]
        pi = all_pi_list[idx]
        if gi in used_gts or pi in used_preds:
            continue
        matched_gt.append(gi)
        matched_pred.append(pi)
        used_gts.add(gi)
        used_preds.add(pi)

    if not matched_gt:
        return None, None

    return torch.tensor(matched_gt), torch.tensor(matched_pred)


def apply_oracle_fgl(outputs, item, patch_size, base_anchor_size):
    """Replace FGL residuals with GT-derived perfect residuals for matched
    predictions, leaving unmatched predictions unchanged.

    This measures the upper bound of FGL refinement quality: "if the FGL
    residuals were perfect (given the model's base predictions and
    classification), how good would the geometry be?"

    The residual is clamped to [-0.5, +0.5] * scale, matching the D-FINE
    weighting function range. If a base prediction is too far from the GT,
    the clamped residual cannot fully correct it — this is by design, as
    it isolates FGL quality from base prediction quality.
    """
    device = outputs.symbols.pred_boxes.data.device

    # --- Symbols (boxes) ---
    gt_boxes = item.sample.boxes.data
    gt_box_labels = item.sample.box_labels.data
    if gt_boxes.shape[0] > 0:
        gt_boxes_pu = gt_boxes / patch_size
        gt_centers = (gt_boxes_pu[:, :2] + gt_boxes_pu[:, 2:]) / 2

        sym_logits = outputs.symbols.pred_logits.data[0]
        sym_labels = sym_logits.sigmoid().argmax(dim=-1)
        sym_centers = outputs.symbols.absolute_centers.data[0]
        sym_shapes = outputs.symbols.learnable_shapes.data[0]
        sym_boxes = outputs.symbols.pred_boxes.data[0]

        gt_idx, pred_idx = _greedy_match(
            sym_labels, gt_box_labels, sym_centers, gt_centers
        )
        if gt_idx is not None:
            gt_idx = gt_idx.to(device)
            pred_idx = pred_idx.to(device)

            w = sym_shapes[pred_idx, 0]
            h = sym_shapes[pred_idx, 1]
            cx = sym_centers[pred_idx, 0]
            cy = sym_centers[pred_idx, 1]
            gx1, gy1, gx2, gy2 = gt_boxes_pu[gt_idx].T

            L_res = ((cx - gx1) / w - 0.5).clamp(-0.5, 0.5)
            T_res = ((cy - gy1) / h - 0.5).clamp(-0.5, 0.5)
            R_res = ((gx2 - cx) / w - 0.5).clamp(-0.5, 0.5)
            B_res = ((gy2 - cy) / h - 0.5).clamp(-0.5, 0.5)

            sym_boxes[pred_idx, 0] = cx - (w / 2 + L_res * w)
            sym_boxes[pred_idx, 1] = cy - (h / 2 + T_res * h)
            sym_boxes[pred_idx, 2] = cx + (w / 2 + R_res * w)
            sym_boxes[pred_idx, 3] = cy + (h / 2 + B_res * h)

    # --- Lines (keypoints) ---
    gt_kps = item.sample.keypoints.data
    gt_kp_labels = item.sample.keypoint_labels.data
    if gt_kps.shape[0] > 0:
        gt_kps_pu = gt_kps / patch_size

        line_logits = outputs.lines.pred_logits.data[0]
        line_labels = line_logits.sigmoid().argmax(dim=-1)
        line_centers = outputs.lines.absolute_centers.data[0]
        line_base_dirs = outputs.lines.raw_directions.data[0]
        line_kps = outputs.lines.pred_keypoints.data[0]

        gt_idx, pred_idx = _greedy_match(
            line_labels, gt_kp_labels, line_kps, gt_kps_pu
        )
        if gt_idx is not None:
            gt_idx = gt_idx.to(device)
            pred_idx = pred_idx.to(device)

            base = line_base_dirs[pred_idx]
            cx = line_centers[pred_idx, 0]
            cy = line_centers[pred_idx, 1]
            scale = base.abs() + base_anchor_size
            gx1, gy1, gx2, gy2 = gt_kps_pu[gt_idx].T

            res_x1 = ((gx1 - cx - base[:, 0]) / scale[:, 0]).clamp(-0.5, 0.5)
            res_y1 = ((gy1 - cy - base[:, 1]) / scale[:, 1]).clamp(-0.5, 0.5)
            res_x2 = ((gx2 - cx - base[:, 2]) / scale[:, 2]).clamp(-0.5, 0.5)
            res_y2 = ((gy2 - cy - base[:, 3]) / scale[:, 3]).clamp(-0.5, 0.5)

            line_kps[pred_idx, 0] = cx + base[:, 0] + res_x1 * scale[:, 0]
            line_kps[pred_idx, 1] = cy + base[:, 1] + res_y1 * scale[:, 1]
            line_kps[pred_idx, 2] = cx + base[:, 2] + res_x2 * scale[:, 2]
            line_kps[pred_idx, 3] = cy + base[:, 3] + res_y2 * scale[:, 3]


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
    if args.force_pyvips or device.type != "cuda":
        item_decoded = det_tf.decode_pyvips(item, device=device)
    else:
        item_decoded = det_tf.decode_nvimgcodec(item, device=device)

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

    # Oracle FGL: replace FGL residuals with GT-derived perfect residuals
    if args.oracle_fgl:
        apply_oracle_fgl(
            outputs, item, args.patch_size, args.base_anchor_size
        )

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


@partial_generator
def create_inference_iterator(
    index_gen: Iterable[int],
    dataset,
    args,
    checkpoint_path: Path,
    sym_idx_to_cat_id: dict,
    line_idx_to_cat_id: dict,
    max_symbols: int,
    max_lines: int,
):
    """Generator that runs inference on a specific GPU based on the current thread's worker_id."""
    current_thread = threading.current_thread()
    worker_id = getattr(current_thread, "worker_id", 0)

    if args.device.startswith("cuda"):
        device = torch.device(f"cuda:{worker_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    # Each worker instantiates its own model here
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

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        # Iterate over the shared, thread-safe index generator
        for i in index_gen:
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
            yield sym_res, line_res

            if device.type == "cuda":
                torch.cuda.empty_cache()


def run_inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if args.oracle_fgl:
        print(">>> ORACLE FGL MODE: replacing FGL residuals with GT-derived targets")

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

    # Determine indices to process
    num_images = len(dataset.images)
    if args.num_samples is not None and args.num_samples < num_images:
        indices = random.sample(range(num_images), args.num_samples)
        print(
            f"Randomly subsampled {args.num_samples} images out of {num_images}."
        )
    else:
        indices = list(range(num_images))

    num_gpus = (
        torch.cuda.device_count() if args.device.startswith("cuda") else 1
    )

    # Wrap indices in a ThreadedGenerator for thread-safe, dynamic distribution
    index_gen = ThreadedGenerator(
        iter(indices), maxsize=num_gpus * 2, name="indices"
    )

    # Create the partial generator instance
    inference_gen = create_inference_iterator(
        index_gen,
        dataset,
        args,
        args.checkpoint,
        sym_idx_to_cat_id,
        line_idx_to_cat_id,
        max_symbols,
        max_lines,
    )

    # Use ParallelGenerator
    parallel_gen = ParallelGenerator(
        inference_gen,
        num_workers=num_gpus,
        maxsize=num_gpus * 2,
        name="inference",
    )

    all_sym_results = []
    all_line_results = []

    print("Running inference...")
    with parallel_gen as gen:
        for sym_res, line_res in tqdm(
            gen, total=len(indices), desc="Inference"
        ):
            all_sym_results.extend(sym_res)
            all_line_results.extend(line_res)

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
    parser.add_argument(
        "--force_pyvips",
        action="store_true",
        help="Force the use of pyvips for image decoding, bypassing nvimgcodec.",
    )
    parser.add_argument(
        "--oracle_fgl",
        action="store_true",
        help="Replace FGL residuals with GT-derived perfect residuals (oracle "
        "upper-bound evaluation). Uses GT from the dataset to compute "
        "optimal FGL targets for matched predictions.",
    )

    args = parser.parse_args()

    if args.var_threshold is None and args.drop_rate is None:
        args.var_threshold = 0.001
    elif args.var_threshold is not None and args.drop_rate is not None:
        raise ValueError("Cannot specify both --var_threshold and --drop_rate")

    run_inference(args)
