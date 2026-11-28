import argparse
import json
import random
import threading
import torch
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
from model.association import (
    AssociationConfig,
    Candidates,
    ScoredDetections,
    associate_candidates,
    select_candidates,
)
from model.line_association import (
    LineAssociationConfig,
    associate_lines as associate_line_chains,
    chain_labels,
    endpoint_agreement,
)
from model.regions import expected_voters
from model.detector import OMRDetector, create_detector, has_shared_class, head_layout
from model.refiner import (
    EndpointRefiner,
    LineQueries,
    TokenGrid,
    load_refiner,
    postprocess_lines,
    refine_queries,
)
from dataset.coco import (
    parse_coco,
    load_coco_sample,
    CocoSymbolAnnotation,
    CocoLineAnnotation,
)
import transform.det as det_tf
from music_types import Batch, BoxDim, CoordDim, DetectionOutput, KeypointDim, NumQueries
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


type Outputs = DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim]


def association_config(args: argparse.Namespace, radius: float, normalized: bool,
                       cannot_link: bool, variance_weights: bool = False) -> AssociationConfig:
    return AssociationConfig(radius=radius, normalized=normalized, cannot_link=cannot_link,
                             variance_weights=variance_weights)


def associate_symbols(outputs: Outputs, args: argparse.Namespace, max_symbols: int) -> ScoredDetections:
    """Patch-set inference for symbols: occupied slots (presence >= floor) are associated, the
    distance between two boxes measured relative to their size (corner distance over
    sqrt(2) x width / height), and the clusters ranked by --cluster_score."""
    k = args.num_shapes
    sym = outputs.symbols
    presence = sym.pred_presence
    floor = args.presence_floor if presence is not None else args.symbol_candidate_floor
    cands = select_candidates(sym.pred_logits.data[0], None if presence is None else presence.data[0], floor)
    boxes = sym.pred_boxes.data[0][cands.index].float()
    size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1 / args.patch_size)
    return associate_candidates(
        cands, boxes, torch.cat([size, size], dim=-1), k,
        outputs.lines.absolute_centers.data[0, ::k].float(), False,
        association_config(args, args.symbol_assoc_radius, True, not args.no_cannot_link),
        args.responsibility_margin, args.cluster_score, max_symbols,
    )


def subset_candidates(cands: Candidates, keep: torch.Tensor) -> Candidates:
    return Candidates(index=cands.index[keep], label=cands.label[keep], score=cands.score[keep],
                      presence=cands.presence[keep])


def associate_lines(outputs: Outputs, args: argparse.Namespace, max_lines: int, refiner: EndpointRefiner | None,
                    grid: TokenGrid | None, chained: frozenset[int] = frozenset()) -> ScoredDetections:
    """Patch-set inference for lines: occupied slots, refined when a refiner is given, then
    associated. Classes in ``chained`` (staff, voltaBracket, octave; --line_association hybrid)
    go through collinear chains cut at endpoint-vote modes (``model.line_association``), ranked
    by support x endpoint agreement; the others by the larger endpoint distance
    (--line_assoc_radius pixels), fused with weights score / var of the refiner's bins, ranked
    by --cluster_score. Exp 042 smoke cache: line AP 0.389 greedy, 0.523 hybrid."""
    k = args.num_shapes
    ln = outputs.lines
    presence = ln.pred_presence
    floor = args.presence_floor if presence is not None else args.line_candidate_floor
    cands = select_candidates(ln.pred_logits.data[0], None if presence is None else presence.data[0], floor)
    endpoints = ln.pred_keypoints.data[0][cands.index].float()
    std: torch.Tensor | None = None
    if refiner is not None and grid is not None:
        refined = refine_queries(
            refiner, grid,
            LineQueries(image=torch.zeros_like(cands.index), source=cands.index // k,
                        label=cands.label, endpoints=endpoints),
            args.refine_steps,
        )
        endpoints, std = refined.endpoints, refined.std
    weighted = std is not None and not args.line_score_fusion
    centers = ln.absolute_centers.data[0, ::k].float()
    spread = std if std is not None else torch.ones_like(endpoints) / args.patch_size
    use_chains = args.line_association == "hybrid" and std is not None and len(chained) > 0
    chain_mask = torch.tensor([int(c) in chained for c in cands.label.tolist()], dtype=torch.bool,
                              device=endpoints.device) if use_chains else torch.zeros_like(cands.label, dtype=torch.bool)
    rest = ~chain_mask
    greedy = associate_candidates(
        subset_candidates(cands, rest), endpoints[rest], spread[rest], k, centers, True,
        association_config(args, args.line_assoc_radius / args.patch_size, False, args.line_cannot_link, weighted),
        args.responsibility_margin, args.cluster_score, max_lines,
    )
    if not use_chains:
        return greedy
    chain_cfg = LineAssociationConfig()
    sel = cands.index[chain_mask]
    objects = associate_line_chains(endpoints[chain_mask], spread[chain_mask], cands.label[chain_mask],
                                    cands.score[chain_mask], cands.presence[chain_mask], centers[sel // k], chain_cfg)
    expected = expected_voters(centers, objects.geometry, True, args.responsibility_margin, chunk=256)
    support = (objects.evidence / expected.clamp(min=1).to(objects.evidence)).clamp(max=1.0)
    chain_score = support * endpoint_agreement(objects, chain_cfg)
    geometry = torch.cat([greedy.geometry, objects.geometry])
    score = torch.cat([greedy.score, chain_score])
    label = torch.cat([greedy.label, objects.label])
    top = score.topk(min(max_lines, score.shape[0])).indices
    return ScoredDetections(geometry=geometry[top], score=score[top], label=label[top])


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
    refiner: EndpointRefiner | None = None,
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

    # Forward pass; with a refiner, also keep the final tokens it reads.
    final: list[torch.Tensor] = []

    def keep_final(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        final.append(output)

    norm = model.backbone.transformer.norm
    handle = norm.register_forward_hook(keep_final) if refiner is not None else None
    try:
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=args.use_amp
        ):
            outputs = model(dropped_item.sample.image)
    finally:
        if handle is not None:
            handle.remove()

    # Oracle FGL: replace FGL residuals with GT-derived perfect residuals
    if args.oracle_fgl:
        apply_oracle_fgl(
            outputs, item, args.patch_size, args.base_anchor_size
        )

    # --- Process Symbols ---
    sym_logits = outputs.symbols.pred_logits.data[0]
    sym_boxes = outputs.symbols.pred_boxes.data[0]

    if args.association:
        assoc = associate_symbols(outputs, args, max_symbols)
        sym_boxes_kept, sym_probs_kept, sym_labels_kept = assoc.geometry, assoc.score, assoc.label
    else:
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

    if args.association:
        grid = None
        if refiner is not None:
            patches = dropped_item.sample.image
            _, height, width = patches.image_shape
            ph, pw = patches.patch_size
            grid = TokenGrid.from_batch(final[0].float(), patches.indices, height // ph, width // pw)
        assoc = associate_lines(outputs, args, max_lines, refiner, grid,
                                chain_labels(dataset.line_categories, dataset.line_cat_id_to_idx))
        line_kps_kept, line_probs_kept, line_labels_kept = assoc.geometry.clone(), assoc.score, assoc.label
    elif refiner is not None:
        # Exp 035 pipeline: refine the candidates above the floor, NMS with merged
        # scores, then keep the top lines by final score.
        patches = dropped_item.sample.image
        _, height, width = patches.image_shape
        ph, pw = patches.patch_size
        grid = TokenGrid.from_batch(
            final[0].float(), patches.indices, height // ph, width // pw
        )
        lines = postprocess_lines(
            refiner,
            grid,
            line_logits,
            line_kps,
            num_shapes=args.num_shapes,
            candidate_floor=args.line_candidate_floor,
            nms_radius=(
                args.line_nms_radius / args.patch_size
                if args.line_nms_radius > 0
                else None
            ),
            merge=not args.no_line_merge,
            max_lines=max_lines,
            num_steps=args.refine_steps,
        )
        line_probs_kept = lines.scores
        line_kps_kept = lines.endpoints.clone()
        line_labels_kept = lines.labels
    else:
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

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    num_shapes, presence = head_layout(checkpoint["model"], dataset.num_symbol_classes)
    if num_shapes != args.num_shapes:
        print(f"The checkpoint has {num_shapes} slots per patch (--num_shapes {args.num_shapes} ignored)")
        args.num_shapes = num_shapes

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
        presence=presence,
        symbol_shared_class=has_shared_class(checkpoint["model"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    refiner = None
    if args.refiner_checkpoint is not None:
        refiner = load_refiner(
            args.refiner_checkpoint,
            model.backbone.patch_embed[1].out_features,
            dataset.num_line_classes,
            device,
        )

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
                refiner,
            )
            yield sym_res, line_res

            if device.type == "cuda":
                torch.cuda.empty_cache()


def run_inference(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if args.oracle_fgl:
        print(">>> ORACLE FGL MODE: replacing FGL residuals with GT-derived targets")
    if args.association:
        print(
            f">>> Association: presence >= {args.presence_floor}, lines {args.line_association} "
            f"(endpoint radius {args.line_assoc_radius} px)"
            f"{' (refined)' if args.refiner_checkpoint is not None else ''}, symbols "
            f"{args.symbol_assoc_radius} x size, cannot-link symbols {not args.no_cannot_link} / lines {args.line_cannot_link}, "
            f"score {args.cluster_score}"
        )
    elif args.refiner_checkpoint is not None:
        print(
            f">>> Line pipeline: refiner {args.refiner_checkpoint}, candidates >= "
            f"{args.line_candidate_floor}, NMS {args.line_nms_radius} px, "
            f"merge {not args.no_line_merge}, steps {args.refine_steps or 'trained'}"
        )

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

    parser.add_argument(
        "--refiner_checkpoint",
        type=Path,
        default=None,
        help="Exp 035 endpoint refiner (refiner.pt). Enables the line pipeline: refine the "
        "candidates above --line_candidate_floor, NMS, noisy-OR merged scores.",
    )
    parser.add_argument(
        "--line_candidate_floor",
        type=float,
        default=0.05,
        help="With a refiner: line queries scoring at least this much are refined.",
    )
    parser.add_argument(
        "--line_nms_radius",
        type=float,
        default=4.0,
        help="With a refiner: a line suppresses same-class lines whose two endpoints are "
        "both within this many pixels (<= 0 disables NMS).",
    )
    parser.add_argument(
        "--no_line_merge",
        action="store_true",
        help="With a refiner: keep the kept line's own score instead of the noisy-OR of "
        "the lines it suppressed.",
    )
    parser.add_argument(
        "--refine_steps",
        type=int,
        default=None,
        help="With a refiner: number of refinement steps (default: the trained number).",
    )

    assoc = parser.add_argument_group(
        "association",
        "Patch-set inference (PATCH_SET_ASSIGNMENT.md section 5): occupied slots are grouped "
        "into clusters (two slots of one patch never share a cluster) instead of top-k / NMS.",
    )
    assoc.add_argument("--association", action="store_true")
    assoc.add_argument(
        "--presence_floor", type=float, default=0.5, help="Occupied slot: presence >= this."
    )
    assoc.add_argument(
        "--symbol_candidate_floor",
        type=float,
        default=0.01,
        help="Without a presence logit: symbol slots scoring at least this much.",
    )
    assoc.add_argument("--line_assoc_radius", type=float, default=4.0, help="Pixels.")
    assoc.add_argument(
        "--line_association",
        choices=["hybrid", "greedy"],
        default="hybrid",
        help="hybrid: staff / voltaBracket / octave by collinear chains cut at endpoint "
        "modes, other classes by endpoint distance; greedy: endpoint distance for all.",
    )
    assoc.add_argument(
        "--symbol_assoc_radius",
        type=float,
        default=0.35,
        help="Relative to symbol size (corner distance / (sqrt(2) x width or height)).",
    )
    assoc.add_argument(
        "--cluster_score", choices=["support_quality", "quality", "noisy_or"], default="support_quality"
    )
    assoc.add_argument("--responsibility_margin", type=float, default=0.5)
    assoc.add_argument("--no_cannot_link", action="store_true", help="Symbols without cannot-link.")
    assoc.add_argument(
        "--line_score_fusion",
        action="store_true",
        help="Fuse refined line endpoints by score only (default: score / variance of the refiner bins).",
    )
    assoc.add_argument(
        "--line_cannot_link",
        action="store_true",
        help="Lines with cannot-link (off: patch-set lines train several on-line slots per patch).",
    )

    args = parser.parse_args()
    if args.refiner_checkpoint is not None and args.oracle_fgl:
        raise ValueError("--oracle_fgl and --refiner_checkpoint cannot be combined")

    if args.var_threshold is None and args.drop_rate is None:
        args.var_threshold = 0.001
    elif args.var_threshold is not None and args.drop_rate is not None:
        raise ValueError("Cannot specify both --var_threshold and --drop_rate")

    run_inference(args)
