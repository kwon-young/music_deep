import argparse
import json
import time
import math
import random
import os
from typing import cast
from pathlib import Path
from typing import Generator, Iterable
from dataclasses import dataclass, replace
import itertools
from itertools import batched
from contextlib import ExitStack, nullcontext
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.assignment import PatchSetAssigner, PatchSetAssignment, PatchSetConfig, SlotStats, slot_stats
from model.detector import DetectorWithTokens, OMRDetector, create_detector, load_detector_state
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion, PatchSetCriterion, VoterWeighting
from model.refiner import (
    DenoisingNoise,
    EndpointRefiner,
    TokenGrid,
    load_refiner,
    refiner_batch,
    refiner_loss,
)
from dataset.coco import parse_coco, load_coco_sample, CocoMetadata, CocoDataset
from dataset.split import split_by_piece
import transform.det as det_tf
from threaded_generator import (
    Monitor,
    ParallelGenerator,
    PartialGenerator,
    ThreadedGenerator,
)
from dataset.synthetic import (
    SYMBOL_NAMES,
    GlyphAtlas,
    SyntheticConfig,
    SyntheticPageGenerator,
    page_to_sample,
)
from metric import compute_map_50, compute_mean_iou, compute_mean_endpoint_error
from visualization import update_plot, reconstruct_image_from_patches
from logger import ExperimentLogger, BaseMetrics
from music_types import (
    DetectionTarget,
    DetectionOutput,
    DetectionLosses,
    DetectionLossWeights,
    Data,
    DetectionSample,
    TensorImage,
    CHW,
    RGB,
    Float1,
    Int255,
    BoundingBoxes,
    ClassLabels,
    Keypoints,
    BatchedData,
    Patches,
    Batch,
    NumPatches,
    PatchDim,
    EmbedDim,
    Embeddings,
    MatchIndices,
    NumQueries,
    BoxDim,
    KeypointDim,
    CoordDim,
)


@dataclass
class DetectionMetrics(BaseMetrics):
    epoch: float
    lr: float
    loss_total: float
    loss_ce: float
    loss_bbox: float
    loss_giou: float
    loss_fgl: float
    loss_line_l1: float
    loss_line_fgl: float
    loss_presence: float
    loss_refine: float
    refine_coarse_l1: float
    refine_l1: float
    refine_masked: float
    sym_presence_pos_high: float
    sym_presence_neg_high: float
    sym_inpatch_dupes: float
    sym_ignored: float
    sym_voters: float
    line_presence_pos_high: float
    line_presence_neg_high: float
    line_ignored: float
    line_voters: float
    map50: float
    miou: float
    line_error: float
    speed: float


@dataclass(frozen=True)
class RefinerParams:
    """Exp 038 (Exp 035 Phase B): the line endpoint refiner trained inside the detector step."""

    init: Path | None
    num_steps: int
    heads: int
    scale: float
    loss_weight: float
    lr_scale: float
    fgl_weight: float
    max_coarse_error: float
    noise: DenoisingNoise
    detach_tokens: bool


@dataclass(frozen=True)
class RefineStats:
    """Refiner loss and, on the real (matched) queries, the masked mean L1 per coordinate of
    the coarse endpoints and of the last refinement step, in patches."""

    loss: float
    coarse_l1: float
    refined_l1: float
    masked: float


NO_REFINE = RefineStats(loss=0.0, coarse_l1=0.0, refined_l1=0.0, masked=0.0)


@dataclass(frozen=True)
class PatchSetParams:
    """Patch-set assignment (PATCH_SET_ASSIGNMENT.md): one-to-one per patch, one-to-many
    across patches, presence logit, quality focal class loss, 1 / N_voters weights."""

    config: PatchSetConfig
    voter_weighting: VoterWeighting
    presence_weighting: VoterWeighting
    presence_alpha: float
    symbol_quality_targets: bool
    line_quality_targets: bool
    refiner_voters: int


NO_SLOT_STATS = SlotStats(positive_high=0.0, negative_high=0.0, duplicates=0.0, ignored=0.0, voters=0.0)


@dataclass
class TrainParams:
    anno_path: Path
    cache_dir: Path | None
    img_dir: Path
    dataset: CocoDataset
    num_images: int | None
    batch_size: int
    accumulation_steps: int
    loader_maxsize: int
    train_maxsize: int
    backbone_size: str
    patch_size: int
    extract_patch_min: int
    extract_patch_max: int
    crop_size: int | None
    channels: int
    num_symbol_classes: int
    num_line_classes: int
    num_shapes: int
    base_anchor_size: float
    cost_class: float
    cost_bbox: float
    cost_giou: float
    radius_patches: float
    top_k: int
    matcher_type: str
    loss_ce: float
    loss_bbox: float
    loss_giou: float
    loss_fgl: float
    loss_line_l1: float
    lr: float
    warmup_epochs: float
    min_lr_ratio: float
    running_loss_half_life: float
    epochs: int
    log_epoch_interval: float
    var_threshold: float | None
    drop_rate: float | None
    topk_patches: int | None
    affine: bool
    morphological_downscale: float | None
    max_translate_frac: float
    max_angle_deg: float
    max_shear_deg: float
    max_scale: float
    log_patches: bool
    compile: bool
    use_sdpa: bool
    use_amp: bool
    prep_device: torch.device
    train_device: torch.device
    match_device: torch.device
    backbone_checkpoint: Path | None
    detector_checkpoint: Path | None
    freeze_backbone: bool
    exp_dir: Path
    stage_name: str
    headless: bool
    use_monitor: bool
    synthetic: SyntheticConfig | None
    synthetic_workers: int
    verovio_root: Path
    refiner: RefinerParams | None
    held_out_image_ids: list[int]
    patch_set: PatchSetParams | None = None
    loss_presence: float = 2.0
    head_lr_scale: float = 1.0


@dataclass
class TrainStepResult:
    step: int
    epoch: float
    lr: float
    batch: BatchedData
    targets: list[DetectionTarget]
    outputs: DetectionOutput
    losses: DetectionLosses
    total_loss: float
    symbols_processed: int
    refine: RefineStats


type DecodedItem = Data[
    CocoMetadata,
    DetectionSample[
        TensorImage[CHW, RGB, Int255],
        BoundingBoxes,
        ClassLabels,
        Keypoints,
        ClassLabels,
    ],
]
type TransformedItem = Data[
    CocoMetadata,
    DetectionSample[
        TensorImage[CHW, RGB, Float1],
        BoundingBoxes,
        ClassLabels,
        Keypoints,
        ClassLabels,
    ],
]


def augment_item(item: DecodedItem, params: TrainParams) -> TransformedItem:
    """Float1 -> optional morphological downscale -> optional affine."""
    item_tf = det_tf.to_float1(item)

    if params.morphological_downscale is not None:
        item_tf = det_tf.random_morphological_downscale(
            item_tf, max_scale_factor=params.morphological_downscale
        )

    if params.affine:
        return det_tf.random_affine(
            item_tf,
            max_translate_frac=params.max_translate_frac,
            max_angle_deg=params.max_angle_deg,
            max_shear_deg=params.max_shear_deg,
            max_scale=params.max_scale,
        )
    return item_tf


def transform_image(index: int, params: TrainParams) -> TransformedItem:
    """Preprocessing of a real page: Load -> Decode/Crop -> augment_item."""
    prep_device = params.prep_device
    crop_size = params.crop_size
    item = load_coco_sample(params.dataset, params.img_dir, index, prep_device)

    if prep_device.type == "cuda":
        item_decoded = det_tf.decode_nvimgcodec(item, device=prep_device)
        if crop_size is not None:
            item_cropped = det_tf.random_crop(item_decoded, crop_size=crop_size)
        else:
            item_cropped = item_decoded
    else:
        if crop_size is not None:
            item_cropped = det_tf.decode_and_crop_pyvips(
                item, crop_size=crop_size, device=prep_device
            )
        else:
            item_cropped = det_tf.decode_pyvips(item, device=prep_device)

    return augment_item(item_cropped, params)


def real_items(params: TrainParams) -> Generator[TransformedItem]:
    """Real pages, reshuffled at every epoch, forever."""
    indices = list(range(len(params.dataset.images)))
    while True:
        random.shuffle(indices)
        for idx in indices:
            yield transform_image(idx, params)


def synthetic_items(
    atlas: GlyphAtlas, config: SyntheticConfig, params: TrainParams
) -> Generator[TransformedItem]:
    """Freshly generated synthetic pages, forever (one generator per thread)."""
    generator = SyntheticPageGenerator(atlas, config)
    for index in itertools.count():
        sample = page_to_sample(generator.generate(), index, params.prep_device)
        yield augment_item(sample, params)


def parallel_synthetic_items(
    config: SyntheticConfig, params: TrainParams, rank: int
) -> ParallelGenerator[TransformedItem]:
    """Synthetic pages from ``params.synthetic_workers`` threads.

    The atlas is shared and fully rendered first (FreeType is not thread-safe);
    each thread gets its own page generator, with seeds far enough apart that no
    two threads or ranks generate the same page.
    """
    atlas = GlyphAtlas(verovio_root=params.verovio_root, fonts=config.fonts)
    atlas.prerender(SYMBOL_NAMES)
    workers = itertools.count()

    def worker_items() -> Generator[TransformedItem]:
        worker = next(workers)
        seed = config.seed + (rank * 64 + worker) * 1_000_000_000
        return synthetic_items(atlas, replace(config, seed=seed), params)

    return ParallelGenerator(
        PartialGenerator(worker_items),
        num_workers=params.synthetic_workers,
        maxsize=2 * params.synthetic_workers,
        name="synthetic_pages",
    )


def log_patch_count(iterator, enabled: bool):
    """Pass-through generator that logs the number of patches if enabled."""
    for item in iterator:
        if enabled:
            num_patches = item.sample.image.data.shape[1]
            print(f"[Pipeline] Number of patches fed to model: {num_patches}")
        yield item


def create_detection_iterator(
    params: TrainParams, rank: int = 0
) -> Generator[
    BatchedData[
        CocoMetadata,
        DetectionSample[
            Patches[Batch, NumPatches, PatchDim],
            list[BoundingBoxes],
            list[ClassLabels],
            list[Keypoints],
            list[ClassLabels],
        ],
    ],
    None,
    None,
]:
    """Creates an infinite Python generator pipeline for detection data."""
    items: Iterable[TransformedItem] = (
        parallel_synthetic_items(params.synthetic, params, rank)
        if params.synthetic is not None
        else real_items(params)
    )

    def _pipeline():
        # Chunk into batches, collate, and extract patches
        for batch_items in batched(items, params.batch_size):
            # 1. Collate, pad to batch max, and extract patches with a random size
            patched_item = det_tf.random_extract_patches_and_collate(
                batch_items,
                min_patch_size=params.extract_patch_min,
                max_patch_size=params.extract_patch_max,
            )

            # 2. Normalize targets to Patch Units (PU) using the extracted patch size
            norm_item = det_tf.normalize_batched_targets(patched_item)

            # 3. Drop patches based on variance
            dropped_item = det_tf.variance_patch_drop(
                norm_item,
                var_threshold=params.var_threshold,
                drop_rate=params.drop_rate,
                topk=params.topk_patches,
            )

            # 4. Resize the surviving patches to the ViT's static capacity (64x64)
            resized_item = det_tf.resize_patches(
                dropped_item,
                target_patch_size=(params.patch_size, params.patch_size),
            )

            # 5. Move to GPU
            final_item = det_tf.to_patches(
                resized_item, device=params.train_device
            )
            yield final_item

    return log_patch_count(_pipeline(), params.log_patches)


def refinement_loss(
    refiner: EndpointRefiner | DDP,
    raw_refiner: EndpointRefiner,
    features: Embeddings[Batch, NumPatches, EmbedDim],
    outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
    line_matches: list[MatchIndices],
    targets: list[DetectionTarget],
    num_shapes: int,
    cfg: RefinerParams,
    max_voters: int | None = None,
) -> tuple[torch.Tensor, RefineStats]:
    """Refiner loss on the step's own (augmented, resized) tokens and line matching.

    With ``cfg.detach_tokens`` the refiner only reads the tokens; otherwise its gradient also
    reaches the backbone (joint training). Coordinates stay in the extraction-patch units of
    the batch, so the window and the +-scale/2 correction are in the same units as at
    inference.
    """
    tokens = features.data.detach() if cfg.detach_tokens else features.data
    _, height, width = features.image_shape
    patch_h, patch_w = features.patch_size
    grid = TokenGrid.from_batch(tokens, features.indices, height // patch_h, width // patch_w)
    batch = refiner_batch(
        num_kept=tokens.shape[1],
        logits=outputs.lines.pred_logits.data,
        keypoints=outputs.lines.pred_keypoints.data,
        matches=line_matches,
        gt_keypoints=[t.keypoints.data for t in targets],
        gt_labels=[t.keypoint_labels.data for t in targets],
        num_shapes=num_shapes,
        max_coarse_error=cfg.max_coarse_error,
        noise=cfg.noise,
        centers=outputs.lines.absolute_centers.data,
        max_voters=max_voters,
    )
    refined = refiner(grid, batch.queries)
    loss, _ = refiner_loss(raw_refiner, refined, batch.target, batch.mask, cfg.fgl_weight)
    with torch.no_grad():
        n = batch.num_real
        mask = batch.mask[:n]
        denom = mask.sum().clamp(min=1.0)
        target = batch.target[:n]
        coarse_l1 = ((batch.queries.endpoints[:n] - target).abs() * mask).sum() / denom
        refined_l1 = ((refined[-1].endpoints[:n] - target).abs() * mask).sum() / denom
    return loss, RefineStats(
        loss=loss.item(), coarse_l1=coarse_l1.item(), refined_l1=refined_l1.item(),
        masked=1.0 - mask.mean().item() if n else 0.0,
    )


def train_step_pipeline(
    iterator: Iterable[
        BatchedData[
            CocoMetadata,
            DetectionSample[
                Patches[Batch, NumPatches, PatchDim],
                list[BoundingBoxes],
                list[ClassLabels],
                list[Keypoints],
                list[ClassLabels],
            ],
        ]
    ],
    model: DetectorWithTokens | DDP,
    matcher: HungarianMatcher | PatchSetAssigner,
    criterion: DFINECriterion,
    refiner: EndpointRefiner | DDP | None,
    raw_refiner: EndpointRefiner | None,
    optimizer: optim.Optimizer,
    params: TrainParams,
    total_symbol_budget: int,
    dataset_symbols: int,
    is_distributed: bool,
) -> Generator[TrainStepResult, None, None]:
    """Executes the forward/backward pass and yields detached results for the main thread."""
    cumulative_symbols = 0
    global_step = 0

    scaler = torch.amp.GradScaler(
        device=params.train_device.type, enabled=params.use_amp
    )

    optimizer.zero_grad()

    for batch in iterator:
        if cumulative_symbols >= total_symbol_budget:
            break

        global_step += 1
        is_accumulating = (global_step % params.accumulation_steps) != 0

        targets = [
            DetectionTarget(
                boxes=b, box_labels=bl, keypoints=k, keypoint_labels=kl
            )
            for b, bl, k, kl in zip(
                batch.sample.boxes,
                batch.sample.box_labels,
                batch.sample.keypoints,
                batch.sample.keypoint_labels,
            )
        ]

        refine_stats = NO_REFINE
        # Prevent DDP from synchronizing gradients during accumulation steps
        with ExitStack() as sync_context:
            if is_accumulating:
                for module in (model, refiner):
                    if isinstance(module, DDP):
                        sync_context.enter_context(module.no_sync())
            with torch.autocast(
                device_type=params.train_device.type,
                dtype=torch.float16,
                enabled=params.use_amp,
            ):
                outputs, features = model(batch.sample.image)
                # One matching, shared by the detector losses and the refiner queries.
                if isinstance(matcher, PatchSetAssigner):
                    assignment = matcher(outputs, targets)
                    losses = criterion(outputs, targets, assignment=assignment)
                    line_matches = assignment.matches()[1]
                else:
                    indices = matcher(outputs, targets)
                    losses = criterion(outputs, targets, indices)
                    line_matches = indices[1]
                total_loss = losses.total()
                if refiner is not None:
                    assert raw_refiner is not None and params.refiner is not None
                    refine_loss, refine_stats = refinement_loss(
                        refiner,
                        raw_refiner,
                        features,
                        outputs,
                        line_matches,
                        targets,
                        params.num_shapes,
                        params.refiner,
                        None if params.patch_set is None else params.patch_set.refiner_voters,
                    )
                    total_loss = total_loss + params.refiner.loss_weight * refine_loss

            # Scale the loss to average gradients over the accumulation steps
            scaled_loss = total_loss / params.accumulation_steps
            scaler.scale(scaled_loss).backward()

        # --- Synchronize Symbol Count ---
        local_symbols = batch.sample.num_symbols
        if is_distributed:
            sym_tensor = torch.tensor(
                [local_symbols], dtype=torch.long, device=params.train_device
            )
            dist.all_reduce(sym_tensor, op=dist.ReduceOp.SUM)
            step_symbols = int(sym_tensor.item())
        else:
            step_symbols = local_symbols

        cumulative_symbols += step_symbols
        current_epoch = cumulative_symbols / dataset_symbols

        # --- Symbol Budget LR Scheduler ---
        warmup_symbols = params.warmup_epochs * dataset_symbols
        if cumulative_symbols < warmup_symbols:
            # Linear Warmup
            current_lr = params.lr * max(
                params.min_lr_ratio, (cumulative_symbols / warmup_symbols)
            )
        else:
            # Cosine Decay
            cosine_progress = (cumulative_symbols - warmup_symbols) / (
                total_symbol_budget - warmup_symbols
            )
            current_lr = (
                params.lr * 0.5 * (1 + math.cos(math.pi * cosine_progress))
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr * float(param_group.get("lr_scale", 1.0))
        # ----------------------------------

        if not is_accumulating:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Cleanly detach the entire nested dataclass structures
        yield TrainStepResult(
            step=global_step,
            epoch=current_epoch,
            lr=current_lr,
            batch=batch,
            targets=targets,
            outputs=outputs.detach(),
            losses=losses.detach(),
            total_loss=total_loss.item(),
            symbols_processed=step_symbols,
            refine=refine_stats,
        )


def patch_set_stats(
    result: TrainStepResult, assignment: PatchSetAssignment, assigner: PatchSetAssigner
) -> tuple[SlotStats, SlotStats]:
    """Presence and in-patch duplicate diagnostics of the logged step's first image."""
    sym, line = result.outputs.symbols, result.outputs.lines
    if sym.pred_presence is None or line.pred_presence is None:
        return NO_SLOT_STATS, NO_SLOT_STATS
    t = result.targets[0]
    k, cfg = assigner.num_slots, assigner.cfg
    return (
        slot_stats(sym.pred_presence.data[0], sym.pred_boxes.data[0], assignment.symbols[0],
                   t.boxes.data, False, k, cfg),
        slot_stats(line.pred_presence.data[0], line.pred_keypoints.data[0], assignment.lines[0],
                   t.keypoints.data, True, k, cfg),
    )


def log_and_save_checkpoint(
    result: TrainStepResult,
    params: TrainParams,
    matcher: HungarianMatcher | PatchSetAssigner,
    logger: ExperimentLogger,
    model: OMRDetector,
    refiner: EndpointRefiner | None,
    optimizer: optim.Optimizer,
    running_loss: float,
    speed: float,
    ax,
    fig,
    plt,
):
    """Handles metrics computation, visualization, and saving the model state."""
    # 1. Compute Metrics
    sym_stats = line_stats = NO_SLOT_STATS
    with torch.no_grad():
        if isinstance(matcher, PatchSetAssigner):
            assignment = matcher(result.outputs, result.targets)
            sym_indices, line_indices = assignment.matches()
            sym_stats, line_stats = patch_set_stats(result, assignment, matcher)
        else:
            sym_indices, line_indices = matcher(result.outputs, result.targets)
        map50 = compute_map_50(
            result.outputs, result.targets, params.num_symbol_classes
        )
        miou = compute_mean_iou(result.outputs, result.targets, sym_indices)
        line_error = compute_mean_endpoint_error(
            result.outputs, result.targets, line_indices
        )

    metrics = DetectionMetrics(
        step=result.step,
        epoch=result.epoch,
        lr=result.lr,
        loss_total=result.total_loss,
        loss_ce=result.losses.loss_ce.item(),
        loss_bbox=result.losses.loss_bbox.item(),
        loss_giou=result.losses.loss_giou.item(),
        loss_fgl=result.losses.loss_fgl.item(),
        loss_line_l1=result.losses.loss_line_l1.item(),
        loss_line_fgl=result.losses.loss_line_fgl.item(),
        loss_presence=result.losses.loss_presence.item(),
        loss_refine=result.refine.loss,
        refine_coarse_l1=result.refine.coarse_l1,
        refine_l1=result.refine.refined_l1,
        refine_masked=result.refine.masked,
        sym_presence_pos_high=sym_stats.positive_high,
        sym_presence_neg_high=sym_stats.negative_high,
        sym_inpatch_dupes=sym_stats.duplicates,
        sym_ignored=sym_stats.ignored,
        sym_voters=sym_stats.voters,
        line_presence_pos_high=line_stats.positive_high,
        line_presence_neg_high=line_stats.negative_high,
        line_ignored=line_stats.ignored,
        line_voters=line_stats.voters,
        map50=map50,
        miou=miou,
        line_error=line_error,
        speed=speed,
    )
    logger.log_metrics(metrics)

    print(
        f"Epoch [{result.epoch:.2f}/{params.epochs}] Step [{result.step}] "
        f"LR: {result.lr:.2e} | "
        f"Loss: {result.total_loss:.4f} (Running: {running_loss:.4f}) | "
        f"CE: {result.losses.loss_ce.item():.4f} | BBox: {result.losses.loss_bbox.item():.4f} | "
        f"GIoU: {result.losses.loss_giou.item():.4f} | FGL: {result.losses.loss_fgl.item():.4f} | "
        f"LineL1: {result.losses.loss_line_l1.item():.4f} | LineFGL: {result.losses.loss_line_fgl.item():.4f} | "
        + (
            f"Presence: {result.losses.loss_presence.item():.4f} (pos>0.5 {sym_stats.positive_high:.2f}/"
            f"{line_stats.positive_high:.2f}, neg>0.5 {sym_stats.negative_high:.4f}/{line_stats.negative_high:.4f}, "
            f"symbol dupes {sym_stats.duplicates:.3f}, voters {sym_stats.voters:.1f}/"
            f"{line_stats.voters:.1f}) | "
            if params.patch_set is not None
            else ""
        )
        + f"mAP@0.5: {map50:.4f} | mIoU: {miou:.4f} | LineErr: {line_error:.4f} | "
        + (
            f"Refine: {result.refine.loss:.4f} (L1 coarse {result.refine.coarse_l1:.4f} -> "
            f"{result.refine.refined_l1:.4f}, masked {result.refine.masked:.2f}) | "
            if refiner is not None
            else ""
        )
        + f"Speed: {speed:.1f} symbols/s"
    )

    # 2. Visualization
    img_tensor = reconstruct_image_from_patches(result.batch.sample.image)
    update_plot(
        ax,
        img_tensor,
        result.targets,
        result.outputs,
        f"Epoch: {result.epoch:.2f}",
        patch_size=params.patch_size,
        indices=(sym_indices, line_indices),
    )

    vis_path = (
        logger.get_visualizations_dir()
        / f"epoch_{int(result.epoch):03d}_step_{result.step:05d}.png"
    )
    plt.savefig(vis_path, dpi=150)

    if not params.headless:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    # 3. Save Checkpoint
    checkpoint = {
        "epoch": result.epoch,
        "step": result.step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "running_loss": running_loss,
    }
    if refiner is not None:
        # load_refiner() reads these two keys, so inference_coco.py --refiner_checkpoint can
        # point at this file directly.
        checkpoint["refiner"] = refiner.state_dict()
        checkpoint["refiner_config"] = refiner.config()
    checkpoint_path = logger.get_checkpoint_dir() / "latest_model.pt"
    torch.save(checkpoint, checkpoint_path)


def train(params: TrainParams):
    # --- DDP Setup & Device Override ---
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # Force all operations to the local GPU
        params.prep_device = device
        params.train_device = device
        params.match_device = device
    else:
        global_rank = 0
        local_rank = 0

    is_main_process = global_rank == 0
    # ----------------------------------------

    if params.headless:
        os.environ["MPLBACKEND"] = "Agg"

    import matplotlib

    if params.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if is_main_process:
        print(f"Using prep_device: {params.prep_device}")
        print(f"Using train_device: {params.train_device}")
        print(f"Using match_device: {params.match_device}")
        print(f"Learning Rate: {params.lr:.2e}")

    logger = (
        ExperimentLogger(params.exp_dir, params.stage_name)
        if is_main_process
        else None
    )

    # 1. Setup Model
    raw_model = create_detector(
        backbone_size=params.backbone_size,
        patch_size=params.patch_size,
        channels=params.channels,
        use_sdpa=params.use_sdpa,
        num_symbol_classes=params.num_symbol_classes,
        num_line_classes=params.num_line_classes,
        num_shapes=params.num_shapes,
        base_anchor_size=params.base_anchor_size,
        presence=params.patch_set is not None,
    ).to(params.train_device)

    loaded_checkpoint = None
    if params.detector_checkpoint:
        assert params.detector_checkpoint.exists()
        if is_main_process:
            print(
                f"Resuming from full detector checkpoint: {params.detector_checkpoint}"
            )
        loaded_checkpoint = torch.load(
            params.detector_checkpoint,
            map_location=params.train_device,
            weights_only=True,
        )
        if load_detector_state(raw_model, loaded_checkpoint["model"]):
            # Another slot count or presence layout (Exp 037 -> patch-set): the head was
            # remapped, so the checkpoint's Adam moments no longer fit its parameters.
            if is_main_process:
                print(
                    f"Remapped the head output layers to K={params.num_shapes}"
                    f"{' with presence' if params.patch_set is not None else ''}; "
                    "the optimizer state is not loaded."
                )
            loaded_checkpoint.pop("optimizer", None)
    elif params.backbone_checkpoint and params.backbone_checkpoint.exists():
        if is_main_process:
            print(f"Loading backbone weights from {params.backbone_checkpoint}")
        checkpoint = torch.load(
            params.backbone_checkpoint,
            map_location=params.train_device,
            weights_only=True,
        )
        raw_model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    elif params.backbone_checkpoint:
        if is_main_process:
            print(
                f"Warning: Checkpoint {params.backbone_checkpoint} not found. Training from scratch."
            )

    if params.freeze_backbone:
        if is_main_process:
            print("Freezing backbone parameters (no fine-tuning).")
        for param in raw_model.backbone.parameters():
            param.requires_grad = False
    else:
        if is_main_process:
            print("Fine-tuning backbone parameters.")

    raw_refiner: EndpointRefiner | None = None
    refiner: EndpointRefiner | DDP | None = None
    if params.refiner is not None:
        dim = raw_model.backbone.patch_embed[1].out_features
        if loaded_checkpoint is not None and "refiner" in loaded_checkpoint:
            assert params.detector_checkpoint is not None
            source = f"checkpoint {params.detector_checkpoint}"
            raw_refiner = load_refiner(
                params.detector_checkpoint, dim, params.num_line_classes, params.train_device
            )
        elif params.refiner.init is not None:
            source = f"init {params.refiner.init}"
            raw_refiner = load_refiner(
                params.refiner.init, dim, params.num_line_classes, params.train_device
            )
        else:
            source = "scratch"
            raw_refiner = EndpointRefiner(
                dim=dim,
                num_classes=params.num_line_classes,
                num_steps=params.refiner.num_steps,
                heads=params.refiner.heads,
                scale=params.refiner.scale,
            ).to(params.train_device)
        raw_refiner.train()
        refiner = (
            DDP(raw_refiner, device_ids=[local_rank], output_device=local_rank)
            if is_distributed
            else raw_refiner
        )
        if is_main_process:
            print(
                f"Line endpoint refiner from {source}: {raw_refiner.config()}, "
                f"loss weight {params.refiner.loss_weight}, LR x{params.refiner.lr_scale}, "
                f"{'detached tokens' if params.refiner.detach_tokens else 'joint (gradients reach the backbone)'}"
            )

    model: DetectorWithTokens | DDP
    # --- Wrap Model in DDP ---
    with_tokens = DetectorWithTokens(raw_model)
    if is_distributed:
        model = DDP(
            with_tokens, device_ids=[local_rank], output_device=local_rank
        )
    else:
        model = with_tokens
    # ------------------------------

    if params.compile:
        # topk yields static shapes, while var_threshold and drop_rate yield dynamic shapes
        use_dynamic = params.topk_patches is None
        if is_main_process:
            print(
                f"Compiling model with torch.compile(dynamic={use_dynamic})..."
            )
        model = cast(
            DetectorWithTokens | DDP, torch.compile(model, dynamic=use_dynamic)
        )

    # 2. Setup Matcher and Criterion
    weights = DetectionLossWeights(
        loss_ce=params.loss_ce,
        loss_bbox=params.loss_bbox,
        loss_giou=params.loss_giou,
        loss_fgl=params.loss_fgl,
        loss_line_l1=params.loss_line_l1,
        loss_presence=params.loss_presence,
    )
    matcher: HungarianMatcher | PatchSetAssigner
    criterion: DFINECriterion
    if params.patch_set is not None:
        matcher = PatchSetAssigner(params.patch_set.config, params.num_shapes)
        criterion = PatchSetCriterion(
            num_symbol_classes=params.num_symbol_classes,
            num_line_classes=params.num_line_classes,
            weights=weights,
            symbol_weights=params.dataset.symbol_weights,
            line_weights=params.dataset.line_weights,
            base_anchor_size=params.base_anchor_size,
            voter_weighting=params.patch_set.voter_weighting,
            presence_weighting=params.patch_set.presence_weighting,
            presence_alpha=params.patch_set.presence_alpha,
            symbol_quality_targets=params.patch_set.symbol_quality_targets,
            line_quality_targets=params.patch_set.line_quality_targets,
            line_perp_sigma=params.patch_set.config.line_perp_sigma,
            line_thickness=params.patch_set.config.line_thickness,
        ).to(params.train_device)
        if is_main_process:
            print(f"Patch-set assignment: {params.patch_set}")
    else:
        matcher = HungarianMatcher(
            cost_class=params.cost_class,
            cost_bbox=params.cost_bbox,
            cost_giou=params.cost_giou,
            calc_device=params.match_device,
            radius_patches=params.radius_patches,
            top_k=params.top_k,
            matcher_type=params.matcher_type,
        )
        criterion = DFINECriterion(
            matcher,
            num_symbol_classes=params.num_symbol_classes,
            num_line_classes=params.num_line_classes,
            weights=weights,
            symbol_weights=params.dataset.symbol_weights,
            line_weights=params.dataset.line_weights,
            base_anchor_size=params.base_anchor_size,
        ).to(params.train_device)

    if params.head_lr_scale != 1.0:
        # The head output layers in their own group (new presence rows and slots, which start
        # from zero weights: Adam moves a parameter by at most ~lr per step).
        head_out = [
            p for m in (raw_model.symbol_head.mlp[-1], raw_model.line_head.mlp[-1]) for p in m.parameters()
        ]
        head_ids = {id(p) for p in head_out}
        groups: list[dict[str, list[torch.nn.Parameter] | float]] = [
            {"params": [p for p in raw_model.parameters() if id(p) not in head_ids]},
            {
                "params": head_out,
                "lr": params.lr * params.head_lr_scale,
                "lr_scale": params.head_lr_scale,
            },
        ]
        optimizer = optim.AdamW(groups, lr=params.lr)
    else:
        optimizer = optim.AdamW(raw_model.parameters(), lr=params.lr)

    def add_refiner_group() -> None:
        if raw_refiner is not None and params.refiner is not None:
            optimizer.add_param_group(
                {
                    "params": list(raw_refiner.parameters()),
                    "lr": params.lr * params.refiner.lr_scale,
                    "lr_scale": params.refiner.lr_scale,
                }
            )

    # A checkpoint saved with the refiner has its group in the optimizer state: add the group
    # first so the state loads into it. Otherwise (e.g. Exp 037 -> Exp 038) the detector state
    # loads alone and the refiner starts with a fresh Adam state.
    checkpoint_has_refiner = loaded_checkpoint is not None and "refiner" in loaded_checkpoint
    if checkpoint_has_refiner:
        add_refiner_group()
    if loaded_checkpoint is not None and "optimizer" in loaded_checkpoint:
        if is_main_process:
            print("Loading optimizer state from checkpoint...")
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])
    if not checkpoint_has_refiner:
        add_refiner_group()

    # Setup Interactive Plotting ONLY on main process
    if is_main_process:
        if not params.headless:
            plt.ion()
        fig, ax = plt.subplots(1, figsize=(8, 8))
        if not params.headless:
            manager = fig.canvas.manager
            if manager:
                manager.set_window_title("OMR Detector Training")
    else:
        fig, ax = None, None

    dataset_symbols = params.dataset.num_symbols
    if is_main_process:
        print(f"Total symbols in COCO dataset: {dataset_symbols} (True Epoch)")
        if params.held_out_image_ids:
            held_out_path = params.exp_dir / "held_out_image_ids.json"
            held_out_path.write_text(json.dumps(params.held_out_image_ids))
            print(
                f"Training on {len(params.dataset.images)} pages; "
                f"{len(params.held_out_image_ids)} held-out pages listed in {held_out_path}"
            )

    model.train()

    total_symbol_budget = dataset_symbols * params.epochs
    if is_main_process:
        print("Training on full dataset.")
        print(
            f"Total Symbol Budget for {params.epochs} epochs: {total_symbol_budget}"
        )

    monitor = Monitor() if (params.use_monitor and is_main_process) else None

    # 1. Data loading thread
    data_iterator = ThreadedGenerator(
        create_detection_iterator(params, global_rank),
        maxsize=params.loader_maxsize,
        name="det_pipeline",
        monitor=monitor,
    )

    # 2. GPU Training thread
    train_iterator = ThreadedGenerator(
        train_step_pipeline(
            data_iterator,
            model,
            matcher,
            criterion,
            refiner,
            raw_refiner,
            optimizer,
            params,
            total_symbol_budget,
            dataset_symbols,
            is_distributed,
        ),
        maxsize=params.train_maxsize,
        name="train_pipeline",
        monitor=monitor,
    )

    last_log_time = time.time()
    total_symbols = 0
    last_log_symbols = 0
    running_loss = None
    last_log_epoch = 0.0
    last_result = None

    ctx = monitor if monitor else nullcontext()

    # 3. Main thread (Metrics & Visualization)
    with ctx:
        for result in train_iterator:
            last_result = result
            total_symbols += result.symbols_processed

            # Half-life decay based on symbols processed
            smoothing = 0.5 ** (
                result.symbols_processed / params.running_loss_half_life
            )
            if running_loss is None:
                running_loss = result.total_loss
            else:
                running_loss = (
                    smoothing * running_loss
                    + (1.0 - smoothing) * result.total_loss
                )

            if is_main_process and (
                result.epoch - last_log_epoch >= params.log_epoch_interval
            ):
                assert logger is not None
                last_log_epoch = result.epoch

                current_time = time.time()
                elapsed = current_time - last_log_time
                speed = (
                    (total_symbols - last_log_symbols) / elapsed
                    if elapsed > 0
                    else 0.0
                )
                last_log_time = current_time
                last_log_symbols = total_symbols

                log_and_save_checkpoint(
                    result,
                    params,
                    matcher,
                    logger,
                    raw_model,
                    raw_refiner,
                    optimizer,
                    running_loss,
                    speed,
                    ax,
                    fig,
                    plt,
                )

        # Ensure the absolute final state is saved
        if is_main_process and last_result is not None:
            assert running_loss is not None
            assert logger is not None
            print("Training complete. Saving final checkpoint and metrics...")

            current_time = time.time()
            elapsed = current_time - last_log_time
            speed = (
                (total_symbols - last_log_symbols) / elapsed
                if elapsed > 0
                else 0.0
            )

            log_and_save_checkpoint(
                last_result,
                params,
                matcher,
                logger,
                raw_model,
                raw_refiner,
                optimizer,
                running_loss,
                speed,
                ax,
                fig,
                plt,
            )

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OMR Detector on Trompa-COCO"
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path(
            "data/trompa-coco/annotations/instances_trainval2017.json"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Directory to store the cached dataset. If None, stores next to anno_path.",
    )
    parser.add_argument(
        "--class_weight_variance_lambda",
        type=float,
        default=0.5,
        help="How strongly intra-class size variability up-weights a class (difficulty proxy).",
    )
    parser.add_argument(
        "--class_weight_size_difficulty_max",
        type=float,
        default=8.0,
        help="Cap on the per-class size-difficulty multiplier.",
    )
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Restrict dataset to first N images. If None, uses the full dataset.",
    )
    parser.add_argument(
        "--include_labels",
        type=str,
        default=None,
        help="Comma-separated list of class names to restrict training to (e.g., 'noteheadBlack,stem').",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights",
    )
    parser.add_argument(
        "--loader_maxsize",
        type=int,
        default=1,
        help="Maximum number of batches to pre-load in the data pipeline queue",
    )
    parser.add_argument(
        "--train_maxsize",
        type=int,
        default=2,
        help="Maximum number of detached outputs to buffer in the training queue",
    )
    parser.add_argument(
        "--backbone_size",
        type=str,
        choices=["nano", "small", "base"],
        default="nano",
        help="Size of the ViT backbone to use.",
    )
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--extract_patch_min", type=int, default=54)
    parser.add_argument("--extract_patch_max", type=int, default=64)
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Square crop size. If not provided, the full image is used.",
    )
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument(
        "--base_anchor_size",
        type=float,
        default=1.0,
        help="Base anchor size in Patch Units",
    )

    # Matcher costs
    parser.add_argument("--cost_class", type=float, default=2.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--cost_giou", type=float, default=2.0)
    parser.add_argument(
        "--radius_patches",
        type=float,
        default=4.0,
        help="Radius in patches for sparse matching",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K fallback for sparse matching",
    )
    parser.add_argument(
        "--matcher_type",
        type=str,
        choices=["scipy", "greedy"],
        default="scipy",
        help="Type of bipartite matching to use",
    )

    # Loss weights
    parser.add_argument("--loss_ce", type=float, default=2.0)
    parser.add_argument("--loss_bbox", type=float, default=5.0)
    parser.add_argument("--loss_giou", type=float, default=2.0)
    parser.add_argument("--loss_fgl", type=float, default=0.05)
    parser.add_argument("--loss_line_l1", type=float, default=5.0)
    parser.add_argument("--loss_presence", type=float, default=2.0)

    patch_set = parser.add_argument_group(
        "patch-set assignment",
        "PATCH_SET_ASSIGNMENT.md: every patch within --responsibility_margin of a GT predicts "
        "it (one-to-one inside the patch), with a presence logit per slot. Loading a K=5 "
        "checkpoint with --num_shapes 8 remaps the head.",
    )
    patch_set.add_argument(
        "--assignment", choices=["global_one_to_one", "patch_set"], default="global_one_to_one"
    )
    patch_set.add_argument("--responsibility_margin", type=float, default=0.5)
    patch_set.add_argument("--ignore_band", type=float, default=0.5)
    patch_set.add_argument(
        "--ignore_quality",
        type=float,
        default=0.5,
        help="IoU / OKS from which an unmatched slot predicting a band or overflowed GT is ignored.",
    )
    patch_set.add_argument(
        "--voter_weighting", choices=["inverse", "inverse_sqrt", "none"], default="inverse"
    )
    patch_set.add_argument(
        "--cost_presence",
        type=float,
        default=0.0,
        help="Weight of the presence focal cost in the in-patch matching (stable slot roles).",
    )
    patch_set.add_argument(
        "--line_on_line",
        type=float,
        default=0.5,
        help="Lines: unmatched slots whose prediction lies on a responsible line (perpendicular "
        "quality >= this, extent ignored) are positives too; 0 = strict in-patch one-to-one.",
    )
    patch_set.add_argument(
        "--line_perp_sigma",
        type=float,
        default=0.1,
        help="Patch units: perpendicular scale of the on-line quality (0.1: within ~7 px).",
    )
    patch_set.add_argument(
        "--presence_weighting",
        choices=["inverse", "inverse_sqrt", "none"],
        default="none",
        help="Weight of a voter in the presence loss ('none': every voter is a full positive, "
        "normalised by the number of voters).",
    )
    patch_set.add_argument(
        "--presence_alpha",
        type=float,
        default=0.5,
        help="Focal alpha of the presence loss (0.5: ~35%% of line slots are positives).",
    )
    patch_set.add_argument(
        "--no_symbol_quality_targets",
        action="store_true",
        help="Symbol class target 1 on positives instead of their IoU (Exp 032 risk).",
    )
    patch_set.add_argument(
        "--no_line_quality_targets",
        action="store_true",
        help="Line class target 1 on positives instead of their on-line quality.",
    )
    patch_set.add_argument(
        "--head_lr_scale",
        type=float,
        default=1.0,
        help="LR multiple for the head output layers (new presence rows and slots).",
    )
    patch_set.add_argument(
        "--refiner_voters", type=int, default=4, help="Refiner queries per GT, spread along it."
    )

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak LR for the Symbol Budget Scheduler",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=0.5,
        help="Number of epochs to linearly warmup the learning rate",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=1e-4,
        help="Minimum LR multiplier at start of warmup",
    )
    parser.add_argument(
        "--running_loss_half_life",
        type=float,
        default=250.0,
        help="Half-life in symbols for running loss smoothing",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--log_epoch_interval",
        type=float,
        default=0.1,
        help="Log and save every X epochs",
    )
    parser.add_argument("--var_threshold", type=float, default=None)
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=None,
        help="Fraction of patches to keep based on variance",
    )
    parser.add_argument(
        "--topk_patches",
        type=int,
        default=None,
        help="Keep exactly K patches with the highest variance. Enables static shapes for torch.compile.",
    )

    # Augmentation params
    parser.add_argument(
        "--affine",
        action="store_true",
        help="Enable random affine augmentations",
    )
    parser.add_argument(
        "--morphological_downscale",
        type=float,
        default=None,
        help="Enable random morphological downscaling for DPI augmentation with the given max scale factor (e.g., 3.0)",
    )
    parser.add_argument(
        "--max_translate_frac",
        type=float,
        default=0.05,
        help="Max translation fraction",
    )
    parser.add_argument(
        "--max_angle_deg",
        type=float,
        default=2.0,
        help="Max rotation angle in degrees",
    )
    parser.add_argument(
        "--max_shear_deg",
        type=float,
        default=2.0,
        help="Max shear angle in degrees",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=1.1,
        help="Max scale factor (e.g. 1.1 for +/- 10%%)",
    )

    parser.add_argument(
        "--log_patches",
        action="store_true",
        help="Log patch count before forward pass",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model (useful for Kaggle/modern GPUs)",
    )
    parser.add_argument(
        "--use_sdpa",
        action="store_true",
        help="Enable scaled_dot_product_attention",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (FP16)",
    )
    parser.add_argument(
        "--prep_device",
        type=str,
        default="cpu",
        help="Device for preprocessing (e.g., cpu, cuda:0)",
    )
    parser.add_argument(
        "--train_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model training (e.g., cuda:0)",
    )
    parser.add_argument(
        "--match_device",
        type=str,
        default="cpu",
        help="Device for the Hungarian Matcher (e.g., cpu, cuda:1)",
    )
    parser.add_argument(
        "--backbone_checkpoint",
        type=Path,
        default=None,
        help="Path to the pre-trained LeJEPA backbone checkpoint",
    )
    parser.add_argument(
        "--detector_checkpoint",
        type=Path,
        default=None,
        help="Path to a full detector checkpoint (model + optimizer) to resume training from.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="If set, the backbone weights will be frozen and only the detection head will be trained.",
    )
    parser.add_argument(
        "--exp_dir", type=Path, default=Path("experiments/default_exp")
    )
    parser.add_argument("--stage_name", type=str, default="train_detection")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable interactive display and use Agg backend",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable the threaded generator monitor dashboard",
    )
    synth = parser.add_argument_group(
        "synthetic pretraining",
        "Train on on-the-fly synthetic pages (dataset.synthetic) instead of the "
        "real images. --anno_path still gives the classes and the epoch size.",
    )
    synth.add_argument("--synthetic", action="store_true")
    synth.add_argument("--synth_workers", type=int, default=2)
    synth.add_argument("--synth_page_size", type=int, default=4096)
    synth.add_argument("--synth_staff_space_min", type=float, default=24.0)
    synth.add_argument("--synth_staff_space_max", type=float, default=96.0)
    synth.add_argument("--synth_objects_per_page", type=int, default=960)
    synth.add_argument("--synth_lines_per_page", type=int, default=800)
    synth.add_argument("--synth_line_max_pair_overlap", type=float, default=0.9)
    synth.add_argument(
        "--synth_fonts", type=str, default="Bravura,Leland,Leipzig,Gootville"
    )
    synth.add_argument("--synth_seed", type=int, default=0)
    synth.add_argument(
        "--verovio_root", type=Path, default=Path("/home/kwon-young/prog/verovio")
    )
    split = parser.add_argument_group(
        "held-out split",
        "Leave a fraction of the pieces out of training (same split by piece as Analysis 005 "
        "and Exp 035 for the same seed); their image ids are written to "
        "<exp_dir>/held_out_image_ids.json for evaluate_coco.py --image_ids.",
    )
    split.add_argument("--holdout_fraction", type=float, default=0.0)
    split.add_argument("--holdout_seed", type=int, default=0)
    refine = parser.add_argument_group(
        "line endpoint refiner (Exp 038, Exp 035 Phase B)",
        "Train the Exp 035 refiner inside the detector step, on the augmented tokens and the "
        "step's line matching. Checkpoints then carry the refiner, so inference_coco.py "
        "--refiner_checkpoint can point at latest_model.pt.",
    )
    refine.add_argument("--line_refiner", action="store_true")
    refine.add_argument(
        "--refiner_init",
        type=Path,
        default=None,
        help="Start from a trained refiner (Phase A refiner.pt); ignored when "
        "--detector_checkpoint already carries one.",
    )
    refine.add_argument("--refiner_num_steps", type=int, default=2)
    refine.add_argument("--refiner_heads", type=int, default=3)
    refine.add_argument(
        "--refiner_scale", type=float, default=3.0, help="Correction range in patches (+-scale/2)."
    )
    refine.add_argument("--refiner_loss_weight", type=float, default=1.0)
    refine.add_argument(
        "--refiner_lr_scale",
        type=float,
        default=1.0,
        help="Refiner LR as a multiple of the detector LR (same schedule).",
    )
    refine.add_argument("--refiner_fgl_weight", type=float, default=0.15)
    refine.add_argument("--refiner_max_coarse_error", type=float, default=1.5)
    refine.add_argument("--refiner_dn_sigma", type=float, default=0.05)
    refine.add_argument("--refiner_dn_uniform", type=float, default=1.0)
    refine.add_argument("--refiner_dn_uniform_prob", type=float, default=0.4)
    refine.add_argument(
        "--refiner_detach_tokens",
        action="store_true",
        help="The refiner reads the tokens without back-propagating into the backbone.",
    )

    args = parser.parse_args()

    # Validate mutual exclusivity of patch dropping options
    drop_options_set = sum(
        [
            args.var_threshold is not None,
            args.drop_rate is not None,
            args.topk_patches is not None,
        ]
    )
    if drop_options_set == 0:
        args.var_threshold = 0.001
    elif drop_options_set > 1:
        raise ValueError(
            "Cannot specify more than one of --var_threshold, --drop_rate, and --topk_patches"
        )

    prep_device = torch.device(args.prep_device)
    train_device = torch.device(args.train_device)
    match_device = torch.device(args.match_device)

    # Parse and cache the dataset before starting training
    dataset = parse_coco(
        args.anno_path,
        cache_dir=args.cache_dir,
        variance_lambda=args.class_weight_variance_lambda,
        size_difficulty_max=args.class_weight_size_difficulty_max,
    )

    synthetic_config = (
        SyntheticConfig(
            page_size=args.synth_page_size,
            staff_space_min=args.synth_staff_space_min,
            staff_space_max=args.synth_staff_space_max,
            objects_per_page=args.synth_objects_per_page,
            lines_per_page=args.synth_lines_per_page,
            line_max_pair_overlap=args.synth_line_max_pair_overlap,
            fonts=tuple(f.strip() for f in args.synth_fonts.split(",")),
            seed=args.synth_seed,
        )
        if args.synthetic
        else None
    )
    if synthetic_config is not None:
        # Synthetic pages sample every class uniformly, so the real-data
        # inverse-frequency weights would over-boost rare classes. Use uniform
        # weights with the same mean, which keeps the overall loss scale.
        dataset.symbol_weights = [
            sum(dataset.symbol_weights) / len(dataset.symbol_weights)
        ] * len(dataset.symbol_weights)
        dataset.line_weights = [
            sum(dataset.line_weights) / len(dataset.line_weights)
        ] * len(dataset.line_weights)

    held_out_image_ids: list[int] = []
    if args.holdout_fraction > 0:
        train_images, held_out_images = split_by_piece(
            dataset.images, args.holdout_seed, args.holdout_fraction
        )
        kept = {img.img_id for img in train_images}
        dataset.images = train_images
        dataset.annotations = {
            img_id: anns
            for img_id, anns in dataset.annotations.items()
            if img_id in kept
        }
        held_out_image_ids = [img.img_id for img in held_out_images]

    refiner_params = (
        RefinerParams(
            init=args.refiner_init,
            num_steps=args.refiner_num_steps,
            heads=args.refiner_heads,
            scale=args.refiner_scale,
            loss_weight=args.refiner_loss_weight,
            lr_scale=args.refiner_lr_scale,
            fgl_weight=args.refiner_fgl_weight,
            max_coarse_error=args.refiner_max_coarse_error,
            noise=DenoisingNoise(
                sigma=args.refiner_dn_sigma,
                uniform=args.refiner_dn_uniform,
                uniform_prob=args.refiner_dn_uniform_prob,
            ),
            detach_tokens=args.refiner_detach_tokens,
        )
        if args.line_refiner
        else None
    )

    # Restrict dataset labels if requested
    if args.include_labels:
        keep_names = [name.strip() for name in args.include_labels.split(",")]
        dataset.restrict_labels(keep_names)

    # Restrict dataset size if requested
    if args.num_images is not None and args.num_images < len(dataset.images):
        kept_img_ids = {img.img_id for img in dataset.images[: args.num_images]}
        dataset.images = dataset.images[: args.num_images]
        dataset.annotations = {
            img_id: anns
            for img_id, anns in dataset.annotations.items()
            if img_id in kept_img_ids
        }
        print(f"Restricting dataset to first {args.num_images} images.")

    params = TrainParams(
        anno_path=args.anno_path,
        cache_dir=args.cache_dir,
        img_dir=args.img_dir,
        dataset=dataset,
        num_images=args.num_images,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        loader_maxsize=args.loader_maxsize,
        train_maxsize=args.train_maxsize,
        backbone_size=args.backbone_size,
        patch_size=args.patch_size,
        extract_patch_min=args.extract_patch_min,
        extract_patch_max=args.extract_patch_max,
        crop_size=args.crop_size,
        channels=args.channels,
        num_symbol_classes=dataset.num_symbol_classes,
        num_line_classes=dataset.num_line_classes,
        num_shapes=args.num_shapes,
        base_anchor_size=args.base_anchor_size,
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        radius_patches=args.radius_patches,
        top_k=args.top_k,
        matcher_type=args.matcher_type,
        loss_ce=args.loss_ce,
        loss_bbox=args.loss_bbox,
        loss_giou=args.loss_giou,
        loss_fgl=args.loss_fgl,
        loss_line_l1=args.loss_line_l1,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        running_loss_half_life=args.running_loss_half_life,
        epochs=args.epochs,
        log_epoch_interval=args.log_epoch_interval,
        var_threshold=args.var_threshold,
        drop_rate=args.drop_rate,
        topk_patches=args.topk_patches,
        affine=args.affine,
        morphological_downscale=args.morphological_downscale,
        max_translate_frac=args.max_translate_frac,
        max_angle_deg=args.max_angle_deg,
        max_shear_deg=args.max_shear_deg,
        max_scale=args.max_scale,
        log_patches=args.log_patches,
        compile=args.compile,
        use_sdpa=args.use_sdpa,
        use_amp=args.use_amp,
        prep_device=prep_device,
        train_device=train_device,
        match_device=match_device,
        backbone_checkpoint=args.backbone_checkpoint,
        detector_checkpoint=args.detector_checkpoint,
        freeze_backbone=args.freeze_backbone,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
        headless=args.headless,
        use_monitor=args.monitor,
        synthetic=synthetic_config,
        synthetic_workers=args.synth_workers,
        verovio_root=args.verovio_root,
        refiner=refiner_params,
        held_out_image_ids=held_out_image_ids,
        patch_set=PatchSetParams(
            config=PatchSetConfig(
                margin=args.responsibility_margin,
                ignore_band=args.ignore_band,
                ignore_quality=args.ignore_quality,
                cost_class=args.cost_class,
                cost_bbox=args.cost_bbox,
                cost_giou=args.cost_giou,
                cost_presence=args.cost_presence,
                line_on_line=args.line_on_line,
                line_perp_sigma=args.line_perp_sigma,
            ),
            voter_weighting=args.voter_weighting,
            presence_weighting=args.presence_weighting,
            presence_alpha=args.presence_alpha,
            symbol_quality_targets=not args.no_symbol_quality_targets,
            line_quality_targets=not args.no_line_quality_targets,
            refiner_voters=args.refiner_voters,
        )
        if args.assignment == "patch_set"
        else None,
        loss_presence=args.loss_presence,
        head_lr_scale=args.head_lr_scale,
    )

    train(params)
