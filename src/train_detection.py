import argparse
import time
import math
import random
import os
from typing import Callable, ContextManager, cast
from pathlib import Path
from typing import Generator, Iterable
from dataclasses import dataclass
from itertools import batched
from contextlib import nullcontext
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.detector import OMRDetector, create_detector
from model.layered_detector import MultiLayerOMRDetector, create_layered_detector
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion, DetectionTargets
from model.layered_criterion import DFINELayeredCriterion
from dataset.coco import parse_coco, load_coco_sample, CocoMetadata, CocoDataset
import transform.det as det_tf
from threaded_generator import ThreadedGenerator, Monitor
from metric import compute_map_50, compute_mean_iou, compute_mean_endpoint_error
from visualization import update_plot, reconstruct_image_from_patches
from logger import ExperimentLogger, BaseMetrics
from music_types import (
    DetectionTarget,
    DetectionOutput,
    DetectionLossWeights,
    LayeredDetectionLosses,
    LayeredDetectionOutput,
    BoxDim,
    KeypointDim,
    CoordDim,
    NumQueries,
    Data,
    DetectionSample,
    TensorImage,
    CHW,
    RGB,
    Float1,
    BoundingBoxes,
    ClassLabels,
    Keypoints,
    BatchedData,
    Patches,
    Batch,
    NumPatches,
    PatchDim,
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
    loss_pre: float
    loss_aux: float
    loss_ddf: float
    map50: float
    miou: float
    line_error: float
    speed: float


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
    loss_ddf: float
    refine_layers: int
    dfine_outer_bins: bool
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
    init_from_single_checkpoint: Path | None
    resume: bool
    freeze_backbone: bool
    exp_dir: Path
    stage_name: str
    headless: bool
    use_monitor: bool


@dataclass
class TrainStepResult:
    step: int
    epoch: float
    lr: float
    batch: BatchedData
    targets: list[DetectionTarget]
    outputs: DetectionOutput
    losses: LayeredDetectionLosses
    total_loss: float
    symbols_processed: int


def transform_image(
    index: int,
    dataset: CocoDataset,
    img_dir: Path,
    crop_size: int | None,
    prep_device: torch.device,
    affine: bool,
    morphological_downscale: float | None,
    max_translate_frac: float,
    max_angle_deg: float,
    max_shear_deg: float,
    max_scale: float,
) -> Data[
    CocoMetadata,
    DetectionSample[
        TensorImage[CHW, RGB, Float1],
        BoundingBoxes,
        ClassLabels,
        Keypoints,
        ClassLabels,
    ],
]:
    """Preprocessing: Load -> Decode/Crop -> Float1 -> Augment."""
    item = load_coco_sample(dataset, img_dir, index, prep_device)

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

    item_tf = det_tf.to_float1(item_cropped)

    if morphological_downscale is not None:
        item_tf = det_tf.random_morphological_downscale(
            item_tf, max_scale_factor=morphological_downscale
        )

    if affine:
        item_aug = det_tf.random_affine(
            item_tf,
            max_translate_frac=max_translate_frac,
            max_angle_deg=max_angle_deg,
            max_shear_deg=max_shear_deg,
            max_scale=max_scale,
        )
    else:
        item_aug = item_tf

    return item_aug


def log_patch_count(iterator, enabled: bool):
    """Pass-through generator that logs the number of patches if enabled."""
    for item in iterator:
        if enabled:
            num_patches = item.sample.image.data.shape[1]
            print(f"[Pipeline] Number of patches fed to model: {num_patches}")
        yield item


def create_detection_iterator(
    params: TrainParams,
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
    num_images = len(params.dataset.images)
    indices = list(range(num_images))

    def _pipeline():
        while True:
            # 1. Shuffle indices at the start of every epoch
            random.shuffle(indices)

            # 2. Map the combined load+transform function over the indices
            transformed_gen = (
                transform_image(
                    idx,
                    params.dataset,
                    params.img_dir,
                    params.crop_size,
                    params.prep_device,
                    affine=params.affine,
                    morphological_downscale=params.morphological_downscale,
                    max_translate_frac=params.max_translate_frac,
                    max_angle_deg=params.max_angle_deg,
                    max_shear_deg=params.max_shear_deg,
                    max_scale=params.max_scale,
                )
                for idx in indices
            )

            # 3. Chunk into batches, collate, and extract patches
            for batch_items in batched(transformed_gen, params.batch_size):
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


type FinalOutput = DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim]
type ForwardLoss = Callable[
    [Patches[Batch, NumPatches, PatchDim], DetectionTargets],
    tuple[FinalOutput, LayeredDetectionLosses],
]


def single_layer_forward_loss(
    model: OMRDetector | DDP, criterion: DFINECriterion
) -> ForwardLoss:
    def forward_loss(
        patches: Patches[Batch, NumPatches, PatchDim], targets: DetectionTargets
    ) -> tuple[FinalOutput, LayeredDetectionLosses]:
        outputs: FinalOutput = model(patches)
        losses = criterion(outputs, targets)
        return outputs, LayeredDetectionLosses(
            layers=[losses], pre=None, loss_ddf=None
        )

    return forward_loss


def layered_forward_loss(
    model: MultiLayerOMRDetector | DDP, criterion: DFINELayeredCriterion
) -> ForwardLoss:
    def forward_loss(
        patches: Patches[Batch, NumPatches, PatchDim], targets: DetectionTargets
    ) -> tuple[FinalOutput, LayeredDetectionLosses]:
        outputs: LayeredDetectionOutput[
            Batch, NumQueries, BoxDim, KeypointDim, CoordDim
        ] = model(patches)
        return outputs.final, criterion(outputs, targets)

    return forward_loss


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
    model: OMRDetector | MultiLayerOMRDetector | DDP,
    forward_loss: ForwardLoss,
    optimizer: optim.Optimizer,
    params: TrainParams,
    total_symbol_budget: int,
    dataset_symbols: int,
    is_distributed: bool,
    start_symbols: int = 0,
    start_step: int = 0,
) -> Generator[TrainStepResult, None, None]:
    """Executes the forward/backward pass and yields detached results for the main thread."""
    cumulative_symbols = start_symbols
    global_step = start_step

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

        # Prevent DDP from synchronizing gradients during accumulation steps
        sync_context: ContextManager[None]
        if isinstance(model, DDP) and is_accumulating:
            sync_context = model.no_sync()
        else:
            sync_context = nullcontext()

        with sync_context:
            with torch.autocast(
                device_type=params.train_device.type,
                dtype=torch.float16,
                enabled=params.use_amp,
            ):
                outputs, losses = forward_loss(batch.sample.image, targets)
                total_loss = losses.total()

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
            param_group["lr"] = current_lr
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
        )


def log_and_save_checkpoint(
    result: TrainStepResult,
    params: TrainParams,
    matcher: HungarianMatcher,
    logger: ExperimentLogger,
    model: OMRDetector | MultiLayerOMRDetector,
    optimizer: optim.Optimizer,
    running_loss: float,
    speed: float,
    ax,
    fig,
    plt,
):
    """Handles metrics computation, visualization, and saving the model state."""
    # 1. Compute Metrics
    with torch.no_grad():
        sym_indices, line_indices = matcher(result.outputs, result.targets)
        map50 = compute_map_50(
            result.outputs, result.targets, params.num_symbol_classes
        )
        miou = compute_mean_iou(result.outputs, result.targets, sym_indices)
        line_error = compute_mean_endpoint_error(
            result.outputs, result.targets, line_indices
        )

    final = result.losses.final
    metrics = DetectionMetrics(
        step=result.step,
        epoch=result.epoch,
        lr=result.lr,
        loss_total=result.total_loss,
        loss_ce=final.loss_ce.item(),
        loss_bbox=final.loss_bbox.item(),
        loss_giou=final.loss_giou.item(),
        loss_fgl=final.loss_fgl.item(),
        loss_line_l1=final.loss_line_l1.item(),
        loss_line_fgl=final.loss_line_fgl.item(),
        loss_pre=(
            result.losses.pre.total().item()
            if result.losses.pre is not None
            else 0.0
        ),
        loss_aux=sum(layer.total().item() for layer in result.losses.layers[:-1]),
        loss_ddf=(
            result.losses.loss_ddf.item()
            if result.losses.loss_ddf is not None
            else 0.0
        ),
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
        f"CE: {metrics.loss_ce:.4f} | BBox: {metrics.loss_bbox:.4f} | "
        f"GIoU: {metrics.loss_giou:.4f} | FGL: {metrics.loss_fgl:.4f} | "
        f"LineL1: {metrics.loss_line_l1:.4f} | LineFGL: {metrics.loss_line_fgl:.4f} | "
        f"Pre: {metrics.loss_pre:.4f} | Aux: {metrics.loss_aux:.4f} | DDF: {metrics.loss_ddf:.4f} | "
        f"mAP@0.5: {map50:.4f} | mIoU: {miou:.4f} | LineErr: {line_error:.4f} | "
        f"Speed: {speed:.1f} symbols/s"
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
        "architecture": {
            "refine_layers": params.refine_layers,
            "dfine_outer_bins": params.dfine_outer_bins,
        },
        "optimizer": optimizer.state_dict(),
        "running_loss": running_loss,
    }
    checkpoint_path = logger.get_checkpoint_dir() / "latest_model.pt"
    torch.save(checkpoint, checkpoint_path)


def wrap_model[M: OMRDetector | MultiLayerOMRDetector](
    raw_model: M,
    params: TrainParams,
    is_distributed: bool,
    local_rank: int,
) -> M | DDP:
    model: M | DDP = (
        DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
        if is_distributed
        else raw_model
    )
    if params.compile:
        # topk yields static shapes, while var_threshold and drop_rate yield dynamic shapes
        use_dynamic = params.topk_patches is None
        if local_rank == 0:
            print(f"Compiling model with torch.compile(dynamic={use_dynamic})...")
        model = cast(M | DDP, torch.compile(model, dynamic=use_dynamic))
    return model


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
    raw_model: OMRDetector | MultiLayerOMRDetector
    if params.refine_layers > 0:
        raw_model = create_layered_detector(
            backbone_size=params.backbone_size,
            patch_size=params.patch_size,
            channels=params.channels,
            use_sdpa=params.use_sdpa,
            num_symbol_classes=params.num_symbol_classes,
            num_line_classes=params.num_line_classes,
            num_shapes=params.num_shapes,
            base_anchor_size=params.base_anchor_size,
            num_refine_layers=params.refine_layers,
            dfine_outer_bins=params.dfine_outer_bins,
        ).to(params.train_device)
    else:
        raw_model = create_detector(
            backbone_size=params.backbone_size,
            patch_size=params.patch_size,
            channels=params.channels,
            use_sdpa=params.use_sdpa,
            num_symbol_classes=params.num_symbol_classes,
            num_line_classes=params.num_line_classes,
            num_shapes=params.num_shapes,
            base_anchor_size=params.base_anchor_size,
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
        raw_model.load_state_dict(loaded_checkpoint["model"], strict=True)
    elif params.init_from_single_checkpoint:
        if not isinstance(raw_model, MultiLayerOMRDetector):
            raise ValueError("--init_from_single_checkpoint needs --refine_layers > 0")
        if is_main_process:
            print(
                f"Initializing backbone and dense heads from: {params.init_from_single_checkpoint}"
            )
        single = torch.load(
            params.init_from_single_checkpoint,
            map_location=params.train_device,
            weights_only=True,
        )
        raw_model.load_single_layer_state(single["model"])
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

    # 2. Setup Matcher and Criterion
    matcher = HungarianMatcher(
        cost_class=params.cost_class,
        cost_bbox=params.cost_bbox,
        cost_giou=params.cost_giou,
        calc_device=params.match_device,
        radius_patches=params.radius_patches,
        top_k=params.top_k,
        matcher_type=params.matcher_type,
    )
    weights = DetectionLossWeights(
        loss_ce=params.loss_ce,
        loss_bbox=params.loss_bbox,
        loss_giou=params.loss_giou,
        loss_fgl=params.loss_fgl,
        loss_line_l1=params.loss_line_l1,
    )
    criterion = DFINECriterion(
        matcher,
        num_symbol_classes=params.num_symbol_classes,
        num_line_classes=params.num_line_classes,
        weights=weights,
        symbol_weights=params.dataset.symbol_weights,
        line_weights=params.dataset.line_weights,
        base_anchor_size=params.base_anchor_size,
        dfine_outer_bins=params.dfine_outer_bins,
    ).to(params.train_device)

    model: OMRDetector | MultiLayerOMRDetector | DDP
    forward_loss: ForwardLoss
    if isinstance(raw_model, MultiLayerOMRDetector):
        layered = wrap_model(raw_model, params, is_distributed, local_rank)
        model = layered
        forward_loss = layered_forward_loss(
            layered, DFINELayeredCriterion(criterion, loss_ddf=params.loss_ddf)
        )
    else:
        single = wrap_model(raw_model, params, is_distributed, local_rank)
        model = single
        forward_loss = single_layer_forward_loss(single, criterion)

    optimizer = optim.AdamW(model.parameters(), lr=params.lr)

    if loaded_checkpoint is not None and "optimizer" in loaded_checkpoint:
        if is_main_process:
            print("Loading optimizer state from checkpoint...")
        optimizer.load_state_dict(loaded_checkpoint["optimizer"])

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

    model.train()

    total_symbol_budget = dataset_symbols * params.epochs
    start_symbols, start_step = 0, 0
    if params.resume:
        if loaded_checkpoint is None:
            raise ValueError("--resume needs --detector_checkpoint")
        start_symbols = int(float(loaded_checkpoint["epoch"]) * dataset_symbols)
        start_step = int(loaded_checkpoint["step"])
        if is_main_process:
            print(
                f"Resuming the schedule at epoch {loaded_checkpoint['epoch']:.2f}, step {start_step}"
            )
    if is_main_process:
        print("Training on full dataset.")
        print(
            f"Total Symbol Budget for {params.epochs} epochs: {total_symbol_budget}"
        )

    monitor = Monitor() if (params.use_monitor and is_main_process) else None

    # 1. Data loading thread
    data_iterator = ThreadedGenerator(
        create_detection_iterator(params),
        maxsize=params.loader_maxsize,
        name="det_pipeline",
        monitor=monitor,
    )

    # 2. GPU Training thread
    train_iterator = ThreadedGenerator(
        train_step_pipeline(
            data_iterator,
            model,
            forward_loss,
            optimizer,
            params,
            total_symbol_budget,
            dataset_symbols,
            is_distributed,
            start_symbols,
            start_step,
        ),
        maxsize=params.train_maxsize,
        name="train_pipeline",
        monitor=monitor,
    )

    last_log_time = time.time()
    total_symbols = 0
    last_log_symbols = 0
    running_loss = None
    last_log_epoch = start_symbols / dataset_symbols
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
    parser.add_argument(
        "--loss_ddf",
        type=float,
        default=1.5,
        help="D-FINE decoupled distillation weight (only with --refine_layers > 0)",
    )

    # D-FINE multi-layer refinement
    parser.add_argument(
        "--refine_layers",
        type=int,
        default=0,
        help="Number of last ViT blocks used as D-FINE refinement layers; "
        "the dense heads read the block before them. 0 = single-layer detector.",
    )
    parser.add_argument(
        "--dfine_outer_bins",
        action="store_true",
        help="Outer FDR bins at +-2a as in D-FINE (legacy: +-a, duplicating the inner bins)",
    )
    parser.add_argument(
        "--init_from_single_checkpoint",
        type=Path,
        default=None,
        help="Initialize a --refine_layers model from a single-layer detector checkpoint "
        "(backbone and dense heads; fresh optimizer).",
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
        "--resume",
        action="store_true",
        help="With --detector_checkpoint: continue its LR schedule and epoch count "
        "(same --epochs) instead of starting a new schedule.",
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
        loss_ddf=args.loss_ddf,
        refine_layers=args.refine_layers,
        dfine_outer_bins=args.dfine_outer_bins,
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
        init_from_single_checkpoint=args.init_from_single_checkpoint,
        resume=args.resume,
        freeze_backbone=args.freeze_backbone,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
        headless=args.headless,
        use_monitor=args.monitor,
    )

    train(params)
