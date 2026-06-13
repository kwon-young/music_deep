import argparse
import time
import math
import random
import os
from pathlib import Path
from typing import Generator, Iterable
from dataclasses import dataclass
from itertools import batched
from contextlib import nullcontext
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.vit import vit_nano
from model.detector import OMRDetector
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion
from dataset.coco import parse_coco, load_coco_sample, CocoMetadata, CocoDataset
import transform.det as det_tf
from threaded_generator import ThreadedGenerator, Monitor
from metric import compute_map_50, compute_mean_iou
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
    BoundingBoxes,
    ClassLabels,
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
    map50: float
    miou: float
    speed: float


@dataclass
class TrainParams:
    anno_path: Path
    cache_dir: Path | None
    img_dir: Path
    dataset: CocoDataset
    batch_size: int
    accumulation_steps: int
    patch_size: int
    crop_size: int | None
    channels: int
    num_classes: int
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
    lr: float
    warmup_ratio: float
    min_lr_ratio: float
    running_loss_half_life: float
    epochs: int
    log_epoch_interval: float
    var_threshold: float
    log_patches: bool
    compile: bool
    use_sdpa: bool
    use_amp: bool
    prep_device: torch.device
    train_device: torch.device
    match_device: torch.device
    backbone_checkpoint: Path | None
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
    losses: DetectionLosses
    total_loss: float
    symbols_processed: int


def transform_image(
    index: int,
    dataset: CocoDataset,
    img_dir: Path,
    patch_size: int,
    crop_size: int | None,
    prep_device: torch.device,
) -> Data[
    CocoMetadata,
    DetectionSample[TensorImage[CHW, RGB, Float1], BoundingBoxes, ClassLabels],
]:
    """Preprocessing: Load -> Decode/Crop -> Float1 -> Pad -> Normalize."""
    item = load_coco_sample(dataset, img_dir, index)

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
    item_padded = det_tf.pad_to_patch_size(
        item_tf, patch_size=(patch_size, patch_size)
    )

    # Normalize boxes using the final padded image dimensions
    item_normalized = det_tf.normalize_boxes(item_padded)

    return item_normalized


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
                    params.patch_size,
                    params.crop_size,
                    params.prep_device,
                )
                for idx in indices
            )

            # 3. Chunk into batches, collate, and extract patches
            for batch_items in batched(transformed_gen, params.batch_size):
                batched_item = det_tf.collate(batch_items)
                patched_item = det_tf.extract_patches(
                    batched_item,
                    patch_size=(params.patch_size, params.patch_size),
                )
                dropped_item = det_tf.variance_patch_drop(
                    patched_item, var_threshold=params.var_threshold
                )

                # Move the final batch to the training device
                final_item = det_tf.to_patches(
                    dropped_item, device=params.train_device
                )
                yield final_item

    return log_patch_count(_pipeline(), params.log_patches)


def train_step_pipeline(
    iterator: Iterable[
        BatchedData[
            CocoMetadata,
            DetectionSample[
                Patches[Batch, NumPatches, PatchDim],
                list[BoundingBoxes],
                list[ClassLabels],
            ],
        ]
    ],
    model: OMRDetector | DDP,
    criterion: DFINECriterion,
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
            DetectionTarget(labels=l, boxes=b)
            for b, l in zip(batch.sample.boxes, batch.sample.labels)
        ]

        # Prevent DDP from synchronizing gradients during accumulation steps
        sync_context = (
            model.no_sync()
            if (is_distributed and is_accumulating)
            else nullcontext()
        )

        with sync_context:
            with torch.autocast(
                device_type=params.train_device.type,
                dtype=torch.float16,
                enabled=params.use_amp,
            ):
                outputs = model(batch.sample.image)
                losses = criterion(outputs, targets)
                total_loss = (
                    losses.loss_ce
                    + losses.loss_bbox
                    + losses.loss_giou
                    + losses.loss_fgl
                )

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
            step_symbols = sym_tensor.item()
        else:
            step_symbols = local_symbols

        cumulative_symbols += step_symbols
        current_epoch = cumulative_symbols / dataset_symbols
        progress = min(1.0, cumulative_symbols / total_symbol_budget)

        # --- Symbol Budget LR Scheduler ---
        if progress < params.warmup_ratio:
            # Linear Warmup
            current_lr = params.lr * max(
                params.min_lr_ratio, (progress / params.warmup_ratio)
            )
        else:
            # Cosine Decay
            cosine_progress = (progress - params.warmup_ratio) / (
                1.0 - params.warmup_ratio
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
    model: OMRDetector,
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
        indices_match = matcher(result.outputs, result.targets)
        map50 = compute_map_50(
            result.outputs, result.targets, params.num_classes
        )
        miou = compute_mean_iou(result.outputs, result.targets, indices_match)

    metrics = DetectionMetrics(
        step=result.step,
        epoch=result.epoch,
        lr=result.lr,
        loss_total=result.total_loss,
        loss_ce=result.losses.loss_ce.item(),
        loss_bbox=result.losses.loss_bbox.item(),
        loss_giou=result.losses.loss_giou.item(),
        loss_fgl=result.losses.loss_fgl.item(),
        map50=map50,
        miou=miou,
        speed=speed,
    )
    logger.log_metrics(metrics)

    print(
        f"Epoch [{result.epoch:.2f}/{params.epochs}] Step [{result.step}] "
        f"LR: {result.lr:.2e} | "
        f"Loss: {result.total_loss:.4f} (Running: {running_loss:.4f}) | "
        f"CE: {result.losses.loss_ce.item():.4f} | BBox: {result.losses.loss_bbox.item():.4f} | "
        f"GIoU: {result.losses.loss_giou.item():.4f} | FGL: {result.losses.loss_fgl.item():.4f} | "
        f"mAP@0.5: {map50:.4f} | mIoU: {miou:.4f} | "
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
        indices=indices_match,
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
    backbone = vit_nano(
        patch_size=params.patch_size,
        channels=params.channels,
        use_sdpa=params.use_sdpa,
    )

    if params.backbone_checkpoint and params.backbone_checkpoint.exists():
        if is_main_process:
            print(f"Loading backbone weights from {params.backbone_checkpoint}")
        checkpoint = torch.load(
            params.backbone_checkpoint,
            map_location=params.train_device,
            weights_only=True,
        )
        backbone.load_state_dict(checkpoint["backbone"], strict=True)
    elif params.backbone_checkpoint:
        if is_main_process:
            print(
                f"Warning: Checkpoint {params.backbone_checkpoint} not found. Training from scratch."
            )

    if params.freeze_backbone:
        if is_main_process:
            print("Freezing backbone parameters (no fine-tuning).")
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        if is_main_process:
            print("Fine-tuning backbone parameters.")

    raw_model = OMRDetector(
        backbone,
        num_classes=params.num_classes,
        num_shapes=params.num_shapes,
        base_anchor_size=params.base_anchor_size,
    ).to(params.train_device)

    model = raw_model

    # --- Wrap Model in DDP ---
    if is_distributed:
        model = DDP(
            raw_model, device_ids=[local_rank], output_device=local_rank
        )
    # ------------------------------

    if params.compile:
        if is_main_process:
            print("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)

    # 2. Setup Matcher and Criterion
    matcher = HungarianMatcher(
        cost_class=params.cost_class,
        cost_bbox=params.cost_bbox,
        cost_giou=params.cost_giou,
        calc_device=params.match_device,
        radius_patches=params.radius_patches,
        top_k=params.top_k,
        patch_size=params.patch_size,
        image_size=params.crop_size if params.crop_size is not None else 3584,
        matcher_type=params.matcher_type,
    )
    weights = DetectionLossWeights(
        loss_ce=params.loss_ce,
        loss_bbox=params.loss_bbox,
        loss_giou=params.loss_giou,
        loss_fgl=params.loss_fgl,
    )
    criterion = DFINECriterion(
        matcher, num_classes=params.num_classes, weights=weights
    ).to(params.train_device)

    optimizer = optim.AdamW(model.parameters(), lr=params.lr)

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
    if is_main_process:
        print("Training on full dataset.")
        print(
            f"Total Symbol Budget for {params.epochs} epochs: {total_symbol_budget}"
        )

    monitor = Monitor() if (params.use_monitor and is_main_process) else None

    # 1. Data loading thread
    data_iterator = ThreadedGenerator(
        create_detection_iterator(params),
        maxsize=4,
        name="det_pipeline",
        monitor=monitor,
    )

    # 2. GPU Training thread
    train_iterator = ThreadedGenerator(
        train_step_pipeline(
            data_iterator,
            model,
            criterion,
            optimizer,
            params,
            total_symbol_budget,
            dataset_symbols,
            is_distributed,
        ),
        maxsize=2,  # Buffer 2 batches of detached outputs
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
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights",
    )
    parser.add_argument("--patch_size", type=int, default=64)
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
    parser.add_argument("--loss_fgl", type=float, default=0.15)

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak LR for the Symbol Budget Scheduler",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of budget used for warmup",
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
    parser.add_argument("--var_threshold", type=float, default=0.001)
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
    prep_device = torch.device(args.prep_device)
    train_device = torch.device(args.train_device)
    match_device = torch.device(args.match_device)

    # Parse and cache the dataset before starting training
    dataset = parse_coco(args.anno_path, cache_dir=args.cache_dir)

    params = TrainParams(
        anno_path=args.anno_path,
        cache_dir=args.cache_dir,
        img_dir=args.img_dir,
        dataset=dataset,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        channels=args.channels,
        num_classes=dataset.num_classes,
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
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        running_loss_half_life=args.running_loss_half_life,
        epochs=args.epochs,
        log_epoch_interval=args.log_epoch_interval,
        var_threshold=args.var_threshold,
        log_patches=args.log_patches,
        compile=args.compile,
        use_sdpa=args.use_sdpa,
        use_amp=args.use_amp,
        prep_device=prep_device,
        train_device=train_device,
        match_device=match_device,
        backbone_checkpoint=args.backbone_checkpoint,
        freeze_backbone=args.freeze_backbone,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
        headless=args.headless,
        use_monitor=args.monitor,
    )

    train(params)
