import argparse
import time
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model.vit import vit_nano
from model.detector import OMRDetector
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion
from dataset.coco import parse_coco, iter_coco, CocoMetadata, CocoDataset
import transform.det as det_tf
from metric import compute_map_50
from visualization import update_plot, reconstruct_image_from_patches
from music_types import (
    DetectionTarget,
    DetectionLossWeights,
    Data,
    DetectionSample,
    PILImage,
    TensorImage,
    HWC,
    CHW,
    RGB,
    Int255,
    Float1,
    BoundingBoxes,
    ClassLabels,
    BatchedData,
    Patches,
    Batch,
    NumPatches,
    PatchDim,
    Absolute,
    NumBoxes,
    BoxDim,
    XYXY,
    TopLeft,
)


@dataclass
class TrainParams:
    anno_path: Path
    img_dir: Path
    dataset: CocoDataset
    patch_size: int
    crop_size: int
    channels: int
    num_classes: int
    num_shapes: int
    cost_class: float
    cost_bbox: float
    cost_giou: float
    loss_ce: float
    loss_bbox: float
    loss_giou: float
    loss_fgl: float
    lr: float
    epochs: int
    log_interval: int
    var_threshold: float
    log_patches: bool
    device: torch.device
    backbone_checkpoint: Path | None
    freeze_backbone: bool


def transform_image(
    item: Data[
        CocoMetadata,
        DetectionSample[
            PILImage[HWC, RGB, Int255],
            BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft],
            ClassLabels,
        ],
    ],
    patch_size: int,
    crop_size: int,
    device: torch.device,
) -> Data[
    CocoMetadata,
    DetectionSample[TensorImage[CHW, RGB, Float1], BoundingBoxes, ClassLabels],
]:
    """Preprocessing: PIL -> Numpy -> Tensor -> Crop -> Device -> Float1 -> Pad -> Normalize."""
    item_np = det_tf.to_numpy(item)
    item_t = det_tf.to_tensor(item_np)

    item_cropped = det_tf.random_crop(item_t, crop_size=crop_size)
    item_gpu = det_tf.to(item_cropped, device=device)

    item_tf = det_tf.to_float1(item_gpu)
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
    """Creates a plain Python generator pipeline for detection data."""
    # 1. Load raw data using the pre-parsed dataset
    raw_gen = iter_coco(params.dataset, params.img_dir)

    # 2. Apply transformations (Crop on CPU, rest on GPU)
    transformed_gen = (
        transform_image(
            item, params.patch_size, params.crop_size, params.device
        )
        for item in raw_gen
    )

    # 3. Collate into batches of size 1 and extract patches
    def _pipeline():
        for item in transformed_gen:
            batched_item = det_tf.collate((item,))
            patched_item = det_tf.extract_patches(
                batched_item, patch_size=(params.patch_size, params.patch_size)
            )
            dropped_item = det_tf.variance_patch_drop(
                patched_item, var_threshold=params.var_threshold
            )
            yield dropped_item

    return log_patch_count(_pipeline(), params.log_patches)


def train(params: TrainParams):
    print(f"Using device: {params.device}")

    # 1. Setup Model
    backbone = vit_nano(patch_size=params.patch_size, channels=params.channels)

    if params.backbone_checkpoint and params.backbone_checkpoint.exists():
        print(f"Loading backbone weights from {params.backbone_checkpoint}")
        checkpoint = torch.load(
            params.backbone_checkpoint, map_location=params.device, weights_only=True
        )
        backbone.load_state_dict(checkpoint["backbone"], strict=True)
    elif params.backbone_checkpoint:
        print(f"Warning: Checkpoint {params.backbone_checkpoint} not found. Training from scratch.")

    if params.freeze_backbone:
        print("Freezing backbone parameters (no fine-tuning).")
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        print("Fine-tuning backbone parameters.")

    model = OMRDetector(
        backbone, num_classes=params.num_classes, num_shapes=params.num_shapes
    ).to(params.device)

    # 2. Setup Matcher and Criterion
    matcher = HungarianMatcher(
        cost_class=params.cost_class,
        cost_bbox=params.cost_bbox,
        cost_giou=params.cost_giou,
    )
    weights = DetectionLossWeights(
        loss_ce=params.loss_ce,
        loss_bbox=params.loss_bbox,
        loss_giou=params.loss_giou,
        loss_fgl=params.loss_fgl,
    )
    criterion = DFINECriterion(
        matcher, num_classes=params.num_classes, weights=weights
    ).to(params.device)

    optimizer = optim.AdamW(model.parameters(), lr=params.lr)

    # Setup Interactive Plotting
    plt.ion()
    fig, ax = plt.subplots(1, figsize=(8, 8))
    manager = fig.canvas.manager
    if manager:
        manager.set_window_title("OMR Detector Training")

    # 3. Training Loop
    start_time = time.time()
    samples = 0
    running_loss = None

    for epoch in range(params.epochs):
        model.train()

        # Re-initialize the generator for each epoch
        iterator = create_detection_iterator(params)

        for step, batch in enumerate(iterator):
            # Reconstruct DetectionTarget for the criterion
            targets = [
                DetectionTarget(labels=l, boxes=b)
                for b, l in zip(batch.sample.boxes, batch.sample.labels)
            ]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch.sample.image)

            # Compute loss
            losses = criterion(outputs, targets)
            total_loss = (
                losses.loss_ce
                + losses.loss_bbox
                + losses.loss_giou
                + losses.loss_fgl
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update metrics
            samples += 1
            if running_loss is None:
                running_loss = total_loss.item()
            else:
                running_loss = 0.99 * running_loss + 0.01 * total_loss.item()

            if step % params.log_interval == 0:
                with torch.no_grad():
                    indices_match = matcher(outputs, targets)
                    map50 = compute_map_50(outputs, targets, params.num_classes)

                elapsed = time.time() - start_time
                speed = samples / elapsed if elapsed > 0 else 0.0
                print(
                    f"Epoch [{epoch}/{params.epochs}] Step [{step}] "
                    f"Loss: {total_loss.item():.4f} (Running: {running_loss:.4f}) | "
                    f"CE: {losses.loss_ce.item():.4f} | BBox: {losses.loss_bbox.item():.4f} | "
                    f"GIoU: {losses.loss_giou.item():.4f} | FGL: {losses.loss_fgl.item():.4f} | "
                    f"mAP@0.5: {map50:.4f} | "
                    f"Speed: {speed:.1f} sample/s"
                )

                # Reconstruct image from patches and update plot
                img_tensor = reconstruct_image_from_patches(batch.sample.image)
                update_plot(
                    ax,
                    img_tensor,
                    targets,
                    outputs,
                    epoch,
                    indices=indices_match,
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)


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
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=4160)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)

    # Matcher costs
    parser.add_argument("--cost_class", type=float, default=2.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--cost_giou", type=float, default=2.0)

    # Loss weights
    parser.add_argument("--loss_ce", type=float, default=2.0)
    parser.add_argument("--loss_bbox", type=float, default=5.0)
    parser.add_argument("--loss_giou", type=float, default=2.0)
    parser.add_argument("--loss_fgl", type=float, default=0.15)

    # Training params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument(
        "--log_patches",
        action="store_true",
        help="Log patch count before forward pass",
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

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse and cache the dataset before starting training
    dataset = parse_coco(args.anno_path)

    params = TrainParams(
        anno_path=args.anno_path,
        img_dir=args.img_dir,
        dataset=dataset,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        channels=args.channels,
        num_classes=dataset.num_classes,
        num_shapes=args.num_shapes,
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        loss_ce=args.loss_ce,
        loss_bbox=args.loss_bbox,
        loss_giou=args.loss_giou,
        loss_fgl=args.loss_fgl,
        lr=args.lr,
        epochs=args.epochs,
        log_interval=args.log_interval,
        var_threshold=args.var_threshold,
        log_patches=args.log_patches,
        device=device,
        backbone_checkpoint=args.backbone_checkpoint,
        freeze_backbone=args.freeze_backbone,
    )

    train(params)
