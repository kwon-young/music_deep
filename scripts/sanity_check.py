import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import sys

sys.path.append("src")

from model.vit import vit_nano
from model.detector import OMRDetector
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion
from dataset.yolo import load_yolo_metadata, load_sample
import transform.det as det_tf
from transform.det import collate
from metric import compute_map_50, compute_mean_iou
from visualization import update_plot
from logger import ExperimentLogger, BaseMetrics
from music_types import (
    DetectionTarget,
    DetectionOutput,
    DetectionLossWeights,
    MatchIndices,
    TensorImage,
    Data,
    PILImage,
    HWC,
    Int255,
    CHW,
    Float1,
    DetectionSample,
    Mode,
    BoundingBoxes,
    ClassLabels,
    Batch,
    NumQueries,
    BoxDim,
    CoordDim,
    BoxShape,
    LabelShape,
    XYXY,
    TopLeft,
    NumClasses,
)


@dataclass
class SanityCheckMetrics(BaseMetrics):
    epoch: int
    loss_total: float
    loss_ce: float
    loss_bbox: float
    loss_giou: float
    loss_fgl: float
    map50: float
    miou: float


@dataclass
class TrainParams:
    img_dir: Path
    lbl_dir: Path
    img_w: int
    img_h: int
    patch_size: int
    channels: int
    num_classes: int
    num_shapes: int
    base_anchor_size: float
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
    conf_thresh: float
    device: torch.device
    exp_dir: Path
    stage_name: str


def transform_image[Meta, M: Mode, B: BoundingBoxes, L: ClassLabels](
    item: Data[Meta, DetectionSample[PILImage[HWC, M, Int255], B, L]],
    device: torch.device,
) -> Data[Meta, DetectionSample[TensorImage[CHW, M, Float1], B, L]]:
    item_np = det_tf.to_numpy(item)
    item_t = det_tf.to_tensor(item_np)
    item_t = det_tf.to(item_t, device=device)
    item_tf = det_tf.to_float1(item_t)
    return item_tf


def train(params: TrainParams):
    device = params.device
    print(f"Using device: {device}")

    logger = ExperimentLogger(params.exp_dir, params.stage_name)

    # 1. Setup Model (using vit_nano for speed)
    backbone = vit_nano(patch_size=params.patch_size, channels=params.channels)
    model = OMRDetector(
        backbone,
        num_classes=params.num_classes,
        num_shapes=params.num_shapes,
        base_anchor_size=params.base_anchor_size,
    ).to(device)

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
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params.lr)

    # 3. Load and Transform Data
    img_dir = params.img_dir
    lbl_dir = params.lbl_dir

    if not img_dir.exists():
        print(
            f"Error: {img_dir} does not exist. Please ensure COCO128 is downloaded."
        )
        return

    # Get the first image
    metadata_gen = load_yolo_metadata(
        img_dir, lbl_dir, params.img_w, params.img_h
    )
    first_metadata = next(metadata_gen)

    print(f"Loading image: {first_metadata.img_path.name}")

    raw_data = load_sample(first_metadata)
    transformed_data = transform_image(raw_data, device)

    # 4. Prepare Batch, Patches, and Centers
    batched_image = collate((transformed_data,))

    patches_obj_batched = det_tf.extract_patches(
        batched_image, patch_size=(params.patch_size, params.patch_size)
    )

    patches_obj = patches_obj_batched.sample.image
    image_tensor = batched_image.sample.image.data  # For plotting

    # Reconstruct DetectionTarget for the criterion
    targets = [
        DetectionTarget(labels=l, boxes=b)
        for b, l in zip(
            patches_obj_batched.sample.boxes, patches_obj_batched.sample.labels
        )
    ]

    print(f"Found {len(targets[0].labels.data)} objects in the image.")

    # 5. Setup Interactive Plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(1, figsize=(8, 8))
    manager = fig.canvas.manager
    if manager:
        manager.set_window_title("OMR Detector Sanity Check")

    # 6. Overfit Loop
    print("Starting sanity check (overfitting a single batch)...")
    model.train()
    for epoch in range(params.epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(patches_obj)

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

        if epoch % params.log_interval == 0:
            # Get matcher indices and mAP for visualization
            with torch.no_grad():
                indices_match = matcher(outputs, targets)
                map50 = compute_map_50(outputs, targets, params.num_classes)
                miou = compute_mean_iou(outputs, targets, indices_match)

            metrics = SanityCheckMetrics(
                step=epoch,
                epoch=epoch,
                loss_total=total_loss.item(),
                loss_ce=losses.loss_ce.item(),
                loss_bbox=losses.loss_bbox.item(),
                loss_giou=losses.loss_giou.item(),
                loss_fgl=losses.loss_fgl.item(),
                map50=map50,
                miou=miou,
            )
            logger.log_metrics(metrics)

            print(
                f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | "
                f"CE: {losses.loss_ce.item():.4f} | "
                f"BBox: {losses.loss_bbox.item():.4f} | "
                f"GIoU: {losses.loss_giou.item():.4f} | "
                f"FGL: {losses.loss_fgl.item():.4f} | "
                f"mAP@0.5: {map50:.4f} | mIoU: {miou:.4f}"
            )

            # Update the plot dynamically using matched indices
            update_plot(
                ax,
                image_tensor,
                targets,
                outputs,
                epoch,
                indices=indices_match,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)  # Brief pause to allow GUI to update

    print(
        "Sanity check complete. If the total loss dropped significantly (near 0), the architecture is learning!"
    )

    # Turn off interactive mode, save the final result with thresholding
    plt.ioff()
    model.eval()
    with torch.no_grad():
        outputs = model(patches_obj)
        update_plot(
            ax,
            image_tensor,
            targets,
            outputs,
            epoch=f"Final (Threshold > {params.conf_thresh})",
            conf_thresh=params.conf_thresh,
            indices=None,
        )

    vis_path = logger.get_visualizations_dir() / "sanity_check_output.png"
    plt.savefig(vis_path, dpi=150)
    print(f"Final visualization saved to {vis_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity check for OMR Detector"
    )
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/coco128/images/train2017")
    )
    parser.add_argument(
        "--lbl_dir", type=Path, default=Path("data/coco128/labels/train2017")
    )
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=80)
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

    # Loss weights
    parser.add_argument("--loss_ce", type=float, default=2.0)
    parser.add_argument("--loss_bbox", type=float, default=5.0)
    parser.add_argument("--loss_giou", type=float, default=2.0)
    parser.add_argument("--loss_fgl", type=float, default=0.15)

    # Training params
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3001)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument(
        "--exp_dir", type=Path, default=Path("experiments/default_exp")
    )
    parser.add_argument("--stage_name", type=str, default="sanity_check")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = TrainParams(
        img_dir=args.img_dir,
        lbl_dir=args.lbl_dir,
        img_w=args.img_w,
        img_h=args.img_h,
        patch_size=args.patch_size,
        channels=args.channels,
        num_classes=args.num_classes,
        num_shapes=args.num_shapes,
        base_anchor_size=args.base_anchor_size,
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
        conf_thresh=args.conf_thresh,
        device=device,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
    )

    train(params)
