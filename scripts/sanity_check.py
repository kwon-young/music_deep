import argparse
import torch
import torch.optim as optim
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass

from model.vit import vit_nano
from model.detector import OMRDetector
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion
from dataset.yolo import load_yolo_metadata, load_sample
import transform.det as det_tf
from transform.det import collate
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
)


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


def transform_image[Meta, M: Mode, B: BoundingBoxes, L: ClassLabels](
    item: Data[Meta, DetectionSample[PILImage[HWC, M, Int255], B, L]],
    device: torch.device
) -> Data[Meta, DetectionSample[TensorImage[CHW, M, Float1], B, L]]:
    item_np = det_tf.to_numpy(item)
    item_t = det_tf.to_tensor(item_np)
    item_t = det_tf.to(item_t, device=device)
    item_tf = det_tf.to_float1(item_t)
    return item_tf


def update_plot(
    ax,
    image_tensor,
    targets: list[DetectionTarget],
    outputs: DetectionOutput,
    img_w,
    img_h,
    epoch,
    conf_thresh=0.5,
    indices: list[MatchIndices] | None = None,
):
    """Clears and redraws the plot with GT and Predictions."""
    ax.clear()
    if isinstance(epoch, int):
        ax.set_title(f"Training Epoch: {epoch:03d}")
    else:
        ax.set_title(f"{epoch}")

    # Convert image tensor to numpy HWC
    img = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)

    # Plot Ground Truth boxes (Green)
    gt_boxes = targets[0].boxes.data.cpu().numpy() * np.array(
        [img_w, img_h, img_w, img_h]
    )
    gt_labels = targets[0].labels.data.cpu().numpy()

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Add GT label text
        ax.text(
            x1,
            y1 - 2,
            f"GT:{label}",
            color="g",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, pad=0, edgecolor="none"),
        )

    # Plot Predicted boxes (Red)
    pred_logits = outputs.pred_logits[0].detach().cpu()  # (P*K, C)
    pred_boxes = outputs.pred_boxes[0].detach().cpu().numpy() * np.array(
        [img_w, img_h, img_w, img_h]
    )

    # Apply sigmoid to get probabilities and find max class prob
    probs = torch.sigmoid(pred_logits)
    max_probs, pred_labels = probs.max(dim=-1)

    if indices is not None:
        # Use Hungarian matched indices (batch size is 1, so we take indices[0])
        src_idx = indices[0].pred_indices.cpu().numpy()
        pred_boxes_kept = pred_boxes[src_idx]
        pred_probs_kept = max_probs[src_idx].numpy()
        pred_labels_kept = pred_labels[src_idx].numpy()
    else:
        # Filter by confidence threshold
        keep = (max_probs > conf_thresh).numpy()
        pred_boxes_kept = pred_boxes[keep]
        pred_probs_kept = max_probs[keep].numpy()
        pred_labels_kept = pred_labels[keep].numpy()

    for box, prob, label in zip(
        pred_boxes_kept, pred_probs_kept, pred_labels_kept
    ):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        # Add Pred label and confidence text
        ax.text(
            x1,
            y2 + 2,
            f"P:{label} {prob:.2f}",
            color="r",
            fontsize=8,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, pad=0, edgecolor="none"),
        )

    ax.axis("off")


def train(params: TrainParams):
    device = params.device
    print(f"Using device: {device}")

    # 1. Setup Model (using vit_nano for speed)
    backbone = vit_nano(patch_size=params.patch_size, channels=params.channels)
    model = OMRDetector(
        backbone, num_classes=params.num_classes, num_shapes=params.num_shapes
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

    from typing import reveal_type
    raw_data = load_sample(first_metadata)
    transformed_data = transform_image(raw_data, device)

    # 4. Prepare Batch, Patches, and Centers
    batched_image = collate((transformed_data,))

    patches_obj_batched = det_tf.extract_patches(
        batched_image, patch_size=(params.patch_size, params.patch_size)
    )

    patches_obj = patches_obj_batched.data.image
    image_tensor = batched_image.data.image.data  # For plotting

    reveal_type(patches_obj_batched)
    # Reconstruct DetectionTarget for the criterion
    targets = [
        DetectionTarget(labels=l, boxes=b)
        for b, l in zip(
            patches_obj_batched.data.boxes, patches_obj_batched.data.labels
        )
    ]
    reveal_type(targets)

    print(f"Found {len(targets[0].labels.data)} objects in the image.")

    # 5. Setup Interactive Plotting
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(1, figsize=(8, 8))
    fig.canvas.manager.set_window_title("OMR Detector Sanity Check")

    # 6. Overfit Loop
    print("Starting sanity check (overfitting a single batch)...")
    model.train()
    for epoch in range(params.epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(patches_obj)

        # Compute loss
        losses = criterion(outputs, targets)
        total_loss = losses.total

        # Backward pass
        total_loss.backward()
        optimizer.step()

        if epoch % params.log_interval == 0:
            print(
                f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | "
                f"CE: {losses.loss_ce.item():.4f} | "
                f"BBox: {losses.loss_bbox.item():.4f} | "
                f"GIoU: {losses.loss_giou.item():.4f} | "
                f"FGL: {losses.loss_fgl.item():.4f}"
            )

            # Get matcher indices for visualization
            with torch.no_grad():
                indices_match = matcher(outputs, targets)

            # Update the plot dynamically using matched indices
            update_plot(
                ax,
                image_tensor,
                targets,
                outputs,
                params.img_w,
                params.img_h,
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
            params.img_w,
            params.img_h,
            epoch=f"Final (Threshold > {params.conf_thresh})",
            conf_thresh=params.conf_thresh,
            indices=None,
        )

    plt.savefig("sanity_check_output.png", dpi=150)
    print("Final visualization saved to sanity_check_output.png")
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
    )

    train(params)
