from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING
from music_types import (
    DetectionTarget,
    DetectionOutput,
    MatchIndices,
    Batch,
    NumQueries,
    BoxDim,
    CoordDim,
    Patches,
    NumPatches,
    PatchDim,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def reconstruct_image_from_patches[B: Batch, N: NumPatches, P: PatchDim](
    patches_obj: Patches[B, N, P],
) -> torch.Tensor:
    """Reconstructs the image tensor from patches, leaving dropped patches as zeros."""
    b, n, d = patches_obj.data.shape
    c, h, w = patches_obj.image_shape
    ph, pw = patches_obj.patch_size

    img = torch.zeros((b, c, h, w), device=patches_obj.data.device)
    grid_w = w // pw

    for batch_idx in range(b):
        for i in range(n):
            patch_idx = patches_obj.indices[batch_idx, i].item()
            py = patch_idx // grid_w
            px = patch_idx % grid_w

            # View as (ph, pw, c) then permute to (c, ph, pw) to avoid color mixing
            patch_data = (
                patches_obj.data[batch_idx, i].view(ph, pw, c).permute(2, 0, 1)
            )
            img[
                batch_idx, :, py * ph : (py + 1) * ph, px * pw : (px + 1) * pw
            ] = patch_data

    return img


def update_plot(
    ax: plt.Axes,
    image_tensor: torch.Tensor,
    targets: list[DetectionTarget],
    outputs: DetectionOutput[Batch, NumQueries, BoxDim, CoordDim],
    epoch: int | str,
    conf_thresh: float = 0.5,
    indices: list[MatchIndices] | None = None,
):
    import matplotlib.patches as patches

    """Clears and redraws the plot with GT and Predictions."""
    ax.clear()
    if isinstance(epoch, int):
        ax.set_title(f"Training Epoch: {epoch:03d}")
    else:
        ax.set_title(f"{epoch}")

    _, _, img_h, img_w = image_tensor.shape

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
        # ax.add_patch(rect)
        # Add GT label text
        # ax.text(
        #     x1,
        #     y1 - 2,
        #     f"GT:{label}",
        #     color="g",
        #     fontsize=8,
        #     bbox=dict(facecolor="white", alpha=0.7, pad=0, edgecolor="none"),
        # )

    # Plot Predicted boxes (Red)
    pred_logits = outputs.pred_logits.data[0].detach().cpu()  # (P*K, C)
    pred_boxes = outputs.pred_boxes.data[0].detach().cpu().numpy() * np.array(
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
        # ax.text(
        #     x1,
        #     y2 + 2,
        #     f"P:{label} {prob:.2f}",
        #     color="r",
        #     fontsize=8,
        #     verticalalignment="top",
        #     bbox=dict(facecolor="white", alpha=0.7, pad=0, edgecolor="none"),
        # )

    ax.axis("off")
