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
    KeypointDim,
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
    outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
    epoch: int | str,
    conf_thresh: float = 0.5,
    indices: tuple[list[MatchIndices], list[MatchIndices]] | None = None,
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
    for box in gt_boxes:
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

    # Plot Ground Truth keypoints (Blue)
    gt_keypoints = targets[0].keypoints.data.cpu().numpy() * np.array(
        [img_w, img_h, img_w, img_h]
    )
    for kp in gt_keypoints:
        x1, y1, x2, y2 = kp
        ax.plot([x1, x2], [y1, y2], color="b", linewidth=2)

    # Plot Predicted boxes (Red)
    sym_logits = outputs.symbols.pred_logits.data[0].detach().cpu()
    sym_boxes = outputs.symbols.pred_boxes.data[
        0
    ].detach().cpu().numpy() * np.array([img_w, img_h, img_w, img_h])
    sym_probs = torch.sigmoid(sym_logits)
    sym_max_probs, _ = sym_probs.max(dim=-1)

    if indices is not None:
        sym_idx = indices[0][0].pred_indices.cpu().numpy()
        sym_boxes_kept = sym_boxes[sym_idx]
    else:
        keep = (sym_max_probs > conf_thresh).numpy()
        sym_boxes_kept = sym_boxes[keep]

    for box in sym_boxes_kept:
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

    # Plot Predicted keypoints (Orange)
    line_logits = outputs.lines.pred_logits.data[0].detach().cpu()
    line_kps = outputs.lines.pred_keypoints.data[
        0
    ].detach().cpu().numpy() * np.array([img_w, img_h, img_w, img_h])
    line_probs = torch.sigmoid(line_logits)
    line_max_probs, _ = line_probs.max(dim=-1)

    if indices is not None:
        line_idx = indices[1][0].pred_indices.cpu().numpy()
        line_kps_kept = line_kps[line_idx]
    else:
        keep = (line_max_probs > conf_thresh).numpy()
        line_kps_kept = line_kps[keep]

    for kp in line_kps_kept:
        x1, y1, x2, y2 = kp
        ax.plot([x1, x2], [y1, y2], color="orange", linewidth=2, linestyle="--")

    ax.axis("off")
