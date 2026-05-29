import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import generalized_box_iou
from .detector import DFINEWeightingFunction
from music_types import (
    DetectionTarget,
    DetectionOutput,
    DetectionLosses,
    DetectionLossWeights,
    MatchIndices,
    FlattenedIndices,
)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class DFINECriterion(nn.Module):
    """
    This class computes the 4 losses for our Dense Patch-as-Predictor model:
    1. Focal Loss (Classification)
    2. L1 Loss (Bounding Box)
    3. GIoU Loss (Bounding Box)
    4. FGL Loss (D-FINE Fine-Grained Localization for edge distributions)
    """

    def __init__(
        self,
        matcher,
        num_classes: int,
        weights: DetectionLossWeights,
        reg_max: int = 32,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weights = weights
        self.reg_max = reg_max
        self.alpha = alpha
        self.gamma = gamma

        # We need the weighting function to map target residuals back to discrete bins
        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def _flatten_indices(self, indices: list[MatchIndices]) -> FlattenedIndices:
        # Permute predictions following the Hungarian Matcher indices
        batch_idx = torch.cat(
            [
                torch.full_like(match.pred_indices, i)
                for i, match in enumerate(indices)
            ]
        )
        src_idx = torch.cat([match.pred_indices for match in indices])
        return FlattenedIndices(batch=batch_idx, src=src_idx)

    def loss_labels(
        self,
        outputs: DetectionOutput,
        flat_idx: FlattenedIndices,
        matched_classes: torch.Tensor,
        num_boxes,
    ) -> torch.Tensor:
        """Classification loss (Focal Loss) applied to ALL predictions."""
        src_logits = outputs.pred_logits

        # Create a target tensor filled with 0s (Background)
        target_classes = torch.zeros_like(src_logits)

        # Set the matched indices to 1.0 for their specific class
        target_classes[flat_idx.batch, flat_idx.src, matched_classes] = 1.0

        # Compute Focal Loss
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="none",
        )
        loss_ce = loss_ce.sum() / num_boxes
        return loss_ce

    def loss_boxes(
        self,
        outputs: DetectionOutput,
        flat_idx: FlattenedIndices,
        matched_boxes: torch.Tensor,
        num_boxes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU loss applied ONLY to matched predictions."""
        # Extract only the matched predictions
        src_boxes = outputs.pred_boxes[flat_idx.batch, flat_idx.src]

        # 1. L1 Loss
        loss_bbox = (
            F.l1_loss(src_boxes, matched_boxes, reduction="none").sum()
            / num_boxes
        )

        # 2. GIoU Loss
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, matched_boxes))
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def loss_fgl(
        self,
        outputs: DetectionOutput,
        flat_idx: FlattenedIndices,
        matched_boxes: torch.Tensor,
        num_boxes,
    ) -> torch.Tensor:
        """D-FINE Fine-Grained Localization Loss applied ONLY to matched predictions."""
        src_edge_logits = outputs.pred_edge_logits[
            flat_idx.batch, flat_idx.src
        ]  # (N_matched, 4, reg_max+1)
        src_centers = outputs.absolute_centers[flat_idx.batch, flat_idx.src]  # (N_matched, 2)
        src_shapes = outputs.learnable_shapes[flat_idx.batch, flat_idx.src]  # (N_matched, 2)

        # Detach the targets so the network doesn't cheat by shrinking the anchors
        with torch.no_grad():
            cx, cy = src_centers[:, 0], src_centers[:, 1]
            w, h = src_shapes[:, 0], src_shapes[:, 1]
            x1, y1, x2, y2 = (
                matched_boxes[:, 0],
                matched_boxes[:, 1],
                matched_boxes[:, 2],
                matched_boxes[:, 3],
            )

            # Reverse the decoding math to get the target residuals
            L_res = (cx - x1) / w - 0.5
            T_res = (cy - y1) / h - 0.5
            R_res = (x2 - cx) / w - 0.5
            B_res = (y2 - cy) / h - 0.5

            target_res = torch.stack(
                [L_res, T_res, R_res, B_res], dim=-1
            )  # (N_matched, 4)

            # Map the continuous target residuals to the discrete bins
            W = self.weighting_fn.w.to(target_res.device)
            target_res = target_res.clamp(W[0].item(), W[-1].item())

            # Find the two closest bins (left and right)
            idx_right = torch.searchsorted(W, target_res).clamp(1, self.reg_max)
            idx_left = idx_right - 1

            W_left = W[idx_left]
            W_right = W[idx_right]

            # Calculate interpolation weights (soft targets)
            w_left = (W_right - target_res) / (W_right - W_left + 1e-6)
            w_right = (target_res - W_left) / (W_right - W_left + 1e-6)

        # Compute Soft Cross-Entropy (outside no_grad so logits get gradients)
        log_probs = F.log_softmax(src_edge_logits, dim=-1)
        loss_left = (
            -torch.gather(log_probs, -1, idx_left.unsqueeze(-1)).squeeze(-1)
            * w_left
        )
        loss_right = (
            -torch.gather(log_probs, -1, idx_right.unsqueeze(-1)).squeeze(-1)
            * w_right
        )

        loss_fgl = (loss_left + loss_right).sum() / num_boxes

        return loss_fgl

    def forward(
        self, outputs: DetectionOutput, targets: list[DetectionTarget]
    ) -> DetectionLosses:
        """
        outputs: DetectionOutput
        targets: list of DetectionTarget
        """
        # 1. Run Hungarian Matcher
        indices = self.matcher(outputs, targets)

        # 2. Compute normalization factor (number of ground truth boxes)
        num_boxes = sum(len(t.labels) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=outputs.pred_logits.device
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 3. Pre-extract matched targets and flatten indices ONCE
        flat_idx = self._flatten_indices(indices)
        matched_classes = torch.cat(
            [t.labels[match.target_indices] for t, match in zip(targets, indices)]
        )
        matched_boxes = torch.cat(
            [t.boxes[match.target_indices] for t, match in zip(targets, indices)],
            dim=0,
        )

        # 4. Compute all raw losses
        raw_loss_ce = self.loss_labels(outputs, flat_idx, matched_classes, num_boxes)
        raw_loss_bbox, raw_loss_giou = self.loss_boxes(
            outputs, flat_idx, matched_boxes, num_boxes
        )
        raw_loss_fgl = self.loss_fgl(outputs, flat_idx, matched_boxes, num_boxes)

        # 5. Apply weights and return dataclass
        return DetectionLosses(
            loss_ce=raw_loss_ce * self.weights.loss_ce,
            loss_bbox=raw_loss_bbox * self.weights.loss_bbox,
            loss_giou=raw_loss_giou * self.weights.loss_giou,
            loss_fgl=raw_loss_fgl * self.weights.loss_fgl,
        )
