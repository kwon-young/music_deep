import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import generalized_box_iou, box_xyxy_to_cxcywh
from .detector import DFINEWeightingFunction
from music_types import (
    DetectionTarget,
    DetectionOutput,
    DetectionLosses,
    DetectionLossWeights,
    MatchIndices,
    FlattenedIndices,
    MatchedOutputs,
    BoundingBoxes,
    ClassLabels,
    Keypoints,
    BoxShape,
    LabelShape,
    KeypointShape,
    XYXY,
    X1Y1X2Y2,
    PatchUnit,
    TopLeft,
    Batch,
    NumQueries,
    BoxDim,
    KeypointDim,
    CoordDim,
    NumSymbolClasses,
    NumLineClasses,
)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | torch.Tensor = 0.25,
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

    if isinstance(alpha, torch.Tensor) or alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def flatten_indices(indices: list[MatchIndices]) -> FlattenedIndices:
    # Permute predictions following the Hungarian Matcher indices
    batch_idx = torch.cat(
        [
            torch.full_like(match.pred_indices, i)
            for i, match in enumerate(indices)
        ]
    )
    src_idx = torch.cat([match.pred_indices for match in indices])
    return FlattenedIndices(batch=batch_idx, src=src_idx)


class DFINECriterion(nn.Module):
    """
    This class computes the losses for our Dense Patch-as-Predictor model:
    1. Focal Loss (Classification)
    2. L1 Loss (Bounding Box / Keypoints)
    3. GIoU Loss (Bounding Box only)
    4. FGL Loss (D-FINE Fine-Grained Localization for edge distributions)
    """

    def __init__(
        self,
        matcher,
        num_symbol_classes: int,
        num_line_classes: int,
        weights: DetectionLossWeights,
        symbol_weights: list[float],
        line_weights: list[float],
        base_anchor_size: float = 1.0,
        reg_max: int = 32,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_symbol_classes = num_symbol_classes
        self.num_line_classes = num_line_classes
        self.weights = weights
        self.base_anchor_size = base_anchor_size
        self.reg_max = reg_max
        self.gamma = gamma

        self.register_buffer("symbol_weights", torch.tensor(symbol_weights, dtype=torch.float32))
        self.register_buffer("line_weights", torch.tensor(line_weights, dtype=torch.float32))

        # We need the weighting function to map target residuals back to discrete bins
        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def loss_labels(
        self,
        src_logits: torch.Tensor,
        flat_idx: FlattenedIndices,
        matched_classes: torch.Tensor,
        num_boxes: float,
        alpha_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Classification loss (Focal Loss) applied to ALL predictions."""
        # Create a target tensor filled with 0s (Background)
        target_classes = torch.zeros_like(src_logits)

        # Set the matched indices to 1.0 for their specific class
        target_classes[flat_idx.batch, flat_idx.src, matched_classes] = 1.0

        # Compute Focal Loss
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes,
            alpha=alpha_weights,
            gamma=self.gamma,
            reduction="none",
        )
        loss_ce = loss_ce.sum() / num_boxes
        return loss_ce

    def loss_boxes(
        self,
        src_boxes: torch.Tensor,
        matched_boxes: torch.Tensor,
        matched_classes: torch.Tensor,
        class_weights: torch.Tensor,
        num_boxes: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU loss applied ONLY to matched predictions."""
        w = class_weights[matched_classes]

        # Convert to CXCYWH for L1 loss
        src_boxes_cxcywh = box_xyxy_to_cxcywh(src_boxes)
        matched_boxes_cxcywh = box_xyxy_to_cxcywh(matched_boxes)

        # 1. L1 Loss
        loss_bbox = F.l1_loss(
            src_boxes_cxcywh, matched_boxes_cxcywh, reduction="none"
        ).sum(dim=-1)
        loss_bbox = (loss_bbox * w).sum() / num_boxes

        # 2. GIoU Loss
        loss_giou = 1 - torch.diag(
            generalized_box_iou(src_boxes, matched_boxes)
        )
        loss_giou = (loss_giou * w).sum() / num_boxes

        return loss_bbox, loss_giou

    def loss_fgl_symbols(
        self,
        src_edge_logits: torch.Tensor,
        src_centers: torch.Tensor,
        src_shapes: torch.Tensor,
        matched_boxes: torch.Tensor,
        matched_classes: torch.Tensor,
        class_weights: torch.Tensor,
        num_boxes: float,
    ) -> torch.Tensor:
        """D-FINE Fine-Grained Localization Loss applied ONLY to matched predictions."""
        w_class = class_weights[matched_classes]

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

        loss_fgl = (loss_left + loss_right).sum(dim=-1)
        loss_fgl = (loss_fgl * w_class).sum() / num_boxes

        return loss_fgl

    def loss_fgl_lines(
        self,
        src_edge_logits: torch.Tensor,
        src_centers: torch.Tensor,
        src_base_dirs: torch.Tensor,
        matched_keypoints: torch.Tensor,
        matched_classes: torch.Tensor,
        class_weights: torch.Tensor,
        num_lines: float,
    ) -> torch.Tensor:
        w_class = class_weights[matched_classes]

        with torch.no_grad():
            cx, cy = src_centers[:, 0], src_centers[:, 1]
            base_x1, base_y1, base_x2, base_y2 = (
                src_base_dirs[:, 0],
                src_base_dirs[:, 1],
                src_base_dirs[:, 2],
                src_base_dirs[:, 3],
            )
            x1, y1, x2, y2 = (
                matched_keypoints[:, 0],
                matched_keypoints[:, 1],
                matched_keypoints[:, 2],
                matched_keypoints[:, 3],
            )

            res_x1 = (x1 - cx - base_x1) / self.base_anchor_size
            res_y1 = (y1 - cy - base_y1) / self.base_anchor_size
            res_x2 = (x2 - cx - base_x2) / self.base_anchor_size
            res_y2 = (y2 - cy - base_y2) / self.base_anchor_size

            target_res = torch.stack([res_x1, res_y1, res_x2, res_y2], dim=-1)

            W = self.weighting_fn.w.to(target_res.device)
            target_res = target_res.clamp(W[0].item(), W[-1].item())

            idx_right = torch.searchsorted(W, target_res).clamp(1, self.reg_max)
            idx_left = idx_right - 1

            W_left = W[idx_left]
            W_right = W[idx_right]

            w_left = (W_right - target_res) / (W_right - W_left + 1e-6)
            w_right = (target_res - W_left) / (W_right - W_left + 1e-6)

        log_probs = F.log_softmax(src_edge_logits, dim=-1)
        loss_left = (
            -torch.gather(log_probs, -1, idx_left.unsqueeze(-1)).squeeze(-1)
            * w_left
        )
        loss_right = (
            -torch.gather(log_probs, -1, idx_right.unsqueeze(-1)).squeeze(-1)
            * w_right
        )

        loss_fgl = (loss_left + loss_right).sum(dim=-1)
        loss_fgl = (loss_fgl * w_class).sum() / num_lines

        return loss_fgl

    def forward(
        self,
        outputs: DetectionOutput[
            Batch, NumQueries, BoxDim, KeypointDim, CoordDim
        ],
        targets: list[
            DetectionTarget[
                BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
                ClassLabels[LabelShape, NumSymbolClasses],
                Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
                ClassLabels[LabelShape, NumLineClasses],
            ]
        ],
    ) -> DetectionLosses:
        sym_indices, line_indices = self.matcher(outputs, targets)

        num_symbols = max(1, sum(len(t.box_labels.data) for t in targets))
        num_lines = max(1, sum(len(t.keypoint_labels.data) for t in targets))

        # --- Symbols ---
        sym_flat_idx = flatten_indices(sym_indices)
        sym_labels = []
        sym_boxes = []
        for t, match in zip(targets, sym_indices):
            sym_labels.append(t.box_labels.data[match.target_indices])
            sym_boxes.append(t.boxes.data[match.target_indices])
        matched_sym_labels = torch.cat(sym_labels)
        matched_sym_boxes = (
            torch.cat(sym_boxes, dim=0)
            if sym_boxes
            else torch.empty(
                (0, 4), device=outputs.symbols.pred_boxes.data.device
            )
        )

        matched_sym_outputs = MatchedOutputs(
            boxes=outputs.symbols.pred_boxes.data[
                sym_flat_idx.batch, sym_flat_idx.src
            ],
            edge_logits=outputs.symbols.pred_edge_logits.data[
                sym_flat_idx.batch, sym_flat_idx.src
            ],
            centers=outputs.symbols.absolute_centers.data[
                sym_flat_idx.batch, sym_flat_idx.src
            ],
            shapes=outputs.symbols.learnable_shapes.data[
                sym_flat_idx.batch, sym_flat_idx.src
            ],
        )

        loss_ce_sym = self.loss_labels(
            outputs.symbols.pred_logits.data,
            sym_flat_idx,
            matched_sym_labels,
            num_symbols,
            self.symbol_weights,
        )
        if len(matched_sym_boxes) > 0:
            loss_bbox_sym, loss_giou_sym = self.loss_boxes(
                matched_sym_outputs.boxes, 
                matched_sym_boxes, 
                matched_sym_labels,
                self.symbol_weights,
                num_symbols
            )
            loss_fgl_sym = self.loss_fgl_symbols(
                matched_sym_outputs.edge_logits,
                matched_sym_outputs.centers,
                matched_sym_outputs.shapes,
                matched_sym_boxes,
                matched_sym_labels,
                self.symbol_weights,
                num_symbols,
            )
        else:
            loss_bbox_sym = torch.tensor(0.0, device=loss_ce_sym.device)
            loss_giou_sym = torch.tensor(0.0, device=loss_ce_sym.device)
            loss_fgl_sym = torch.tensor(0.0, device=loss_ce_sym.device)

        # --- Lines ---
        line_flat_idx = flatten_indices(line_indices)
        line_labels = []
        line_keypoints = []
        for t, match in zip(targets, line_indices):
            line_labels.append(t.keypoint_labels.data[match.target_indices])
            line_keypoints.append(t.keypoints.data[match.target_indices])
        matched_line_labels = torch.cat(line_labels)
        matched_line_keypoints = (
            torch.cat(line_keypoints, dim=0)
            if line_keypoints
            else torch.empty(
                (0, 4), device=outputs.lines.pred_keypoints.data.device
            )
        )

        loss_ce_line = self.loss_labels(
            outputs.lines.pred_logits.data,
            line_flat_idx,
            matched_line_labels,
            num_lines,
            self.line_weights,
        )

        if len(matched_line_keypoints) > 0:
            matched_line_kp_preds = outputs.lines.pred_keypoints.data[
                line_flat_idx.batch, line_flat_idx.src
            ]
            w_line = self.line_weights[matched_line_labels]
            
            loss_l1_line = F.l1_loss(
                matched_line_kp_preds,
                matched_line_keypoints,
                reduction="none",
            ).sum(dim=-1)
            loss_l1_line = (loss_l1_line * w_line).sum() / num_lines

            loss_fgl_line = self.loss_fgl_lines(
                outputs.lines.pred_endpoint_logits.data[
                    line_flat_idx.batch, line_flat_idx.src
                ],
                outputs.lines.absolute_centers.data[
                    line_flat_idx.batch, line_flat_idx.src
                ],
                outputs.lines.raw_directions.data[
                    line_flat_idx.batch, line_flat_idx.src
                ],
                matched_line_keypoints,
                matched_line_labels,
                self.line_weights,
                num_lines,
            )
        else:
            loss_l1_line = torch.tensor(0.0, device=loss_ce_line.device)
            loss_fgl_line = torch.tensor(0.0, device=loss_ce_line.device)

        return DetectionLosses(
            loss_ce=(loss_ce_sym + loss_ce_line) * self.weights.loss_ce,
            loss_bbox=loss_bbox_sym * self.weights.loss_bbox,
            loss_giou=loss_giou_sym * self.weights.loss_giou,
            loss_fgl=loss_fgl_sym * self.weights.loss_fgl,
            loss_line_l1=loss_l1_line * self.weights.loss_line_l1,
            loss_line_fgl=loss_fgl_line * self.weights.loss_fgl,
        )
