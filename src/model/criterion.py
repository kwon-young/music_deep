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
    EmbedDim,
    LineQueries,
    RefinedLines,
    LineRefinerLosses,
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


def line_fgl_loss(
    edge_logits: torch.Tensor,
    centers: torch.Tensor,
    base_dirs: torch.Tensor,
    target_keypoints: torch.Tensor,
    line_weights: torch.Tensor,
    bins: torch.Tensor,
    base_anchor_size: float,
    num_lines: float,
) -> torch.Tensor:
    """D-FINE FGL for line endpoints.

    The target residual of each endpoint coordinate is expressed in the frame of
    the stage-0 prediction: relative to `center + base_dir`, in units of
    `|base_dir| + base_anchor_size`. It is split between its two neighbouring bins.
    """
    with torch.no_grad():
        scale = base_dirs.abs() + base_anchor_size
        target_res = (target_keypoints - centers.repeat(1, 2) - base_dirs) / scale
        target_res = target_res.clamp(bins[0], bins[-1])

        idx_right = torch.searchsorted(bins, target_res).clamp(1, len(bins) - 1)
        idx_left = idx_right - 1

        bins_left = bins[idx_left]
        bins_right = bins[idx_right]

        w_left = (bins_right - target_res) / (bins_right - bins_left + 1e-6)
        w_right = (target_res - bins_left) / (bins_right - bins_left + 1e-6)

    log_probs = F.log_softmax(edge_logits, dim=-1)
    loss_left = (
        -torch.gather(log_probs, -1, idx_left.unsqueeze(-1)).squeeze(-1)
        * w_left
    )
    loss_right = (
        -torch.gather(log_probs, -1, idx_right.unsqueeze(-1)).squeeze(-1)
        * w_right
    )

    loss_fgl = (loss_left + loss_right).sum(dim=-1)
    return (loss_fgl * line_weights).sum() / num_lines


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

    symbol_weights: torch.Tensor
    line_weights: torch.Tensor

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

        self.register_buffer(
            "symbol_weights", torch.tensor(symbol_weights, dtype=torch.float32)
        )
        self.register_buffer(
            "line_weights", torch.tensor(line_weights, dtype=torch.float32)
        )

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
        return line_fgl_loss(
            src_edge_logits,
            src_centers,
            src_base_dirs,
            matched_keypoints,
            class_weights[matched_classes],
            self.weighting_fn.w,
            self.base_anchor_size,
            num_lines,
        )

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
                num_symbols,
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


class LineRefinerCriterion(nn.Module):
    """Deep supervision of the line refiner: L1 + FGL on the output of every layer.

    Matching is done once on the stage-0 predictions and shared by all layers.
    """

    line_weights: torch.Tensor

    def __init__(
        self,
        line_weights: list[float],
        loss_line_l1: float,
        loss_line_fgl: float,
        base_anchor_size: float = 1.0,
        reg_max: int = 32,
    ):
        super().__init__()
        self.loss_line_l1 = loss_line_l1
        self.loss_line_fgl = loss_line_fgl
        self.base_anchor_size = base_anchor_size
        self.register_buffer(
            "line_weights", torch.tensor(line_weights, dtype=torch.float32)
        )
        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward[B: Batch, Q: NumQueries, D: EmbedDim](
        self,
        refined: RefinedLines[B, Q, KeypointDim],
        queries: LineQueries[B, Q, D, KeypointDim, CoordDim],
        target_keypoints: Keypoints[
            tuple[B, Q, KeypointDim], X1Y1X2Y2, PatchUnit, TopLeft
        ],
        target_labels: ClassLabels[tuple[B, Q], NumLineClasses],
    ) -> LineRefinerLosses:
        weights = self.line_weights[target_labels.data].flatten()
        num_lines = float(max(1, weights.numel()))
        centers = queries.absolute_centers.data.flatten(0, 1)
        base_dirs = queries.raw_directions.data.flatten(0, 1)
        targets = target_keypoints.data.flatten(0, 1)

        losses_l1: list[torch.Tensor] = []
        losses_fgl: list[torch.Tensor] = []
        for keypoints, logits in zip(refined.keypoints, refined.endpoint_logits):
            l1 = F.l1_loss(
                keypoints.data.flatten(0, 1), targets, reduction="none"
            ).sum(dim=-1)
            losses_l1.append(
                (l1 * weights).sum() / num_lines * self.loss_line_l1
            )
            fgl = line_fgl_loss(
                logits.data.flatten(0, 1),
                centers,
                base_dirs,
                targets,
                weights,
                self.weighting_fn.w,
                self.base_anchor_size,
                num_lines,
            )
            losses_fgl.append(fgl * self.loss_line_fgl)

        return LineRefinerLosses(
            loss_line_l1=losses_l1, loss_line_fgl=losses_fgl
        )
