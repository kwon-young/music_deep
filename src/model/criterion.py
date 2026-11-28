import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import box_xyxy_to_cxcywh, paired_generalized_box_iou
from typing import Literal

from .assignment import PatchSetAssignment, SlotLabels, on_line_quality, pair_quality
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
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU loss applied ONLY to matched predictions."""
        w = class_weights[matched_classes]
        if sample_weights is not None:
            w = w * sample_weights

        # Convert to CXCYWH for L1 loss
        src_boxes_cxcywh = box_xyxy_to_cxcywh(src_boxes)
        matched_boxes_cxcywh = box_xyxy_to_cxcywh(matched_boxes)

        # 1. L1 Loss
        loss_bbox = F.l1_loss(
            src_boxes_cxcywh, matched_boxes_cxcywh, reduction="none"
        ).sum(dim=-1)
        loss_bbox = (loss_bbox * w).sum() / num_boxes

        # 2. GIoU Loss
        loss_giou = 1 - paired_generalized_box_iou(src_boxes, matched_boxes)
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
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """D-FINE Fine-Grained Localization Loss applied ONLY to matched predictions."""
        w_class = class_weights[matched_classes]
        if sample_weights is not None:
            w_class = w_class * sample_weights

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
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_class = class_weights[matched_classes]
        if sample_weights is not None:
            w_class = w_class * sample_weights

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

            scale_x1 = torch.abs(base_x1) + self.base_anchor_size
            scale_y1 = torch.abs(base_y1) + self.base_anchor_size
            scale_x2 = torch.abs(base_x2) + self.base_anchor_size
            scale_y2 = torch.abs(base_y2) + self.base_anchor_size

            res_x1 = (x1 - cx - base_x1) / scale_x1
            res_y1 = (y1 - cy - base_y1) / scale_y1
            res_x2 = (x2 - cx - base_x2) / scale_x2
            res_y2 = (y2 - cy - base_y2) / scale_y2

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
        indices: tuple[list[MatchIndices], list[MatchIndices]] | None = None,
    ) -> DetectionLosses:
        """``indices``: the (symbol, line) matching of ``outputs`` to ``targets`` when the caller
        already computed it (e.g. to share the line assignment with the endpoint refiner)."""
        sym_indices, line_indices = (
            self.matcher(outputs, targets) if indices is None else indices
        )

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
            loss_presence=torch.zeros_like(loss_ce_line),
        )


type VoterWeighting = Literal["inverse", "inverse_sqrt", "none"]


def quality_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    alpha: float | torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """GFL quality focal loss with the focal alpha: ``alpha_t |q - p|^gamma BCE(p, q)``, where
    alpha_t is alpha on ``positive`` entries and 1 - alpha elsewhere. With q in {0, 1} it is
    exactly ``sigmoid_focal_loss``; with a soft q the positive entry is pulled to q, not 1."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    alpha_t = torch.where(positive, alpha, 1 - alpha) if isinstance(alpha, torch.Tensor) else (
        torch.where(positive, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    )
    return alpha_t * (target - p).abs().pow(gamma) * ce


def voter_weights(voters: torch.Tensor, mode: VoterWeighting) -> torch.Tensor:
    n = voters.clamp(min=1).float()
    if mode == "inverse":
        return 1 / n
    if mode == "inverse_sqrt":
        return n.rsqrt()
    return torch.ones_like(n)


class PatchSetCriterion(DFINECriterion):
    """Losses of the patch-set assignment (PATCH_SET_ASSIGNMENT.md, section 3).

    Every slot is positive, ignored or negative (``model.assignment``), and the presence and
    class losses share that labelling:

    * presence: binary focal loss, target 1 on positives and 0 on negatives;
    * class: quality focal loss on the per-class sigmoids, target q = IoU / OKS of the slot's
      own (detached) prediction on its GT class (1 with quality targets off), 0 elsewhere,
      with the per-class alpha weights. Lines use ``line_quality`` (on the line, extent
      ignored) instead of OKS: the coarse OKS of a far voter is ~0 (median 0.006 on a full
      page), which would train the class score of real positives toward 0;
    * regression (L1, GIoU, FGL): on every positive.

    Positive slots are weighted by ``voter_weights`` of their GT's voter count, so a GT weighs
    about what its single match weighed before, whatever its number of voters; ignored slots
    get no presence or class loss. All terms are normalised by the number of GT.
    """

    def __init__(
        self,
        num_symbol_classes: int,
        num_line_classes: int,
        weights: DetectionLossWeights,
        symbol_weights: list[float],
        line_weights: list[float],
        base_anchor_size: float = 1.0,
        reg_max: int = 32,
        gamma: float = 2.0,
        presence_alpha: float = 0.5,
        voter_weighting: VoterWeighting = "inverse",
        presence_weighting: VoterWeighting = "none",
        symbol_quality_targets: bool = True,
        line_quality_targets: bool = True,
        line_thickness: float = 1 / 16,
        line_perp_sigma: float = 0.1,
    ) -> None:
        super().__init__(
            None, num_symbol_classes, num_line_classes, weights, symbol_weights, line_weights,
            base_anchor_size, reg_max, gamma,
        )
        self.presence_alpha = presence_alpha
        self.voter_weighting: VoterWeighting = voter_weighting
        self.presence_weighting: VoterWeighting = presence_weighting
        self.symbol_quality_targets = symbol_quality_targets
        self.line_quality_targets = line_quality_targets
        self.line_thickness = line_thickness
        self.line_perp_sigma = line_perp_sigma

    def line_quality(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Class target of a line positive: how well its prediction lies on its GT line
        (perpendicular, extent ignored). A voter cannot see a far endpoint, and the coarse OKS of
        far voters is ~0; where the line ends is the job of the per-endpoint spread at fusion."""
        return on_line_quality(pred.detach().float(), gt.float(), self.line_perp_sigma).clamp(min=0.01)

    def score_losses(
        self,
        logits: torch.Tensor,
        presence: torch.Tensor,
        labels: list[SlotLabels],
        flat: FlattenedIndices,
        classes: torch.Tensor,
        quality: torch.Tensor,
        weight: torch.Tensor,
        voters: torch.Tensor,
        alpha: torch.Tensor,
        num_gt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(class loss, presence loss) over all slots of one modality. The presence loss uses
        its own voter weighting: with "none" every voter is a full positive (presence is a
        per-slot question) and the loss is normalised by the number of voters, as in RetinaNet;
        otherwise by the number of GT like the other terms."""
        ignored = torch.stack([s.ignored for s in labels]).to(logits.device)
        base = (~ignored).to(logits.dtype)
        row = base.clone()
        row[flat.batch, flat.src] = weight.to(logits.dtype)
        presence_row = base.clone()
        presence_row[flat.batch, flat.src] = voter_weights(voters, self.presence_weighting).to(logits.dtype)
        presence_norm = max(1.0, float(voters.numel())) if self.presence_weighting == "none" else num_gt
        target = torch.zeros_like(logits)
        positive = torch.zeros_like(logits, dtype=torch.bool)
        target[flat.batch, flat.src, classes] = quality.to(logits.dtype)
        positive[flat.batch, flat.src, classes] = True
        loss_cls = quality_focal_loss(logits, target, positive, alpha, self.gamma)
        loss_cls = (loss_cls * row.unsqueeze(-1)).sum() / num_gt
        is_voter = torch.zeros_like(presence, dtype=torch.bool)
        is_voter[flat.batch, flat.src] = True
        loss_presence = quality_focal_loss(
            presence, is_voter.to(presence.dtype), is_voter, self.presence_alpha, self.gamma
        )
        return loss_cls, (loss_presence * presence_row).sum() / presence_norm

    def forward(
        self,
        outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
        targets: list[
            DetectionTarget[
                BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
                ClassLabels[LabelShape, NumSymbolClasses],
                Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
                ClassLabels[LabelShape, NumLineClasses],
            ]
        ],
        indices: tuple[list[MatchIndices], list[MatchIndices]] | None = None,
        assignment: PatchSetAssignment | None = None,
    ) -> DetectionLosses:
        if assignment is None:
            raise ValueError("PatchSetCriterion needs the patch-set assignment")
        sym_presence, line_presence = outputs.symbols.pred_presence, outputs.lines.pred_presence
        if sym_presence is None or line_presence is None:
            raise ValueError("PatchSetCriterion needs a detector with presence logits")
        num_symbols = max(1, sum(len(t.box_labels.data) for t in targets))
        num_lines = max(1, sum(len(t.keypoint_labels.data) for t in targets))
        device = outputs.symbols.pred_boxes.data.device

        def gather(labels: list[SlotLabels], gt: list[torch.Tensor], gt_labels: list[torch.Tensor]
                   ) -> tuple[FlattenedIndices, torch.Tensor, torch.Tensor, torch.Tensor]:
            flat = flatten_indices([s.match for s in labels])
            flat = FlattenedIndices(batch=flat.batch.to(device), src=flat.src.to(device))
            idx = [s.match.target_indices.to(device) for s in labels]
            boxes = torch.cat([g.to(device)[i] for g, i in zip(gt, idx)]) if idx else torch.empty(0, 4, device=device)
            classes = torch.cat([g.to(device)[i] for g, i in zip(gt_labels, idx)])
            voters = torch.cat([s.voters.to(device)[i] for s, i in zip(labels, idx)])
            return flat, boxes, classes, voters

        # --- Symbols ---
        sym_flat, sym_gt, sym_cls, sym_voters = gather(
            assignment.symbols, [t.boxes.data for t in targets], [t.box_labels.data for t in targets]
        )
        sym_w = voter_weights(sym_voters, self.voter_weighting)
        sym_pred = outputs.symbols.pred_boxes.data[sym_flat.batch, sym_flat.src]
        sym_q = (
            pair_quality(sym_pred.detach().float(), sym_gt.float(), False).clamp(min=0.01)
            if self.symbol_quality_targets else torch.ones_like(sym_w)
        )
        loss_ce_sym, loss_pres_sym = self.score_losses(
            outputs.symbols.pred_logits.data, sym_presence.data, assignment.symbols, sym_flat,
            sym_cls, sym_q, sym_w, sym_voters, self.symbol_weights, num_symbols,
        )
        if sym_gt.shape[0] > 0:
            loss_bbox_sym, loss_giou_sym = self.loss_boxes(
                sym_pred, sym_gt, sym_cls, self.symbol_weights, num_symbols, sym_w
            )
            loss_fgl_sym = self.loss_fgl_symbols(
                outputs.symbols.pred_edge_logits.data[sym_flat.batch, sym_flat.src],
                outputs.symbols.absolute_centers.data[sym_flat.batch, sym_flat.src],
                outputs.symbols.learnable_shapes.data[sym_flat.batch, sym_flat.src],
                sym_gt, sym_cls, self.symbol_weights, num_symbols, sym_w,
            )
        else:
            loss_bbox_sym = loss_giou_sym = loss_fgl_sym = torch.zeros((), device=device)

        # --- Lines ---
        line_flat, line_gt, line_cls, line_voters = gather(
            assignment.lines, [t.keypoints.data for t in targets], [t.keypoint_labels.data for t in targets]
        )
        line_w = voter_weights(line_voters, self.voter_weighting)
        line_pred = outputs.lines.pred_keypoints.data[line_flat.batch, line_flat.src]
        line_q = (
            self.line_quality(line_pred, line_gt) if self.line_quality_targets else torch.ones_like(line_w)
        )
        loss_ce_line, loss_pres_line = self.score_losses(
            outputs.lines.pred_logits.data, line_presence.data, assignment.lines, line_flat,
            line_cls, line_q, line_w, line_voters, self.line_weights, num_lines,
        )
        if line_gt.shape[0] > 0:
            w_line = self.line_weights[line_cls] * line_w
            loss_l1_line = (F.l1_loss(line_pred, line_gt, reduction="none").sum(dim=-1) * w_line).sum() / num_lines
            loss_fgl_line = self.loss_fgl_lines(
                outputs.lines.pred_endpoint_logits.data[line_flat.batch, line_flat.src],
                outputs.lines.absolute_centers.data[line_flat.batch, line_flat.src],
                outputs.lines.raw_directions.data[line_flat.batch, line_flat.src],
                line_gt, line_cls, self.line_weights, num_lines, line_w,
            )
        else:
            loss_l1_line = loss_fgl_line = torch.zeros((), device=device)

        return DetectionLosses(
            loss_ce=(loss_ce_sym + loss_ce_line) * self.weights.loss_ce,
            loss_bbox=loss_bbox_sym * self.weights.loss_bbox,
            loss_giou=loss_giou_sym * self.weights.loss_giou,
            loss_fgl=loss_fgl_sym * self.weights.loss_fgl,
            loss_line_l1=loss_l1_line * self.weights.loss_line_l1,
            loss_line_fgl=loss_fgl_line * self.weights.loss_fgl,
            loss_presence=(loss_pres_sym + loss_pres_line) * self.weights.loss_presence,
        )
