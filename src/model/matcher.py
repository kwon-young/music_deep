import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from .box_ops import box_xyxy_to_cxcywh
from music_types import (
    DetectionTarget,
    DetectionOutput,
    MatchIndices,
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


@torch.jit.script
def greedy_matcher_gpu(
    sorted_rows: torch.Tensor,
    sorted_cols: torch.Tensor,
    num_queries: int,
    num_targets: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A fully GPU-native greedy bipartite matcher compiled via TorchScript.
    Avoids CPU-GPU syncs and Python loop overhead.
    """
    device = sorted_rows.device
    assigned_preds = torch.zeros(num_queries, dtype=torch.bool, device=device)
    assigned_targets = torch.zeros(num_targets, dtype=torch.bool, device=device)

    # Pre-allocate max possible size to avoid dynamic list appending on GPU
    match_preds = torch.empty_like(sorted_rows)
    match_targets = torch.empty_like(sorted_cols)

    count = 0
    for i in range(sorted_rows.size(0)):
        r = sorted_rows[i]
        c = sorted_cols[i]
        if not assigned_preds[r] and not assigned_targets[c]:
            match_preds[count] = r
            match_targets[count] = c
            assigned_preds[r] = True
            assigned_targets[c] = True
            count += 1
            if count == num_targets:
                break

    return match_preds[:count], match_targets[:count]


def elementwise_generalized_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Generalized IoU element-wise for two 1D lists of boxes.
    boxes1 and boxes2 must have the same shape [K, 4].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union

    lt_hull = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_hull = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh_hull = (rb_hull - lt_hull).clamp(min=0)
    area_hull = wh_hull[:, 0] * wh_hull[:, 1]

    return iou - (area_hull - union) / area_hull


class HungarianMatcher(nn.Module):
    """
    This class computes a sparse assignment between the targets and the predictions of the network.
    It uses box-to-box distance to prune impossible matches, drastically speeding up training on large crops
    while keeping VRAM usage extremely low.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        calc_device: torch.device | None = None,
        radius_patches: float = 4.0,
        top_k: int = 10,
        matcher_type: str = "scipy",
    ):
        super().__init__()
        self.matcher_type = matcher_type
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.calc_device = calc_device

        self.top_k = top_k
        self.radius_patches = radius_patches

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs can't be 0"
        )

    def _get_valid_pairs_boxes(
        self, out_bbox: torch.Tensor, tgt_bbox: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the dense distance matrix to find valid pairs for boxes.
        """
        dx = torch.clamp(
            torch.max(
                tgt_bbox[None, :, 0] - out_bbox[:, None, 2],
                out_bbox[:, None, 0] - tgt_bbox[None, :, 2],
            ),
            min=0,
        )
        dy = torch.clamp(
            torch.max(
                tgt_bbox[None, :, 1] - out_bbox[:, None, 3],
                out_bbox[:, None, 1] - tgt_bbox[None, :, 3],
            ),
            min=0,
        )
        box_distances = torch.sqrt(dx**2 + dy**2)

        valid_mask = box_distances <= self.radius_patches

        # Top-K Fallback: Ensure every GT box has at least K valid predictions
        topk = min(self.top_k, len(out_bbox))
        topk_idx = torch.topk(
            box_distances, k=topk, dim=0, largest=False
        ).indices
        valid_mask.scatter_(0, topk_idx, True)

        return cast(tuple[torch.Tensor, torch.Tensor], torch.where(valid_mask))

    def _get_valid_pairs_keypoints(
        self, out_kp: torch.Tensor, tgt_kp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the true Point-to-Line-Segment distance.
        Uses the midpoint of the predicted line (patch center) to the GT line segment.
        """
        # 1. Midpoint of predictions (Patch Centers) -> P: [N, 2]
        p_x = (out_kp[:, 0] + out_kp[:, 2]) / 2.0
        p_y = (out_kp[:, 1] + out_kp[:, 3]) / 2.0
        P = torch.stack([p_x, p_y], dim=-1)

        # 2. Target line segments A -> B -> [M, 2]
        A = tgt_kp[:, :2]
        B = tgt_kp[:, 2:]

        AB = B - A
        AB_squared = (
            (AB**2).sum(dim=-1).clamp(min=1e-6)
        )  # Avoid division by zero

        # 3. Vectorized projection of all N points onto all M segments
        AP = P.unsqueeze(1) - A.unsqueeze(0)  # [N, M, 2]
        dot_AP_AB = (AP * AB.unsqueeze(0)).sum(dim=-1)  # [N, M]

        # Calculate projection scalar 't' and clamp to [0, 1] to keep it on the segment
        t = (dot_AP_AB / AB_squared.unsqueeze(0)).clamp(min=0.0, max=1.0)

        # 4. Find the exact closest point on the segment and compute distance
        Proj = A.unsqueeze(0) + t.unsqueeze(-1) * AB.unsqueeze(0)  # [N, M, 2]
        kp_distances = torch.norm(P.unsqueeze(1) - Proj, p=2, dim=-1)  # [N, M]

        # 5. Apply radius and Top-K fallback
        valid_mask = kp_distances <= self.radius_patches

        topk = min(self.top_k, len(out_kp))
        topk_idx = torch.topk(
            kp_distances, k=topk, dim=0, largest=False
        ).indices
        valid_mask.scatter_(0, topk_idx, True)

        return cast(tuple[torch.Tensor, torch.Tensor], torch.where(valid_mask))

    def _match_modality(
        self,
        out_prob: torch.Tensor,
        out_coords: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_coords: torch.Tensor,
        is_box: bool,
        calc_device: torch.device,
    ) -> MatchIndices:
        num_queries = len(out_prob)
        num_targets = len(tgt_ids)

        if num_targets == 0:
            return MatchIndices(
                pred_indices=torch.empty(0, dtype=torch.int64),
                target_indices=torch.empty(0, dtype=torch.int64),
            )

        if is_box:
            row_idx, col_idx = self._get_valid_pairs_boxes(
                out_coords, tgt_coords
            )
        else:
            row_idx, col_idx = self._get_valid_pairs_keypoints(
                out_coords, tgt_coords
            )

        valid_out_prob = out_prob[row_idx]
        valid_tgt_ids = tgt_ids[col_idx]
        valid_out_coords = out_coords[row_idx]
        valid_tgt_coords = tgt_coords[col_idx]

        # Classification Cost
        prob_for_target = valid_out_prob[
            torch.arange(len(row_idx)), valid_tgt_ids
        ]
        neg_cost_class = (
            (1 - self.alpha)
            * (prob_for_target**self.gamma)
            * (-(1 - prob_for_target + 1e-8).log())
        )
        pos_cost_class = (
            self.alpha
            * ((1 - prob_for_target) ** self.gamma)
            * (-(prob_for_target + 1e-8).log())
        )
        cost_class_1d = pos_cost_class - neg_cost_class

        if is_box:
            # BBox L1 Cost
            valid_out_cxcywh = box_xyxy_to_cxcywh(valid_out_coords)
            valid_tgt_cxcywh = box_xyxy_to_cxcywh(valid_tgt_coords)
            cost_coord_1d = F.l1_loss(
                valid_out_cxcywh, valid_tgt_cxcywh, reduction="none"
            ).sum(dim=-1)

            # GIoU Cost
            cost_giou_1d = -elementwise_generalized_box_iou(
                valid_out_coords, valid_tgt_coords
            )

            valid_costs = (
                self.cost_bbox * cost_coord_1d
                + self.cost_class * cost_class_1d
                + self.cost_giou * cost_giou_1d
            )
        else:
            # Keypoint L1 Cost
            cost_coord_1d = F.l1_loss(
                valid_out_coords, valid_tgt_coords, reduction="none"
            ).sum(dim=-1)

            valid_costs = (
                self.cost_bbox * cost_coord_1d + self.cost_class * cost_class_1d
            )

        if self.matcher_type == "scipy":
            sparse_cost_matrix = csr_matrix(
                (
                    valid_costs.cpu().numpy(),
                    (row_idx.cpu().numpy(), col_idx.cpu().numpy()),
                ),
                shape=(num_queries, num_targets),
            )

            row_ind, col_ind = min_weight_full_bipartite_matching(
                sparse_cost_matrix
            )

            return MatchIndices(
                pred_indices=torch.as_tensor(row_ind, dtype=torch.int64),
                target_indices=torch.as_tensor(col_ind, dtype=torch.int64),
            )
        elif self.matcher_type == "greedy":
            if len(valid_costs) == 0:
                row_ind = torch.empty(0, dtype=torch.int64, device=calc_device)
                col_ind = torch.empty(0, dtype=torch.int64, device=calc_device)
            else:
                _, sort_idx = torch.sort(valid_costs)
                sorted_rows = row_idx[sort_idx]
                sorted_cols = col_idx[sort_idx]

                row_ind, col_ind = greedy_matcher_gpu(
                    sorted_rows, sorted_cols, num_queries, num_targets
                )

            return MatchIndices(
                pred_indices=row_ind,
                target_indices=col_ind,
            )
        else:
            raise ValueError(f"Unknown matcher_type: {self.matcher_type}")

    @torch.no_grad()
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
    ) -> tuple[list[MatchIndices], list[MatchIndices]]:
        bs = outputs.symbols.pred_logits.data.shape[0]
        calc_device = (
            self.calc_device
            if self.calc_device is not None
            else outputs.symbols.pred_logits.data.device
        )

        sym_out_prob = F.sigmoid(outputs.symbols.pred_logits.data).to(
            calc_device
        )
        sym_out_bbox = outputs.symbols.pred_boxes.data.to(calc_device)

        line_out_prob = F.sigmoid(outputs.lines.pred_logits.data).to(
            calc_device
        )
        line_out_kp = outputs.lines.pred_keypoints.data.to(calc_device)

        sym_indices = []
        line_indices = []

        for b in range(bs):
            # Symbols
            sym_tgt_ids = targets[b].box_labels.data.to(calc_device)
            sym_tgt_bbox = targets[b].boxes.data.to(calc_device)
            sym_match = self._match_modality(
                sym_out_prob[b],
                sym_out_bbox[b],
                sym_tgt_ids,
                sym_tgt_bbox,
                True,
                calc_device,
            )
            sym_indices.append(sym_match)

            # Lines
            line_tgt_ids = targets[b].keypoint_labels.data.to(calc_device)
            line_tgt_kp = targets[b].keypoints.data.to(calc_device)
            line_match = self._match_modality(
                line_out_prob[b],
                line_out_kp[b],
                line_tgt_ids,
                line_tgt_kp,
                False,
                calc_device,
            )
            line_indices.append(line_match)

        return sym_indices, line_indices
