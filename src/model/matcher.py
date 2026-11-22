import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from .box_ops import box_xyxy_to_cxcywh, elementwise_generalized_box_iou
from music_types import (
    DetectionTarget,
    DetectionOutput,
    LineOutput,
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
        max_pair_elements: int = 1 << 24,
    ):
        super().__init__()
        self.max_pair_elements = max_pair_elements
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

    def _valid_pairs(
        self,
        distances: Callable[[slice], torch.Tensor],
        num_queries: int,
        num_targets: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(query, target) pairs within `radius_patches`, plus the `top_k` closest
        queries of every target.

        Both criteria are per target, so targets are processed in chunks that keep
        every [chunk, num_queries] intermediate under `max_pair_elements` (dense
        pages reach ~50k queries x ~2k targets). Targets-major layout makes the
        top-k reduction contiguous (~9x faster than along dim 0). The pairs are
        returned in the (query, target) row-major order of a dense `torch.where`.
        """
        chunk = max(1, self.max_pair_elements // max(1, num_queries))
        top_k = min(self.top_k, num_queries)
        rows: list[torch.Tensor] = []
        cols: list[torch.Tensor] = []
        for start in range(0, num_targets, chunk):
            distance = distances(slice(start, start + chunk))
            valid_mask = distance <= self.radius_patches
            topk_idx = torch.topk(distance, k=top_k, dim=-1, largest=False).indices
            valid_mask.scatter_(-1, topk_idx, True)
            col, row = torch.where(valid_mask)
            rows.append(row)
            cols.append(col + start)
        row, col = torch.cat(rows), torch.cat(cols)
        order = torch.argsort(row * num_targets + col)
        return row[order], col[order]

    def _get_valid_pairs_boxes(
        self, out_bbox: torch.Tensor, tgt_bbox: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Box-to-box gap distance."""

        def distances(targets: slice) -> torch.Tensor:
            tgt = tgt_bbox[targets]
            dx = torch.clamp(
                torch.max(
                    tgt[:, None, 0] - out_bbox[None, :, 2],
                    out_bbox[None, :, 0] - tgt[:, None, 2],
                ),
                min=0,
            )
            dy = torch.clamp(
                torch.max(
                    tgt[:, None, 1] - out_bbox[None, :, 3],
                    out_bbox[None, :, 1] - tgt[:, None, 3],
                ),
                min=0,
            )
            return torch.sqrt(dx**2 + dy**2)

        return self._valid_pairs(distances, len(out_bbox), len(tgt_bbox))

    def _get_valid_pairs_keypoints(
        self, out_kp: torch.Tensor, tgt_kp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance from the midpoint of each predicted line to each GT segment."""
        P = (out_kp[:, :2] + out_kp[:, 2:]) / 2.0

        def distances(targets: slice) -> torch.Tensor:
            A = tgt_kp[targets, :2]
            AB = tgt_kp[targets, 2:] - A
            AB_squared = (AB**2).sum(dim=-1).clamp(min=1e-6)
            AP = P.unsqueeze(0) - A.unsqueeze(1)
            t = ((AP * AB.unsqueeze(1)).sum(dim=-1) / AB_squared.unsqueeze(1)).clamp(
                min=0.0, max=1.0
            )
            Proj = A.unsqueeze(1) + t.unsqueeze(-1) * AB.unsqueeze(1)
            return torch.norm(P.unsqueeze(0) - Proj, p=2, dim=-1)

        return self._valid_pairs(distances, len(out_kp), len(tgt_kp))

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

        sym_indices = []
        for b in range(bs):
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

        return sym_indices, self.match_lines(outputs.lines, targets)

    @torch.no_grad()
    def match_lines(
        self,
        lines: LineOutput[Batch, NumQueries, KeypointDim, CoordDim],
        targets: list[
            DetectionTarget[
                BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
                ClassLabels[LabelShape, NumSymbolClasses],
                Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
                ClassLabels[LabelShape, NumLineClasses],
            ]
        ],
    ) -> list[MatchIndices]:
        calc_device = (
            self.calc_device
            if self.calc_device is not None
            else lines.pred_logits.data.device
        )
        line_out_prob = F.sigmoid(lines.pred_logits.data).to(calc_device)
        line_out_kp = lines.pred_keypoints.data.to(calc_device)

        line_indices = []
        for b in range(line_out_prob.shape[0]):
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

        return line_indices
