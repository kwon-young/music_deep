import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .box_ops import box_xyxy_to_cxcywh, generalized_box_iou
from music_types import (
    DetectionTarget,
    DetectionOutput,
    MatchIndices,
    BoundingBoxes,
    ClassLabels,
    BoxShape,
    LabelShape,
    XYXY,
    Float1,
    TopLeft,
    Batch,
    NumQueries,
    BoxDim,
    CoordDim,
    NumClasses,
)


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        calc_device: torch.device | None = None,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.calc_device = calc_device
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs can't be 0"
        )

    @torch.no_grad()
    def forward(
        self,
        outputs: DetectionOutput[Batch, NumQueries, BoxDim, CoordDim],
        targets: list[
            DetectionTarget[
                BoundingBoxes[BoxShape, XYXY, Float1, TopLeft],
                ClassLabels[LabelShape, NumClasses],
            ]
        ],
    ) -> list[MatchIndices]:
        """
        Params:
            outputs: DetectionOutput containing predictions
            targets: This is a list of DetectionTarget (len(targets) = batch_size)

        Returns:
            A list of size batch_size, containing MatchIndices where:
                - pred_indices is the indices of the selected predictions (in order)
                - target_indices is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs.pred_logits.data.shape[:2]

        calc_device = self.calc_device if self.calc_device is not None else outputs.pred_logits.data.device

        out_prob = F.sigmoid(outputs.pred_logits.data.flatten(0, 1)).to(
            calc_device
        )
        out_bbox = outputs.pred_boxes.data.flatten(0, 1).to(calc_device)

        tgt_ids = torch.cat([v.labels.data for v in targets]).to(calc_device)
        tgt_bbox = torch.cat([v.boxes.data for v in targets]).to(calc_device)

        # Handle edge case where there are no targets in the batch
        if len(tgt_ids) == 0:
            return [
                MatchIndices(
                    pred_indices=torch.empty(0, dtype=torch.int64),
                    target_indices=torch.empty(0, dtype=torch.int64),
                )
                for _ in range(bs)
            ]

        # 1. Compute the classification cost (Focal Loss approximation)
        out_prob = out_prob[:, tgt_ids]
        neg_cost_class = (
            (1 - self.alpha)
            * (out_prob**self.gamma)
            * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = (
            self.alpha
            * ((1 - out_prob) ** self.gamma)
            * (-(out_prob + 1e-8).log())
        )
        cost_class = pos_cost_class - neg_cost_class

        # 2. Compute the L1 cost between boxes
        # L1 cost is typically computed on [cx, cy, w, h] format
        out_bbox_cxcywh = box_xyxy_to_cxcywh(out_bbox)
        tgt_bbox_cxcywh = box_xyxy_to_cxcywh(tgt_bbox)
        cost_bbox = torch.cdist(out_bbox_cxcywh, tgt_bbox_cxcywh, p=1)

        # 3. Compute the GIoU cost between boxes (requires [x1, y1, x2, y2] format)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1)

        # Handle potential NaNs
        C = torch.nan_to_num(C, nan=1e6)

        sizes = [len(v.boxes.data) for v in targets]

        # Run Hungarian Matching (linear_sum_assignment)
        indices = [
            linear_sum_assignment(c[i].cpu().numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            MatchIndices(
                pred_indices=torch.as_tensor(i, dtype=torch.int64),
                target_indices=torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
