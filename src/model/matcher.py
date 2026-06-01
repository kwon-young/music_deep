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
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs can't be 0"
        )

    @torch.no_grad()
    def forward(
        self,
        outputs: DetectionOutput,
        targets: list[
            DetectionTarget[
                BoundingBoxes[BoxShape, XYXY, Float1, TopLeft],
                ClassLabels[LabelShape],
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
        bs, num_queries = outputs.pred_logits.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # Using Focal Loss approximation for probabilities
        out_prob = F.sigmoid(
            outputs.pred_logits.flatten(0, 1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs.pred_boxes.flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v.labels.data for v in targets])
        tgt_bbox = torch.cat([v.boxes.data for v in targets])

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
        C = C.view(bs, num_queries, -1).cpu()

        # Handle potential NaNs
        C = torch.nan_to_num(C, nan=1e6)

        sizes = [len(v.boxes.data) for v in targets]

        # Run Hungarian Matching (linear_sum_assignment)
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            MatchIndices(
                pred_indices=torch.as_tensor(i, dtype=torch.int64),
                target_indices=torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
