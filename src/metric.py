import torch
from model.box_ops import box_iou
from music_types import (
    DetectionTarget,
    DetectionOutput,
    MatchIndices,
    BoundingBoxes,
    ClassLabels,
    Keypoints,
    Batch,
    NumQueries,
    BoxDim,
    KeypointDim,
    CoordDim,
    BoxShape,
    LabelShape,
    KeypointShape,
    XYXY,
    X1Y1X2Y2,
    TopLeft,
    PatchUnit,
    NumSymbolClasses,
    NumLineClasses,
)


@torch.no_grad()
def compute_map_50(
    outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
    targets: list[
        DetectionTarget[
            BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumSymbolClasses],
            Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumLineClasses],
        ]
    ],
    num_classes: int,
) -> float:
    """Computes the Mean Average Precision at IoU threshold 0.5 for symbols."""
    if num_classes == 0:
        return 0.0

    pred_logits = outputs.symbols.pred_logits.data
    pred_boxes = outputs.symbols.pred_boxes.data
    probs = torch.sigmoid(pred_logits)
    max_probs, pred_labels = probs.max(dim=-1)

    aps: list[float] = []

    for c in range(num_classes):
        # Gather all predictions for class c across the batch
        class_preds: list[tuple[float, int, torch.Tensor]] = []
        for b in range(len(targets)):
            mask = pred_labels[b] == c
            if not mask.any():
                continue
            b_probs = max_probs[b][mask]
            b_boxes = pred_boxes[b][mask]
            for p, box in zip(b_probs, b_boxes):
                class_preds.append((p.item(), b, box))

        # Sort predictions by confidence descending
        class_preds.sort(key=lambda x: x[0], reverse=True)

        total_gt = 0
        gt_matched: list[torch.Tensor] = []
        gt_boxes_per_img: dict[int, torch.Tensor] = {}

        # Gather all ground truth boxes for class c
        for b, target in enumerate(targets):
            gt_labels = target.box_labels.data
            gt_boxes = target.boxes.data
            mask = gt_labels == c
            c_gt_boxes = gt_boxes[mask]
            total_gt += len(c_gt_boxes)
            gt_boxes_per_img[b] = c_gt_boxes
            gt_matched.append(torch.zeros(len(c_gt_boxes), dtype=torch.bool))

        if total_gt == 0:
            continue
        if len(class_preds) == 0:
            aps.append(0.0)
            continue

        tps = torch.zeros(len(class_preds))
        fps = torch.zeros(len(class_preds))

        # Match predictions to ground truth
        for i, (prob, b, pred_box) in enumerate(class_preds):
            c_gt_boxes = gt_boxes_per_img[b]
            if len(c_gt_boxes) == 0:
                fps[i] = 1
                continue

            ious, _ = box_iou(pred_box.unsqueeze(0), c_gt_boxes)
            max_iou, max_idx = ious.squeeze(0).max(dim=0)

            if max_iou >= 0.5 and not gt_matched[b][max_idx]:
                tps[i] = 1
                gt_matched[b][max_idx] = True
            else:
                fps[i] = 1

        # Compute Precision-Recall curve
        tps_cum = torch.cumsum(tps, dim=0)
        fps_cum = torch.cumsum(fps, dim=0)
        recalls = tps_cum / total_gt
        precisions = tps_cum / (tps_cum + fps_cum)

        # Compute exact Area Under Curve (all-point interpolation)
        precisions = torch.cat(
            [torch.tensor([0.0]), precisions, torch.tensor([0.0])]
        )
        recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])

        indices = torch.where(recalls[1:] != recalls[:-1])[0]
        ap = torch.sum(
            (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
        )
        aps.append(ap.item())

    if len(aps) == 0:
        return 0.0
    return sum(aps) / len(aps)


@torch.no_grad()
def compute_mean_iou(
    outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
    targets: list[
        DetectionTarget[
            BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumSymbolClasses],
            Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumLineClasses],
        ]
    ],
    indices: list[MatchIndices],
) -> float:
    """Computes the average IoU of the Hungarian-matched boxes (symbols)."""
    ious = []
    pred_boxes = outputs.symbols.pred_boxes.data

    for b, (target, match) in enumerate(zip(targets, indices)):
        if len(match.pred_indices) == 0:
            continue

        matched_preds = pred_boxes[b][match.pred_indices]
        matched_gts = target.boxes.data[match.target_indices]

        # Compute pairwise IoU and extract the diagonal (matched pairs)
        iou_matrix, _ = box_iou(matched_preds, matched_gts)
        ious.append(torch.diag(iou_matrix).mean().item())

    return sum(ious) / len(ious) if ious else 0.0


@torch.no_grad()
def compute_mean_endpoint_error(
    outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
    targets: list[
        DetectionTarget[
            BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumSymbolClasses],
            Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumLineClasses],
        ]
    ],
    indices: list[MatchIndices],
) -> float:
    """Computes the average L2 distance of the Hungarian-matched keypoints (lines)."""
    errors = []
    pred_keypoints = outputs.lines.pred_keypoints.data

    for b, (target, match) in enumerate(zip(targets, indices)):
        if len(match.pred_indices) == 0:
            continue

        matched_preds = pred_keypoints[b][match.pred_indices]
        matched_gts = target.keypoints.data[match.target_indices]

        dist = torch.norm(matched_preds - matched_gts, p=2, dim=-1)
        errors.append(dist.mean().item())

    return sum(errors) / len(errors) if errors else 0.0
