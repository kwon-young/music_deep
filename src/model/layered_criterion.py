import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import elementwise_box_iou
from .criterion import (
    DFINECriterion,
    DetectionTargets,
    count_lines,
    count_symbols,
    flatten_indices,
)
from music_types import (
    Batch,
    BoxDim,
    CoordDim,
    DetectionOutput,
    FlattenedIndices,
    KeypointDim,
    LayeredDetectionLosses,
    LayeredDetectionOutput,
    MatchIndices,
    ModalityMatch,
    NumQueries,
)


def go_indices(matches: list[MatchIndices]) -> MatchIndices:
    """D-FINE `_get_go_indices` for one image: the union of the matchings of all layers.

    Every prediction matched in at least one layer is kept, paired with the target
    it was matched to most often. A target can therefore get several predictions.
    """
    pairs = torch.cat(
        [torch.stack([m.pred_indices, m.target_indices], dim=1) for m in matches]
    )
    if len(pairs) == 0:
        return MatchIndices(pred_indices=pairs[:, 0], target_indices=pairs[:, 1])
    unique, counts = torch.unique(pairs, dim=0, return_counts=True)
    unique = unique[torch.argsort(counts, descending=True, stable=True)]
    unique = unique[torch.argsort(unique[:, 0], stable=True)]
    first = torch.ones(len(unique), dtype=torch.bool, device=unique.device)
    first[1:] = unique[1:, 0] != unique[:-1, 0]
    kept = unique[first]
    return MatchIndices(pred_indices=kept[:, 0], target_indices=kept[:, 1])


def ddf_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    matched: FlattenedIndices,
    matched_quality: torch.Tensor,
    temperature: float = 5.0,
) -> torch.Tensor:
    """D-FINE Decoupled Distillation Focal loss over all queries of one layer.

    KL from the final layer's edge distributions (teacher, detached) to this
    layer's, weighted by the teacher's confidence, or by the localization quality
    for matched queries. Matched and unmatched queries are averaged separately
    and recombined with sqrt-count weights, so the many background queries of a
    dense head do not drown the matched ones.
    """
    batch = student_logits.shape[0]
    weight = teacher_scores.detach().sigmoid().max(dim=-1).values.clone()
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask[matched.batch, matched.src] = True
    weight[matched.batch, matched.src] = matched_quality.detach().to(weight.dtype)

    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits.detach() / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    loss = weight.unsqueeze(-1) * temperature**2 * kl

    edge_mask = mask.unsqueeze(-1).expand_as(loss).to(loss.dtype)
    num_pos = edge_mask.sum()
    num_neg = edge_mask.numel() - num_pos
    loss_pos = (loss * edge_mask).sum() / num_pos.clamp(min=1.0)
    loss_neg = (loss * (1.0 - edge_mask)).sum() / num_neg.clamp(min=1.0)

    batch_scale = 8.0 / batch
    weight_pos = (num_pos * batch_scale).sqrt()
    weight_neg = (num_neg * batch_scale).sqrt()
    return (loss_pos * weight_pos + loss_neg * weight_neg) / (
        weight_pos + weight_neg
    ).clamp(min=1e-6)


class DFINELayeredCriterion(nn.Module):
    """D-FINE loss for a MultiLayerOMRDetector.

    - every layer (and the pre-prediction) is matched on its own predictions and
      classified with that matching (focal loss, not VFL: see Exp 032);
    - L1 / GIoU / FGL use the union of all matchings (GO indices);
    - symbol FGL is weighted by the IoU of the prediction;
    - the pre-prediction gets classification and L1 / GIoU only;
    - DDF distils the final distributions into every intermediate layer.
    """

    def __init__(
        self,
        base: DFINECriterion,
        loss_ddf: float = 1.5,
        ddf_temperature: float = 5.0,
    ):
        super().__init__()
        self.base = base
        self.loss_ddf = loss_ddf
        self.ddf_temperature = ddf_temperature

    def match(
        self,
        outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
        targets: DetectionTargets,
    ) -> ModalityMatch:
        symbols, lines = self.base.matcher(outputs, targets)
        return ModalityMatch(
            symbols=symbols,
            lines=lines,
            num_symbols=count_symbols(targets),
            num_lines=count_lines(targets),
        )

    def forward(
        self,
        outputs: LayeredDetectionOutput[
            Batch, NumQueries, BoxDim, KeypointDim, CoordDim
        ],
        targets: DetectionTargets,
    ) -> LayeredDetectionLosses:
        layer_matches = [self.match(layer, targets) for layer in outputs.layers]
        pre_match = self.match(outputs.pre, targets)
        everything = [*layer_matches, pre_match]

        go_symbols = [
            go_indices([m.symbols[b] for m in everything]) for b in range(len(targets))
        ]
        go_lines = [
            go_indices([m.lines[b] for m in everything]) for b in range(len(targets))
        ]
        regression = ModalityMatch(
            symbols=go_symbols,
            lines=go_lines,
            num_symbols=max(1, sum(len(m.pred_indices) for m in go_symbols)),
            num_lines=max(1, sum(len(m.pred_indices) for m in go_lines)),
        )

        layers = [
            self.base.layer_losses(
                layer, targets, match, regression, with_fgl=True, iou_weighted_fgl=True
            )
            for layer, match in zip(outputs.layers, layer_matches)
        ]
        pre = self.base.layer_losses(
            outputs.pre, targets, pre_match, regression, with_fgl=False
        )

        loss_ddf = (
            self.distillation(outputs, targets, regression) * self.loss_ddf
            if len(outputs.layers) > 1
            else None
        )
        return LayeredDetectionLosses(layers=layers, pre=pre, loss_ddf=loss_ddf)

    def distillation(
        self,
        outputs: LayeredDetectionOutput[
            Batch, NumQueries, BoxDim, KeypointDim, CoordDim
        ],
        targets: DetectionTargets,
        regression: ModalityMatch,
    ) -> torch.Tensor:
        teacher = outputs.final
        symbols = flatten_indices(regression.symbols)
        lines = flatten_indices(regression.lines)
        target_boxes = torch.cat(
            [t.boxes.data[m.target_indices] for t, m in zip(targets, regression.symbols)]
        )

        total = torch.zeros((), device=teacher.symbols.pred_logits.data.device)
        for student in outputs.layers[:-1]:
            quality = (
                elementwise_box_iou(
                    student.symbols.pred_boxes.data[symbols.batch, symbols.src].detach(),
                    target_boxes,
                )
                .nan_to_num(0.0)
                .clamp(0.0, 1.0)
            )
            total = total + ddf_loss(
                student.symbols.pred_edge_logits.data,
                teacher.symbols.pred_edge_logits.data,
                teacher.symbols.pred_logits.data,
                symbols,
                quality,
                self.ddf_temperature,
            )
            total = total + ddf_loss(
                student.lines.pred_endpoint_logits.data,
                teacher.lines.pred_endpoint_logits.data,
                teacher.lines.pred_logits.data,
                lines,
                torch.ones(len(lines.src), device=total.device),
                self.ddf_temperature,
            )
        return total
