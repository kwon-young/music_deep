"""Patch-set assignment: one-to-one inside a patch, one-to-many across patches
(PATCH_SET_ASSIGNMENT.md, sections 1-3).

Every slot of every kept patch is labelled positive (matched to a GT the patch is responsible
for), ignored (it predicts a GT of the patch's ignore band, or one dropped by overflow) or
negative (everything else, in-patch duplicates included). A patch p is responsible for a GT g
when the distance between p's unit cell and g is at most ``margin``; when more than K GT of a
modality fall in one patch, the K with the fewest expected voters stay (a staff line with 40
voters can lose one, a dot with 2 cannot), the rest are ignored in that patch.

Inside a patch the K slots are matched to its (at most K) responsible GT by a batched greedy
assignment on the matcher's cost (focal class cost, L1, GIoU for symbols): K argmin rounds over
a (P, K, K) cost tensor, no Python loop over patches.

Query q of an image is slot q % K of kept patch q // K, as in the detector output.
"""

from dataclasses import dataclass

import torch

from model.box_ops import box_xyxy_to_cxcywh
from model.regions import cell_distance, point_box_distance, point_segment_distance
from music_types import (
    Batch,
    BoxDim,
    CoordDim,
    DetectionOutput,
    DetectionTarget,
    KeypointDim,
    MatchIndices,
    NumQueries,
)


@dataclass(frozen=True)
class PatchSetConfig:
    """``margin`` / ``ignore_band`` in patch units (Analysis 009: 0.5 / 0.5). ``ignore_quality``:
    an unmatched slot whose IoU (symbols) or OKS (lines) with a band or overflowed GT of its
    patch reaches it is ignored instead of negative. Line OKS uses the evaluator formula with
    area = length x ``line_thickness`` (patch units, ~4 px at 64 px). ``cost_presence`` adds the
    focal cost of the slot's presence to the in-patch matching cost, so a slot that already
    claims an object keeps it (stable slot roles, as DETR's score term in its matching cost).

    Lines (``line_on_line`` > 0): besides its matched slot, every other slot of the patch whose
    prediction lies on a responsible line (``on_line_quality`` >= ``line_on_line``) is a
    positive too. Coarse line endpoints are off by patches along the line (median 2.9 patches
    for the worse end of matched slots in the Exp 042 smoke), so which of several such slots
    wins the L1 match carries no learnable meaning; refined lines never crowd (Analysis 009),
    so lines do not need in-patch one-to-one and are associated without cannot-link."""

    margin: float = 0.5
    ignore_band: float = 0.5
    ignore_quality: float = 0.5
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    cost_presence: float = 0.0
    alpha: float = 0.25
    gamma: float = 2.0
    line_thickness: float = 1 / 16
    oks_sigma: float = 0.1
    line_on_line: float = 0.5
    line_perp_sigma: float = 0.1


@dataclass(frozen=True)
class SlotLabels:
    """Labelling of the Q = P x K slots of one modality of one image.

    ``match``: the positives, ``pred_indices`` (queries) and ``target_indices`` (GT, repeated
    across patches). ``ignored`` (Q,) bool. ``voters`` (G,): positives per GT. ``expected``
    (G,): kept patches responsible for the GT before overflow. ``overflowed``: number of
    (patch, GT) responsibilities dropped by overflow."""

    match: MatchIndices
    ignored: torch.Tensor
    voters: torch.Tensor
    expected: torch.Tensor
    overflowed: int


@dataclass(frozen=True)
class PatchSetAssignment:
    symbols: list[SlotLabels]
    lines: list[SlotLabels]

    def matches(self) -> tuple[list[MatchIndices], list[MatchIndices]]:
        """The positives in the global matcher's format (metrics, plots, the refiner)."""
        return [s.match for s in self.symbols], [s.match for s in self.lines]


def pair_quality(pred: torch.Tensor, gt: torch.Tensor, is_line: bool,
                 line_thickness: float = 1 / 16, sigma: float = 0.1) -> torch.Tensor:
    """Element-wise quality of predictions (..., 4) against GT (..., 4): IoU of boxes, or the
    COCO line OKS (mean over the two endpoints of exp(-d^2 / (2 (2 sigma)^2 area)))."""
    if is_line:
        d2 = (pred - gt).unflatten(-1, (2, 2)).square().sum(dim=-1)
        length = (gt[..., 2:] - gt[..., :2]).norm(dim=-1)
        area = (length * line_thickness).clamp(min=line_thickness**2)
        e = d2 / (2 * sigma) ** 2 / area.unsqueeze(-1) / 2
        return torch.exp(-e).mean(dim=-1)
    lt = torch.maximum(pred[..., :2], gt[..., :2])
    rb = torch.minimum(pred[..., 2:], gt[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_p = (pred[..., 2:] - pred[..., :2]).clamp(min=0).prod(dim=-1)
    area_g = (gt[..., 2:] - gt[..., :2]).clamp(min=0).prod(dim=-1)
    return inter / (area_p + area_g - inter).clamp(min=1e-9)


def on_line_quality(pred: torch.Tensor, gt: torch.Tensor, sigma: float) -> torch.Tensor:
    """Element-wise (..., 4) quality of a predicted segment lying on a GT segment, ignoring
    extent: exp(-d^2 / (2 sigma^2)) with d the larger perpendicular distance of the two
    predicted endpoints to the GT line, 0 when the projections do not overlap the GT segment."""
    c, e = gt[..., :2], gt[..., 2:]
    length = (e - c).norm(dim=-1, keepdim=True).clamp(min=1e-6)
    u = (e - c) / length
    n = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    a, b = pred[..., :2] - c, pred[..., 2:] - c
    d = torch.maximum((a * n).sum(-1).abs(), (b * n).sum(-1).abs())
    ta, tb = (a * u).sum(-1), (b * u).sum(-1)
    overlap = torch.minimum(torch.maximum(ta, tb), length.squeeze(-1)) - torch.minimum(ta, tb).clamp(min=0)
    return torch.where(overlap > 0, torch.exp(-d.square() / (2 * sigma**2)), 0.0)


def broadcast_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalized IoU of broadcastable boxes (..., 4)."""
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    inter = (torch.minimum(a[..., 2:], b[..., 2:]) - torch.maximum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    union = area_a + area_b - inter
    hull = (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    return inter / union.clamp(min=1e-9) - (hull - union) / hull.clamp(min=1e-9)


def focal_cost(prob: torch.Tensor, cfg: PatchSetConfig) -> torch.Tensor:
    """The matcher's focal cost of calling a slot with probability ``prob`` positive."""
    neg = (1 - cfg.alpha) * prob.pow(cfg.gamma) * -(1 - prob + 1e-8).log()
    pos = cfg.alpha * (1 - prob).pow(cfg.gamma) * -(prob + 1e-8).log()
    return pos - neg


def greedy_slot_assignment(cost: torch.Tensor) -> torch.Tensor:
    """(P, K, R) cost (inf = forbidden) -> (P, K) column assigned to each slot, -1 if none.

    Each round takes, in every patch at once, the cheapest remaining (slot, column) pair and
    removes its row and column, so a patch with r <= K finite columns gets all r assigned."""
    p, k, r = cost.shape
    cost = cost.clone()
    out = torch.full((p, k), -1, dtype=torch.long, device=cost.device)
    rows = torch.arange(p, device=cost.device)
    for _ in range(min(k, r)):
        best, flat = cost.view(p, k * r).min(dim=1)
        ok = torch.isfinite(best)
        slot, col = flat // r, flat % r
        out[rows[ok], slot[ok]] = col[ok]
        cost[rows[ok], slot[ok], :] = torch.inf
        cost[rows[ok], :, col[ok]] = torch.inf
    return out


def responsible_gt(distance: torch.Tensor, center_distance: torch.Tensor, margin: float,
                   num_slots: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """From (P, G) cell and centre distances: ``kept`` (P, R) GT index per patch (R = min(K, G),
    -1 = none), ``responsible`` (P, G) before overflow and ``overflow`` (P, G) the dropped
    responsibilities. Priority: fewest expected voters, then nearest to the patch centre."""
    responsible = distance <= margin
    expected = responsible.sum(dim=0)
    key = expected.to(distance.dtype)[None, :] * 1e3 + center_distance
    key = torch.where(responsible, key, torch.inf)
    r = min(num_slots, distance.shape[1])
    value, kept = key.topk(r, dim=1, largest=False)
    kept = torch.where(torch.isfinite(value), kept, -1)
    keep = torch.zeros_like(responsible)
    valid = kept >= 0
    keep[torch.arange(kept.shape[0], device=kept.device)[:, None].expand_as(kept)[valid], kept[valid]] = True
    return kept, responsible, responsible & ~keep


def assign_image(
    centers: torch.Tensor,
    probs: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_labels: torch.Tensor,
    is_line: bool,
    num_slots: int,
    cfg: PatchSetConfig,
    presence: torch.Tensor | None = None,
) -> SlotLabels:
    """``centers`` (P, 2) kept patch centres, ``probs`` (P*K, C) class probabilities, ``pred``
    (P*K, 4) geometry, ``gt`` (G, 4) and ``gt_labels`` (G,), all in patch units; ``presence``
    (P*K,) presence probabilities, if the detector has them."""
    device = pred.device
    num_p, k = centers.shape[0], num_slots
    num_q, num_g = num_p * k, gt.shape[0]
    if num_g == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return SlotLabels(
            match=MatchIndices(pred_indices=empty, target_indices=empty),
            ignored=torch.zeros(num_q, dtype=torch.bool, device=device),
            voters=empty, expected=empty, overflowed=0,
        )
    distance = cell_distance(centers, gt, is_line)
    center_distance = point_segment_distance(centers, gt) if is_line else point_box_distance(centers, gt)
    kept, responsible, overflow = responsible_gt(distance, center_distance, cfg.margin, k)
    band = (distance > cfg.margin) & (distance <= cfg.margin + cfg.ignore_band)

    # (P, K, R) in-patch cost
    valid = kept >= 0
    safe = kept.clamp(min=0)
    slot_pred = pred.view(num_p, k, 4)
    slot_probs = probs.view(num_p, k, -1)
    gt_pk = gt[safe]  # (P, R, 4)
    labels = gt_labels[safe]  # (P, R)
    prob = slot_probs.gather(2, labels[:, None, :].expand(-1, k, -1))  # (P, K, R)
    a, b = slot_pred[:, :, None, :], gt_pk[:, None, :, :]
    if is_line:
        cost = cfg.cost_class * focal_cost(prob, cfg) + cfg.cost_bbox * (a - b).abs().sum(-1)
    else:
        l1 = (box_xyxy_to_cxcywh(a) - box_xyxy_to_cxcywh(b)).abs().sum(-1)
        cost = cfg.cost_class * focal_cost(prob, cfg) + cfg.cost_bbox * l1 - cfg.cost_giou * broadcast_giou(a, b)
    if presence is not None and cfg.cost_presence:
        cost = cost + cfg.cost_presence * focal_cost(presence.view(num_p, k, 1), cfg)
    cost = torch.where(valid[:, None, :], cost, torch.inf)
    column = greedy_slot_assignment(cost)

    if is_line and cfg.line_on_line > 0:
        on_line = on_line_quality(a, b, cfg.line_perp_sigma)  # (P, K, R)
        on_line = torch.where(valid[:, None, :], on_line, 0.0)
        best_q, best_r = on_line.max(dim=-1)
        extra = (column < 0) & (best_q >= cfg.line_on_line)
        column = torch.where(extra, best_r, column)

    matched = column >= 0
    patch_idx, slot_idx = matched.nonzero(as_tuple=True)
    target = kept[patch_idx, column[patch_idx, slot_idx]]
    query = patch_idx * k + slot_idx

    # ignore: unmatched slots that predict a band / overflowed GT of their patch well
    ignored = torch.zeros(num_q, dtype=torch.bool, device=device)
    pair_p, pair_g = (band | overflow).nonzero(as_tuple=True)
    if pair_p.numel():
        free = ~matched[pair_p]  # (n, K)
        q = pair_quality(slot_pred[pair_p], gt[pair_g][:, None, :].expand(-1, k, -1), is_line,
                         cfg.line_thickness, cfg.oks_sigma)
        hit = free & (q >= cfg.ignore_quality)
        hit_p, hit_k = hit.nonzero(as_tuple=True)
        ignored[pair_p[hit_p] * k + hit_k] = True
    return SlotLabels(
        match=MatchIndices(pred_indices=query, target_indices=target),
        ignored=ignored,
        voters=torch.bincount(target, minlength=num_g),
        expected=responsible.sum(dim=0),
        overflowed=int(overflow.sum().item()),
    )


class PatchSetAssigner:
    """Labels every slot of a detector output (no gradients)."""

    def __init__(self, cfg: PatchSetConfig, num_slots: int) -> None:
        self.cfg = cfg
        self.num_slots = num_slots

    @torch.no_grad()
    def __call__(
        self,
        outputs: DetectionOutput[Batch, NumQueries, BoxDim, KeypointDim, CoordDim],
        targets: list[DetectionTarget],
    ) -> PatchSetAssignment:
        k = self.num_slots
        # line queries are anchored at their patch centre
        centers = outputs.lines.absolute_centers.data[:, ::k].float()
        symbols: list[SlotLabels] = []
        lines: list[SlotLabels] = []
        sym_presence, line_presence = outputs.symbols.pred_presence, outputs.lines.pred_presence
        for b, t in enumerate(targets):
            symbols.append(assign_image(
                centers[b], outputs.symbols.pred_logits.data[b].float().sigmoid(),
                outputs.symbols.pred_boxes.data[b].float(), t.boxes.data.float(),
                t.box_labels.data, False, k, self.cfg,
                None if sym_presence is None else sym_presence.data[b].float().sigmoid(),
            ))
            lines.append(assign_image(
                centers[b], outputs.lines.pred_logits.data[b].float().sigmoid(),
                outputs.lines.pred_keypoints.data[b].float(), t.keypoints.data.float(),
                t.keypoint_labels.data, True, k, self.cfg,
                None if line_presence is None else line_presence.data[b].float().sigmoid(),
            ))
        return PatchSetAssignment(symbols=symbols, lines=lines)


@dataclass(frozen=True)
class SlotStats:
    """Smoke-test diagnostics of one modality (PATCH_SET_ASSIGNMENT.md, Phase 2): fraction of
    positives / negatives with presence > 0.5, in-patch duplicates (negatives with presence >
    0.5 whose prediction reaches quality 0.5 against a GT matched by another slot of their
    patch) per confident positive, ignored fraction of slots, mean voters per GT. Duplicates
    are only meaningful for symbols: line slots lying on a matched line are positives, and the
    coarse line OKS is ~0 anyway; line duplicates are measured after refinement and association."""

    positive_high: float
    negative_high: float
    duplicates: float
    ignored: float
    voters: float


def slot_stats(presence: torch.Tensor, pred: torch.Tensor, labels: SlotLabels, gt: torch.Tensor,
               is_line: bool, num_slots: int, cfg: PatchSetConfig) -> SlotStats:
    """``presence`` (Q,) logits and ``pred`` (Q, 4) of one image."""
    k = num_slots
    num_p = presence.shape[0] // k
    high = presence.float().sigmoid() > 0.5
    positive = torch.zeros_like(high)
    positive[labels.match.pred_indices] = True
    negative = ~positive & ~labels.ignored
    table = torch.full((num_p * k,), -1, dtype=torch.long, device=pred.device)
    table[labels.match.pred_indices] = labels.match.target_indices
    table = table.view(num_p, k)
    duplicates = 0
    if gt.shape[0]:
        own = gt[table.clamp(min=0)]  # (P, K, 4): GT of each slot of the patch
        q = pair_quality(pred.view(num_p, k, 1, 4).float(), own[:, None].float(), is_line,
                         cfg.line_thickness, cfg.oks_sigma)  # (P, K slots, K matched)
        q = torch.where((table >= 0)[:, None, :], q, 0.0)
        dup = (negative & high).view(num_p, k) & (q.amax(dim=-1) >= 0.5)
        duplicates = int(dup.sum().item())
    confident = int((positive & high).sum().item())
    return SlotStats(
        positive_high=float(high[positive].float().mean().item()) if positive.any() else 0.0,
        negative_high=float(high[negative].float().mean().item()) if negative.any() else 0.0,
        duplicates=duplicates / max(1, confident),
        ignored=float(labels.ignored.float().mean().item()),
        voters=float(labels.voters.float().mean().item()) if labels.voters.numel() else 0.0,
    )
