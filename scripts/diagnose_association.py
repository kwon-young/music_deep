"""Analysis 011 (patch-set Phase 1): association instead of NMS, on a frozen one-to-one detector.

Reads the slot cache of ``scripts/build_slot_cache.py`` and, per variant, clusters the candidates
of every page (``model.association``), writes COCO predictions and evaluates them with the
pycocotools setup of ``evaluate_coco.py``. Besides AP it reports, from the same per-GT matches
(IoU / OKS 0.5):

* recall of coincident symbol pairs (same class, < 1 px: shared noteheads of two voices) and of
  crowded same-class pairs (both corners / endpoints within 64 px);
* staff recall split by end type (both ends on a barline or not, Analysis 010);
* confident duplicates (score >= 0.3, IoU / OKS >= 0.5 with an already matched GT);
* calibration of the predicted std (|error| / std of the matched candidates), which the
  normalised association distance relies on;
* the spread of the duplicates of one symbol across patches (sets the symbol radius).

On a one-to-one detector (no presence logit) presence is the class score (support = summed
class score / expected voters). On a patch-set detector (Exp 042+: the cache has ``presence``)
the patch-set variants run instead: candidates by presence floor, lines fused with weights
score / variance of the refiner bins and associated without cannot-link, symbols with
cannot-link at a size-relative radius, clusters ranked by support x quality. The summary also
counts confident unmatched staff predictions per page (fragments) and the association time.

Usage:
    PYTHONPATH=src:scripts mamba run -n pytorch python scripts/diagnose_association.py \
        --cache_dir experiments/037_029_recipe_from_023/slot_cache \
        --out_dir experiments/analysis_011_association
"""

import argparse
import contextlib
import gc
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.coco import parse_coco
from evaluate_coco import fast_computeOks, fast_evaluateImg, summarize_at_maxdet
from model.association import AssociationConfig, Clusters, Detections, associate, support_ratio
from model.line_association import LineAssociationConfig, associate_lines, chain_labels, endpoint_agreement
from model.refiner import deduplicate_lines
from model.regions import expected_voters

type Modality = Literal["symbols", "lines"]
type Ranking = Literal["noisy_or", "quality", "support_quality", "slot", "support_cut"]
type JsonValue = int | float | str | bool | None | list[JsonValue] | dict[str, JsonValue]

PS = 64.0
MARGIN = 0.5
CROWD_PX = 64.0
CONFIDENT = 0.3


@dataclass(frozen=True)
class Variant:
    name: str
    modality: Modality
    method: Literal["topk", "nms", "associate", "chains", "hybrid"]
    ranking: Ranking
    floor: float
    association: AssociationConfig | None = None
    chains: LineAssociationConfig | None = None
    refined: bool = True
    presence: bool = False
    size_scale: bool = False
    chain_labels: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Page:
    image_id: int
    width: int
    height: int
    grid_w: int
    num_shapes: int
    centers: torch.Tensor
    symbols: dict[str, torch.Tensor]
    lines: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Kept:
    """Final detections of one page and modality (pixels)."""

    geometry: torch.Tensor
    label: torch.Tensor
    score: torch.Tensor


def variants() -> list[Variant]:
    plain_line = AssociationConfig(radius=4 / PS, normalized=False)
    plain_sym = AssociationConfig(radius=8 / PS, normalized=False)
    norm = AssociationConfig(radius=3.0, normalized=True, variance_weights=True)
    norm1 = AssociationConfig(radius=1.0, normalized=True, variance_weights=True)
    no_cl_line = AssociationConfig(radius=4 / PS, normalized=False, cannot_link=False)
    no_cl_sym = AssociationConfig(radius=8 / PS, normalized=False, cannot_link=False)
    return [
        Variant("lines_nms4_noisyor", "lines", "nms", "noisy_or", 0.05),
        Variant("lines_assoc4px_noisyor", "lines", "associate", "noisy_or", 0.05, plain_line),
        Variant("lines_assoc4px_nocl_noisyor", "lines", "associate", "noisy_or", 0.05, no_cl_line),
        Variant("lines_assoc3sd_noisyor", "lines", "associate", "noisy_or", 0.05, norm),
        Variant("lines_assoc1sd_noisyor", "lines", "associate", "noisy_or", 0.05, norm1),
        Variant("lines_assoc3sd_quality", "lines", "associate", "quality", 0.05, norm),
        Variant("lines_assoc3sd_support", "lines", "associate", "support_quality", 0.05, norm),
        Variant("lines_coarse_assoc3sd_noisyor", "lines", "associate", "noisy_or", 0.05, norm, refined=False),
        Variant("symbols_topk", "symbols", "topk", "slot", 0.0),
        Variant("symbols_assoc8px_noisyor", "symbols", "associate", "noisy_or", 0.01, plain_sym),
        Variant("symbols_assoc8px_nocl_noisyor", "symbols", "associate", "noisy_or", 0.01, no_cl_sym),
        Variant("symbols_assoc3sd_noisyor", "symbols", "associate", "noisy_or", 0.01, norm),
        Variant("symbols_assoc1sd_noisyor", "symbols", "associate", "noisy_or", 0.01, norm1),
        Variant("symbols_assoc3sd_quality", "symbols", "associate", "quality", 0.01, norm),
        Variant("symbols_assoc3sd_support", "symbols", "associate", "support_quality", 0.01, norm),
    ]


def patch_set_variants() -> list[Variant]:
    """Variants for a patch-set detector (presence floor 0.5)."""
    ivw = AssociationConfig(radius=4 / PS, normalized=False, cannot_link=False, variance_weights=True)
    score_w = AssociationConfig(radius=4 / PS, normalized=False, cannot_link=False)
    ivw_1sd = AssociationConfig(radius=1.0, normalized=True, cannot_link=False, variance_weights=True)
    ivw_cl = AssociationConfig(radius=4 / PS, normalized=False, cannot_link=True, variance_weights=True)
    rel = AssociationConfig(radius=0.35, normalized=True)
    rel_nocl = AssociationConfig(radius=0.35, normalized=True, cannot_link=False)
    return [
        Variant("ps_lines_nms4_noisyor", "lines", "nms", "noisy_or", 0.5, presence=True),
        Variant("ps_lines_assoc4px_ivw_support", "lines", "associate", "support_quality", 0.5, ivw, presence=True),
        Variant("ps_lines_assoc4px_ivw_quality", "lines", "associate", "quality", 0.5, ivw, presence=True),
        Variant("ps_lines_assoc4px_score_support", "lines", "associate", "support_quality", 0.5, score_w, presence=True),
        Variant("ps_lines_assoc1sd_ivw_support", "lines", "associate", "support_quality", 0.5, ivw_1sd, presence=True),
        Variant("ps_lines_assoc4px_ivw_cl_support", "lines", "associate", "support_quality", 0.5, ivw_cl, presence=True),
        Variant("ps_lines_chains_support_cut", "lines", "chains", "support_cut", 0.5, presence=True,
                chains=LineAssociationConfig()),
        Variant("ps_lines_chains_support_quality", "lines", "chains", "support_quality", 0.5, presence=True,
                chains=LineAssociationConfig()),
        Variant("ps_lines_chains_presence_support_cut", "lines", "chains", "support_cut", 0.5, presence=True,
                chains=LineAssociationConfig(weighting="presence", certain_std=24 / 64, min_cut_weight=1.5)),
        Variant("ps_lines_hybrid", "lines", "hybrid", "support_quality", 0.5, ivw, presence=True,
                chains=LineAssociationConfig()),
        Variant("ps_lines_hybrid_presence_cuts", "lines", "hybrid", "support_quality", 0.5, ivw, presence=True,
                chains=LineAssociationConfig(weighting="presence", certain_std=24 / 64, min_cut_weight=1.5)),
        Variant("ps_symbols_topk", "symbols", "topk", "slot", 0.5, presence=True),
        Variant("ps_symbols_assocrel_support", "symbols", "associate", "support_quality", 0.5, rel,
                presence=True, size_scale=True),
        Variant("ps_symbols_assocrel_quality", "symbols", "associate", "quality", 0.5, rel,
                presence=True, size_scale=True),
        Variant("ps_symbols_assocrel_nocl_support", "symbols", "associate", "support_quality", 0.5, rel_nocl,
                presence=True, size_scale=True),
    ]


def load_page(path: Path) -> Page:
    d: dict[str, object] = torch.load(path, weights_only=True)

    def tensors(key: str) -> dict[str, torch.Tensor]:
        value = d[key]
        assert isinstance(value, dict)
        return {str(k): v for k, v in value.items() if isinstance(v, torch.Tensor)}

    cells = d["cells"]
    grid_w, num_shapes = d["grid_w"], d["num_shapes"]
    image_id, width, height = d["image_id"], d["width"], d["height"]
    assert isinstance(cells, torch.Tensor)
    assert isinstance(grid_w, int) and isinstance(num_shapes, int)
    assert isinstance(image_id, int) and isinstance(width, int) and isinstance(height, int)
    c = cells.long()
    centers = torch.stack([(c % grid_w).float() + 0.5, (c // grid_w).float() + 0.5], dim=-1)
    return Page(image_id, width, height, grid_w, num_shapes, centers, tensors("symbols"), tensors("lines"))


def candidates(page: Page, v: Variant) -> Detections:
    slots = page.lines if v.modality == "lines" else page.symbols
    prob = slots["logits"].float().sigmoid()
    score, label = prob.max(dim=-1)
    presence = slots["presence"].float().sigmoid() if v.presence else score
    keep = presence >= v.floor
    use_refined = v.modality == "lines" and v.refined
    geometry = slots["refined" if use_refined else "geometry"].float()
    std = slots["refined_std" if use_refined else "std"].float()
    if v.size_scale:
        # distance relative to the box size: corner offsets over sqrt(2) x width / height
        size = (geometry[:, 2:] - geometry[:, :2]).clamp(min=1 / PS)
        std = torch.cat([size, size], dim=-1)
    return Detections(
        geometry=geometry[keep],
        std=std[keep],
        label=label[keep],
        score=score[keep],
        presence=presence[keep],
        patch=slots["query"][keep].long() // page.num_shapes,
    )


def cluster_score(page: Page, v: Variant, clusters: Clusters) -> torch.Tensor:
    if v.ranking == "noisy_or":
        return clusters.noisy_or
    if v.ranking == "quality":
        return clusters.quality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected = expected_voters(
        page.centers.to(device), clusters.geometry.to(device), v.modality == "lines", MARGIN, chunk=256
    ).cpu()
    return support_ratio(clusters, expected) * clusters.quality


def subset(dets: Detections, keep: torch.Tensor) -> Detections:
    return Detections(geometry=dets.geometry[keep], std=dets.std[keep], label=dets.label[keep],
                      score=dets.score[keep], presence=dets.presence[keep], patch=dets.patch[keep])


def chain_objects(page: Page, dets: Detections, cfg: LineAssociationConfig, by_agreement: bool
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chain-and-cut lines, scored by support x endpoint agreement (or x best class score)."""
    objects = associate_lines(dets.geometry, dets.std, dets.label, dets.score, dets.presence,
                              page.centers[dets.patch], cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected = expected_voters(page.centers.to(device), objects.geometry.to(device), True, MARGIN, chunk=256).cpu()
    support = (objects.evidence.cpu() / expected.clamp(min=1).float()).clamp(max=1.0)
    agreement = endpoint_agreement(objects, cfg).cpu()
    score = support * (agreement if by_agreement else objects.quality.cpu())
    return objects.geometry.cpu(), objects.label.cpu(), score


def run_variant(page: Page, v: Variant, cap: int) -> Kept:
    dets = candidates(page, v)
    if v.method == "topk":
        score, geometry, label = dets.score, dets.geometry, dets.label
    elif v.method == "chains":
        assert v.chains is not None
        geometry, label, score = chain_objects(page, dets, v.chains, v.ranking == "support_cut")
    elif v.method == "hybrid":
        # chains for the long collinear classes, endpoint association for the others
        assert v.chains is not None and v.association is not None
        chained = torch.tensor([int(c) in v.chain_labels for c in dets.label.tolist()], dtype=torch.bool)
        g1, l1, s1 = chain_objects(page, subset(dets, chained), v.chains, True)
        rest = subset(dets, ~chained)
        clusters = associate(rest, v.association)
        g2, l2, s2 = clusters.geometry, clusters.label, cluster_score(page, v, clusters)
        geometry, label, score = torch.cat([g1, g2]), torch.cat([l1, l2]), torch.cat([s1, s2])
    elif v.method == "nms":
        deduped = deduplicate_lines(dets.geometry * PS, dets.score, dets.label, 4.0, merge=True)
        score, geometry, label = deduped.score, dets.geometry[deduped.keep], dets.label[deduped.keep]
    else:
        assert v.association is not None
        clusters = associate(dets, v.association)
        score, geometry, label = cluster_score(page, v, clusters), clusters.geometry, clusters.label
    top = score.topk(min(cap, score.shape[0])).indices
    return Kept(geometry[top] * PS, label[top], score[top])


def coco_results(page: Page, kept: Kept, modality: Modality, idx_to_cat: dict[int, int]) -> list[JsonValue]:
    out: list[JsonValue] = []
    w, h = float(page.width), float(page.height)
    for g, label, score in zip(kept.geometry.tolist(), kept.label.tolist(), kept.score.tolist()):
        x1, y1, x2, y2 = (min(max(g[0], 0.0), w), min(max(g[1], 0.0), h),
                          min(max(g[2], 0.0), w), min(max(g[3], 0.0), h))
        entry: dict[str, JsonValue] = {"image_id": page.image_id, "category_id": idx_to_cat[label], "score": score}
        if modality == "symbols":
            entry["bbox"] = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        else:
            entry["keypoints"] = [x1, y1, 1, x2, y2, 1]
        out.append(entry)
    return out


def evaluate(coco_gt: COCO, results: str, modality: Modality, cat_ids: list[int],
             img_ids: list[int]) -> COCOeval:
    """Same setup as ``evaluate_coco.evaluate_modality``, but keeps the COCOeval object."""
    iou_type = "bbox" if modality == "symbols" else "keypoints"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)  # a JSON file: one variant in memory at a time
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        if iou_type == "keypoints":
            coco_eval.computeOks = fast_computeOks.__get__(coco_eval, COCOeval)
            coco_eval.params.kpt_oks_sigmas = np.array([0.1, 0.1])
        coco_eval.evaluateImg = fast_evaluateImg.__get__(coco_eval, COCOeval)
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = img_ids
        coco_eval.params.maxDets = [20, 100, 2000]
        coco_eval.evaluate()
        coco_eval.accumulate()
    return coco_eval


def confident_unmatched(coco_eval: COCOeval, cat_id: int) -> int:
    """Detections of ``cat_id`` scoring >= CONFIDENT left unmatched at the first threshold
    (staff: fragments with a wrong extent, or false positives)."""
    count = 0
    for e in coco_eval.evalImgs:
        if e is None or e["aRng"] != coco_eval.params.areaRng[0] or e["category_id"] != cat_id:
            continue
        dtm = e["dtMatches"][0]
        count += sum(1 for m, s in zip(dtm, e["dtScores"]) if m == 0 and s >= CONFIDENT)
    return count


def matched_gt(coco_eval: COCOeval) -> tuple[set[int], int]:
    """GT ids matched at the first threshold (0.5), area 'all', last maxDet; and the number of
    confident duplicates (unmatched detections scoring >= CONFIDENT that overlap a same-class GT
    at >= 0.5, i.e. whose GT was taken by a higher-scored detection)."""
    hit: set[int] = set()
    dupes = 0
    for e in coco_eval.evalImgs:
        if e is None or e["aRng"] != coco_eval.params.areaRng[0]:
            continue
        gt_ids = e["gtIds"]
        gtm = e["gtMatches"][0]
        hit.update(int(g) for g, m in zip(gt_ids, gtm) if m > 0)
        ious = coco_eval.ious.get((e["image_id"], e["category_id"]))
        if ious is None or len(ious) == 0:
            continue
        dtm = e["dtMatches"][0]
        scores = e["dtScores"]
        best = np.asarray(ious).max(axis=1) if np.asarray(ious).ndim == 2 else np.zeros(len(dtm))
        for d in range(len(dtm)):
            if dtm[d] == 0 and scores[d] >= CONFIDENT and d < len(best) and best[d] >= 0.5:
                dupes += 1
    return hit, dupes


def ap_summary(coco_eval: COCOeval, names: dict[int, str], common: set[str]) -> dict[str, JsonValue]:
    stats = summarize_at_maxdet(coco_eval, max_dets=2000)
    precision = coco_eval.eval["precision"]
    per_class: dict[str, float] = {}
    for k, cat in enumerate(coco_eval.params.catIds):
        p = precision[:, :, k, 0, 2]
        p = p[p > -1]
        if len(p):
            per_class[names[cat]] = float(np.mean(p))
    com = [v for n, v in per_class.items() if n in common]
    return {
        "AP": stats[0], "AP50": stats[1], "AP75": stats[2],
        "AP_common": float(np.mean(com)) if com else -1.0,
        "per_class": {n: round(v, 4) for n, v in per_class.items()},
    }


def corner_pairs(ids: Sequence[int], geometry: np.ndarray, cats: Sequence[int], is_line: bool,
                 lo: float, hi: float) -> list[tuple[int, int]]:
    """Same-class pairs whose two points (corners / endpoints, either order for lines) are all
    within ``hi`` px, and not all within ``lo`` px."""
    out: list[tuple[int, int]] = []
    g = geometry.reshape(-1, 2, 2)
    for c in set(cats):
        idx = [i for i, k in enumerate(cats) if k == c]
        if len(idx) < 2:
            continue
        p = g[idx]
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)).max(-1)
        if is_line:
            swapped = np.sqrt(((p[:, None] - p[None, :, ::-1]) ** 2).sum(-1)).max(-1)
            d = np.minimum(d, swapped)
        a, b = np.nonzero(np.triu(d <= hi, k=1))
        for i, j in zip(a.tolist(), b.tolist()):
            if d[i, j] >= lo:
                out.append((ids[idx[i]], ids[idx[j]]))
    return out


@dataclass(frozen=True)
class GtSets:
    coincident: list[tuple[int, int]]
    crowded: dict[Modality, list[tuple[int, int]]]
    staff_barline: set[int]
    staff_other: set[int]


def gt_sets(coco_gt: COCO, img_ids: list[int], line_cats: set[int], names: dict[int, str]) -> GtSets:
    coincident: list[tuple[int, int]] = []
    crowded: dict[Modality, list[tuple[int, int]]] = {"symbols": [], "lines": []}
    staff_barline: set[int] = set()
    staff_other: set[int] = set()
    for img in img_ids:
        anns = coco_gt.imgToAnns[img]
        for modality in ("symbols", "lines"):
            is_line = modality == "lines"
            sel = [a for a in anns if (a["category_id"] in line_cats) == is_line]
            if not sel:
                continue
            if is_line:
                geo = np.array([[a["keypoints"][0], a["keypoints"][1], a["keypoints"][3], a["keypoints"][4]] for a in sel])
            else:
                geo = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in sel])
            ids = [int(a["id"]) for a in sel]
            cats = [int(a["category_id"]) for a in sel]
            if not is_line:
                coincident += corner_pairs(ids, geo, cats, False, 0.0, 1.0)
            crowded[modality] += corner_pairs(ids, geo, cats, is_line, 1.0, CROWD_PX)
            if is_line:
                bars = geo[[names[c] == "barLine" for c in cats]]
                for i, c in enumerate(cats):
                    if names[c] != "staff":
                        continue
                    ends = geo[i].reshape(2, 2)
                    on_bar = [
                        bool(np.any((np.abs((bars[:, 0] + bars[:, 2]) / 2 - x) <= 12)
                                    & (np.minimum(bars[:, 1], bars[:, 3]) - 12 <= y)
                                    & (y <= np.maximum(bars[:, 1], bars[:, 3]) + 12)))
                        for x, y in ends
                    ] if len(bars) else [False, False]
                    (staff_barline if all(on_bar) else staff_other).add(ids[i])
    return GtSets(coincident, crowded, staff_barline, staff_other)


def pair_recall(pairs: list[tuple[int, int]], hit: set[int]) -> float:
    return sum(a in hit and b in hit for a, b in pairs) / max(1, len(pairs))


def calibration(page: Page, coco_gt: COCO, cat_to_idx: dict[Modality, dict[int, int]],
                line_cats: set[int], acc: dict[str, list[np.ndarray]]) -> None:
    """|error| / std per coordinate of the slots (score >= 0.05) nearest to a same-class GT."""
    anns = coco_gt.imgToAnns[page.image_id]
    pairs: tuple[tuple[Modality, dict[str, torch.Tensor]], ...] = (
        ("symbols", page.symbols), ("lines", page.lines)
    )
    for modality, slots in pairs:
        is_line = modality == "lines"
        sel = [a for a in anns if (a["category_id"] in line_cats) == is_line]
        if not sel:
            continue
        if is_line:
            gt = torch.tensor([[a["keypoints"][0], a["keypoints"][1], a["keypoints"][3], a["keypoints"][4]] for a in sel]) / PS
        else:
            gt = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in sel]) / PS
        gt_label = torch.tensor([cat_to_idx[modality][a["category_id"]] for a in sel])
        score, label = slots["logits"].float().sigmoid().max(dim=-1)
        keep = score >= 0.05
        geoms = [("coarse", slots["geometry"], slots["std"])]
        if is_line:
            geoms.append(("refined", slots["refined"], slots["refined_std"]))
        for kind, geometry, std in geoms:
            g, s, lab = geometry[keep].float(), std[keep].float().clamp(min=1 / PS), label[keep]
            if g.shape[0] == 0:
                continue
            d = (g.view(-1, 1, 2, 2) - gt.view(1, -1, 2, 2)).norm(dim=-1).amax(dim=-1)
            d = torch.where(lab[:, None] == gt_label[None, :], d, torch.inf)
            near, j = d.min(dim=1)
            ok = near < 0.5
            err = (g[ok] - gt[j[ok]]).abs()
            acc.setdefault(f"{modality}_{kind}_z", []).append((err / s[ok]).flatten().numpy())
            acc.setdefault(f"{modality}_{kind}_err_px", []).append((err * PS).flatten().numpy())


def box_area(b: torch.Tensor) -> torch.Tensor:
    return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)


def symbol_spread(page: Page, coco_gt: COCO, cat_to_idx: dict[int, int], line_cats: set[int],
                  acc: list[np.ndarray]) -> None:
    """Corner distance (px) between slots of different patches assigned (nearest centre, same
    class, IoU >= 0.5) to the same symbol GT."""
    sel = [a for a in coco_gt.imgToAnns[page.image_id] if a["category_id"] not in line_cats]
    if not sel:
        return
    gt = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in sel]) / PS
    gt_label = torch.tensor([cat_to_idx[a["category_id"]] for a in sel])
    score, label = page.symbols["logits"].float().sigmoid().max(dim=-1)
    keep = score >= 0.05
    g, lab = page.symbols["geometry"][keep].float(), label[keep]
    patch = page.symbols["query"][keep].long() // page.num_shapes
    if g.shape[0] == 0:
        return
    lt = torch.maximum(g[:, None, :2], gt[None, :, :2])
    rb = torch.minimum(g[:, None, 2:], gt[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    iou = inter / (box_area(g)[:, None] + box_area(gt)[None] - inter).clamp(min=1e-9)
    iou = torch.where(lab[:, None] == gt_label[None], iou, 0.0)
    best, j = iou.max(dim=1)
    for k in j[best >= 0.5].unique().tolist():
        m = ((j == k) & (best >= 0.5)).nonzero().squeeze(1)
        if m.shape[0] < 2:
            continue
        p = g[m].view(-1, 2, 2)
        d = (p[:, None] - p[None]).norm(dim=-1).amax(-1)
        other = patch[m][:, None] != patch[m][None]
        iu = torch.triu_indices(m.shape[0], m.shape[0], 1)
        sel_pairs = other[iu[0], iu[1]]
        size = float(((gt[k, 2] - gt[k, 0]) * (gt[k, 3] - gt[k, 1])).clamp(min=1e-9).sqrt())
        acc.append(np.stack([(d[iu[0], iu[1]][sel_pairs] * PS).numpy(),
                             (d[iu[0], iu[1]][sel_pairs] / size).numpy()], axis=-1))


def in_patch_duplicates(page: Page, acc: dict[str, int]) -> None:
    """Pairs of slots of one patch, same class, all points within 4 px (lines, refined) or
    8 px (symbols): the duplicates that cannot-link forces into two clusters."""
    for modality, slots, key, radius in (("lines", page.lines, "refined", 4 / PS),
                                         ("symbols", page.symbols, "geometry", 8 / PS)):
        score, label = slots["logits"].float().sigmoid().max(dim=-1)
        patch = slots["query"].long() // page.num_shapes
        for floor in (0.05, 0.3):
            keep = (score >= floor).nonzero().squeeze(1)
            acc[f"{modality}_slots_{floor}"] = acc.get(f"{modality}_slots_{floor}", 0) + keep.shape[0]
            if keep.shape[0] < 2:
                continue
            order = keep[patch[keep].argsort()]
            pts = slots[key][order].float().view(-1, 2, 2)
            pp, ll = patch[order], label[order]
            # slots of one patch are consecutive: only compare within a band of K
            k = page.num_shapes
            n = order.shape[0]
            count = 0
            for off in range(1, k):
                a, b = torch.arange(n - off), torch.arange(off, n)
                if a.numel() == 0:
                    break
                close = (pts[a] - pts[b]).norm(dim=-1).amax(-1) <= radius
                count += int((close & (pp[a] == pp[b]) & (ll[a] == ll[b])).sum())
            acc[f"{modality}_inpatch_dupes_{floor}"] = acc.get(f"{modality}_inpatch_dupes_{floor}", 0) + count


def quantiles(parts: list[np.ndarray]) -> dict[str, JsonValue]:
    if not parts:
        return {}
    v = np.concatenate(parts)
    out: dict[str, JsonValue] = {"n": int(v.shape[0])}
    for q in (50, 90, 99):
        out[f"p{q}"] = round(float(np.percentile(v, q)), 3)
    return out


def calibration_summary(parts: list[np.ndarray]) -> dict[str, JsonValue]:
    if not parts:
        return {}
    z = np.concatenate(parts)
    out: dict[str, JsonValue] = {f"within_{k}sd": round(float((z <= k).mean()), 4) for k in (1, 2, 3)}
    out["median_z"] = round(float(np.median(z)), 3)
    return out


@dataclass
class PagePart:
    """Everything the page loop produces for one page, saved so that a run can resume."""

    results: dict[str, list[JsonValue]]
    calib: dict[str, list[torch.Tensor]]
    spread: list[torch.Tensor]
    inpatch: dict[str, int]
    seconds: dict[str, float] | None = None


def save_part(path: Path, part: PagePart) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"results": json.dumps(part.results), "calib": part.calib, "spread": part.spread,
                "inpatch": part.inpatch, "seconds": json.dumps(part.seconds or {})}, tmp)
    tmp.replace(path)


def load_part(path: Path) -> PagePart | None:
    if not path.exists():
        return None
    d: dict[str, object] = torch.load(path, weights_only=True)
    results, calib, spread, inpatch = d["results"], d["calib"], d["spread"], d["inpatch"]
    assert isinstance(results, str) and isinstance(calib, dict)
    assert isinstance(spread, list) and isinstance(inpatch, dict)
    parsed = json.loads(results)
    assert isinstance(parsed, dict)
    raw_seconds = d.get("seconds")
    seconds = json.loads(raw_seconds) if isinstance(raw_seconds, str) else {}
    assert isinstance(seconds, dict)
    return PagePart(
        seconds={str(k): float(v) for k, v in seconds.items()},
        results={str(k): v for k, v in parsed.items() if isinstance(v, list)},
        calib={str(k): [t for t in v if isinstance(t, torch.Tensor)] for k, v in calib.items() if isinstance(v, list)},
        spread=[t for t in spread if isinstance(t, torch.Tensor)],
        inpatch={str(k): int(v) for k, v in inpatch.items()},
    )


def write_predictions(part_paths: list[Path], names: list[str], preds_dir: Path) -> None:
    """One COCO results file per variant, streamed from the page parts in a single pass."""
    if not names:
        return
    tmp = {name: preds_dir / f"{name}.json.tmp" for name in names}
    files = {name: tmp[name].open("w") for name in names}
    first = dict.fromkeys(names, True)
    try:
        for f in files.values():
            f.write("[")
        for path in part_paths:
            part = load_part(path)
            assert part is not None, f"missing page part {path}"
            for name in names:
                for item in part.results[name]:
                    files[name].write(("" if first[name] else ",") + json.dumps(item))
                    first[name] = False
        for f in files.values():
            f.write("]")
    finally:
        for f in files.values():
            f.close()
    for name in names:
        tmp[name].replace(preds_dir / f"{name}.json")


def run(cache_dir: Path, out_dir: Path, anno_path: Path, held_out: Path, only: set[str] | None,
        max_pages: int | None, variant_set: Literal["auto", "one_to_one", "patch_set"] = "auto") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        coco_gt = COCO(str(anno_path))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    line_cat_list = [int(c["id"]) for c in cats if "keypoints" in c and "start" in c["keypoints"]]
    sym_cat_list = [int(c["id"]) for c in cats if int(c["id"]) not in set(line_cat_list)]
    line_cats = set(line_cat_list)
    names = {int(c["id"]): str(c["name"]) for c in cats}
    cat_ids: dict[Modality, list[int]] = {"symbols": sym_cat_list, "lines": line_cat_list}
    dataset = parse_coco(anno_path, cache_dir=Path("data/cache"))
    cat_to_idx: dict[Modality, dict[int, int]] = {
        "symbols": dict(dataset.symbol_cat_id_to_idx), "lines": dict(dataset.line_cat_id_to_idx),
    }
    cat_ids = {m: [c for c in ids if c in cat_to_idx[m]] for m, ids in cat_ids.items()}
    idx_to_cat: dict[Modality, dict[int, int]] = {m: {i: c for c, i in d.items()} for m, d in cat_to_idx.items()}
    caps: dict[Modality, int] = {"symbols": 0, "lines": 0}
    for img, anns in coco_gt.imgToAnns.items():
        caps["lines"] = max(caps["lines"], sum(a["category_id"] in line_cats for a in anns))
        caps["symbols"] = max(caps["symbols"], sum(a["category_id"] not in line_cats for a in anns))
    caps = {m: c + 100 for m, c in caps.items()}
    paths = sorted(cache_dir.glob("*.pt"), key=lambda p: int(p.stem))
    if max_pages is not None:
        paths = paths[:max_pages]
    img_ids = [int(p.stem) for p in paths]
    held = set(json.loads(held_out.read_text()))
    held_ids = [i for i in img_ids if i in held]
    common: dict[Modality, set[str]] = {
        "lines": {"barLine", "beam", "ledgerLines", "staff", "stem", "system"},
        "symbols": {names[c] for c in sym_cat_list
                    if sum(1 for a in coco_gt.loadAnns(coco_gt.getAnnIds(catIds=[c]))) >= 1000},
    }
    sets = gt_sets(coco_gt, img_ids, line_cats, names)
    first = load_page(paths[0])
    patch_set = "presence" in first.lines if variant_set == "auto" else variant_set == "patch_set"
    candidates_list = patch_set_variants() if patch_set else variants()
    print(f"{'patch-set' if patch_set else 'one-to-one'} cache, K = {first.num_shapes}", flush=True)
    chain_idx = chain_labels([c for c in cats if int(c["id"]) in line_cats], cat_to_idx["lines"])
    chosen = [replace(v, chain_labels=chain_idx) for v in candidates_list if only is None or v.name in only]
    seconds: dict[str, float] = {}
    calib: dict[str, list[np.ndarray]] = {}
    spread: list[np.ndarray] = []
    inpatch: dict[str, int] = {}
    t0 = time.monotonic()
    parts_dir = out_dir / "pages"
    parts_dir.mkdir(exist_ok=True)
    for n, path in enumerate(paths, start=1):
        part_path = parts_dir / f"{path.stem}.pt"
        part = load_part(part_path)
        if part is None or not all(v.name in part.results for v in chosen):
            page = load_page(path)
            part = PagePart({}, {}, [], {}, {})
            assert part.seconds is not None
            for v in chosen:
                start = time.monotonic()
                kept = run_variant(page, v, caps[v.modality])
                part.seconds[v.name] = time.monotonic() - start
                part.results[v.name] = coco_results(page, kept, v.modality, idx_to_cat[v.modality])
            acc: dict[str, list[np.ndarray]] = {}
            calibration(page, coco_gt, cat_to_idx, line_cats, acc)
            part.calib.update({k: [torch.from_numpy(a) for a in parts] for k, parts in acc.items()})
            page_spread: list[np.ndarray] = []
            symbol_spread(page, coco_gt, cat_to_idx["symbols"], line_cats, page_spread)
            part.spread.extend(torch.from_numpy(a) for a in page_spread)
            in_patch_duplicates(page, part.inpatch)
            save_part(part_path, part)
        for k, tensors in part.calib.items():
            calib.setdefault(k, []).extend(t.numpy() for t in tensors)
        spread.extend(t.numpy() for t in part.spread)
        for k, c in part.inpatch.items():
            inpatch[k] = inpatch.get(k, 0) + c
        for k, t in (part.seconds or {}).items():
            seconds[k] = seconds.get(k, 0.0) + t
        if n % 25 == 0 or n == len(paths):
            print(f"{n}/{len(paths)} pages, {(time.monotonic() - t0) / n:.1f} s/page", flush=True)

    summary: dict[str, JsonValue] = {
        "pages": len(paths),
        "calibration": {k: calibration_summary(v) for k, v in calib.items() if k.endswith("_z")},
        "error_px": {k: quantiles(v) for k, v in calib.items() if k.endswith("_err_px")},
        "symbol_duplicate_spread_px": quantiles([a[:, 0] for a in spread]),
        "symbol_duplicate_spread_rel_size": quantiles([a[:, 1] for a in spread]),
        "in_patch_duplicates": {k: v for k, v in inpatch.items()},
        "gt_sets": {
            "coincident_pairs": len(sets.coincident),
            "crowded_symbol_pairs": len(sets.crowded["symbols"]),
            "crowded_line_pairs": len(sets.crowded["lines"]),
            "staff_barline_ends": len(sets.staff_barline),
            "staff_other_ends": len(sets.staff_other),
        },
    }
    preds_dir, variants_dir = out_dir / "preds", out_dir / "variants"
    preds_dir.mkdir(exist_ok=True)
    variants_dir.mkdir(exist_ok=True)
    write_predictions([parts_dir / f"{p.stem}.pt" for p in paths],
                      [v.name for v in chosen if not (preds_dir / f"{v.name}.json").exists()], preds_dir)
    for v in chosen:
        done = variants_dir / f"{v.name}.json"
        if done.exists():
            summary[v.name] = json.loads(done.read_text())
            continue
        row: dict[str, JsonValue] = {}
        for split, ids in (("all", img_ids), ("held_out", held_ids)):
            if not ids:
                continue
            ev = evaluate(coco_gt, str(preds_dir / f"{v.name}.json"), v.modality, cat_ids[v.modality], ids)
            hit, dupes = matched_gt(ev)
            entry = ap_summary(ev, names, common[v.modality]) | {
                "confident_dupes": dupes,
                "crowded_pair_recall": round(pair_recall(sets.crowded[v.modality], hit), 4)
                if split == "all" else None,
            }
            if v.modality == "symbols" and split == "all":
                entry["coincident_pair_recall"] = round(pair_recall(sets.coincident, hit), 4)
            entry["seconds_per_page"] = round(seconds.get(v.name, 0.0) / max(1, len(paths)), 4)
            if v.modality == "lines":
                staff = [c for c in cat_ids["lines"] if names[c] == "staff"]
                entry["staff_confident_unmatched_per_page"] = round(
                    sum(confident_unmatched(ev, c) for c in staff) / max(1, len(ids)), 2
                )
            if v.modality == "lines" and split == "all":
                entry["staff_recall_barline_ends"] = round(len(sets.staff_barline & hit) / max(1, len(sets.staff_barline)), 4)
                entry["staff_recall_other_ends"] = round(len(sets.staff_other & hit) / max(1, len(sets.staff_other)), 4)
            row[split] = entry
            del ev
            gc.collect()
        summary[v.name] = row
        done.write_text(json.dumps(row))
        a = row["all"]
        assert isinstance(a, dict)
        print(f"{v.name:<34} AP {a['AP']:.4f} AP50 {a['AP50']:.4f} AP75 {a['AP75']:.4f} "
              f"common {a['AP_common']:.4f} dupes {a['confident_dupes']} crowd {a['crowded_pair_recall']}"
              + (f" coincident {a['coincident_pair_recall']}" if "coincident_pair_recall" in a else "")
              + (f" staff bar/other {a['staff_recall_barline_ends']}/{a['staff_recall_other_ends']}"
                 if "staff_recall_barline_ends" in a else "")
              + (f" staff fragments/page {a['staff_confident_unmatched_per_page']}"
                 if "staff_confident_unmatched_per_page" in a else ""), flush=True)
    (out_dir / "association_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("calibration", "error_px", "symbol_duplicate_spread_px",
                                      "symbol_duplicate_spread_rel_size", "in_patch_duplicates", "gt_sets")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--held_out", type=Path,
                        default=Path("experiments/037_029_recipe_from_023/refiner_phase_a/held_out_image_ids.json"))
    parser.add_argument("--only", nargs="+", default=None, help="Variant names to run.")
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--variant_set", choices=["auto", "one_to_one", "patch_set"], default="auto",
                        help="auto: patch-set variants when the cache has presence.")
    args = parser.parse_args()
    variant_set: Literal["auto", "one_to_one", "patch_set"] = args.variant_set
    run(Path(args.cache_dir), Path(args.out_dir), Path(args.anno_path), Path(args.held_out),
        None if args.only is None else set(args.only), args.max_pages, variant_set)


if __name__ == "__main__":
    main()
