"""Exp 035 diagnostic: where does the remaining line AP go? (TIDE-style, with OKS)

On the held-out pages of Phase A, with the refined endpoints, this replays the COCO matching
(per page and class, detections in score order take the unmatched ground truth of highest
OKS >= t) and labels every error, following TIDE (Bolya et al., 2020) with OKS for IoU:

* false positives: ``dupe`` (OKS >= t with a same-class GT already taken by a higher score),
  ``loc`` (same class, 0.1 <= OKS < t), ``cls`` (OKS >= t with a GT of another class),
  ``both`` (another class, 0.1 <= OKS < t), ``bkg`` (OKS < 0.1 with every GT);
* ``missed`` ground truths (not matched, not covered by a cls/loc error), split by asking
  **all** queries of the page (every slot of every patch, refined), not only the kept ones:
  ``below_cutoff`` (a same-class query with OKS >= t exists but is not in the top
  ``max_lines``: under-scored), ``taken`` (such queries are kept but matched to other GTs),
  ``wrong_class`` (only other-class queries reach OKS >= t), ``none`` (no query does).

Each error type's cost is the AP gained by fixing only that type with an oracle (TIDE's dAP):
dupe/bkg/both are removed, cls/loc detections take their GT when it is free (else removed),
missed GTs are removed from the GT count.

Usage:
    PYTHONPATH=src:scripts mamba run -n pytorch python scripts/diagnose_line_errors.py
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from dataset.coco import CocoLineAnnotation, parse_coco
from model.refiner import (
    EndpointRefiner,
    LineQueries,
    QualityHead,
    TokenGrid,
    deduplicate_lines,
    line_oks,
)
from dataset.split import held_out_pieces
from train_endpoint_refiner import PageRecord, build_tables
from train_quality_branch import best_oks, line_areas, load_refiner

FP_TYPES = ("dupe", "loc", "cls", "both", "bkg")
MISS_TYPES = ("below_cutoff", "taken", "wrong_class", "none")
BACKGROUND_OKS = 0.1


@dataclass(frozen=True)
class Config:
    phase_a_dir: Path
    anno_path: Path
    cache_dir: Path | None
    out_dir: Path
    device: str
    seed: int
    holdout_fraction: float
    num_shapes: int
    patch_size: int
    sigma: float
    thresholds: tuple[float, ...]
    alpha: float
    line_areas: Path
    candidate_floor: float
    nms_radius: float


@dataclass
class ClassDetections:
    """Detections of one class over all held-out pages, after matching at one threshold."""

    score: list[float] = field(default_factory=list)
    tp: list[bool] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    fix_target: list[tuple[int, int, int]] = field(default_factory=list)
    num_gt: int = 0
    missed: Counter[str] = field(default_factory=Counter)
    covered: int = 0


@dataclass(frozen=True)
class PageGeometry:
    """Per held-out page: kept detections and all-query candidates, as numpy arrays."""

    page: int
    score: np.ndarray
    label: np.ndarray
    oks: np.ndarray
    gt_label: np.ndarray
    kept_query: np.ndarray
    all_label: np.ndarray
    all_oks: np.ndarray


def coco_ap(score: Sequence[float], tp: Sequence[bool], num_gt: int) -> float:
    """COCO 101-point interpolated AP of one class at one threshold."""
    if num_gt == 0:
        return float("nan")
    if len(score) == 0:
        return 0.0
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = np.asarray(tp, dtype=np.float64)[order]
    tps = np.cumsum(hits)
    fps = np.cumsum(1.0 - hits)
    recall = tps / num_gt
    precision = tps / np.maximum(tps + fps, np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0).mean())


def refine_all(
    refiner: EndpointRefiner,
    grid: TokenGrid,
    queries: LineQueries,
    chunk: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refined endpoints and final endpoint states of every query."""
    endpoints: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, queries.image.shape[0], chunk):
            part = LineQueries(
                queries.image[start : start + chunk],
                queries.source[start : start + chunk],
                queries.label[start : start + chunk],
                queries.endpoints[start : start + chunk],
            )
            last = refiner(grid, part)[-1]
            endpoints.append(last.endpoints)
            states.append(last.state)
    return torch.cat(endpoints), torch.cat(states)


def page_geometry(
    cfg: Config,
    records: Sequence[PageRecord],
    areas: Sequence[torch.Tensor],
    grid: TokenGrid,
    token_offsets: Sequence[int],
    refiner: EndpointRefiner,
    head: QualityHead | None,
    page: int,
    max_lines: int,
    device: torch.device,
) -> dict[str, PageGeometry]:
    """Geometry of one page for each ranking/post-processing variant (see ``variant_names``)."""
    r = records[page]
    probs = r.logits.float().sigmoid()
    best, labels = probs.max(dim=-1)
    num = best.shape[0]
    all_queries = LineQueries(
        image=torch.full((num,), page, dtype=torch.long, device=device),
        source=(torch.arange(num) // cfg.num_shapes + token_offsets[page]).to(device),
        label=labels.to(device),
        endpoints=r.keypoints.to(device),
    )
    refined, state = refine_all(refiner, grid, all_queries)
    ps = float(cfg.patch_size)
    gt_px = r.gt_keypoints.to(device) * ps
    gt_label = r.gt_labels.to(device)
    area = areas[page].to(device)
    if gt_label.shape[0]:
        raw_oks = line_oks(refined * ps, gt_px, area, cfg.sigma)
        coarse_oks = line_oks(all_queries.endpoints * ps, gt_px, area, cfg.sigma)
    else:
        raw_oks = refined.new_zeros(num, 0)
        coarse_oks = raw_oks
    cls_score = best.to(device)
    top = cls_score.topk(min(max_lines, num)).indices

    def geometry(kept: torch.Tensor, score: torch.Tensor, oks: torch.Tensor) -> PageGeometry:
        """``kept`` query indices with their final ``score`` (aligned with ``kept``)."""
        return PageGeometry(
            page=page,
            score=score.cpu().numpy(),
            label=all_queries.label[kept].cpu().numpy(),
            oks=oks[kept].cpu().numpy(),
            gt_label=gt_label.cpu().numpy(),
            kept_query=kept.cpu().numpy(),
            all_label=all_queries.label.cpu().numpy(),
            all_oks=oks.cpu().numpy(),
        )

    out = {
        "coarse endpoints, cls": geometry(top, cls_score[top], coarse_oks),
        "refined endpoints, cls": geometry(top, cls_score[top], raw_oks),
    }
    if head is not None:
        with torch.no_grad():
            quality = head(state).sigmoid()
        combined = cls_score.pow(1 - cfg.alpha) * quality.pow(cfg.alpha)
        out[f"refined endpoints, cls*quality (alpha {cfg.alpha:g})"] = geometry(
            top, combined[top], raw_oks
        )
    # Analysis 007 post-processing: refine the candidates above the floor, NMS with
    # noisy-OR merged scores, then keep the top lines by merged score.
    candidates = (cls_score >= cfg.candidate_floor).nonzero().squeeze(1)
    deduped = deduplicate_lines(
        refined[candidates] * ps,
        cls_score[candidates],
        all_queries.label[candidates],
        cfg.nms_radius,
        merge=True,
    )
    keep = deduped.keep[:max_lines]
    out[nms_variant(cfg)] = geometry(candidates[keep], deduped.score[:max_lines], raw_oks)
    return out


def nms_variant(cfg: Config) -> str:
    return (
        f"refined endpoints, score >= {cfg.candidate_floor:g}, "
        f"NMS {cfg.nms_radius:g}px + noisy-OR"
    )


def variant_names(cfg: Config, with_quality: bool) -> list[str]:
    names = ["coarse endpoints, cls", "refined endpoints, cls"]
    if with_quality:
        names.append(f"refined endpoints, cls*quality (alpha {cfg.alpha:g})")
    return names + [nms_variant(cfg)]


def match_page(geo: PageGeometry, t: float, per_class: dict[int, ClassDetections]) -> None:
    """COCO greedy matching per class, then TIDE labels for FPs and missed GTs."""
    num_gt = geo.gt_label.shape[0]
    same = geo.label[:, None] == geo.gt_label[None, :] if num_gt else np.zeros((len(geo.label), 0), bool)
    oks_same = np.where(same, geo.oks, -1.0)
    taken = np.zeros(num_gt, dtype=bool)
    order = np.argsort(-geo.score, kind="mergesort")
    is_tp = np.zeros(len(geo.score), dtype=bool)
    for d in order:
        if num_gt == 0:
            break
        candidates = np.where(~taken & (oks_same[d] >= t), oks_same[d], -1.0)
        g = int(candidates.argmax())
        if candidates[g] >= t:
            taken[g] = True
            is_tp[d] = True

    covered = np.zeros(num_gt, dtype=bool)
    for d in order:
        c = int(geo.label[d])
        cd = per_class.setdefault(c, ClassDetections())
        cd.score.append(float(geo.score[d]))
        cd.tp.append(bool(is_tp[d]))
        if is_tp[d] or num_gt == 0:
            cd.kind.append("tp" if is_tp[d] else "bkg")
            cd.fix_target.append((-1, -1, -1))
            continue
        row_same = oks_same[d]
        row_other = np.where(~same[d], geo.oks[d], -1.0)
        if row_same.max() >= t:
            kind, g = "dupe", int(row_same.argmax())
        elif row_other.max() >= t:
            kind, g = "cls", int(row_other.argmax())
        elif row_same.max() >= BACKGROUND_OKS:
            kind, g = "loc", int(row_same.argmax())
        elif row_other.max() >= BACKGROUND_OKS:
            kind, g = "both", int(row_other.argmax())
        else:
            kind, g = "bkg", -1
        cd.kind.append(kind)
        if kind in ("cls", "loc") and not taken[g]:
            covered[g] = True
            cd.fix_target.append((geo.page, g, int(geo.gt_label[g])))
        else:
            cd.fix_target.append((-1, -1, -1))

    kept_set = np.zeros(len(geo.all_label), dtype=bool)
    kept_set[geo.kept_query] = True
    for g in range(num_gt):
        c = int(geo.gt_label[g])
        cd = per_class.setdefault(c, ClassDetections())
        cd.num_gt += 1
        if taken[g]:
            continue
        if covered[g]:
            cd.covered += 1
            continue
        good = geo.all_oks[:, g] >= t
        good_same = good & (geo.all_label == c)
        if good_same.any():
            kind = "taken" if (good_same & kept_set).any() else "below_cutoff"
        elif good.any():
            kind = "wrong_class"
        else:
            kind = "none"
        cd.missed[kind] += 1


def fixed_ap(cd: ClassDetections, fix: str, moved_in: list[float]) -> float:
    """AP of one class after an oracle fix of one error type (TIDE)."""
    score: list[float] = []
    tp: list[bool] = []
    num_gt = cd.num_gt
    claimed: set[tuple[int, int]] = set()
    for s, hit, kind, target in zip(cd.score, cd.tp, cd.kind, cd.fix_target):
        if kind == fix and fix in ("dupe", "bkg", "both"):
            continue
        if kind == fix and fix == "loc":
            key = (target[0], target[1])
            if target[0] >= 0 and key not in claimed:
                claimed.add(key)
                score.append(s)
                tp.append(True)
            continue
        if kind == fix and fix == "cls":
            continue
        score.append(s)
        tp.append(hit)
    if fix == "cls":
        score += moved_in
        tp += [True] * len(moved_in)
    if fix == "missed":
        num_gt -= sum(cd.missed.values())
    return coco_ap(score, tp, num_gt)


def report(
    per_class: dict[int, ClassDetections], names: dict[int, str], common: set[str]
) -> dict[str, dict[str, float | int | dict[str, int]]]:
    moved: dict[int, list[float]] = {c: [] for c in per_class}
    for cd in per_class.values():
        claimed: set[tuple[int, int]] = set()
        for s, kind, target in zip(cd.score, cd.kind, cd.fix_target):
            if kind == "cls" and target[0] >= 0 and (target[0], target[1]) not in claimed:
                claimed.add((target[0], target[1]))
                moved.setdefault(target[2], []).append(s)

    out: dict[str, dict[str, float | int | dict[str, int]]] = {}
    for c, cd in sorted(per_class.items()):
        if cd.num_gt == 0:
            continue
        base = coco_ap(cd.score, cd.tp, cd.num_gt)
        row: dict[str, float | int | dict[str, int]] = {"num_gt": cd.num_gt, "ap": base}
        for fix in FP_TYPES + ("missed",):
            row[f"dAP_{fix}"] = fixed_ap(cd, fix, moved.get(c, [])) - base
        kinds = Counter(cd.kind)
        confident = Counter(k for k, s in zip(cd.kind, cd.score) if s >= 0.3)
        row["fp"] = {k: kinds[k] for k in FP_TYPES}
        row["fp_score_ge_0.3"] = {k: confident[k] for k in FP_TYPES}
        row["tp"] = kinds["tp"]
        row["tp_score_lt_0.3"] = sum(1 for k, s in zip(cd.kind, cd.score) if k == "tp" and s < 0.3)
        row["missed"] = {k: cd.missed[k] for k in MISS_TYPES}
        row["covered_by_cls_loc"] = cd.covered
        out[names[c]] = row

    def mean_over(keys: set[str]) -> dict[str, float | int | dict[str, int]]:
        rows = [out[k] for k in out if k in keys]
        agg: dict[str, float | int | dict[str, int]] = {"num_gt": sum(int(str(r["num_gt"])) for r in rows)}
        for key in ["ap"] + [f"dAP_{f}" for f in FP_TYPES + ("missed",)]:
            agg[key] = float(np.mean([float(str(r[key])) for r in rows]))
        return agg

    out["mean, all classes"] = mean_over(set(names.values()))
    out["mean, common classes"] = mean_over(common)
    return out


def print_table(title: str, rows: dict[str, dict[str, float | int | dict[str, int]]]) -> None:
    print(f"\n{title}")
    header = f"{'class':<22}{'nGT':>7}{'AP':>7}" + "".join(f"{'d' + f:>9}" for f in FP_TYPES + ("missed",))
    print(header + "   missed: below_cutoff/taken/wrong_class/none | conf. FP (>=0.3) dupe/loc/cls/both/bkg | TP<0.3")
    for name, row in rows.items():
        line = f"{name:<22}{int(str(row['num_gt'])):>7}{float(str(row['ap'])):>7.3f}"
        line += "".join(f"{float(str(row['dAP_' + f])):>+9.3f}" for f in FP_TYPES + ("missed",))
        missed = row.get("missed")
        conf = row.get("fp_score_ge_0.3")
        if isinstance(missed, dict) and isinstance(conf, dict):
            line += "   " + "/".join(str(missed[k]) for k in MISS_TYPES)
            line += " | " + "/".join(str(conf[k]) for k in FP_TYPES)
            line += f" | {row['tp_score_lt_0.3']}"
        print(line)


def run(cfg: Config) -> None:
    device = torch.device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = parse_coco(cfg.anno_path, cache_dir=cfg.cache_dir)
    raw: list[dict[str, object]] = torch.load(cfg.phase_a_dir / "features.pt", weights_only=True)
    records = [PageRecord.from_dict(d) for d in raw]
    del raw
    areas = line_areas(cfg.anno_path, cfg.line_areas, records)
    grid, _, pieces, token_offsets = build_tables(records, cfg.num_shapes, device)
    refiner = load_refiner(
        cfg.phase_a_dir / "refiner.pt", grid.tokens.shape[1], dataset.num_line_classes, device
    )
    head: QualityHead | None = None
    head_path = cfg.phase_a_dir / "quality" / "quality_head.pt"
    if head_path.exists():
        head = QualityHead(grid.tokens.shape[1]).to(device)
        saved: dict[str, object] = torch.load(head_path, map_location=device, weights_only=True)
        state = saved["quality_head"]
        assert isinstance(state, dict)
        head.load_state_dict(state)
        head.eval()

    max_lines = 100 + max(
        sum(1 for a in anns if isinstance(a, CocoLineAnnotation)) for anns in dataset.annotations.values()
    )
    holdout = held_out_pieces(pieces, cfg.seed, cfg.holdout_fraction)
    test_pages = [i for i, r in enumerate(records) if r.piece in holdout]
    names = {dataset.line_cat_id_to_idx[int(c["id"])]: str(c["name"]) for c in dataset.line_categories}
    common = {"barLine", "beam", "ledgerLines", "staff", "stem", "system"}

    variants = variant_names(cfg, with_quality=head is not None)
    # Matching is per page, so each page's (queries x GT) OKS tables are dropped right away.
    per_key: dict[str, dict[int, ClassDetections]] = {
        f"{variant}, OKS {t:g}": {} for variant in variants for t in cfg.thresholds
    }
    for p in test_pages:
        geo = page_geometry(cfg, records, areas, grid, token_offsets, refiner, head, p, max_lines, device)
        for variant in variants:
            for t in cfg.thresholds:
                match_page(geo[variant], t, per_key[f"{variant}, OKS {t:g}"])
    summary: dict[str, dict[str, dict[str, float | int | dict[str, int]]]] = {}
    for key, per_class in per_key.items():
        rows = report(per_class, names, common)
        summary[key] = rows
        print_table(key, rows)
    (cfg.out_dir / "line_error_diagnosis.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {cfg.out_dir / 'line_error_diagnosis.json'}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--phase_a_dir", type=Path, default=Path("experiments/035_endpoint_refiner/phase_a"))
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out_dir", type=Path, default=Path("experiments/035_endpoint_refiner/phase_a/diagnosis"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--line_areas",
        type=Path,
        default=Path("experiments/035_endpoint_refiner/phase_a/quality/line_areas.pt"),
        help="Cache of the GT line areas (read from the annotation JSON when missing).",
    )
    parser.add_argument("--candidate_floor", type=float, default=0.05)
    parser.add_argument("--nms_radius", type=float, default=4.0)
    args = parser.parse_args()
    return Config(
        phase_a_dir=Path(args.phase_a_dir),
        anno_path=Path(args.anno_path),
        cache_dir=None if args.cache_dir is None else Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        device=str(args.device),
        seed=int(args.seed),
        holdout_fraction=float(args.holdout_fraction),
        num_shapes=int(args.num_shapes),
        patch_size=int(args.patch_size),
        sigma=float(args.sigma),
        thresholds=tuple(float(t) for t in args.thresholds),
        alpha=float(args.alpha),
        line_areas=Path(args.line_areas),
        candidate_floor=float(args.candidate_floor),
        nms_radius=float(args.nms_radius),
    )


if __name__ == "__main__":
    run(parse_args())
