"""Exp 035 Phase A+: a localization-quality branch on the frozen Phase A refiner.

The coarse score of a line query comes from a one-to-one head: only the Hungarian-matched
query is a positive, so the score says little about how well a query is placed (Analysis
003: for 34.7% of staff segments the top-scored prediction is not the best-localized one).
Here a small head reads the refiner's final endpoint states and predicts the OKS of the
refined line. Unlike the Exp 032 quality target, it is trained on **every** query kept at
inference (top ``max_lines`` per page), with the OKS against the best same-class ground truth
as a soft target, so duplicates are ranked by their placement.

Evaluation on the held-out pieces of Phase A:

* correlation between predicted and true OKS;
* "top-scored is not the best-localized": among the queries with OKS >= 0.5 to a ground
  truth, how often the highest-scored one is not the best-placed one;
* COCO line predictions with score = cls^(1 - alpha) * quality^alpha for several alpha,
  for ``evaluate_coco.py --image_ids`` (all with the refined endpoints).

Usage:
    PYTHONPATH=src:scripts mamba run -n pytorch python scripts/train_quality_branch.py
"""

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset.coco import CocoDataset, CocoLineAnnotation, parse_coco
from model.refiner import EndpointRefiner, LineQueries, QualityHead, TokenGrid, line_oks
from train_endpoint_refiner import (
    PageRecord,
    build_tables,
    coco_lines,
    held_out_pieces,
    page_queries,
)


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
    train_steps: int
    batch: int
    lr: float
    weight_decay: float
    alphas: tuple[float, ...]


@dataclass(frozen=True)
class QueryTable:
    """One row per query kept at inference, refined by the frozen Phase A refiner."""

    page: torch.Tensor
    cls_score: torch.Tensor
    label: torch.Tensor
    refined: torch.Tensor
    state: torch.Tensor
    oks: torch.Tensor
    coarse_oks: torch.Tensor

    def select(self, mask: torch.Tensor) -> "QueryTable":
        return QueryTable(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})


@dataclass(frozen=True)
class RankingStats:
    num_gt: int
    not_best_localized: float
    median_oks_gap: float


def load_refiner(path: Path, dim: int, num_classes: int, device: torch.device) -> EndpointRefiner:
    saved: dict[str, object] = torch.load(path, map_location=device, weights_only=True)
    config = saved["config"]
    state = saved["refiner"]
    assert isinstance(config, dict) and isinstance(state, dict)
    refiner = EndpointRefiner(
        dim=dim,
        num_classes=num_classes,
        num_steps=int(str(config["num_steps"])),
        heads=int(str(config["heads"])),
        scale=float(str(config["scale"])),
    ).to(device)
    refiner.load_state_dict(state)
    refiner.eval()
    return refiner


def line_areas(anno_path: Path, cache: Path, records: Sequence[PageRecord]) -> list[torch.Tensor]:
    """COCO ``area`` of every line annotation of each page, in the order of ``gt_keypoints``.

    The parsed dataset cache drops ``area``, so it is read once from the annotation JSON with
    the same line filter as ``parse_coco``.
    """
    if cache.exists():
        by_image: dict[int, torch.Tensor] = torch.load(cache, weights_only=True)
    else:
        print(f"Reading line areas from {anno_path} (one-off)...")
        coco = json.loads(anno_path.read_text())
        line_cats = {
            int(c["id"])
            for c in coco["categories"]
            if "keypoints" in c and "start" in c["keypoints"] and "end" in c["keypoints"]
        }
        wanted = {r.image_id for r in records}
        lists: dict[int, list[float]] = {i: [] for i in wanted}
        for ann in coco["annotations"]:
            image = int(ann["image_id"])
            if (
                image in wanted
                and int(ann["category_id"]) in line_cats
                and "keypoints" in ann
                and len(ann["keypoints"]) >= 6
            ):
                lists[image].append(float(ann["area"]))
        del coco
        by_image = {i: torch.tensor(a, dtype=torch.float32) for i, a in lists.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(by_image, cache)
    areas = [by_image[r.image_id] for r in records]
    for r, a in zip(records, areas):
        assert a.shape[0] == r.gt_labels.shape[0], f"image {r.image_id}: area count mismatch"
    return areas


def best_oks(
    pred_px: torch.Tensor,
    label: torch.Tensor,
    gt_px: torch.Tensor,
    gt_label: torch.Tensor,
    area: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """(Q, T) OKS, zero across classes."""
    if gt_label.numel() == 0:
        return pred_px.new_zeros(pred_px.shape[0], 0)
    oks = line_oks(pred_px, gt_px, area, sigma)
    return oks * (label.unsqueeze(1) == gt_label.unsqueeze(0))


def row_max(x: torch.Tensor) -> torch.Tensor:
    """Max over ground truths, 0 for a page without any."""
    return x.amax(dim=1) if x.shape[1] > 0 else x.new_zeros(x.shape[0])


def build_queries(
    cfg: Config,
    records: Sequence[PageRecord],
    areas: Sequence[torch.Tensor],
    grid: TokenGrid,
    token_offsets: Sequence[int],
    refiner: EndpointRefiner,
    max_lines: int,
    device: torch.device,
) -> QueryTable:
    pages = list(range(len(records)))
    queries, scores = page_queries(records, pages, token_offsets, cfg.num_shapes, max_lines, device)
    refined_parts: list[torch.Tensor] = []
    state_parts: list[torch.Tensor] = []
    chunk = 16384
    with torch.no_grad():
        for start in range(0, queries.image.shape[0], chunk):
            part = LineQueries(
                queries.image[start : start + chunk],
                queries.source[start : start + chunk],
                queries.label[start : start + chunk],
                queries.endpoints[start : start + chunk],
            )
            last = refiner(grid, part)[-1]
            refined_parts.append(last.endpoints)
            state_parts.append(last.state.half())
    refined = torch.cat(refined_parts)

    ps = float(cfg.patch_size)
    oks = torch.zeros_like(scores, device=device)
    coarse_oks = torch.zeros_like(oks)
    for i, r in enumerate(records):
        rows = (queries.image == i).nonzero().squeeze(1)
        gt_px = r.gt_keypoints.to(device) * ps
        gt_label = r.gt_labels.to(device)
        area = areas[i].to(device)
        label = queries.label[rows]
        oks[rows] = row_max(best_oks(refined[rows] * ps, label, gt_px, gt_label, area, cfg.sigma))
        coarse_oks[rows] = row_max(
            best_oks(queries.endpoints[rows] * ps, label, gt_px, gt_label, area, cfg.sigma)
        )
    return QueryTable(
        page=queries.image,
        cls_score=scores.to(device),
        label=queries.label,
        refined=refined,
        state=torch.cat(state_parts),
        oks=oks,
        coarse_oks=coarse_oks,
    )


def train_head(head: QualityHead, table: QueryTable, cfg: Config, device: torch.device) -> list[float]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train_steps)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    metrics = (cfg.out_dir / "metrics.jsonl").open("w")
    losses: list[float] = []
    head.train()
    for step in range(cfg.train_steps):
        idx = torch.randint(0, table.page.shape[0], (cfg.batch,), device=device, generator=generator)
        logits = head(table.state[idx].float())
        loss = F.binary_cross_entropy_with_logits(logits, table.oks[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            mean = sum(losses[-500:]) / 500
            metrics.write(json.dumps({"step": step + 1, "bce": mean}) + "\n")
            print(f"  step {step + 1}: BCE {mean:.4f}", flush=True)
    metrics.close()
    head.eval()
    return losses


def predict_quality(head: QualityHead, state: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([head(s.float()).sigmoid() for s in state.split(65536)])


def ranks(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort()
    out = torch.empty_like(x)
    out[order] = torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
    return out


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))


def ranking_stats(
    cfg: Config,
    records: Sequence[PageRecord],
    areas: Sequence[torch.Tensor],
    table: QueryTable,
    score: torch.Tensor,
    pages: Sequence[int],
    device: torch.device,
) -> RankingStats:
    """Among queries with OKS >= 0.5 to a ground truth, is the top-scored one the best-placed?"""
    ps = float(cfg.patch_size)
    wrong = 0
    total = 0
    gaps: list[torch.Tensor] = []
    for i in pages:
        r = records[i]
        rows = (table.page == i).nonzero().squeeze(1)
        oks = best_oks(
            table.refined[rows] * ps,
            table.label[rows],
            r.gt_keypoints.to(device) * ps,
            r.gt_labels.to(device),
            areas[i].to(device),
            cfg.sigma,
        )
        candidate = oks >= 0.5
        has = candidate.any(dim=0)
        if not bool(has.any()):
            continue
        masked_score = torch.where(candidate, score[rows].unsqueeze(1), -1.0)
        top = masked_score.argmax(dim=0)
        best = torch.where(candidate, oks, -1.0).amax(dim=0)
        top_oks = oks.gather(0, top.unsqueeze(0)).squeeze(0)
        gap = (best - top_oks)[has]
        wrong += int((gap > 1e-6).sum())
        total += int(has.sum())
        gaps.append(gap[gap > 1e-6])
    all_gaps = torch.cat(gaps) if gaps else torch.zeros(1)
    return RankingStats(
        num_gt=total,
        not_best_localized=wrong / max(1, total),
        median_oks_gap=float(all_gaps.median()) if all_gaps.numel() else 0.0,
    )


def class_names(dataset: CocoDataset) -> dict[int, str]:
    return {dataset.line_cat_id_to_idx[int(c["id"])]: str(c["name"]) for c in dataset.line_categories}


def run(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = parse_coco(cfg.anno_path, cache_dir=cfg.cache_dir)
    raw: list[dict[str, object]] = torch.load(cfg.phase_a_dir / "features.pt", weights_only=True)
    records = [PageRecord.from_dict(d) for d in raw]
    del raw
    areas = line_areas(cfg.anno_path, cfg.out_dir / "line_areas.pt", records)

    grid, _, pieces, token_offsets = build_tables(records, cfg.num_shapes, device)
    refiner = load_refiner(
        cfg.phase_a_dir / "refiner.pt", grid.tokens.shape[1], dataset.num_line_classes, device
    )
    max_lines = 100 + max(
        sum(1 for a in anns if isinstance(a, CocoLineAnnotation)) for anns in dataset.annotations.values()
    )
    print("Refining the top queries of every page...")
    table = build_queries(cfg, records, areas, grid, token_offsets, refiner, max_lines, device)

    holdout = held_out_pieces(pieces, cfg.seed, cfg.holdout_fraction)
    test_pages = [i for i, r in enumerate(records) if r.piece in holdout]
    is_test = torch.isin(table.page, torch.tensor(test_pages, device=device))
    train_table = table.select(~is_test)
    test_table = table.select(is_test)
    print(
        f"queries: train {train_table.page.shape[0]}, held-out {test_table.page.shape[0]} "
        f"({len(test_pages)} pages); train OKS > 0.5: {(train_table.oks > 0.5).float().mean():.3f}"
    )

    head = QualityHead(grid.tokens.shape[1]).to(device)
    train_head(head, train_table, cfg, device)
    torch.save({"quality_head": head.state_dict()}, cfg.out_dir / "quality_head.pt")

    quality = predict_quality(head, test_table.state)
    cls = test_table.cls_score
    print(
        f"held-out OKS correlation (Pearson / Spearman): "
        f"cls {pearson(cls, test_table.oks):.3f} / {pearson(ranks(cls), ranks(test_table.oks)):.3f}, "
        f"quality {pearson(quality, test_table.oks):.3f} / "
        f"{pearson(ranks(quality), ranks(test_table.oks)):.3f}"
    )

    variants: dict[str, torch.Tensor] = {"cls": cls}
    for alpha in cfg.alphas:
        variants[f"alpha_{alpha:g}"] = cls.pow(1 - alpha) * quality.pow(alpha)
    variants["quality"] = quality

    ranking: dict[str, RankingStats] = {}
    for name, score in variants.items():
        ranking[name] = ranking_stats(cfg, records, areas, test_table, score, test_pages, device)
        print(
            f"  {name:<12} top-scored not best-localized: "
            f"{ranking[name].not_best_localized * 100:5.1f}% of {ranking[name].num_gt} GT "
            f"(median OKS gap {ranking[name].median_oks_gap:.3f})"
        )
    names = class_names(dataset)
    staff_idx = next(k for k, v in names.items() if v == "staff")
    staff = test_table.label == staff_idx
    staff_oks = ranks(test_table.oks[staff])
    print(
        f"  staff queries, Spearman with OKS: cls {pearson(ranks(cls[staff]), staff_oks):.3f}, "
        f"quality {pearson(ranks(quality[staff]), staff_oks):.3f}"
    )

    idx_to_cat = {v: k for k, v in dataset.line_cat_id_to_idx.items()}
    for name, score in variants.items():
        out = cfg.out_dir / name
        out.mkdir(exist_ok=True)
        results = coco_lines(
            records, test_table.refined, test_table.label, score, test_table.page, idx_to_cat, cfg.patch_size
        )
        (out / "preds_lines.json").write_text(json.dumps(results))
    summary = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "queries": {"train": int(train_table.page.shape[0]), "held_out": int(test_table.page.shape[0])},
        "ranking": {name: asdict(s) for name, s in ranking.items()},
    }
    (cfg.out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved predictions for {list(variants)} under {cfg.out_dir}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--phase_a_dir", type=Path, default=Path("experiments/035_endpoint_refiner/phase_a"))
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out_dir", type=Path, default=Path("experiments/035_endpoint_refiner/phase_a/quality"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.1, help="OKS sigma, as in evaluate_coco.py.")
    parser.add_argument("--train_steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
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
        train_steps=int(args.train_steps),
        batch=int(args.batch),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        alphas=tuple(float(a) for a in args.alphas),
    )


if __name__ == "__main__":
    run(parse_args())
