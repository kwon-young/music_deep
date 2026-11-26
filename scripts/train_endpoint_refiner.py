"""Exp 035 Phase A: train the endpoint refiner on a frozen Exp 029 detector.

The backbone and the coarse line head are frozen, so their outputs on a page never change:
one pass over the pages caches the final tokens, the coarse line predictions and the
Hungarian assignment of every ground-truth line (``features.pt``). The refiner then trains on
that cache in minutes, on the pieces of the training split (same split by piece as Analysis
005), and is evaluated on the held-out pieces:

* endpoint error of the coarse head vs each refinement step, grouped like Analysis 005;
* COCO line predictions of the held-out pages, coarse and refined, for ``evaluate_coco.py
  --image_ids``.

Training queries come in two kinds, one of each per sampled ground-truth line:

* real: the matched query's coarse endpoints and predicted class (what inference will see);
  endpoints the coarse head misses by more than ``max_coarse_error`` patches are left out of
  the loss, since the window cannot reach them;
* denoising: the ground truth plus noise shaped like the head's errors, with the true class.

Usage:
    PYTHONPATH=src mamba run -n pytorch python scripts/train_endpoint_refiner.py --use_sdpa
"""

import argparse
import json
import math
import random
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch

import transform.det as det_tf
from dataset.coco import CocoDataset, CocoLineAnnotation, load_coco_sample, parse_coco
from model.detector import OMRDetector, create_detector
from model.matcher import HungarianMatcher
from model.refiner import EndpointRefiner, LineQueries, TokenGrid, refiner_loss
from music_types import Batch, NumPatches, PatchDim, Patches
from threaded_generator import ThreadedGenerator


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    anno_path: Path
    img_dir: Path
    cache_dir: Path | None
    out_dir: Path
    backbone_size: str
    patch_size: int
    channels: int
    num_shapes: int
    base_anchor_size: float
    var_threshold: float
    use_sdpa: bool
    device: str
    num_pages: int | None
    holdout_fraction: float
    seed: int
    cost_class: float
    cost_bbox: float
    radius_patches: float
    top_k: int
    num_steps: int
    heads: int
    scale: float
    train_steps: int
    batch_lines: int
    lr: float
    weight_decay: float
    warmup_steps: int
    fgl_weight: float
    max_coarse_error: float
    dn_sigma: float
    dn_uniform: float
    dn_uniform_prob: float
    score_threshold: float


@dataclass(frozen=True)
class PageInput:
    piece: str
    image_id: int
    width: int
    height: int
    patches: Patches[Batch, NumPatches, PatchDim]
    keypoints: torch.Tensor
    keypoint_labels: torch.Tensor


@dataclass(frozen=True)
class PageRecord:
    """Everything the refiner needs from one page, on CPU (coordinates in patch units)."""

    piece: str
    image_id: int
    width: int
    height: int
    grid_h: int
    grid_w: int
    tokens: torch.Tensor
    cells: torch.Tensor
    logits: torch.Tensor
    keypoints: torch.Tensor
    gt_keypoints: torch.Tensor
    gt_labels: torch.Tensor
    match: torch.Tensor

    def to_dict(self) -> dict[str, str | int | torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def from_dict(d: dict[str, object]) -> "PageRecord":
        def tensor(key: str) -> torch.Tensor:
            value = d[key]
            assert isinstance(value, torch.Tensor)
            return value

        def integer(key: str) -> int:
            value = d[key]
            assert isinstance(value, int)
            return value

        piece = d["piece"]
        assert isinstance(piece, str)
        return PageRecord(
            piece=piece,
            image_id=integer("image_id"),
            width=integer("width"),
            height=integer("height"),
            grid_h=integer("grid_h"),
            grid_w=integer("grid_w"),
            tokens=tensor("tokens"),
            cells=tensor("cells"),
            logits=tensor("logits"),
            keypoints=tensor("keypoints"),
            gt_keypoints=tensor("gt_keypoints"),
            gt_labels=tensor("gt_labels"),
            match=tensor("match"),
        )


@dataclass(frozen=True)
class LineTable:
    """One row per matched ground-truth line, on the training device."""

    image: torch.Tensor
    piece: torch.Tensor
    source: torch.Tensor
    pred_label: torch.Tensor
    score: torch.Tensor
    coarse: torch.Tensor
    gt_label: torch.Tensor
    target: torch.Tensor
    ambiguous: torch.Tensor
    reach: torch.Tensor

    def map(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "LineTable":
        return LineTable(**{fd.name: f(getattr(self, fd.name)) for fd in fields(self)})

    def real_queries(self) -> LineQueries:
        return LineQueries(self.image, self.source, self.pred_label, self.coarse)


@dataclass(frozen=True)
class ErrorStats:
    n: int
    median_px: float
    p90_px: float
    mean_px: float
    worse_2px: float


def piece_name(file_name: str) -> str:
    return re.sub(r"_\d+$", "", Path(file_name).stem)


def load_model(cfg: Config, dataset: CocoDataset, device: torch.device) -> OMRDetector:
    model = create_detector(
        backbone_size=cfg.backbone_size,
        patch_size=cfg.patch_size,
        channels=cfg.channels,
        use_sdpa=cfg.use_sdpa,
        num_symbol_classes=dataset.num_symbol_classes,
        num_line_classes=dataset.num_line_classes,
        num_shapes=cfg.num_shapes,
        base_anchor_size=cfg.base_anchor_size,
    ).to(device)
    checkpoint: dict[str, object] = torch.load(
        cfg.checkpoint, map_location=device, weights_only=True
    )
    state = checkpoint["model"]
    assert isinstance(state, dict)
    model.load_state_dict(state)
    model.eval()
    return model


def page_inputs(dataset: CocoDataset, cfg: Config, indices: Sequence[int]) -> Iterator[PageInput]:
    """Decodes, patches and drops on CPU; the main thread moves the kept patches to the GPU."""
    cpu = torch.device("cpu")
    patch_size = (cfg.patch_size, cfg.patch_size)
    for i in indices:
        item = load_coco_sample(dataset, cfg.img_dir, i, cpu)
        decoded = det_tf.decode_pyvips(item, device=cpu)
        padded = det_tf.pad_to_patch_size(det_tf.to_float1(decoded), patch_size=patch_size)
        patched = det_tf.extract_patches(det_tf.collate((padded,)), patch_size=patch_size)
        dropped = det_tf.variance_patch_drop(patched, var_threshold=cfg.var_threshold)
        meta = item.metadata
        yield PageInput(
            piece=piece_name(meta.file_name),
            image_id=meta.img_id,
            width=meta.width,
            height=meta.height,
            patches=dropped.sample.image,
            keypoints=item.sample.keypoints.data,
            keypoint_labels=item.sample.keypoint_labels.data,
        )


def match_lines(
    matchers: tuple[HungarianMatcher, HungarianMatcher],
    logits: torch.Tensor,
    keypoints: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_keypoints: torch.Tensor,
) -> torch.Tensor:
    """Query index assigned to each ground truth by the training matcher (-1 if none)."""
    cpu = torch.device("cpu")
    match = torch.full_like(gt_labels, -1)
    if gt_labels.numel() == 0:
        return match
    probs = logits.float().sigmoid()
    hungarian, greedy = matchers
    try:
        indices = hungarian._match_modality(probs, keypoints, gt_labels, gt_keypoints, False, cpu)
    except ValueError:
        indices = greedy._match_modality(probs, keypoints, gt_labels, gt_keypoints, False, cpu)
    match[indices.target_indices.cpu()] = indices.pred_indices.cpu()
    return match


def extract(model: OMRDetector, dataset: CocoDataset, cfg: Config, device: torch.device) -> list[PageRecord]:
    indices = list(range(len(dataset.images)))
    random.Random(cfg.seed).shuffle(indices)
    if cfg.num_pages is not None:
        indices = indices[: cfg.num_pages]

    final: list[torch.Tensor] = []

    def keep_final(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        final.append(output)

    handle = model.backbone.transformer.norm.register_forward_hook(keep_final)
    matchers = tuple(
        HungarianMatcher(
            cost_class=cfg.cost_class,
            cost_bbox=cfg.cost_bbox,
            calc_device=torch.device("cpu"),
            radius_patches=cfg.radius_patches,
            top_k=cfg.top_k,
            matcher_type=kind,
        )
        for kind in ("scipy", "greedy")
    )
    assert len(matchers) == 2
    records: list[PageRecord] = []
    loader = ThreadedGenerator(page_inputs(dataset, cfg, indices), maxsize=2, name="pages")
    ps = float(cfg.patch_size)
    with torch.no_grad(), loader as pages:
        for n, page in enumerate(pages, start=1):
            patches = page.patches
            on_device = replace(
                patches, data=patches.data.to(device), indices=patches.indices.to(device)
            )
            final.clear()
            outputs = model.forward(on_device)
            logits = outputs.lines.pred_logits.data[0].cpu()
            keypoints = outputs.lines.pred_keypoints.data[0].cpu()
            gt_keypoints = page.keypoints / ps
            _, height, width = patches.image_shape
            records.append(
                PageRecord(
                    piece=page.piece,
                    image_id=page.image_id,
                    width=page.width,
                    height=page.height,
                    grid_h=height // patches.patch_size[0],
                    grid_w=width // patches.patch_size[1],
                    tokens=final[0][0].half().cpu(),
                    cells=patches.indices[0].cpu(),
                    logits=logits.half(),
                    keypoints=keypoints,
                    gt_keypoints=gt_keypoints,
                    gt_labels=page.keypoint_labels,
                    match=match_lines(
                        (matchers[0], matchers[1]), logits, keypoints, page.keypoint_labels, gt_keypoints
                    ),
                )
            )
            if n % 10 == 0 or n == len(indices):
                print(f"  features: {n}/{len(indices)} pages", flush=True)
    handle.remove()
    return records


def load_or_extract(cfg: Config, dataset: CocoDataset, device: torch.device) -> list[PageRecord]:
    cache = cfg.out_dir / "features.pt"
    if cache.exists():
        print(f"Loading cached features from {cache}")
        raw: list[dict[str, object]] = torch.load(cache, weights_only=True)
        return [PageRecord.from_dict(d) for d in raw]
    print("Extracting frozen features...")
    model = load_model(cfg, dataset, device)
    records = extract(model, dataset, cfg, device)
    del model
    torch.cuda.empty_cache()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save([r.to_dict() for r in records], cache)
    print(f"Saved {cache}")
    return records


def held_out_pieces(pieces: Sequence[str], cfg: Config) -> set[str]:
    """Same split as Analysis 005: sorted pieces, shuffled with the seed, first fraction held out."""
    order = list(range(len(pieces)))
    random.Random(cfg.seed).shuffle(order)
    num_holdout = max(1, round(cfg.holdout_fraction * len(pieces)))
    return {pieces[k] for k in order[:num_holdout]}


def endpoint_ambiguity(gt: torch.Tensor, labels: torch.Tensor, grid_w: int) -> torch.Tensor:
    """(T, 2): another same-class line has the same endpoint (start/end) in the same patch."""
    num = labels.shape[0]
    points = gt.view(num, 2, 2)
    cell = points[..., 1].floor().long() * grid_w + points[..., 0].floor().long()
    which = torch.arange(2).expand(num, 2)
    key = (labels.unsqueeze(1) * 2 + which) * (1 << 32) + cell
    _, inverse, counts = key.flatten().unique(return_inverse=True, return_counts=True)
    return (counts[inverse] > 1).view(num, 2)


def build_tables(
    records: Sequence[PageRecord], num_shapes: int, device: torch.device
) -> tuple[TokenGrid, LineTable, list[str], list[int]]:
    pieces = sorted({r.piece for r in records})
    piece_ids = {p: k for k, p in enumerate(pieces)}
    grids: list[TokenGrid] = []
    rows: list[LineTable] = []
    token_offsets: list[int] = []
    offset = 0
    for i, r in enumerate(records):
        grids.append(TokenGrid.from_batch(r.tokens.unsqueeze(0), r.cells.unsqueeze(0), r.grid_h, r.grid_w))
        token_offsets.append(offset)
        matched = (r.match >= 0).nonzero().squeeze(1)
        query = r.match[matched]
        logits = r.logits[query].float()
        rows.append(
            LineTable(
                image=torch.full_like(query, i),
                piece=torch.full_like(query, piece_ids[r.piece]),
                source=query // num_shapes + offset,
                pred_label=logits.argmax(dim=-1),
                score=logits.sigmoid().amax(dim=-1),
                coarse=r.keypoints[query],
                gt_label=r.gt_labels[matched],
                target=r.gt_keypoints[matched],
                ambiguous=endpoint_ambiguity(r.gt_keypoints, r.gt_labels, r.grid_w)[matched],
                reach=torch.zeros(matched.shape[0], 2),
            )
        )
        offset += r.tokens.shape[0]
    grid = TokenGrid.concat(grids).to(device)
    table = LineTable(
        **{fd.name: torch.cat([getattr(t, fd.name) for t in rows]) for fd in fields(LineTable)}
    ).map(lambda t: t.to(device))
    source_center = grid.centers(table.source, table.image)
    reach = torch.linalg.vector_norm(
        table.target.view(-1, 2, 2) - source_center.unsqueeze(1), dim=-1
    )
    table = LineTable(**{**{fd.name: getattr(table, fd.name) for fd in fields(table)}, "reach": reach})
    return grid, table, pieces, token_offsets


def denoising_noise(shape: tuple[int, int], cfg: Config, device: torch.device) -> torch.Tensor:
    """Per endpoint: small Gaussian (precise head) or uniform up to ``dn_uniform`` (tail)."""
    num = shape[0]
    gaussian = torch.randn(num, 2, 2, device=device) * cfg.dn_sigma
    uniform = (torch.rand(num, 2, 2, device=device) * 2 - 1) * cfg.dn_uniform
    pick = torch.rand(num, 2, 1, device=device) < cfg.dn_uniform_prob
    return torch.where(pick, uniform, gaussian).reshape(shape)


def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.train_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(
    refiner: EndpointRefiner, grid: TokenGrid, table: LineTable, cfg: Config, device: torch.device
) -> None:
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    within = (table.coarse - table.target).view(-1, 2, 2).abs().amax(dim=-1) <= cfg.max_coarse_error
    real_mask = within.float().repeat_interleave(2, dim=1)
    print(
        f"train lines {table.image.shape[0]}; coarse endpoints within "
        f"{cfg.max_coarse_error} patches: {within.float().mean():.3f}"
    )
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    metrics = (cfg.out_dir / "metrics.jsonl").open("w")
    refiner.train()
    start = time.time()
    window: list[tuple[float, list[float], list[float]]] = []
    for step in range(cfg.train_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        idx = torch.randint(0, table.image.shape[0], (cfg.batch_lines,), device=device, generator=generator)
        batch = table.map(lambda t: t[idx])
        noisy = batch.target + denoising_noise((cfg.batch_lines, 4), cfg, device)
        queries = LineQueries.concat(
            [
                batch.real_queries(),
                LineQueries(batch.image, batch.source, batch.gt_label, noisy),
            ]
        )
        target = torch.cat([batch.target, batch.target])
        mask = torch.cat([real_mask[idx], torch.ones_like(batch.target)])
        outputs = refiner(grid, queries)
        loss, l1_steps = refiner_loss(refiner, outputs, target, mask, cfg.fgl_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            n = cfg.batch_lines
            real_l1 = [
                ((o.endpoints[:n] - batch.target).abs() * real_mask[idx]).sum().item()
                / max(1.0, real_mask[idx].sum().item())
                for o in outputs
            ]
        window.append((loss.item(), l1_steps, real_l1))
        if (step + 1) % 100 == 0 or step + 1 == cfg.train_steps:
            k = len(window)
            mean_loss = sum(w[0] for w in window) / k
            l1_px = [sum(w[1][s] for w in window) / k * cfg.patch_size for s in range(cfg.num_steps)]
            real_px = [sum(w[2][s] for w in window) / k * cfg.patch_size for s in range(cfg.num_steps)]
            elapsed = time.time() - start
            record = {
                "step": step + 1,
                "lr": lr_at(step, cfg),
                "loss": mean_loss,
                "l1_px_per_step": l1_px,
                "real_l1_px_per_step": real_px,
                "elapsed_s": elapsed,
            }
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            if (step + 1) % 1000 == 0 or step + 1 == cfg.train_steps:
                steps_txt = " ".join(f"{v:.2f}" for v in real_px)
                print(
                    f"  step {step + 1}: loss {mean_loss:.4f}, real L1 per step "
                    f"(px) {steps_txt}, {elapsed:.0f}s",
                    flush=True,
                )
            window = []
    metrics.close()


def refine(
    refiner: EndpointRefiner, grid: TokenGrid, queries: LineQueries, chunk: int = 16384
) -> list[torch.Tensor]:
    """Endpoints after each step, (N, 4) per step, without gradients."""
    refiner.eval()
    per_step: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, queries.image.shape[0], chunk):
            part = LineQueries(
                queries.image[start : start + chunk],
                queries.source[start : start + chunk],
                queries.label[start : start + chunk],
                queries.endpoints[start : start + chunk],
            )
            outputs = refiner(grid, part)
            for s, out in enumerate(outputs):
                if len(per_step) <= s:
                    per_step.append([])
                per_step[s].append(out.endpoints)
    return [torch.cat(parts) for parts in per_step]


def endpoint_errors(pred: torch.Tensor, target: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(N, 2) Euclidean error per endpoint, in pixels."""
    return torch.linalg.vector_norm((pred - target).view(-1, 2, 2), dim=-1) * patch_size


def stats(errors: torch.Tensor, baseline: torch.Tensor) -> ErrorStats:
    return ErrorStats(
        n=int(errors.numel()),
        median_px=errors.median().item(),
        p90_px=errors.quantile(0.9).item(),
        mean_px=errors.mean().item(),
        worse_2px=(errors > baseline + 2.0).float().mean().item(),
    )


def evaluation_groups(
    table: LineTable, dataset: CocoDataset, cfg: Config
) -> list[tuple[str, torch.Tensor]]:
    """(N, 2) endpoint masks."""
    confident = (table.score >= cfg.score_threshold).unsqueeze(1).expand(-1, 2)
    within = (table.coarse - table.target).view(-1, 2, 2).abs().amax(dim=-1) <= cfg.max_coarse_error
    groups: list[tuple[str, torch.Tensor]] = [
        ("lines, all matched", torch.ones_like(confident)),
        (f"lines, score >= {cfg.score_threshold}", confident),
        ("  coarse within window", confident & within),
        ("  unambiguous", confident & ~table.ambiguous),
        ("  ambiguous", confident & table.ambiguous),
    ]
    names = {
        dataset.line_cat_id_to_idx[int(cat["id"])]: str(cat["name"]) for cat in dataset.line_categories
    }
    for idx, name in sorted(names.items()):
        is_class = (table.gt_label == idx).unsqueeze(1).expand(-1, 2)
        groups.append((f"  {name}", confident & is_class))
        if name == "staff":
            groups.append(("  staff, unambiguous", confident & is_class & ~table.ambiguous))
    for low, high in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, math.inf)]:
        groups.append(
            (f"  reach {low}-{high} patches", confident & (table.reach >= low) & (table.reach < high))
        )
    return groups


def coco_lines(
    records: Sequence[PageRecord],
    keypoints: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    query_image: torch.Tensor,
    idx_to_cat: dict[int, int],
    patch_size: int,
) -> list[dict[str, int | float | list[float]]]:
    results: list[dict[str, int | float | list[float]]] = []
    kps = (keypoints * patch_size).cpu().tolist()
    for kp, label, score, image in zip(kps, labels.tolist(), scores.tolist(), query_image.tolist()):
        r = records[image]
        x1, y1, x2, y2 = kp
        results.append(
            {
                "image_id": r.image_id,
                "category_id": idx_to_cat[label],
                "keypoints": [
                    max(0.0, min(x1, float(r.width))),
                    max(0.0, min(y1, float(r.height))),
                    1,
                    max(0.0, min(x2, float(r.width))),
                    max(0.0, min(y2, float(r.height))),
                    1,
                ],
                "score": score,
            }
        )
    return results


def page_queries(
    records: Sequence[PageRecord],
    pages: Sequence[int],
    token_offsets: Sequence[int],
    num_shapes: int,
    max_lines: int,
    device: torch.device,
) -> tuple[LineQueries, torch.Tensor]:
    """Top ``max_lines`` line queries of each page by score, as inference keeps them."""
    parts: list[LineQueries] = []
    scores: list[torch.Tensor] = []
    for i in pages:
        r = records[i]
        probs = r.logits.float().sigmoid()
        best, labels = probs.max(dim=-1)
        top_scores, top = best.topk(min(max_lines, best.shape[0]))
        parts.append(
            LineQueries(
                image=torch.full_like(top, i),
                source=top // num_shapes + token_offsets[i],
                label=labels[top],
                endpoints=r.keypoints[top],
            )
        )
        scores.append(top_scores)
    queries = LineQueries.concat(parts)
    return (
        LineQueries(
            queries.image.to(device),
            queries.source.to(device),
            queries.label.to(device),
            queries.endpoints.to(device),
        ),
        torch.cat(scores),
    )


def run(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = parse_coco(cfg.anno_path, cache_dir=cfg.cache_dir)
    records = load_or_extract(cfg, dataset, device)

    grid, table, pieces, token_offsets = build_tables(records, cfg.num_shapes, device)
    holdout = held_out_pieces(pieces, cfg)
    holdout_ids = torch.tensor([k for k, p in enumerate(pieces) if p in holdout], device=device)
    is_test = torch.isin(table.piece, holdout_ids)
    train_table = table.map(lambda t: t[~is_test])
    test_table = table.map(lambda t: t[is_test])
    total_gt = sum(r.gt_labels.shape[0] for r in records)
    print(
        f"{len(records)} pages, {len(pieces)} pieces ({len(holdout)} held out); "
        f"{table.image.shape[0]}/{total_gt} lines matched; "
        f"train {train_table.image.shape[0]}, held-out {test_table.image.shape[0]}"
    )

    refiner = EndpointRefiner(
        dim=grid.tokens.shape[1],
        num_classes=dataset.num_line_classes,
        num_steps=cfg.num_steps,
        heads=cfg.heads,
        scale=cfg.scale,
    ).to(device)
    print(f"refiner parameters: {sum(p.numel() for p in refiner.parameters()):,}")
    train(refiner, grid, train_table, cfg, device)
    torch.save({"refiner": refiner.state_dict(), "config": {k: str(v) for k, v in asdict(cfg).items()}},
               cfg.out_dir / "refiner.pt")

    refined = refine(refiner, grid, test_table.real_queries())
    head = endpoint_errors(test_table.coarse, test_table.target, cfg.patch_size)
    errors = {"head": head} | {
        f"step {s + 1}": endpoint_errors(p, test_table.target, cfg.patch_size)
        for s, p in enumerate(refined)
    }
    report: dict[str, dict[str, ErrorStats]] = {}
    for name, mask in evaluation_groups(test_table, dataset, cfg):
        if int(mask.sum()) < 20:
            continue
        report[name] = {src: stats(err[mask], head[mask]) for src, err in errors.items()}

    sources = list(errors)
    last = sources[-1]
    print(f"{'held-out endpoints (median / p90 px)':<38}{'n':>8}" + "".join(f"{s:>16}" for s in sources)
          + f"{'worse>2px':>11}")
    for name, by_source in report.items():
        row = f"{name:<38}{by_source['head'].n:>8}"
        row += "".join(f"{by_source[s].median_px:>8.1f} /{by_source[s].p90_px:>6.1f}" for s in sources)
        row += f"{by_source[last].worse_2px * 100:>10.1f}%"
        print(row)

    test_pages = [i for i, r in enumerate(records) if r.piece in holdout]
    max_lines = 100 + max(
        sum(1 for a in anns if isinstance(a, CocoLineAnnotation)) for anns in dataset.annotations.values()
    )
    queries, scores = page_queries(records, test_pages, token_offsets, cfg.num_shapes, max_lines, device)
    refined_pages = refine(refiner, grid, queries)[-1]
    idx_to_cat = {v: k for k, v in dataset.line_cat_id_to_idx.items()}
    for name, keypoints in (("coarse", queries.endpoints), ("refined", refined_pages)):
        out = cfg.out_dir / name
        out.mkdir(exist_ok=True)
        results = coco_lines(
            records, keypoints, queries.label, scores, queries.image, idx_to_cat, cfg.patch_size
        )
        (out / "preds_lines.json").write_text(json.dumps(results))
    image_ids = [records[i].image_id for i in test_pages]
    (cfg.out_dir / "held_out_image_ids.json").write_text(json.dumps(image_ids))

    summary = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "lines": {
            "total_gt": total_gt,
            "matched": int(table.image.shape[0]),
            "train": int(train_table.image.shape[0]),
            "held_out": int(test_table.image.shape[0]),
            "held_out_pages": len(test_pages),
        },
        "groups": {
            name: {src: asdict(s) for src, s in by_source.items()} for name, by_source in report.items()
        },
    }
    (cfg.out_dir / "refiner_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {cfg.out_dir / 'refiner_summary.json'} and COCO predictions for {len(test_pages)} pages")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/029_size_difficulty_balanced_loss/train_detection/checkpoints/latest_model.pt"),
    )
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out_dir", type=Path, default=Path("experiments/035_endpoint_refiner/phase_a"))
    parser.add_argument("--backbone_size", choices=["nano", "small", "base"], default="nano")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_pages", type=int, default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost_class", type=float, default=2.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--radius_patches", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=2, help="Refinement steps (re-centred windows).")
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--scale", type=float, default=3.0, help="Correction range in patches (±scale/2).")
    parser.add_argument("--train_steps", type=int, default=8000)
    parser.add_argument("--batch_lines", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--fgl_weight", type=float, default=0.15)
    parser.add_argument("--max_coarse_error", type=float, default=1.5)
    parser.add_argument("--dn_sigma", type=float, default=0.05)
    parser.add_argument("--dn_uniform", type=float, default=1.0)
    parser.add_argument("--dn_uniform_prob", type=float, default=0.4)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    args = parser.parse_args()
    return Config(
        checkpoint=Path(args.checkpoint),
        anno_path=Path(args.anno_path),
        img_dir=Path(args.img_dir),
        cache_dir=None if args.cache_dir is None else Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        backbone_size=str(args.backbone_size),
        patch_size=int(args.patch_size),
        channels=int(args.channels),
        num_shapes=int(args.num_shapes),
        base_anchor_size=float(args.base_anchor_size),
        var_threshold=float(args.var_threshold),
        use_sdpa=bool(args.use_sdpa),
        device=str(args.device),
        num_pages=None if args.num_pages is None else int(args.num_pages),
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
        cost_class=float(args.cost_class),
        cost_bbox=float(args.cost_bbox),
        radius_patches=float(args.radius_patches),
        top_k=int(args.top_k),
        num_steps=int(args.num_steps),
        heads=int(args.heads),
        scale=float(args.scale),
        train_steps=int(args.train_steps),
        batch_lines=int(args.batch_lines),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        fgl_weight=float(args.fgl_weight),
        max_coarse_error=float(args.max_coarse_error),
        dn_sigma=float(args.dn_sigma),
        dn_uniform=float(args.dn_uniform),
        dn_uniform_prob=float(args.dn_uniform_prob),
        score_threshold=float(args.score_threshold),
    )


if __name__ == "__main__":
    run(parse_args())
