"""Analysis 005: is the localization limit of the detector a reach or a resolution problem?

On a frozen detector, small MLP probes regress the position of every ground-truth point
(both endpoints of a line, top-left and bottom-right corners of a box) from:

* ``head``: the detector's own prediction of the point, taken from the closest confident
  query of the same class (no probe, the baseline the model achieves today);
* ``matched@<depth>``: the final token of that same query, with its slot as input;
* ``local@<tap>``: the token of the patch that *contains* the point, read at several depths
  (0 = patch embedding output, depth = final normalized output);
* ``center``: no model at all, the center of the containing patch.

All sources are evaluated on the same points of held-out pieces (split by piece, so no page
of a held-out score is seen by the probes). Reading the result:

* ``local@final`` close to 1-2 px while ``matched`` and ``head`` are far worse, with the gap
  growing with the reach (distance between the predicting token and the point): the
  information exists where the evidence is, so a second stage that samples tokens at the
  predicted endpoints is enough.
* ``local@0`` good but ``local@final`` bad: the backbone discards fine geometry with depth.
* ``local@0`` already bad: the patch embedding throws it away, so the second stage also
  needs raw pixels around the endpoint.

Usage (no GPU work is done at import time):
    PYTHONPATH=src mamba run -n pytorch python scripts/probe_token_localization.py --use_sdpa
"""

import argparse
import json
import random
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

import transform.det as det_tf
from dataset.coco import CocoDataset, load_coco_sample, parse_coco
from model.detector import OMRDetector, create_detector
from music_types import Batch, NumPatches, PatchDim, Patches
from threaded_generator import ThreadedGenerator

SYMBOL = 0
LINE = 1


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
    max_points_per_page: int
    score_threshold: float
    taps: tuple[int, ...]
    holdout_fraction: float
    seed: int
    probe_hidden: int
    probe_steps: int
    probe_batch: int
    probe_lr: float


@dataclass(frozen=True)
class PageInput:
    piece: str
    patches: Patches[Batch, NumPatches, PatchDim]
    boxes: torch.Tensor
    box_labels: torch.Tensor
    keypoints: torch.Tensor
    keypoint_labels: torch.Tensor


@dataclass(frozen=True)
class PointTable:
    """One row per ground-truth point; coordinates in patch units, features on CPU in fp16."""

    piece: torch.Tensor
    kind: torch.Tensor
    label: torch.Tensor
    which: torch.Tensor
    size: torch.Tensor
    point: torch.Tensor
    head_point: torch.Tensor
    local_valid: torch.Tensor
    local_ambiguous: torch.Tensor
    local_center: torch.Tensor
    local_feats: torch.Tensor
    matched_valid: torch.Tensor
    matched_center: torch.Tensor
    matched_slot: torch.Tensor
    matched_feats: torch.Tensor

    def map(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "PointTable":
        return PointTable(
            piece=f(self.piece),
            kind=f(self.kind),
            label=f(self.label),
            which=f(self.which),
            size=f(self.size),
            point=f(self.point),
            head_point=f(self.head_point),
            local_valid=f(self.local_valid),
            local_ambiguous=f(self.local_ambiguous),
            local_center=f(self.local_center),
            local_feats=f(self.local_feats),
            matched_valid=f(self.matched_valid),
            matched_center=f(self.matched_center),
            matched_slot=f(self.matched_slot),
            matched_feats=f(self.matched_feats),
        )

    @staticmethod
    def concat(tables: Sequence["PointTable"]) -> "PointTable":
        return PointTable(
            piece=torch.cat([t.piece for t in tables]),
            kind=torch.cat([t.kind for t in tables]),
            label=torch.cat([t.label for t in tables]),
            which=torch.cat([t.which for t in tables]),
            size=torch.cat([t.size for t in tables]),
            point=torch.cat([t.point for t in tables]),
            head_point=torch.cat([t.head_point for t in tables]),
            local_valid=torch.cat([t.local_valid for t in tables]),
            local_ambiguous=torch.cat([t.local_ambiguous for t in tables]),
            local_center=torch.cat([t.local_center for t in tables]),
            local_feats=torch.cat([t.local_feats for t in tables]),
            matched_valid=torch.cat([t.matched_valid for t in tables]),
            matched_center=torch.cat([t.matched_center for t in tables]),
            matched_slot=torch.cat([t.matched_slot for t in tables]),
            matched_feats=torch.cat([t.matched_feats for t in tables]),
        )


@dataclass(frozen=True)
class ProbeData:
    feats: torch.Tensor
    cond: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class ErrorStats:
    n: int
    median_px: float
    p90_px: float
    mean_px: float


class TokenTaps:
    """Records the backbone's residual stream at chosen depths during a forward pass.

    Tap 0 is the patch embedding output, tap d in [1, depth) the stream after block d (read
    as the input of block d + 1), and tap ``depth`` the final normalized tokens the heads see.
    """

    def __init__(self, model: OMRDetector, taps: Sequence[int]) -> None:
        backbone = model.backbone
        layers = backbone.transformer.layers
        self.depth = len(layers)
        self.features: dict[int, torch.Tensor] = {}
        self.handles: list[RemovableHandle] = []
        for tap in taps:
            if tap == 0:
                self.handles.append(
                    backbone.patch_embed.register_forward_hook(self._output_hook(tap))
                )
            elif 0 < tap < self.depth:
                block = layers[tap]
                assert isinstance(block, nn.ModuleList)
                self.handles.append(
                    block[0].register_forward_pre_hook(self._input_hook(tap))
                )
            elif tap == self.depth:
                self.handles.append(
                    backbone.transformer.norm.register_forward_hook(
                        self._output_hook(tap)
                    )
                )
            else:
                raise ValueError(f"tap {tap} outside [0, {self.depth}]")

    def _output_hook(
        self, tap: int
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(
            module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            self.features[tap] = output

        return hook

    def _input_hook(
        self, tap: int
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            self.features[tap] = args[0]

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


class Probe(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, feats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.norm(feats), cond], dim=-1))


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


def page_to(page: PageInput, device: torch.device) -> PageInput:
    """Moves a page decoded on CPU to the device; only the kept patches travel."""
    patches = replace(
        page.patches,
        data=page.patches.data.to(device),
        indices=page.patches.indices.to(device),
    )
    return replace(
        page,
        patches=patches,
        boxes=page.boxes.to(device),
        box_labels=page.box_labels.to(device),
        keypoints=page.keypoints.to(device),
        keypoint_labels=page.keypoint_labels.to(device),
    )


def page_inputs(
    dataset: CocoDataset,
    cfg: Config,
    indices: Sequence[int],
    device: torch.device,
) -> Iterator[PageInput]:
    patch_size = (cfg.patch_size, cfg.patch_size)
    for i in indices:
        item = load_coco_sample(dataset, cfg.img_dir, i, device)
        decoded = det_tf.decode_pyvips(item, device=device)
        padded = det_tf.pad_to_patch_size(
            det_tf.to_float1(decoded), patch_size=patch_size
        )
        patched = det_tf.extract_patches(
            det_tf.collate((padded,)), patch_size=patch_size
        )
        dropped = det_tf.variance_patch_drop(patched, var_threshold=cfg.var_threshold)
        yield PageInput(
            piece=piece_name(item.metadata.file_name),
            patches=dropped.sample.image,
            boxes=item.sample.boxes.data,
            box_labels=item.sample.box_labels.data,
            keypoints=item.sample.keypoints.data,
            keypoint_labels=item.sample.keypoint_labels.data,
        )


def match_objects(
    logits: torch.Tensor,
    pred_geom: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_geom: torch.Tensor,
    score_threshold: float,
) -> torch.Tensor:
    """Index of the query that predicts each ground truth, or -1.

    The query is the closest (L1 over the 4 coordinates) among the confident queries of the
    same class. Several ground truths may share a query: the probe asks which token the
    detector relies on, not for a one-to-one assignment.
    """
    scores, classes = logits.sigmoid().max(dim=-1)
    query = torch.full_like(gt_labels, -1)
    for c in gt_labels.unique().tolist():
        candidates = ((classes == c) & (scores >= score_threshold)).nonzero().squeeze(1)
        if candidates.numel() == 0:
            continue
        gt_idx = (gt_labels == c).nonzero().squeeze(1)
        for chunk in gt_idx.split(1024):
            dist = torch.cdist(gt_geom[chunk], pred_geom[candidates], p=1.0)
            query[chunk] = candidates[dist.argmin(dim=1)]
    return query


def cell_centers(cells: torch.Tensor, grid_w: int) -> torch.Tensor:
    return torch.stack([cells % grid_w + 0.5, cells // grid_w + 0.5], dim=-1).float()


def modality_points(
    kind: int,
    gt_geom: torch.Tensor,
    gt_labels: torch.Tensor,
    logits: torch.Tensor,
    pred_geom: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, ...]:
    """Per-point geometry for one modality, in patch units (features gathered later)."""
    query = match_objects(logits, pred_geom, gt_labels, gt_geom, cfg.score_threshold)
    num = gt_labels.shape[0]
    head = pred_geom[query.clamp(min=0)]
    point = torch.cat([gt_geom[:, :2], gt_geom[:, 2:]])
    head_point = torch.cat([head[:, :2], head[:, 2:]])
    size = torch.linalg.vector_norm(gt_geom[:, 2:] - gt_geom[:, :2], dim=-1).repeat(2)
    which = torch.arange(2, device=gt_labels.device).repeat_interleave(num)
    return (
        torch.full_like(which, kind),
        gt_labels.repeat(2),
        which,
        size,
        point,
        head_point,
        query.repeat(2),
    )


def extract_page(
    model: OMRDetector,
    taps: TokenTaps,
    tap_order: Sequence[int],
    page: PageInput,
    piece_id: int,
    cfg: Config,
    rng: torch.Generator,
) -> PointTable:
    # forward() rather than __call__ keeps the DetectionOutput type; the tap hooks live on
    # submodules, which are still invoked through __call__.
    outputs = model.forward(page.patches)
    num_shapes = model.symbol_head.num_shapes
    ps = float(cfg.patch_size)

    _, height, width = page.patches.image_shape
    ph, pw = page.patches.patch_size
    grid_h, grid_w = height // ph, width // pw
    kept_cells = page.patches.indices[0]
    device = kept_cells.device
    cell_to_pos = torch.full((grid_h * grid_w,), -1, dtype=torch.long, device=device)
    cell_to_pos[kept_cells] = torch.arange(kept_cells.shape[0], device=device)

    parts = [
        modality_points(
            SYMBOL,
            page.boxes / ps,
            page.box_labels,
            outputs.symbols.pred_logits.data[0],
            outputs.symbols.pred_boxes.data[0],
            cfg,
        ),
        modality_points(
            LINE,
            page.keypoints / ps,
            page.keypoint_labels,
            outputs.lines.pred_logits.data[0],
            outputs.lines.pred_keypoints.data[0],
            cfg,
        ),
    ]
    kind, label, which, size, point, head_point, query = (
        torch.cat(column) for column in zip(*parts)
    )

    col = point[:, 0].floor().long().clamp(0, grid_w - 1)
    row = point[:, 1].floor().long().clamp(0, grid_h - 1)
    local_cell = row * grid_w + col
    local_pos = cell_to_pos[local_cell]

    num_labels = max(int(label.max().item()) + 1 if label.numel() else 1, 1)
    group = ((kind * num_labels + label) * 2 + which) * (grid_h * grid_w) + local_cell
    _, inverse, counts = group.unique(return_inverse=True, return_counts=True)

    matched_pos = query.clamp(min=0) // num_shapes

    features = torch.stack([taps.features[t][0] for t in tap_order], dim=1)
    final = taps.features[taps.depth][0]

    keep = torch.randperm(point.shape[0], generator=rng)[: cfg.max_points_per_page]
    keep = keep.to(device)
    local_keep = local_pos[keep]
    matched_keep = matched_pos[keep]
    table = PointTable(
        piece=torch.full_like(keep, piece_id),
        kind=kind[keep],
        label=label[keep],
        which=which[keep],
        size=size[keep],
        point=point[keep],
        head_point=head_point[keep],
        local_valid=local_keep >= 0,
        local_ambiguous=counts[inverse][keep] > 1,
        local_center=cell_centers(local_cell[keep], grid_w),
        local_feats=features[local_keep.clamp(min=0)].half(),
        matched_valid=query[keep] >= 0,
        matched_center=cell_centers(kept_cells[matched_keep], grid_w),
        matched_slot=query[keep].clamp(min=0) % num_shapes,
        matched_feats=final[matched_keep].half(),
    )
    return table.map(lambda t: t.cpu())


def collect_points(
    model: OMRDetector, dataset: CocoDataset, cfg: Config, device: torch.device
) -> tuple[PointTable, list[str]]:
    indices = list(range(len(dataset.images)))
    random.Random(cfg.seed).shuffle(indices)
    if cfg.num_pages is not None:
        indices = indices[: cfg.num_pages]

    pieces = sorted({piece_name(dataset.images[i].file_name) for i in indices})
    piece_ids = {p: k for k, p in enumerate(pieces)}

    taps = TokenTaps(model, sorted(set(cfg.taps) | {len(model.backbone.transformer.layers)}))
    rng = torch.Generator().manual_seed(cfg.seed)
    tables: list[PointTable] = []
    loader = ThreadedGenerator(
        page_inputs(dataset, cfg, indices, torch.device("cpu")),
        maxsize=2,
        name="pages",
    )
    with torch.no_grad(), loader as pages:
        for n, page in enumerate(pages, start=1):
            tables.append(
                extract_page(
                    model,
                    taps,
                    cfg.taps,
                    page_to(page, device),
                    piece_ids[page.piece],
                    cfg,
                    rng,
                )
            )
            if n % 10 == 0 or n == len(indices):
                print(f"  features: {n}/{len(indices)} pages", flush=True)
    taps.remove()
    return PointTable.concat(tables), pieces


def split_by_piece(
    table: PointTable, pieces: Sequence[str], cfg: Config
) -> torch.Tensor:
    """Boolean mask of held-out rows."""
    order = list(range(len(pieces)))
    random.Random(cfg.seed).shuffle(order)
    num_holdout = max(1, round(cfg.holdout_fraction * len(pieces)))
    holdout = torch.tensor(order[:num_holdout], dtype=torch.long)
    return torch.isin(table.piece, holdout)


def condition(
    table: PointTable,
    dataset: CocoDataset,
    num_shapes: int,
    with_slot: bool,
) -> torch.Tensor:
    global_label = table.label + table.kind * dataset.num_symbol_classes
    parts = [
        F.one_hot(global_label, dataset.num_symbol_classes + dataset.num_line_classes),
        F.one_hot(table.which, 2),
    ]
    if with_slot:
        parts.append(F.one_hot(table.matched_slot, num_shapes))
    return torch.cat(parts, dim=-1).float()


def fit_probe(
    data: ProbeData, cfg: Config, device: torch.device
) -> tuple[Probe, float]:
    probe = Probe(data.feats.shape[-1], data.cond.shape[-1], cfg.probe_hidden).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.probe_steps)
    feats = data.feats.to(device)
    cond = data.cond.to(device)
    target = data.target.to(device)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    recent: list[float] = []
    for _ in range(cfg.probe_steps):
        idx = torch.randint(
            0, feats.shape[0], (cfg.probe_batch,), device=device, generator=generator
        )
        pred: torch.Tensor = probe(feats[idx].float(), cond[idx])
        loss = F.l1_loss(pred, target[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        recent = (recent + [loss.item()])[-100:]
    return probe, sum(recent) / len(recent)


def predict(probe: Probe, data: ProbeData, device: torch.device) -> torch.Tensor:
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for feats, cond in zip(data.feats.split(16384), data.cond.split(16384)):
            pred: torch.Tensor = probe(feats.to(device).float(), cond.to(device))
            preds.append(pred.cpu())
    return torch.cat(preds)


def error_px(pred: torch.Tensor, target: torch.Tensor, patch_size: int) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target, dim=-1) * patch_size


def stats(errors: torch.Tensor) -> ErrorStats:
    return ErrorStats(
        n=int(errors.numel()),
        median_px=errors.median().item(),
        p90_px=errors.quantile(0.9).item(),
        mean_px=errors.mean().item(),
    )


def evaluation_groups(
    table: PointTable, dataset: CocoDataset, patch_size: int
) -> list[tuple[str, torch.Tensor]]:
    is_symbol = table.kind == SYMBOL
    is_line = table.kind == LINE
    unambiguous = ~table.local_ambiguous
    reach = torch.linalg.vector_norm(table.point - table.matched_center, dim=-1)
    diag_px = table.size * patch_size

    groups: list[tuple[str, torch.Tensor]] = [
        ("symbols", is_symbol),
        ("symbols, unambiguous", is_symbol & unambiguous),
    ]
    for low, high in [(0, 32), (32, 64), (64, 128), (128, float("inf"))]:
        groups.append(
            (f"symbols, diag {low}-{high} px", is_symbol & (diag_px >= low) & (diag_px < high))
        )
    groups += [("lines", is_line), ("lines, unambiguous", is_line & unambiguous)]
    line_names = {
        dataset.line_cat_id_to_idx[int(cat["id"])]: str(cat["name"])
        for cat in dataset.line_categories
    }
    for idx, name in sorted(line_names.items()):
        groups.append((f"line {name}", is_line & (table.label == idx)))
    for low, high in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, float("inf"))]:
        groups.append(
            (f"lines, reach {low}-{high} patches", is_line & (reach >= low) & (reach < high))
        )
    return groups


def run(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    dataset = parse_coco(cfg.anno_path, cache_dir=cfg.cache_dir)
    model = load_model(cfg, dataset, device)
    depth = len(model.backbone.transformer.layers)
    num_shapes = model.symbol_head.num_shapes

    print("Extracting frozen features...")
    table, pieces = collect_points(model, dataset, cfg, device)
    del model
    total = table.point.shape[0]
    common = table.local_valid & table.matched_valid
    print(
        f"{total} points; local patch kept {table.local_valid.float().mean():.3f}, "
        f"matched query found {table.matched_valid.float().mean():.3f}, "
        f"both {common.float().mean():.3f}"
    )

    table = table.map(lambda t: t[common])
    holdout = split_by_piece(table, pieces, cfg)
    train = table.map(lambda t: t[~holdout])
    test = table.map(lambda t: t[holdout])
    print(f"train {train.point.shape[0]} points, held-out {test.point.shape[0]} points")

    errors: dict[str, torch.Tensor] = {
        "head": error_px(test.head_point, test.point, cfg.patch_size),
    }
    train_losses: dict[str, float] = {}

    def probe_source(
        name: str, select: Callable[[PointTable], ProbeData]
    ) -> None:
        print(f"Training probe {name}...")
        probe, loss = fit_probe(select(train), cfg, device)
        train_losses[name] = loss * cfg.patch_size
        test_data = select(test)
        errors[name] = error_px(predict(probe, test_data, device), test_data.target, cfg.patch_size)

    probe_source(
        f"matched@{depth}",
        lambda t: ProbeData(
            feats=t.matched_feats,
            cond=condition(t, dataset, num_shapes, with_slot=True),
            target=t.point - t.matched_center,
        ),
    )
    def local_tap(k: int) -> Callable[[PointTable], ProbeData]:
        def select(t: PointTable) -> ProbeData:
            return ProbeData(
                feats=t.local_feats[:, k],
                cond=condition(t, dataset, num_shapes, with_slot=False),
                target=t.point - t.local_center,
            )

        return select

    for k, tap in sorted(enumerate(cfg.taps), key=lambda kt: -kt[1]):
        probe_source(f"local@{tap}", local_tap(k))
    errors["center"] = error_px(test.local_center, test.point, cfg.patch_size)

    report: dict[str, dict[str, ErrorStats]] = {}
    for name, mask in evaluation_groups(test, dataset, cfg.patch_size):
        if int(mask.sum()) < 20:
            continue
        report[name] = {source: stats(err[mask]) for source, err in errors.items()}

    sources = list(errors)
    header = f"{'group (median / p90 px)':<36}{'n':>8}" + "".join(f"{s:>17}" for s in sources)
    print(header)
    for name, by_source in report.items():
        row = f"{name:<36}{by_source[sources[0]].n:>8}"
        row += "".join(
            f"{by_source[s].median_px:>8.1f} /{by_source[s].p90_px:>6.1f}" for s in sources
        )
        print(row)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "points": {
            "total": total,
            "common": int(common.sum()),
            "train": int(train.point.shape[0]),
            "held_out": int(test.point.shape[0]),
        },
        "probe_train_l1_px": train_losses,
        "groups": {
            name: {source: asdict(s) for source, s in by_source.items()}
            for name, by_source in report.items()
        },
    }
    out = cfg.out_dir / "token_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/029_size_difficulty_balanced_loss/train_detection/checkpoints/latest_model.pt"
        ),
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument("--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments/029_size_difficulty_balanced_loss/analysis_005_token_probe"),
    )
    parser.add_argument("--backbone_size", choices=["nano", "small", "base"], default="nano")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_pages", type=int, default=None)
    parser.add_argument("--max_points_per_page", type=int, default=2000)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument(
        "--taps",
        type=int,
        nargs="+",
        default=[0, 6, 12],
        help="Backbone depths to probe for the local token (0 = patch embedding, "
        "depth = final output).",
    )
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe_hidden", type=int, default=512)
    parser.add_argument("--probe_steps", type=int, default=4000)
    parser.add_argument("--probe_batch", type=int, default=4096)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
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
        max_points_per_page=int(args.max_points_per_page),
        score_threshold=float(args.score_threshold),
        taps=tuple(int(t) for t in args.taps),
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
        probe_hidden=int(args.probe_hidden),
        probe_steps=int(args.probe_steps),
        probe_batch=int(args.probe_batch),
        probe_lr=float(args.probe_lr),
    )


if __name__ == "__main__":
    run(parse_args())
