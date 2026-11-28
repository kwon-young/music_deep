"""Analysis 011 (patch-set Phase 1) cache: every occupied slot of a frozen detector, both heads.

For each page, the slots whose best class probability is >= ``--floor`` are stored with their
class logits, geometry, the standard deviation of each coordinate (from the spread of its D-FINE
edge distribution) and, for lines, the refined endpoints and their own spread (Exp 035 refiner).
The association experiments of ``scripts/diagnose_association.py`` read this cache, so they run
without the detector. One file per page: ``<out_dir>/<image_id>.pt``.

Usage:
    PYTHONPATH=src:scripts mamba run -n pytorch python scripts/build_slot_cache.py \
        --checkpoint experiments/037_029_recipe_from_023/train_detection/checkpoints/latest_model.pt \
        --refiner experiments/037_029_recipe_from_023/refiner_phase_a/refiner.pt \
        --out_dir experiments/037_029_recipe_from_023/slot_cache
"""

import argparse
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

import transform.det as det_tf
from dataset.coco import CocoDataset, load_coco_sample, parse_coco
from dataset.split import piece_name
from model.detector import OMRDetector, create_detector, head_layout
from model.refiner import EndpointRefiner, LineQueries, TokenGrid, distribution_std, load_refiner
from music_types import Patches
from threaded_generator import ThreadedGenerator


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    refiner: Path
    anno_path: Path
    img_dir: Path
    cache_dir: Path
    out_dir: Path
    backbone_size: str
    patch_size: int
    channels: int
    num_shapes: int
    base_anchor_size: float
    var_threshold: float
    floor: float
    presence_floor: float
    use_sdpa: bool
    max_pages: int | None


@dataclass(frozen=True)
class PageInput:
    image_id: int
    piece: str
    width: int
    height: int
    patches: Patches


@dataclass(frozen=True)
class SlotSet:
    """Occupied slots of one head of one page; geometry in patch units (x1, y1, x2, y2)."""

    query: torch.Tensor
    logits: torch.Tensor
    geometry: torch.Tensor
    std: torch.Tensor
    presence: torch.Tensor | None


def load_model(cfg: Config, dataset: CocoDataset, device: torch.device) -> tuple[OMRDetector, Config]:
    """The detector, with its slot count and presence read from the checkpoint (``cfg`` is
    returned with that ``num_shapes``)."""
    checkpoint: dict[str, object] = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    state = checkpoint["model"]
    assert isinstance(state, dict)
    num_shapes, presence = head_layout(state, dataset.num_symbol_classes)
    cfg = replace(cfg, num_shapes=num_shapes)
    model = create_detector(
        backbone_size=cfg.backbone_size,
        patch_size=cfg.patch_size,
        channels=cfg.channels,
        use_sdpa=cfg.use_sdpa,
        num_symbol_classes=dataset.num_symbol_classes,
        num_line_classes=dataset.num_line_classes,
        num_shapes=cfg.num_shapes,
        base_anchor_size=cfg.base_anchor_size,
        presence=presence,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def page_inputs(dataset: CocoDataset, cfg: Config, indices: Sequence[int]) -> Iterator[PageInput]:
    """Same preprocessing as the Exp 035 Phase A cache: pyvips, padding, variance drop."""
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
            image_id=meta.img_id,
            piece=piece_name(meta.file_name),
            width=meta.width,
            height=meta.height,
            patches=dropped.sample.image,
        )


def occupied(logits: torch.Tensor, presence: torch.Tensor | None, floor: float, presence_floor: float) -> torch.Tensor:
    """Slots whose best class probability reaches ``floor`` or whose presence reaches
    ``presence_floor``."""
    keep = logits.float().sigmoid().amax(dim=-1) >= floor
    if presence is not None:
        keep = keep | (presence.float().sigmoid() >= presence_floor)
    return keep.nonzero().squeeze(1)


def symbol_slots(model: OMRDetector, logits: torch.Tensor, boxes: torch.Tensor,
                 edge_logits: torch.Tensor, shapes: torch.Tensor, presence: torch.Tensor | None,
                 floor: float, presence_floor: float) -> SlotSet:
    """Box edges are ``x1 = cx - (w/2 + r_L w)`` etc., so the std of x1 is std(r_L) * w."""
    keep = occupied(logits, presence, floor, presence_floor)
    spread = distribution_std(edge_logits[keep], model.symbol_head.weighting_fn.w)
    w, h = shapes[keep, 0:1], shapes[keep, 1:2]
    return SlotSet(
        query=keep.int().cpu(),
        logits=logits[keep].half().cpu(),
        geometry=boxes[keep].float().cpu(),
        std=(spread * torch.cat([w, h, w, h], dim=-1)).half().cpu(),
        presence=None if presence is None else presence[keep].half().cpu(),
    )


def line_slots(model: OMRDetector, logits: torch.Tensor, keypoints: torch.Tensor,
               edge_logits: torch.Tensor, base_dirs: torch.Tensor, base_anchor_size: float,
               presence: torch.Tensor | None, floor: float, presence_floor: float) -> SlotSet:
    """Endpoint coordinates are ``base + r * (|base| + anchor)``, so their std is std(r) * scale."""
    keep = occupied(logits, presence, floor, presence_floor)
    spread = distribution_std(edge_logits[keep], model.line_head.weighting_fn.w)
    scale = base_dirs[keep].abs() + base_anchor_size
    return SlotSet(
        query=keep.int().cpu(),
        logits=logits[keep].half().cpu(),
        geometry=keypoints[keep].float().cpu(),
        std=(spread * scale).half().cpu(),
        presence=None if presence is None else presence[keep].half().cpu(),
    )


def refine(refiner: EndpointRefiner, grid: TokenGrid, lines: SlotSet, num_shapes: int,
           device: torch.device, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Refined endpoints (N, 4) and the std of each refined coordinate (N, 4)."""
    endpoints: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    labels = lines.logits.float().argmax(dim=-1)
    for start in range(0, lines.query.shape[0], chunk):
        query = lines.query[start : start + chunk].long().to(device)
        part = LineQueries(
            image=torch.zeros_like(query),
            source=query // num_shapes,
            label=labels[start : start + chunk].to(device),
            endpoints=lines.geometry[start : start + chunk].to(device),
        )
        last = refiner(grid, part)[-1]
        endpoints.append(last.endpoints.float().cpu())
        stds.append((distribution_std(last.bin_logits, refiner.weighting.w) * refiner.scale).half().cpu())
    if not endpoints:
        return lines.geometry.clone(), lines.std.clone()
    return torch.cat(endpoints), torch.cat(stds)


def run(cfg: Config) -> None:
    device = torch.device("cuda")
    dataset = parse_coco(cfg.anno_path, cache_dir=cfg.cache_dir)
    model, cfg = load_model(cfg, dataset, device)
    refiner = load_refiner(cfg.refiner, model.backbone.patch_embed[1].out_features,
                           dataset.num_line_classes, device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    indices = [i for i, m in enumerate(dataset.images) if not (cfg.out_dir / f"{m.img_id}.pt").exists()]
    if cfg.max_pages is not None:
        indices = indices[: cfg.max_pages]
    final: list[torch.Tensor] = []

    def keep_final(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        final.append(output)

    handle = model.backbone.transformer.norm.register_forward_hook(keep_final)
    t0 = time.monotonic()
    loader = ThreadedGenerator(page_inputs(dataset, cfg, indices), maxsize=2, name="pages")
    with torch.no_grad(), loader as pages:
        for n, page in enumerate(pages, start=1):
            patches = page.patches
            on_device = replace(patches, data=patches.data.to(device), indices=patches.indices.to(device))
            final.clear()
            out = model.forward(on_device)
            sym = out.symbols
            symbols = symbol_slots(
                model, sym.pred_logits.data[0], sym.pred_boxes.data[0],
                sym.pred_edge_logits.data[0], sym.learnable_shapes.data[0],
                None if sym.pred_presence is None else sym.pred_presence.data[0], cfg.floor, cfg.presence_floor,
            )
            ln = out.lines
            lines = line_slots(
                model, ln.pred_logits.data[0], ln.pred_keypoints.data[0],
                ln.pred_endpoint_logits.data[0], ln.raw_directions.data[0],
                cfg.base_anchor_size, None if ln.pred_presence is None else ln.pred_presence.data[0],
                cfg.floor, cfg.presence_floor,
            )
            _, height, width = patches.image_shape
            grid_h, grid_w = height // patches.patch_size[0], width // patches.patch_size[1]
            grid = TokenGrid.from_batch(final[0].float(), on_device.indices, grid_h, grid_w)
            refined, refined_std = refine(refiner, grid, lines, cfg.num_shapes, device)
            torch.save(
                {
                    "image_id": page.image_id,
                    "piece": page.piece,
                    "width": page.width,
                    "height": page.height,
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "num_shapes": cfg.num_shapes,
                    "cells": patches.indices[0].int().cpu(),
                    "symbols": asdict(symbols),
                    "lines": asdict(lines) | {"refined": refined, "refined_std": refined_std},
                },
                cfg.out_dir / f"{page.image_id}.pt",
            )
            if n % 10 == 0 or n == len(indices):
                rate = (time.monotonic() - t0) / n
                print(f"{n}/{len(indices)} pages, {rate:.1f} s/page, last: "
                      f"{symbols.query.shape[0]} symbol / {lines.query.shape[0]} line slots", flush=True)
    handle.remove()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--backbone_size", choices=["nano", "small", "base"], default="nano")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--floor", type=float, default=0.01)
    parser.add_argument(
        "--presence_floor", type=float, default=0.1, help="Also keep slots with presence >= this."
    )
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--max_pages", type=int, default=None)
    args = parser.parse_args()
    return Config(
        checkpoint=Path(args.checkpoint),
        refiner=Path(args.refiner),
        anno_path=Path(args.anno_path),
        img_dir=Path(args.img_dir),
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        backbone_size=str(args.backbone_size),
        patch_size=int(args.patch_size),
        channels=int(args.channels),
        num_shapes=int(args.num_shapes),
        base_anchor_size=float(args.base_anchor_size),
        var_threshold=float(args.var_threshold),
        floor=float(args.floor),
        presence_floor=float(args.presence_floor),
        use_sdpa=bool(args.use_sdpa),
        max_pages=None if args.max_pages is None else int(args.max_pages),
    )


if __name__ == "__main__":
    run(parse_args())
