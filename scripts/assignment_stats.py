"""Analysis 009: ground-truth statistics for the patch-set assignment (PATCH_SET_ASSIGNMENT.md,
Phase 0). No model is involved.

For each extraction size, responsibility margin m and slot count K, on the kept patches of every
trainval page (same decode, padding, grid and variance drop as training):

* how many GT objects each patch is responsible for (overflow when more than K);
* how many patches vote for each GT after overflow (0 = structurally missed);
* which same-class pairs of nearby GT (crowded pairs) share at least one patch where both hold
  a slot, i.e. are kept apart by the cannot-link rule.

Usage:
    PYTHONPATH=src mamba run -n pytorch python scripts/assignment_stats.py \
        --out_dir experiments/analysis_009_patch_set_assignment
    # re-print the report from the saved counts:
    PYTHONPATH=src mamba run -n pytorch python scripts/assignment_stats.py \
        --out_dir experiments/analysis_009_patch_set_assignment --report_only
"""

import argparse
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from dataset.coco import CocoDataset, CocoLineAnnotation, parse_coco
from music_types import LazyImage
from transform.core import (
    decode_pyvips_img,
    extract_patches_img,
    pad_to_patch_size_img,
    stack_tensor_img,
    to_float1_img,
    variance_patch_drop_indices,
)

SIZES = (54, 64, 74)
MARGINS = (0.0, 0.25, 0.5)
SLOTS: tuple[int | None, ...] = (5, 8, None)
CROWD_PX = (8.0, 16.0, 32.0, 64.0, 128.0)
CLASS_CROWD = CROWD_PX.index(64.0)
VOTER_BINS = 257
COUNT_BINS = 65
CHUNK = 512

type Modality = Literal["symbols", "lines"]
MODALITIES: tuple[Modality, ...] = ("symbols", "lines")
type JsonValue = int | float | str | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True)
class ConfigKey:
    modality: Modality
    size: int
    margin: float
    slots: int | None

    def name(self) -> str:
        k = "inf" if self.slots is None else str(self.slots)
        return f"{self.modality}/s{self.size}/m{self.margin}/k{k}"


@dataclass(frozen=True)
class PatchKey:
    modality: Modality
    size: int
    margin: float

    def name(self) -> str:
        return f"{self.modality}/s{self.size}/m{self.margin}"


@dataclass
class ConfigCounts:
    """Per class: voter histogram (clipped), GT overflowed in at least one patch, and per crowd
    distance the crowded pairs kept apart by some shared patch."""

    voters: torch.Tensor
    overflowed: torch.Tensor
    protected: torch.Tensor

    @staticmethod
    def zeros(num_classes: int) -> "ConfigCounts":
        return ConfigCounts(
            voters=torch.zeros(num_classes, VOTER_BINS, dtype=torch.int64),
            overflowed=torch.zeros(num_classes, dtype=torch.int64),
            protected=torch.zeros(num_classes, len(CROWD_PX), dtype=torch.int64),
        )

    def add(self, other: "ConfigCounts") -> None:
        self.voters += other.voters
        self.overflowed += other.overflowed
        self.protected += other.protected


@dataclass
class ModalityCounts:
    """Per class: GT count, crowded pairs per crowd distance, near-identical GT pairs (< 1 px)."""

    gt: torch.Tensor
    pairs: torch.Tensor
    near_identical: torch.Tensor

    @staticmethod
    def zeros(num_classes: int) -> "ModalityCounts":
        return ModalityCounts(
            gt=torch.zeros(num_classes, dtype=torch.int64),
            pairs=torch.zeros(num_classes, len(CROWD_PX), dtype=torch.int64),
            near_identical=torch.zeros(num_classes, dtype=torch.int64),
        )

    def add(self, other: "ModalityCounts") -> None:
        self.gt += other.gt
        self.pairs += other.pairs
        self.near_identical += other.near_identical


@dataclass
class Counts:
    configs: dict[ConfigKey, ConfigCounts]
    patch_load: dict[PatchKey, torch.Tensor]
    kept_patches: dict[int, int]
    modalities: dict[Modality, ModalityCounts]
    pages: int

    def add(self, other: "Counts") -> None:
        for key, value in other.configs.items():
            self.configs[key].add(value)
        for pkey, hist in other.patch_load.items():
            self.patch_load[pkey] += hist
        for size, kept in other.kept_patches.items():
            self.kept_patches[size] += kept
        for modality, mcounts in other.modalities.items():
            self.modalities[modality].add(mcounts)
        self.pages += other.pages


def empty_counts(num_classes: dict[Modality, int]) -> Counts:
    return Counts(
        configs={
            ConfigKey(mod, size, margin, slots): ConfigCounts.zeros(num_classes[mod])
            for mod in MODALITIES
            for size in SIZES
            for margin in MARGINS
            for slots in SLOTS
        },
        patch_load={
            PatchKey(mod, size, margin): torch.zeros(COUNT_BINS, dtype=torch.int64)
            for mod in MODALITIES
            for size in SIZES
            for margin in MARGINS
        },
        kept_patches={size: 0 for size in SIZES},
        modalities={mod: ModalityCounts.zeros(num_classes[mod]) for mod in MODALITIES},
        pages=0,
    )


@dataclass(frozen=True)
class PageObjects:
    """GT of one page in pixels: (N, 4) xyxy boxes or x1y1x2y2 segments, and (N,) labels."""

    geometry: torch.Tensor
    labels: torch.Tensor


def page_objects(dataset: CocoDataset, index: int) -> dict[Modality, PageObjects]:
    boxes: list[list[float]] = []
    box_labels: list[int] = []
    lines: list[list[float]] = []
    line_labels: list[int] = []
    for ann in dataset.annotations.get(dataset.images[index].img_id, []):
        if isinstance(ann, CocoLineAnnotation):
            lines.append(ann.keypoints[:4])
            line_labels.append(dataset.line_cat_id_to_idx[ann.category_id])
        else:
            x, y, w, h = ann.bbox
            boxes.append([x, y, x + w, y + h])
            box_labels.append(dataset.symbol_cat_id_to_idx[ann.category_id])
    return {
        "symbols": PageObjects(
            torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4),
            torch.tensor(box_labels, dtype=torch.int64),
        ),
        "lines": PageObjects(
            torch.tensor(lines, dtype=torch.float64).reshape(-1, 4),
            torch.tensor(line_labels, dtype=torch.int64),
        ),
    }


def kept_cell_centers(
    dataset: CocoDataset, img_dir: Path, index: int, var_threshold: float
) -> dict[int, torch.Tensor]:
    """(P, 2) centres, in patch units, of the patches kept by the training variance drop."""
    meta = dataset.images[index]
    image = to_float1_img(
        decode_pyvips_img(
            LazyImage(img_dir / meta.file_name, meta.width, meta.height), torch.device("cpu")
        )
    )
    centers: dict[int, torch.Tensor] = {}
    for size in SIZES:
        padded = pad_to_patch_size_img(image, (size, size))
        patches = extract_patches_img(stack_tensor_img([padded]), (size, size))
        kept = variance_patch_drop_indices(patches.data, var_threshold=var_threshold)[0]
        grid_w = patches.image_shape[2] // size
        centers[size] = torch.stack([kept % grid_w, kept // grid_w], dim=-1).double() + 0.5
    return centers


def point_box_distance(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """(P, N) distance from points (P, 2) to boxes (N, 4)."""
    px, py = points[:, 0:1], points[:, 1:2]
    dx = (boxes[None, :, 0] - px).maximum(px - boxes[None, :, 2]).clamp(min=0)
    dy = (boxes[None, :, 1] - py).maximum(py - boxes[None, :, 3]).clamp(min=0)
    return torch.hypot(dx, dy)


def point_segment_distance(points: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """(P, N) distance from points (P, 2) to segments (N, 4)."""
    a = segments[:, :2]
    ab = segments[:, 2:] - a
    ap = points[:, None, :] - a[None]
    t = ((ap * ab[None]).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)[None]).clamp(0, 1)
    return (ap - t[..., None] * ab[None]).norm(dim=-1)


def cell_box_distance(centers: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """(P, N) distance between unit cells centred on ``centers`` and boxes."""
    cx, cy = centers[:, 0:1], centers[:, 1:2]
    dx = (boxes[None, :, 0] - (cx + 0.5)).maximum((cx - 0.5) - boxes[None, :, 2]).clamp(min=0)
    dy = (boxes[None, :, 1] - (cy + 0.5)).maximum((cy - 0.5) - boxes[None, :, 3]).clamp(min=0)
    return torch.hypot(dx, dy)


def cell_segment_distance(centers: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """(P, N) distance between unit cells and segments: 0 when the segment crosses the cell
    (Liang-Barsky clipping), otherwise the closest of endpoint-to-cell and corner-to-segment."""
    cells = torch.cat([centers - 0.5, centers + 0.5], dim=-1)
    endpoint = torch.minimum(
        point_box_distance(segments[:, :2], cells), point_box_distance(segments[:, 2:], cells)
    ).T
    corner = torch.stack(
        [
            point_segment_distance(centers + torch.tensor(offset, dtype=centers.dtype), segments)
            for offset in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))
        ]
    ).amin(dim=0)
    start = segments[:, :2]
    delta = segments[:, 2:] - start
    enter = torch.zeros(endpoint.shape, dtype=centers.dtype)
    leave = torch.ones(endpoint.shape, dtype=centers.dtype)
    for axis in range(2):
        a = start[None, :, axis]
        d = delta[None, :, axis]
        lo = cells[:, axis : axis + 1]
        hi = cells[:, axis + 2 : axis + 3]
        parallel = d.abs() < 1e-12
        safe = torch.where(parallel, torch.ones_like(d), d)
        t1 = (lo - a) / safe
        t2 = (hi - a) / safe
        inside = (a >= lo) & (a <= hi)
        never = parallel & ~inside
        enter = torch.where(parallel, enter, enter.maximum(t1.minimum(t2)))
        leave = torch.where(parallel, leave, leave.minimum(t1.maximum(t2)))
        leave = torch.where(never, torch.full_like(leave, -1.0), leave)
    crosses = enter <= leave
    return torch.where(crosses, torch.zeros_like(endpoint), endpoint.minimum(corner))


@dataclass(frozen=True)
class Pairs:
    """Responsibility candidates (patch, GT) within the largest margin, with the cell distance
    and the patch-centre distance (overflow tie-break)."""

    patch: torch.Tensor
    gt: torch.Tensor
    cell_distance: torch.Tensor
    center_distance: torch.Tensor


def candidate_pairs(centers: torch.Tensor, geometry: torch.Tensor, modality: Modality) -> Pairs:
    max_margin = max(MARGINS)
    patch: list[torch.Tensor] = []
    gt: list[torch.Tensor] = []
    cell: list[torch.Tensor] = []
    center: list[torch.Tensor] = []
    for start in range(0, geometry.shape[0], CHUNK):
        part = geometry[start : start + CHUNK]
        if modality == "symbols":
            dist = cell_box_distance(centers, part)
        else:
            dist = cell_segment_distance(centers, part)
        p, g = (dist <= max_margin + 1e-9).nonzero(as_tuple=True)
        if modality == "symbols":
            cdist = point_box_distance(centers[p], part)[torch.arange(p.shape[0]), g]
        else:
            cdist = point_segment_distance(centers[p], part)[torch.arange(p.shape[0]), g]
        patch.append(p)
        gt.append(g + start)
        cell.append(dist[p, g])
        center.append(cdist)
    return Pairs(torch.cat(patch), torch.cat(gt), torch.cat(cell), torch.cat(center))


@dataclass(frozen=True)
class Assignment:
    """Kept (patch, GT) pairs after overflow, with each pair's slot rank in its patch."""

    patch: torch.Tensor
    gt: torch.Tensor
    rank: torch.Tensor
    overflowed_gt: torch.Tensor


def assign(
    pairs: Pairs, margin: float, slots: int | None, num_patches: int, num_gt: int
) -> Assignment:
    """Overflow rule of the plan: a patch keeps the ``slots`` GT with the fewest expected
    voters (ties: nearest to the patch centre); the others are ignored in that patch."""
    inside = pairs.cell_distance <= margin + 1e-9
    patch, gt, center = pairs.patch[inside], pairs.gt[inside], pairs.center_distance[inside]
    expected = torch.bincount(gt, minlength=num_gt)
    order = torch.argsort(center, stable=True)
    order = order[torch.argsort(expected[gt[order]], stable=True)]
    order = order[torch.argsort(patch[order], stable=True)]
    patch, gt = patch[order], gt[order]
    load = torch.bincount(patch, minlength=num_patches)
    rank = torch.arange(patch.shape[0]) - (torch.cumsum(load, 0) - load)[patch]
    keep = rank < (slots if slots is not None else patch.shape[0] + 1)
    overflowed = torch.zeros(num_gt, dtype=torch.bool)
    overflowed[gt[~keep]] = True
    return Assignment(patch[keep], gt[keep], rank[keep], overflowed)


def shared_slot_keys(assignment: Assignment, num_patches: int, num_gt: int) -> torch.Tensor:
    """Sorted unique keys a * num_gt + b (a < b) of GT pairs holding slots in the same patch."""
    if assignment.patch.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64)
    width = int(assignment.rank.max().item()) + 1
    table = torch.full((num_patches, width), -1, dtype=torch.int64)
    table[assignment.patch, assignment.rank] = assignment.gt
    table = table[(table >= 0).sum(dim=1) >= 2]
    i, j = torch.triu_indices(width, width, offset=1)
    a, b = table[:, i], table[:, j]
    valid = (a >= 0) & (b >= 0)
    lo, hi = a.minimum(b)[valid], a.maximum(b)[valid]
    return torch.unique(lo * num_gt + hi)


@dataclass(frozen=True)
class CrowdedPairs:
    """Same-class GT pairs (a < b) whose matching endpoints / corners are both within the
    largest crowd distance, with that distance in pixels."""

    a: torch.Tensor
    b: torch.Tensor
    distance: torch.Tensor


def crowded_pairs(objects: PageObjects, modality: Modality) -> CrowdedPairs:
    a_list: list[torch.Tensor] = []
    b_list: list[torch.Tensor] = []
    d_list: list[torch.Tensor] = []
    for label in objects.labels.unique().tolist():
        index = (objects.labels == label).nonzero().squeeze(1)
        geom = objects.geometry[index]
        first, second = geom[:, :2], geom[:, 2:]
        dist = torch.cdist(first, first).maximum(torch.cdist(second, second))
        if modality == "lines":
            dist = dist.minimum(torch.cdist(first, second).maximum(torch.cdist(second, first)))
        i, j = torch.triu_indices(index.shape[0], index.shape[0], offset=1)
        close = dist[i, j] <= max(CROWD_PX)
        a_list.append(index[i[close]])
        b_list.append(index[j[close]])
        d_list.append(dist[i[close], j[close]])
    if not a_list:
        empty = torch.empty(0, dtype=torch.int64)
        return CrowdedPairs(empty, empty, torch.empty(0, dtype=torch.float64))
    return CrowdedPairs(torch.cat(a_list), torch.cat(b_list), torch.cat(d_list))


def page_counts(
    dataset: CocoDataset,
    img_dir: Path,
    index: int,
    var_threshold: float,
    num_classes: dict[Modality, int],
) -> Counts:
    counts = empty_counts(num_classes)
    counts.pages = 1
    centers = kept_cell_centers(dataset, img_dir, index, var_threshold)
    for size in SIZES:
        counts.kept_patches[size] = centers[size].shape[0]
    for modality, objects in page_objects(dataset, index).items():
        num_gt = objects.labels.shape[0]
        classes = num_classes[modality]
        mcounts = counts.modalities[modality]
        mcounts.gt += torch.bincount(objects.labels, minlength=classes)
        if num_gt == 0:
            continue
        crowd = crowded_pairs(objects, modality)
        crowd_label = objects.labels[crowd.a]
        crowd_keys = crowd.a * num_gt + crowd.b
        for d, px in enumerate(CROWD_PX):
            mcounts.pairs[:, d] += torch.bincount(
                crowd_label[crowd.distance <= px], minlength=classes
            )
        mcounts.near_identical += torch.bincount(
            crowd_label[crowd.distance < 1.0], minlength=classes
        )
        for size in SIZES:
            cells = centers[size]
            pairs = candidate_pairs(cells, objects.geometry / size, modality)
            for margin in MARGINS:
                load = torch.bincount(
                    pairs.patch[pairs.cell_distance <= margin + 1e-9], minlength=cells.shape[0]
                )
                counts.patch_load[PatchKey(modality, size, margin)] += torch.bincount(
                    load.clamp(max=COUNT_BINS - 1), minlength=COUNT_BINS
                )
                for slots in SLOTS:
                    assignment = assign(pairs, margin, slots, cells.shape[0], num_gt)
                    config = counts.configs[ConfigKey(modality, size, margin, slots)]
                    voters = torch.bincount(assignment.gt, minlength=num_gt)
                    config.voters += torch.bincount(
                        objects.labels * VOTER_BINS + voters.clamp(max=VOTER_BINS - 1),
                        minlength=classes * VOTER_BINS,
                    ).view(classes, VOTER_BINS)
                    config.overflowed += torch.bincount(
                        objects.labels[assignment.overflowed_gt], minlength=classes
                    )
                    shared = torch.isin(
                        crowd_keys, shared_slot_keys(assignment, cells.shape[0], num_gt)
                    )
                    for d, px in enumerate(CROWD_PX):
                        config.protected[:, d] += torch.bincount(
                            crowd_label[shared & (crowd.distance <= px)], minlength=classes
                        )
    return counts


def collect(
    dataset: CocoDataset, img_dir: Path, var_threshold: float, workers: int
) -> Counts:
    num_classes: dict[Modality, int] = {
        "symbols": dataset.num_symbol_classes,
        "lines": dataset.num_line_classes,
    }
    total = empty_counts(num_classes)
    start = time.time()

    def one(index: int) -> Counts:
        return page_counts(dataset, img_dir, index, var_threshold, num_classes)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results: Iterator[Counts] = pool.map(one, range(len(dataset.images)))
        for done, page in enumerate(results, start=1):
            total.add(page)
            if done % 25 == 0 or done == len(dataset.images):
                print(f"{done}/{len(dataset.images)} pages, {time.time() - start:.0f} s", flush=True)
    return total


def to_json(counts: Counts, names: dict[Modality, list[str]]) -> dict[str, JsonValue]:
    def rows(t: torch.Tensor) -> JsonValue:
        values: list[JsonValue] = [int(v) for v in t.flatten().tolist()]
        return values if t.dim() == 1 else [rows(r) for r in t]

    return {
        "sizes": [int(s) for s in SIZES],
        "margins": [float(m) for m in MARGINS],
        "slots": ["inf" if k is None else int(k) for k in SLOTS],
        "crowd_px": [float(d) for d in CROWD_PX],
        "pages": counts.pages,
        "names": {mod: [str(n) for n in names[mod]] for mod in MODALITIES},
        "kept_patches": {str(s): counts.kept_patches[s] for s in SIZES},
        "modalities": {
            mod: {
                "gt": rows(m.gt),
                "pairs": rows(m.pairs),
                "near_identical": rows(m.near_identical),
            }
            for mod, m in counts.modalities.items()
        },
        "patch_load": {key.name(): rows(hist) for key, hist in counts.patch_load.items()},
        "configs": {
            key.name(): {
                "voters": rows(c.voters),
                "overflowed": rows(c.overflowed),
                "protected": rows(c.protected),
            }
            for key, c in counts.configs.items()
        },
    }


def tensor(value: JsonValue) -> torch.Tensor:
    assert isinstance(value, list)
    return torch.tensor(value, dtype=torch.int64)


def field(value: JsonValue, key: str) -> JsonValue:
    assert isinstance(value, dict)
    return value[key]


def quantile(hist: torch.Tensor, q: float) -> int:
    """Smallest value v such that at least a fraction q of the histogram is <= v."""
    total = int(hist.sum().item())
    if total == 0:
        return 0
    cumulative = torch.cumsum(hist, 0)
    return int((cumulative >= q * total).nonzero()[0].item())


def pct(num: int | float, den: int | float) -> str:
    return f"{100.0 * num / den:6.2f}%" if den else "     - "


def report(data: dict[str, JsonValue]) -> str:
    lines: list[str] = []
    pages = data["pages"]
    assert isinstance(pages, int)
    kept = field(data, "kept_patches")
    lines.append(f"Pages: {pages}")
    for size in SIZES:
        n = field(kept, str(size))
        assert isinstance(n, int)
        lines.append(f"  extraction {size} px: {n / pages:.0f} kept patches per page")
    for modality in MODALITIES:
        mdata = field(field(data, "modalities"), modality)
        gt = tensor(field(mdata, "gt"))
        pairs = tensor(field(mdata, "pairs"))
        near = tensor(field(mdata, "near_identical"))
        names = field(field(data, "names"), modality)
        assert isinstance(names, list)
        total_gt = int(gt.sum().item())
        lines.append("")
        lines.append(f"=== {modality}: {total_gt} GT ===")
        lines.append(
            "crowded same-class pairs (both endpoints/corners within d): "
            + ", ".join(f"{d:.0f} px: {int(pairs[:, i].sum())}" for i, d in enumerate(CROWD_PX))
            + f"; near-identical (< 1 px): {int(near.sum())}"
        )
        lines.append("")
        lines.append(
            "size  m     K   | patch load p50 p90 p99 max  >K patches | GT: no voter (of which"
            " no kept patch)  1 voter  overflowed  voters p10 p50 | crowd kept apart "
            + " / ".join(f"{d:.0f}" for d in CROWD_PX)
            + " px"
        )
        for size in SIZES:
            for margin in MARGINS:
                load = tensor(field(field(data, "patch_load"), PatchKey(modality, size, margin).name()))
                base = tensor(
                    field(field(field(data, "configs"), ConfigKey(modality, size, margin, None).name()), "voters")
                ).sum(0)
                for slots in SLOTS:
                    config = field(field(data, "configs"), ConfigKey(modality, size, margin, slots).name())
                    voters = tensor(field(config, "voters")).sum(0)
                    overflowed = int(tensor(field(config, "overflowed")).sum().item())
                    protected = tensor(field(config, "protected")).sum(0)
                    over = int(load[slots + 1 :].sum().item()) if slots is not None else 0
                    k = "inf" if slots is None else str(slots)
                    lines.append(
                        f"{size:4d}  {margin:<4}  {k:>3} | "
                        f"{quantile(load, 0.5):14d} {quantile(load, 0.9):3d} {quantile(load, 0.99):3d} "
                        f"{len(load) - 1 if load[-1] > 0 else int(load.nonzero().max()):3d}  "
                        f"{pct(over, int(load.sum()))} | "
                        f"{pct(int(voters[0]), total_gt)} ({pct(int(base[0]), total_gt)})  "
                        f"{pct(int(voters[1]), total_gt)}  {pct(overflowed, total_gt)}  "
                        f"{quantile(voters, 0.1):10d} {quantile(voters, 0.5):3d} | "
                        + " / ".join(
                            pct(int(protected[i]), int(pairs[:, i].sum())) for i in range(len(CROWD_PX))
                        )
                    )
            lines.append("")
        for size, margin in ((64, 0.0), (64, 0.25), (64, 0.5)):
            for slots in (5, 8):
                config = field(field(data, "configs"), ConfigKey(modality, size, margin, slots).name())
                voters = tensor(field(config, "voters"))
                class_overflow = tensor(field(config, "overflowed"))
                protected = tensor(field(config, "protected"))
                lines.append(f"--- {modality} per class, extraction {size} px, m = {margin}, K = {slots} ---")
                lines.append(
                    f"{'class':<28} {'GT':>7} {'no voter':>9} {'1 voter':>8} {'overflow':>9} "
                    f"{'p10':>4} {'p50':>4} {'pairs' + str(int(CROWD_PX[CLASS_CROWD])):>8} {'kept apart':>10}"
                )
                order = torch.argsort(gt, descending=True)
                for c in order.tolist():
                    n = int(gt[c])
                    if n == 0:
                        continue
                    zero = int(voters[c, 0])
                    unprotected = int(pairs[c, CLASS_CROWD] - protected[c, CLASS_CROWD])
                    if modality == "symbols" and n < 1000 and zero == 0 and unprotected == 0:
                        continue
                    lines.append(
                        f"{str(names[c]):<28} {n:7d} {pct(zero, n):>9} {pct(int(voters[c, 1]), n):>8} "
                        f"{pct(int(class_overflow[c]), n):>9} {quantile(voters[c], 0.1):4d} "
                        f"{quantile(voters[c], 0.5):4d} {int(pairs[c, CLASS_CROWD]):8d} "
                        f"{pct(int(protected[c, CLASS_CROWD]), int(pairs[c, CLASS_CROWD])):>10}"
                    )
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anno_path", type=Path, default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--report_only", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    counts_path = out_dir / "assignment_stats.json"
    if args.report_only:
        loaded = json.loads(counts_path.read_text())
        assert isinstance(loaded, dict)
        data: dict[str, JsonValue] = loaded
    else:
        dataset = parse_coco(args.anno_path, args.cache_dir)
        if args.max_pages is not None:
            dataset.images = dataset.images[: args.max_pages]
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // args.workers))
        counts = collect(dataset, args.img_dir, args.var_threshold, args.workers)
        names: dict[Modality, list[str]] = {
            "symbols": [str(c["name"]) for c in dataset.symbol_categories],
            "lines": [str(c["name"]) for c in dataset.line_categories],
        }
        data = to_json(counts, names)
        out_dir.mkdir(parents=True, exist_ok=True)
        counts_path.write_text(json.dumps(data))
    text = report(data)
    (out_dir / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
