"""Line association for patch-set voters: collinear chains, cut at endpoint-vote modes.

A line voter sees the part of the line around its patch: it places the line well across (its
perpendicular position) but its far endpoint is often a guess (Exp 042: 44% of refined voter
endpoints are > 0.5 patch off, with errors of 100-600 px). Endpoint distances therefore split
one line into many clusters, and averaging endpoints mixes good votes with guesses. Instead:

1. **Chains**: per class, voters that lie on each other's line (perpendicular distance of each
   one's anchor to the other's line <= ``perp_tol``, anchors within ``link`` patches) are
   connected; a connected component is a whole line of the page, e.g. a staff line across its
   barlines. The anchor is the point of the voter's segment nearest its own patch centre, the
   part it actually sees.
2. **Cuts**: all endpoints of a chain are projected on the chain's axis; the 1-D modes of the
   *certain* votes (refined spread <= ``certain_std``) are the cuts. Two abutting segments
   share one cut (a barline).
3. **Objects**: one per interval between consecutive cuts that holds voter anchors; its
   endpoints are the weighted means of the votes of its two cuts, so an endpoint is never
   averaged across voters that disagree. A chain with fewer than two cuts yields its most
   certain voter's segment.

Coordinates are in patch units. Each object carries the evidence (summed presence of its
voters), the best voter class score and the weight of its weaker cut (endpoint agreement).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class LineAssociationConfig:
    """Distances in patch units (64 px). ``min_cut_weight`` is in the units of the vote weights:
    presence / spread^2 (spread in patch units) with ``weighting`` "inverse_variance", presence
    with "presence"."""

    perp_tol: float = 5 / 64
    link: float = 3.0
    cut_radius: float = 4 / 64
    certain_std: float = 16 / 64
    min_cut_weight: float = 8.0
    weighting: Literal["inverse_variance", "presence"] = "inverse_variance"
    min_std: float = 1 / 64


@dataclass(frozen=True)
class LineObjects:
    """Associated lines: geometry (N, 4), label, evidence (summed member presence), quality
    (best member class score), cut_weight (weight of the weaker cut; the chosen voter's weight
    for a chain without cuts) and size (members)."""

    geometry: torch.Tensor
    label: torch.Tensor
    evidence: torch.Tensor
    quality: torch.Tensor
    cut_weight: torch.Tensor
    size: torch.Tensor


def anchors(segments: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """(N, 2) point of each segment nearest its patch centre, kept 0.25 patch inside the segment
    (or at its middle when shorter than 0.5 patch)."""
    a, b = segments[:, :2], segments[:, 2:]
    length = (b - a).norm(dim=-1).clamp(min=1e-6)
    u = (b - a) / length[:, None]
    t = ((centers - a) * u).sum(-1)
    margin = torch.minimum(torch.full_like(length, 0.25), length / 2)
    return a + u * torch.minimum(torch.maximum(t, margin), length - margin)[:, None]


def collinear_chains(segments: torch.Tensor, anchor: torch.Tensor, perp_tol: float, link: float,
                     chunk: int = 2048) -> np.ndarray:
    """(N,) connected-component index of "mutually on each other's line, anchors close"."""
    n = segments.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    d = segments[:, 2:] - segments[:, :2]
    u = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    normal = torch.stack([-u[:, 1], u[:, 0]], dim=-1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for start in range(0, n, chunk):
        rel = anchor[None, :, :] - anchor[start : start + chunk, None, :]
        ok = (
            (rel.norm(dim=-1) <= link)
            & ((rel * normal[start : start + chunk, None, :]).sum(-1).abs() <= perp_tol)
            & ((rel * normal[None, :, :]).sum(-1).abs() <= perp_tol)
        )
        r, c = ok.nonzero(as_tuple=True)
        rows.append((r + start).cpu().numpy())
        cols.append(c.cpu().numpy())
    r_all, c_all = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(r_all.shape[0], dtype=np.int8), (r_all, c_all)), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    return component.astype(np.int64)


def vote_modes(position: torch.Tensor, weight: torch.Tensor, radius: float, min_weight: float) -> list[float]:
    """Sorted 1-D modes: repeatedly take the densest window of +-``radius`` (by summed weight)
    among the votes not yet claimed, if it weighs at least ``min_weight``; the votes within
    2 x ``radius`` of a mode are claimed."""
    order = position.argsort()
    t, w = position[order], weight[order]
    cumulative = torch.cat([torch.zeros(1, dtype=w.dtype, device=w.device), w.cumsum(0)])
    lo = torch.searchsorted(t, t - radius, right=False)
    hi = torch.searchsorted(t, t + radius, right=True)
    density = cumulative[hi] - cumulative[lo]
    alive = torch.ones_like(t, dtype=torch.bool)
    modes: list[float] = []
    for k in density.argsort(descending=True).tolist():
        if not alive[k]:
            continue
        if density[k] < min_weight:
            break
        window = (t >= t[k] - radius) & (t <= t[k] + radius) & alive
        total = w[window].sum()
        if total < min_weight:
            continue
        mode = float((t[window] * w[window]).sum() / total)
        modes.append(mode)
        alive &= ~((t >= mode - 2 * radius) & (t <= mode + 2 * radius))
    return sorted(modes)


def chain_axis(anchor: torch.Tensor, direction: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Unit axis of a chain: principal direction of its anchors (the members' mean direction for
    a single or degenerate chain), oriented along the weighted majority of member directions."""
    centred = anchor - anchor.mean(dim=0)
    if anchor.shape[0] < 2 or float(centred.norm()) < 1e-3:
        axis = direction[0]
    else:
        axis = torch.linalg.svd(centred, full_matrices=False)[2][0]
    return -axis if float(((direction @ axis) * weight).sum()) < 0 else axis


def associate_lines(
    geometry: torch.Tensor,
    std: torch.Tensor,
    label: torch.Tensor,
    score: torch.Tensor,
    presence: torch.Tensor,
    centers: torch.Tensor,
    cfg: LineAssociationConfig,
) -> LineObjects:
    """``geometry`` / ``std`` (N, 4) refined endpoints and their spread, ``label`` / ``score``
    argmax class and its probability, ``presence`` (N,), ``centers`` (N, 2) the patch centre of
    each candidate; all in patch units."""
    out_geometry: list[torch.Tensor] = []
    out_label: list[int] = []
    out_evidence: list[float] = []
    out_quality: list[float] = []
    out_cut: list[float] = []
    out_size: list[int] = []
    g64, s64 = geometry.double(), std.double().clamp(min=cfg.min_std)
    p64, q64 = presence.double(), score.double()
    for c in torch.unique(label).tolist():
        idx = (label == c).nonzero().squeeze(1)
        seg, sd, pres, qual = g64[idx], s64[idx], p64[idx], q64[idx]
        a, b = seg[:, :2], seg[:, 2:]
        direction = (b - a) / (b - a).norm(dim=-1, keepdim=True).clamp(min=1e-9)
        anchor = anchors(seg, centers[idx].double())
        component = torch.from_numpy(collinear_chains(seg.float(), anchor.float(), cfg.perp_tol, cfg.link))
        spread = torch.stack([sd[:, :2].square().sum(-1), sd[:, 2:].square().sum(-1)], dim=-1)  # (n, 2) var
        for comp in torch.unique(component).tolist():
            mem = (component == comp).nonzero().squeeze(1)
            axis = chain_axis(anchor[mem], direction[mem], pres[mem])
            origin = anchor[mem].mean(dim=0)
            points = torch.cat([a[mem], b[mem]])
            t = (points - origin) @ axis
            var = torch.cat([spread[mem, 0], spread[mem, 1]])
            vote_presence = torch.cat([pres[mem], pres[mem]])
            weight = vote_presence / var if cfg.weighting == "inverse_variance" else vote_presence
            certain = var.sqrt() <= cfg.certain_std
            cuts = vote_modes(t[certain], weight[certain], cfg.cut_radius, cfg.min_cut_weight) if bool(certain.any()) else []
            if len(cuts) < 2:
                best = int(spread[mem].sum(-1).argmin())
                out_geometry.append(seg[mem[best]])
                out_label.append(c)
                out_evidence.append(float(pres[mem].sum()))
                out_quality.append(float(qual[mem].max()))
                out_cut.append(float(weight[best]))
                out_size.append(int(mem.shape[0]))
                continue
            cut_t = torch.tensor(cuts, dtype=torch.float64, device=t.device)
            near = (t[certain][None, :] - cut_t[:, None]).abs() <= cfg.cut_radius
            cut_w = near.double() * weight[certain][None, :]
            cut_mass = cut_w.sum(dim=1)
            cut_xy = (cut_w @ points[certain]) / cut_mass[:, None].clamp(min=1e-12)
            interval = torch.searchsorted(cut_t, (anchor[mem] - origin) @ axis) - 1
            inside = (interval >= 0) & (interval < len(cuts) - 1)
            for k in torch.unique(interval[inside]).tolist():
                members = inside & (interval == k)
                out_geometry.append(torch.cat([cut_xy[k], cut_xy[k + 1]]))
                out_label.append(c)
                out_evidence.append(float(pres[mem][members].sum()))
                out_quality.append(float(qual[mem][members].max()))
                out_cut.append(float(torch.minimum(cut_mass[k], cut_mass[k + 1])))
                out_size.append(int(members.sum()))
    device = geometry.device
    if not out_geometry:
        empty = torch.zeros(0, device=device)
        return LineObjects(torch.zeros(0, 4, device=device), torch.zeros(0, dtype=torch.long, device=device),
                           empty, empty, empty, torch.zeros(0, dtype=torch.long, device=device))
    return LineObjects(
        geometry=torch.stack(out_geometry).float().to(device),
        label=torch.tensor(out_label, dtype=torch.long, device=device),
        evidence=torch.tensor(out_evidence, dtype=torch.float32, device=device),
        quality=torch.tensor(out_quality, dtype=torch.float32, device=device),
        cut_weight=torch.tensor(out_cut, dtype=torch.float32, device=device),
        size=torch.tensor(out_size, dtype=torch.long, device=device),
    )


CHAIN_CLASSES = frozenset({"staff", "voltaBracket", "octave"})
"""Long horizontal line classes that break into collinear abutting segments: associated by chains
and cuts. Stems, barlines, beams and ledger lines have no collinear neighbour and their voters
mostly agree on both ends, so endpoint association suits them (Exp 042: barLine 0.79 vs 0.60,
stem 0.76 vs 0.70, staff 0.04 vs 0.69). ``system`` lines are vertical and line up across
consecutive systems, so chains merge them (Exp 043 test: 0.999 with endpoints, 0.101 chained)."""


def chain_labels(line_categories: list[dict[str, object]], cat_id_to_idx: dict[int, int],
                 names: frozenset[str] = CHAIN_CLASSES) -> frozenset[int]:
    """Class indices of the chain-associated line classes."""
    out: set[int] = set()
    for cat in line_categories:
        cat_id, name = cat.get("id"), cat.get("name")
        if isinstance(cat_id, int) and name in names and cat_id in cat_id_to_idx:
            out.add(cat_id_to_idx[cat_id])
    return frozenset(out)


CUT_SATURATION = {"inverse_variance": 200.0, "presence": 10.0}
"""Cut weight from which endpoint agreement counts as full (Exp 042 prototype)."""


def endpoint_agreement(objects: LineObjects, cfg: LineAssociationConfig) -> torch.Tensor:
    """(N,) in [0, 1]: weight of each object's weaker cut, saturating at ``CUT_SATURATION``."""
    return (objects.cut_weight / CUT_SATURATION[cfg.weighting]).clamp(max=1.0)
