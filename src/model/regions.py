"""Responsibility regions of the patch-set assignment (PATCH_SET_ASSIGNMENT.md, section 1).

All coordinates are in patch units: a patch's cell is the unit square centred on its centre, and
a patch is responsible for an object when the distance between its cell and the object (box or
segment) is at most the margin m.
"""

import torch


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
            point_segment_distance(centers + torch.tensor(offset, dtype=centers.dtype, device=centers.device), segments)
            for offset in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))
        ]
    ).amin(dim=0)
    start = segments[:, :2]
    delta = segments[:, 2:] - start
    enter = torch.zeros(endpoint.shape, dtype=centers.dtype, device=centers.device)
    leave = torch.ones(endpoint.shape, dtype=centers.dtype, device=centers.device)
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


def cell_distance(centers: torch.Tensor, geometry: torch.Tensor, is_line: bool) -> torch.Tensor:
    """(P, N) distance between the cells centred on ``centers`` and boxes or segments."""
    return cell_segment_distance(centers, geometry) if is_line else cell_box_distance(centers, geometry)


def expected_voters(
    centers: torch.Tensor, geometry: torch.Tensor, is_line: bool, margin: float, chunk: int = 1024
) -> torch.Tensor:
    """(N,) number of kept patches (``centers``) whose region contains each object."""
    counts = [
        (cell_distance(centers, geometry[start : start + chunk], is_line) <= margin).sum(dim=0)
        for start in range(0, geometry.shape[0], chunk)
    ]
    return torch.cat(counts) if counts else geometry.new_zeros(0, dtype=torch.long)
