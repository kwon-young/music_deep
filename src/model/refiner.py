"""Second-stage line endpoint refiner (Exp 035).

Analysis 005 showed that the token whose patch contains a line endpoint locates it to about
1.5 px, while the token that predicts the line degrades with its distance to the endpoint
(8.8 px median beyond 16 patches). The refiner reads, for each coarse endpoint, the 3x3
window of kept tokens around it and predicts an absolute correction of at most
``scale / 2`` patches, independent of the line length. It runs ``num_steps`` times, each
step re-centring the window on the previous estimate.

Coordinates are in patch units with the top-left image corner at (0, 0), so the patch of
grid cell (row, col) spans [col, col + 1) x [row, row + 1) and its center is at +0.5.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import DFINEWeightingFunction

WINDOW_OFFSETS = (-1, 0, 1)


@dataclass(frozen=True)
class TokenGrid:
    """Kept tokens of several images, flattened, with a dense cell -> token lookup.

    ``lookup[offset[i] + row * width[i] + col]`` is the index in ``tokens`` of the kept patch
    at (row, col) of image i, or -1 when variance dropping removed that patch.
    """

    tokens: torch.Tensor
    lookup: torch.Tensor
    offset: torch.Tensor
    height: torch.Tensor
    width: torch.Tensor

    @staticmethod
    def from_batch(
        tokens: torch.Tensor, cells: torch.Tensor, grid_h: int, grid_w: int
    ) -> "TokenGrid":
        """``tokens`` (B, P, D) of the kept patches whose grid cells are ``cells`` (B, P)."""
        batch, num_kept, dim = tokens.shape
        device = tokens.device
        num_cells = grid_h * grid_w
        first_cell = torch.arange(batch, device=device) * num_cells
        lookup = torch.full((batch * num_cells,), -1, dtype=torch.long, device=device)
        lookup[(first_cell.unsqueeze(1) + cells).flatten()] = torch.arange(
            batch * num_kept, device=device
        )
        return TokenGrid(
            tokens=tokens.reshape(batch * num_kept, dim),
            lookup=lookup,
            offset=first_cell,
            height=torch.full((batch,), grid_h, dtype=torch.long, device=device),
            width=torch.full((batch,), grid_w, dtype=torch.long, device=device),
        )

    @staticmethod
    def concat(grids: Sequence["TokenGrid"]) -> "TokenGrid":
        lookups: list[torch.Tensor] = []
        offsets: list[torch.Tensor] = []
        num_tokens = 0
        num_cells = 0
        for grid in grids:
            lookups.append(torch.where(grid.lookup >= 0, grid.lookup + num_tokens, -1))
            offsets.append(grid.offset + num_cells)
            num_tokens += grid.tokens.shape[0]
            num_cells += grid.lookup.shape[0]
        return TokenGrid(
            tokens=torch.cat([g.tokens for g in grids]),
            lookup=torch.cat(lookups),
            offset=torch.cat(offsets),
            height=torch.cat([g.height for g in grids]),
            width=torch.cat([g.width for g in grids]),
        )

    def to(self, device: torch.device) -> "TokenGrid":
        return TokenGrid(
            tokens=self.tokens.to(device),
            lookup=self.lookup.to(device),
            offset=self.offset.to(device),
            height=self.height.to(device),
            width=self.width.to(device),
        )

    def centers(self, token: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Patch centers of global token indices ``token`` belonging to images ``image``."""
        cell = self.cell_of(token, image)
        width = self.width[image]
        return torch.stack([cell % width + 0.5, cell // width + 0.5], dim=-1).float()

    def cell_of(self, token: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Inverse of the lookup, for tokens of the given images."""
        inverse = torch.empty_like(self.tokens[:, 0], dtype=torch.long)
        kept = (self.lookup >= 0).nonzero().squeeze(1)
        inverse[self.lookup[kept]] = kept
        return inverse[token] - self.offset[image]

    def window(
        self, image: torch.Tensor, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The 3x3 cells around each point: token index (M, 9), validity (M, 9), centers (M, 9, 2).

        A cell is invalid when it lies outside the image or its patch was dropped; its token
        index is then 0 and must be masked by the caller.
        """
        safe = points.detach().clamp(-1e4, 1e4)
        col = safe[:, 0].floor().long()
        row = safe[:, 1].floor().long()
        delta = torch.tensor(WINDOW_OFFSETS, device=points.device)
        rows = (row[:, None, None] + delta[None, :, None]).expand(-1, 3, 3).reshape(-1, 9)
        cols = (col[:, None, None] + delta[None, None, :]).expand(-1, 3, 3).reshape(-1, 9)
        height = self.height[image].unsqueeze(1)
        width = self.width[image].unsqueeze(1)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        cell = self.offset[image].unsqueeze(1) + rows.clamp(min=0) * width + cols.clamp(min=0)
        token = torch.where(inside, self.lookup[cell.clamp(max=self.lookup.shape[0] - 1)], -1)
        valid = token >= 0
        centers = torch.stack([cols + 0.5, rows + 0.5], dim=-1).float()
        return token.clamp(min=0), valid, centers


@dataclass(frozen=True)
class LineQueries:
    """Lines to refine: which image, which token predicted them, class, coarse endpoints (N, 4)."""

    image: torch.Tensor
    source: torch.Tensor
    label: torch.Tensor
    endpoints: torch.Tensor

    @staticmethod
    def concat(queries: Sequence["LineQueries"]) -> "LineQueries":
        return LineQueries(
            image=torch.cat([q.image for q in queries]),
            source=torch.cat([q.source for q in queries]),
            label=torch.cat([q.label for q in queries]),
            endpoints=torch.cat([q.endpoints for q in queries]),
        )


@dataclass(frozen=True)
class RefinedEndpoints:
    """Output of one refinement step; ``start`` is where the step read its window from and
    ``state`` (N, 2, D) what the step knows about each endpoint after reading it."""

    endpoints: torch.Tensor
    start: torch.Tensor
    bin_logits: torch.Tensor
    state: torch.Tensor


class FourierOffsets(nn.Module):
    """Embeds 2D offsets (in patches) with sin/cos at geometric periods plus the raw value."""

    freqs: torch.Tensor

    def __init__(self, dim: int, num_freqs: int, min_period: float, max_period: float) -> None:
        super().__init__()
        periods = torch.logspace(
            math.log10(min_period), math.log10(max_period), num_freqs
        )
        self.register_buffer("freqs", 2 * math.pi / periods)
        self.proj = nn.Linear(4 * num_freqs + 2, dim)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        angles = offsets.unsqueeze(-1) * self.freqs
        feats = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.proj(torch.cat([feats, offsets], dim=-1))


class RefineStep(nn.Module):
    """One cross-attention read of the window plus an FFN, then a binned absolute correction."""

    def __init__(self, dim: int, heads: int, num_bins: int) -> None:
        super().__init__()
        self.heads = heads
        self.key_pos = FourierOffsets(dim, num_freqs=12, min_period=1 / 16, max_period=8.0)
        # Always-valid key: lets attention read nothing when the whole window was dropped.
        self.null_key = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        self.to_bins = nn.Linear(dim, 2 * num_bins)
        # Uniform bins at init: the expected correction is exactly 0 (W is symmetric).
        nn.init.zeros_(self.to_bins.weight)
        nn.init.zeros_(self.to_bins.bias)

    def forward(
        self,
        state: torch.Tensor,
        grid: TokenGrid,
        image: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``state`` (M, D) per endpoint, ``points`` (M, 2); returns new state and bin logits."""
        num, dim = state.shape
        token, valid, centers = grid.window(image, points)
        keys = self.norm_kv(grid.tokens[token].float())
        keys = keys + self.key_pos(centers - points.unsqueeze(1))
        keys = torch.cat([self.null_key.expand(num, 1, dim), keys], dim=1)
        mask = torch.cat([valid.new_ones(num, 1), valid], dim=1)

        # One query against 10 keys: elementwise attention; fused SDPA or bmm kernels waste
        # nearly all their tiles on such tiny matrices.
        dim_head = dim // self.heads
        q = self.to_q(self.norm_q(state)).view(num, self.heads, dim_head)
        k, v = (t.view(num, -1, self.heads, dim_head) for t in self.to_kv(keys).chunk(2, dim=-1))
        logits = (q.unsqueeze(1) * k).sum(dim=-1) * dim_head**-0.5
        weights = logits.masked_fill(~mask.unsqueeze(-1), float("-inf")).softmax(dim=1)
        read = (weights.unsqueeze(-1) * v).sum(dim=1)
        state = state + self.to_out(read.reshape(num, dim))
        state = state + self.ffn(state)
        return state, self.to_bins(self.out_norm(state)).view(num, 2, -1)


class EndpointRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_steps: int = 2,
        heads: int = 3,
        reg_max: int = 32,
        scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.weighting = DFINEWeightingFunction(reg_max=reg_max)
        self.source_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.class_emb = nn.Embedding(num_classes, dim)
        self.end_emb = nn.Embedding(2, dim)
        self.direction = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.steps = nn.ModuleList(
            [RefineStep(dim, heads, reg_max + 1) for _ in range(num_steps)]
        )

    def initial_state(self, grid: TokenGrid, queries: LineQueries) -> torch.Tensor:
        """(N, 2, D): what the line is, which end, and where the other end is."""
        points = queries.endpoints.detach().view(-1, 2, 2)
        to_other = points.flip(1) - points
        length = torch.linalg.vector_norm(to_other, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = torch.cat(
            [to_other.sign() * to_other.abs().log1p(), to_other / length], dim=-1
        )
        line = self.source_proj(grid.tokens[queries.source].float()) + self.class_emb(
            queries.label
        )
        return line.unsqueeze(1) + self.end_emb.weight.unsqueeze(0) + self.direction(direction)

    def forward(self, grid: TokenGrid, queries: LineQueries) -> list[RefinedEndpoints]:
        num = queries.endpoints.shape[0]
        state = self.initial_state(grid, queries).reshape(2 * num, -1)
        image = queries.image.repeat_interleave(2)
        points = queries.endpoints.detach().reshape(2 * num, 2)
        outputs: list[RefinedEndpoints] = []
        for step in self.steps:
            start = points.detach()
            read: tuple[torch.Tensor, torch.Tensor] = step(state, grid, image, start)
            state, logits = read
            points = start + self.weighting(logits.softmax(dim=-1)) * self.scale
            outputs.append(
                RefinedEndpoints(
                    endpoints=points.reshape(num, 4),
                    start=start.reshape(num, 4),
                    bin_logits=logits.reshape(num, 4, -1),
                    state=state.reshape(num, 2, -1),
                )
            )
        return outputs


class QualityHead(nn.Module):
    """Predicts the localization quality (OKS logit) of a refined line from its two endpoint
    states, so the final score can rank duplicates by how well they are placed."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``state`` (N, 2, D) -> (N,) logits."""
        return self.net(state.flatten(1)).squeeze(-1)


def line_oks(
    pred: torch.Tensor, gt: torch.Tensor, area: torch.Tensor, sigma: float = 0.1
) -> torch.Tensor:
    """(Q, T) COCO keypoint OKS between lines (Q, 4) and (T, 4), as in ``evaluate_coco.py``:
    per endpoint exp(-d^2 / (2 (2 sigma)^2 area)), averaged over the two endpoints.
    Coordinates and ``area`` must be in the same (pixel) units."""
    d2 = (pred.view(-1, 1, 2, 2) - gt.view(1, -1, 2, 2)).pow(2).sum(dim=-1)
    e = d2 / (2 * sigma) ** 2 / (area.view(1, -1, 1) + torch.finfo(pred.dtype).eps) / 2
    return torch.exp(-e).mean(dim=-1)


def fgl_loss(
    bin_logits: torch.Tensor, target_residual: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """D-FINE fine-grained localization loss: cross-entropy against the two bins around the
    target, weighted by linear interpolation. ``target_residual`` is in units of the
    correction scale (the range of ``weights``), shape (..., ) matching ``bin_logits[..., 0]``.
    """
    with torch.no_grad():
        target = target_residual.clamp(weights[0].item(), weights[-1].item())
        right = torch.searchsorted(weights, target.contiguous()).clamp(1, weights.shape[0] - 1)
        left = right - 1
        w_left = (weights[right] - target) / (weights[right] - weights[left] + 1e-6)
        w_right = 1.0 - w_left
    log_probs = F.log_softmax(bin_logits, dim=-1)
    return -(
        log_probs.gather(-1, left.unsqueeze(-1)).squeeze(-1) * w_left
        + log_probs.gather(-1, right.unsqueeze(-1)).squeeze(-1) * w_right
    )


def refiner_loss(
    refiner: EndpointRefiner,
    outputs: Sequence[RefinedEndpoints],
    target: torch.Tensor,
    mask: torch.Tensor,
    fgl_weight: float,
) -> tuple[torch.Tensor, list[float]]:
    """Deep-supervised L1 + FGL over all steps, on the coordinates where ``mask`` is 1.

    Returns the total loss and each step's masked L1 in patches (for logging).
    """
    denom = mask.sum().clamp(min=1.0)
    total = target.new_zeros(())
    l1_per_step: list[float] = []
    for out in outputs:
        l1 = ((out.endpoints - target).abs() * mask).sum() / denom
        residual = (target - out.start) / refiner.scale
        fgl = (fgl_loss(out.bin_logits, residual, refiner.weighting.w) * mask).sum() / denom
        total = total + l1 + fgl_weight * fgl
        l1_per_step.append(l1.item())
    return total, l1_per_step
