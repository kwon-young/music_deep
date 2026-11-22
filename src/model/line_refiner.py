import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import DFINEWeightingFunction
from music_types import (
    Batch,
    NumPatches,
    NumQueries,
    EmbedDim,
    KeypointDim,
    CoordDim,
    NumBins,
    X1Y1X2Y2,
    PatchUnit,
    TopLeft,
    Embeddings,
    TensorData,
    TokenMap,
    LineOutput,
    LineQueries,
    RefinedLines,
    Keypoints,
    EdgeLogits,
    Coordinates,
)


@dataclass(frozen=True)
class LineRefinerConfig:
    dim: int = 192
    num_layers: int = 2
    heads: int = 6
    points: int = 4
    mlp_ratio: int = 4
    offset_scale: float = 0.5
    num_shapes: int = 5
    reg_max: int = 32
    base_anchor_size: float = 1.0


def dense_token_map[B: Batch, D: EmbedDim](
    features: Embeddings[B, NumPatches, D],
) -> TokenMap[B, D]:
    _, height, width = features.image_shape
    patch_h, patch_w = features.patch_size
    grid_h, grid_w = height // patch_h, width // patch_w
    batch, _, dim = features.data.shape

    dense = features.data.new_zeros(batch, grid_h * grid_w, dim).scatter(
        1, features.indices.unsqueeze(-1).expand(-1, -1, dim), features.data
    )
    return TokenMap(dense.transpose(1, 2).reshape(batch, dim, grid_h, grid_w))


def gather_line_queries[B: Batch, D: EmbedDim](
    features: Embeddings[B, NumPatches, D],
    lines: LineOutput[B, NumQueries, KeypointDim, CoordDim],
    query_indices: torch.Tensor,
    num_shapes: int,
) -> LineQueries[B, NumQueries, D, KeypointDim, CoordDim]:
    """Selects line queries `p * num_shapes + k` given as a (B, Q) index tensor."""
    dim = features.data.shape[-1]
    patch_indices = query_indices // num_shapes

    def take(values: torch.Tensor) -> torch.Tensor:
        trailing = values.shape[2:]
        index = query_indices.reshape(*query_indices.shape, *([1] * len(trailing)))
        return torch.gather(values, 1, index.expand(-1, -1, *trailing))

    return LineQueries(
        content=TensorData(
            torch.gather(
                features.data, 1, patch_indices.unsqueeze(-1).expand(-1, -1, dim)
            )
        ),
        shape_ids=TensorData(query_indices % num_shapes),
        absolute_centers=Coordinates(take(lines.absolute_centers.data)),
        raw_directions=Coordinates(take(lines.raw_directions.data)),
        endpoint_logits=EdgeLogits(take(lines.pred_endpoint_logits.data)),
    )


def decode_endpoints(
    centers: torch.Tensor,
    base_dirs: torch.Tensor,
    endpoint_logits: torch.Tensor,
    bins: torch.Tensor,
    base_anchor_size: float,
) -> torch.Tensor:
    """Same decoding as LineHead: center + base_dir + E[W(n)] * (|base_dir| + anchor)."""
    scale = base_dirs.abs() + base_anchor_size
    residual = (F.softmax(endpoint_logits, dim=-1) * bins).sum(dim=-1) * scale
    return torch.cat([centers, centers], dim=-1) + base_dirs + residual


class EndpointDeformableAttention(nn.Module):
    """D-FINE deformable cross-attention, with the two line endpoints as reference points.

    Each head samples `points` locations around each endpoint. Offsets are expressed in
    units of that endpoint's scale `|base_dir| + anchor`, the same frame as the FDR bins.
    """

    def __init__(self, dim: int, heads: int, points: int, offset_scale: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.points = points
        self.offset_scale = offset_scale

        self.sampling_offsets = nn.Linear(dim, heads * 2 * points * 2)
        self.attention_weights = nn.Linear(dim, heads * 2 * points)
        self.output_proj = nn.Linear(dim, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.heads, dtype=torch.float32) * (
            2.0 * math.pi / self.heads
        )
        directions = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        directions = directions / directions.abs().max(dim=-1, keepdim=True).values
        radii = torch.arange(1, self.points + 1, dtype=torch.float32)
        grid = directions[:, None, None, :] * radii[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(
                grid.expand(-1, 2, -1, -1).flatten()
            )

        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward[B: Batch, D: EmbedDim](
        self,
        query: torch.Tensor,
        endpoints: torch.Tensor,
        scales: torch.Tensor,
        value: TokenMap[B, D],
    ) -> torch.Tensor:
        batch, num_queries, _ = query.shape
        _, dim, grid_h, grid_w = value.data.shape
        num_samples = 2 * self.points

        offsets = self.sampling_offsets(query).view(
            batch, num_queries, self.heads, 2, self.points, 2
        )
        reference = endpoints.view(batch, num_queries, 1, 2, 1, 2)
        extent = scales.view(batch, num_queries, 1, 2, 1, 2)
        locations = reference + offsets * extent * (self.offset_scale / self.points)

        grid_size = locations.new_tensor([grid_w, grid_h])
        grid = (locations / grid_size * 2.0 - 1.0).permute(0, 2, 1, 3, 4, 5)
        grid = grid.reshape(batch * self.heads, num_queries, num_samples, 2)

        weights = self.attention_weights(query).view(
            batch, num_queries, self.heads, num_samples
        )
        weights = weights.softmax(dim=-1).permute(0, 2, 1, 3)
        weights = weights.reshape(batch * self.heads, 1, num_queries, num_samples)

        head_values = value.data.reshape(
            batch * self.heads, dim // self.heads, grid_h, grid_w
        )
        sampled = F.grid_sample(
            head_values,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        out = (sampled * weights).sum(dim=-1)
        out = out.reshape(batch, dim, num_queries).transpose(1, 2)
        return self.output_proj(out)


class LineRefinerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        points: int,
        mlp_ratio: int,
        offset_scale: float,
        num_bins: int,
    ):
        super().__init__()
        self.cross_attn = EndpointDeformableAttention(
            dim, heads, points, offset_scale
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

        self.delta_hidden = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.delta_out = nn.Linear(dim, 4 * num_bins)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    def forward[B: Batch, D: EmbedDim](
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        endpoints: torch.Tensor,
        scales: torch.Tensor,
        value: TokenMap[B, D],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended = self.cross_attn(query + query_pos, endpoints, scales, value)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))
        return query, self.delta_out(self.delta_hidden(query))


class LineRefiner(nn.Module):
    """Iterative D-FINE refinement of the stage-0 line endpoints.

    Every layer keeps the stage-0 frame (`base_dirs`, scale `|base_dir| + anchor`),
    adds residual logits to the previous distribution, and samples the token map
    around the current (detached) endpoints.
    """

    def __init__(self, config: LineRefinerConfig):
        super().__init__()
        self.config = config
        dim = config.dim

        self.value_proj = nn.Linear(dim, dim)
        self.shape_embed = nn.Embedding(config.num_shapes, dim)
        self.query_pos = nn.Sequential(
            nn.Linear(4, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.layers = nn.ModuleList(
            [
                LineRefinerLayer(
                    dim,
                    config.heads,
                    config.points,
                    config.mlp_ratio,
                    config.offset_scale,
                    config.reg_max + 1,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.weighting_fn = DFINEWeightingFunction(reg_max=config.reg_max)

    def forward[B: Batch, Q: NumQueries, D: EmbedDim](
        self,
        features: Embeddings[B, NumPatches, D],
        queries: LineQueries[B, Q, D, KeypointDim, CoordDim],
    ) -> RefinedLines[B, Q, KeypointDim]:
        value = dense_token_map(
            replace(features, data=self.value_proj(features.data))
        )
        grid_h, grid_w = value.data.shape[-2:]
        bins = self.weighting_fn.w
        anchor = self.config.base_anchor_size

        centers = queries.absolute_centers.data
        base_dirs = queries.raw_directions.data
        scales = base_dirs.abs() + anchor
        logits = queries.endpoint_logits.data
        batch, num_queries, coords, num_bins = logits.shape
        normalizer = scales.new_tensor([grid_w, grid_h, grid_w, grid_h])

        endpoints = decode_endpoints(centers, base_dirs, logits, bins, anchor)
        query = queries.content.data + self.shape_embed(queries.shape_ids.data)

        keypoints: list[
            Keypoints[tuple[B, Q, KeypointDim], X1Y1X2Y2, PatchUnit, TopLeft]
        ] = []
        endpoint_logits: list[EdgeLogits[tuple[B, Q, KeypointDim, NumBins]]] = []
        for layer in self.layers:
            reference = endpoints.detach()
            query_pos = self.query_pos(reference / normalizer)
            query, delta = layer(query, query_pos, reference, scales, value)
            logits = logits + delta.view(batch, num_queries, coords, num_bins)
            endpoints = decode_endpoints(centers, base_dirs, logits, bins, anchor)
            keypoints.append(Keypoints(endpoints))
            endpoint_logits.append(EdgeLogits(logits))

        return RefinedLines(keypoints=keypoints, endpoint_logits=endpoint_logits)
