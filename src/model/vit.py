import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from typing import Any
from functools import lru_cache
from music_types import (
    Patches,
    Embeddings,
    Batch,
    NumPatches,
    PatchDim,
    EmbedDim,
    FlatViewPatches,
    FlatViewEmbeddings,
    BatchView,
    View,
)


def pair[T](t: T | tuple[T, T]) -> tuple[T, T]:
    return t if isinstance(t, tuple) else (t, t)


def apply_pope(q, k, freqs):
    q_mag = F.softplus(q)
    k_mag = F.softplus(k)

    q_phase = freqs.unsqueeze(1)
    k_phase = freqs.unsqueeze(1)

    q_rotated = torch.stack(
        [q_mag * q_phase.cos(), q_mag * q_phase.sin()], dim=-1
    ).flatten(start_dim=-2)
    k_rotated = torch.stack(
        [k_mag * k_phase.cos(), k_mag * k_phase.sin()], dim=-1
    ).flatten(start_dim=-2)

    return q_rotated, k_rotated


@lru_cache(maxsize=32)
def get_2d_pope_frequencies(
    grid_h, grid_w, dim_head, base=10000.0, device="cpu"
):
    dim_y = dim_head // 2
    dim_x = dim_head - dim_y

    inv_freq_y = 1.0 / (
        base ** (torch.arange(dim_y, device=device).float() / dim_y)
    )
    inv_freq_x = 1.0 / (
        base ** (torch.arange(dim_x, device=device).float() / dim_x)
    )

    pos_y = torch.arange(grid_h, device=device).float()
    pos_x = torch.arange(grid_w, device=device).float()

    freqs_y = torch.einsum("i, j -> ij", pos_y, inv_freq_y)
    freqs_x = torch.einsum("i, j -> ij", pos_x, inv_freq_x)

    freqs_y = freqs_y.unsqueeze(1).expand(-1, grid_w, -1)
    freqs_x = freqs_x.unsqueeze(0).expand(grid_h, -1, -1)

    freqs = torch.cat((freqs_y, freqs_x), dim=-1).reshape(-1, dim_head)
    return freqs


def compute_freqs(patches: Patches, dim_head: int) -> torch.Tensor:
    """Computes and gathers frequencies for the given patches."""
    c, h, w = patches.image_shape
    ph, pw = patches.patch_size
    grid_h, grid_w = h // ph, w // pw

    # using string for device to ensure hashability
    base_freqs = get_2d_pope_frequencies(
        grid_h, grid_w, dim_head, device=str(patches.data.device)
    )

    freqs = base_freqs.unsqueeze(0).expand(patches.batch_size, -1, -1)

    # Gather only the frequencies for the kept patches
    kept_freqs = torch.gather(
        freqs, 1, patches.indices.unsqueeze(-1).expand(-1, -1, dim_head)
    )

    return kept_freqs


class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, freqs=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2), qkv
        )

        if freqs is not None:
            q, k = apply_pope(q, k, freqs)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).flatten(2)
        return self.to_out(out)


class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, freqs=None):
        for attn, ff in self.layers:
            x = attn(x, freqs=freqs) + x
            x = ff(x) + x

        return self.norm(x)


class BaseViT(Module):
    """Base class holding the parameters and core logic. Does not define forward()."""
    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_height, patch_width = pair(patch_size)
        self.dim_head = dim_head

        patch_dim = channels * patch_height * patch_width

        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

    def _forward_impl[B: Batch, N: NumPatches](
        self, patches: Embeddings[B, N, Any]
    ) -> Embeddings[B, N, EmbedDim]:
        freqs = compute_freqs(patches, self.dim_head)
        x_data = patches.data

        x = self.patch_embed(x_data)
        x = self.dropout(x)
        x = self.transformer(x, freqs=freqs)

        return Embeddings(
            data=x,
            indices=patches.indices,
            image_shape=patches.image_shape,
            patch_size=patches.patch_size,
        )


class ViT(BaseViT):
    """Standard ViT that processes generic Patches."""
    def forward[B: Batch, N: NumPatches](
        self, patches: Patches[B, N, PatchDim]
    ) -> Embeddings[B, N, EmbedDim]:
        return self._forward_impl(patches)


class ViewViT(BaseViT):
    """Specialized ViT that strictly processes FlatViewPatches."""
    def forward[B: Batch, BV: BatchView, V: View, N: NumPatches](
        self, patches: FlatViewPatches[B, BV, V, N, PatchDim]
    ) -> FlatViewEmbeddings[B, BV, V, N, EmbedDim]:
        
        out = self._forward_impl(patches)

        return FlatViewEmbeddings(
            data=out.data,
            indices=out.indices,
            image_shape=out.image_shape,
            patch_size=out.patch_size,
            num_views=patches.num_views,
            original_batch_size=patches.original_batch_size,
        )


def vit_nano(
    patch_size: int | tuple[int, int] = 16,
    **kwargs: Any,
) -> ViT:
    return ViT(
        patch_size=patch_size,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        **kwargs,
    )


def vit_small(
    patch_size: int | tuple[int, int] = 16,
    **kwargs: Any,
) -> ViT:
    return ViT(
        patch_size=patch_size,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs,
    )


def vit_base(
    patch_size: int | tuple[int, int] = 16,
    **kwargs: Any,
) -> ViT:
    return ViT(
        patch_size=patch_size,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )
