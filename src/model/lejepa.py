import torch
import torch.nn as nn
from model.vit import Transformer, compute_freqs
from music_types import (
    FlatViewEmbeddings,
    Embeddings,
    Batch,
    BatchView,
    View,
    NumPatches,
    EmbedDim,
)

type ProjDim = int


class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer = Transformer(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            use_sdpa=use_sdpa,
        )

    def forward(self, context: Embeddings, target: Embeddings) -> Embeddings:
        B, N_total = target.indices.shape
        B, N_ctx = context.indices.shape
        N_mask = N_total - N_ctx

        # 1. Find mask indices (indices in target that are NOT in context)
        max_idx = target.indices.max().item() + 1
        dense_mask = torch.ones(
            (B, max_idx), dtype=torch.bool, device=target.indices.device
        )
        dense_mask.scatter_(1, context.indices, False)
        is_mask = torch.gather(dense_mask, 1, target.indices)
        mask_indices = target.indices[is_mask].view(B, N_mask)

        # 2. Create mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        # 3. Concatenate context and mask tokens
        pred_data = torch.cat([context.data, mask_tokens], dim=1)
        pred_indices = torch.cat([context.indices, mask_indices], dim=1)

        pred_embeddings = Embeddings(
            data=pred_data,
            indices=pred_indices,
            image_shape=target.image_shape,
            patch_size=target.patch_size,
        )

        # 4. Compute POPE frequencies and pass through Transformer
        freqs = compute_freqs(pred_embeddings, self.dim_head)
        out_data = self.transformer(pred_embeddings.data, freqs=freqs)

        # 5. Extract only the predictions for the mask tokens
        mask_out_data = out_data[:, N_ctx:, :]

        return Embeddings(
            data=mask_out_data,
            indices=mask_indices,
            image_shape=target.image_shape,
            patch_size=target.patch_size,
        )


class ProjectorMLP(nn.Module):
    def __init__(
        self,
        in_features: EmbedDim,
        hidden_features: int,
        out_features: ProjDim,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward[B: Batch, BV: BatchView, V: View, N: NumPatches](
        self, x: FlatViewEmbeddings[B, BV, V, N, EmbedDim]
    ) -> FlatViewEmbeddings[B, BV, V, N, ProjDim]:
        b, n, _ = x.data.shape

        flat_x = x.data.reshape(b * n, -1)
        out_data = self.net(flat_x)

        out_data = out_data.view(b, n, -1)

        return FlatViewEmbeddings(
            data=out_data,
            indices=x.indices,
            image_shape=x.image_shape,
            patch_size=x.patch_size,
            num_views=x.num_views,
            original_batch_size=x.original_batch_size,
        )
