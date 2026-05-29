import torch
import torch.nn as nn
from music_types import Patches, Embeddings, Batch, NumPatches, EmbedDim

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

    def forward[B: Batch, N: NumPatches](
        self, x: Embeddings[B, N, EmbedDim]
    ) -> Embeddings[B, N, ProjDim]:
        orig_shape = x.data.shape
        
        # Flatten the batch and patch dimensions to pass through the Linear/BatchNorm layers
        flat_x = x.data.reshape(-1, orig_shape[-1])
        out_data = self.net(flat_x)
        
        # Reshape back to (Batch, NumPatches, ProjDim)
        out_data = out_data.view(*orig_shape[:-1], -1)
        
        return Embeddings(
            data=out_data,
            indices=x.indices,
            image_shape=x.image_shape,
            patch_size=x.patch_size,
        )
