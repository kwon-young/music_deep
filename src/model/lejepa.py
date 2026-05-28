import torch
import torch.nn as nn
from music_types import Patches


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

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ProjectorMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
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

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        x = self.net(x)
        return x.view(*orig_shape[:-1], -1)


class LeJEPAEncoder(nn.Module):
    def __init__(self, vit_model, embed_dim, proj_dim=16):
        super().__init__()
        self.backbone = vit_model
        self.proj = ProjectorMLP(embed_dim, 2048, proj_dim)

    def forward(self, patches: Patches):
        emb = self.backbone(patches)
        proj = self.proj(emb)
        return emb, proj
