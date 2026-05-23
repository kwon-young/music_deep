import torch
from torch import nn
from torch.nn import Module, ModuleList
from typing import Any


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2), qkv
        )

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

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class PatchRearrange(Module):
    def __init__(self, patch_height, patch_width):
        super().__init__()
        self.ph = patch_height
        self.pw = patch_width

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unflatten(2, (h // self.ph, self.ph)).unflatten(
            4, (w // self.pw, self.pw)
        )
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(b, -1, self.ph * self.pw * c)


class ViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        num_keep_patches=None,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.num_keep_patches = num_keep_patches
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (
            image_width // patch_width
        )
        patch_dim = channels * patch_height * patch_width

        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )
        self.num_cls_tokens = 1 if pool == "cls" else 0

        self.patch_rearrange = PatchRearrange(patch_height, patch_width)
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(self.num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches + self.num_cls_tokens, dim)
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.patch_rearrange(img)
        num_patches = x.shape[1]

        if self.num_keep_patches is not None and self.num_keep_patches < num_patches:
            variances = x.var(dim=-1)
            _, indices = variances.topk(self.num_keep_patches, dim=-1)
            indices, _ = indices.sort(dim=-1)
            x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        else:
            indices = None

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if indices is not None:
            pos = self.pos_embedding.expand(batch, -1, -1)
            cls_pos = pos[:, :self.num_cls_tokens, :]
            patch_pos = pos[:, self.num_cls_tokens : self.num_cls_tokens + num_patches, :]
            
            patch_pos = torch.gather(patch_pos, 1, indices.unsqueeze(-1).expand(-1, -1, patch_pos.shape[-1]))
            
            pos = torch.cat((cls_pos, patch_pos), dim=1)
            x = x + pos
        else:
            seq = x.shape[1]
            x = x + self.pos_embedding[:seq]

        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def vit_nano(
    image_size: int | tuple[int, int],
    num_classes: int,
    patch_size: int | tuple[int, int] = 16,
    **kwargs: Any,
) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        **kwargs,
    )


def vit_small(
    image_size: int | tuple[int, int],
    num_classes: int,
    patch_size: int | tuple[int, int] = 16,
    **kwargs: Any,
) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs,
    )


def vit_base(
    image_size: int | tuple[int, int],
    num_classes: int,
    patch_size: int | tuple[int, int] = 16,
    **kwargs: Any,
) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )
