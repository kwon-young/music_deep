import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DFINEWeightingFunction(nn.Module):
    """
    D-FINE's non-uniform weighting function W(n) for the FDR bins.
    Converts bin probabilities into actual edge offsets.
    """

    def __init__(self, reg_max: int = 32, a: float = 0.5, c: float = 0.25):
        super().__init__()
        self.reg_max = reg_max
        num_bins = reg_max + 1

        # Pre-compute the weighting function W(n) for all bins
        w = torch.zeros(num_bins)
        N = reg_max

        w[0] = -a
        w[N] = a

        for n in range(1, N):
            if n < N / 2:
                w[n] = c - c * ((a / c) + 1) ** ((N - 2 * n) / (N - 2))
            else:
                w[n] = -c + c * ((a / c) + 1) ** ((-N + 2 * n) / (N - 2))

        # Register as a buffer so it moves to the correct device automatically
        self.register_buffer("w", w)

    def forward(self, edge_probs: torch.Tensor) -> torch.Tensor:
        # edge_probs shape: (..., 4, reg_max + 1)
        # Expected offset is the weighted sum of probabilities
        return (edge_probs * self.w).sum(dim=-1)


class DFINEDenseHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_shapes: int = 5,
        reg_max: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_shapes = num_shapes
        self.reg_max = reg_max
        self.num_bins = reg_max + 1

        # C (classes) + 2 (center offsets) + 4*N (edge bins)
        self.preds_per_shape = num_classes + 2 + (4 * self.num_bins)
        self.out_dim = num_shapes * self.preds_per_shape

        # Shared Learnable Shapes (Dynamic Anchors): [K, 2] for (width, height)
        self.learnable_shapes = nn.Parameter(torch.full((num_shapes, 2), 0.1))

        # The Dense MLP applied to each patch token
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_dim),
        )

        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, _ = patch_tokens.shape

        raw_preds = self.mlp(patch_tokens)
        preds = raw_preds.view(B, P, self.num_shapes, self.preds_per_shape)

        classes = preds[..., : self.num_classes]
        center_offsets = preds[..., self.num_classes : self.num_classes + 2]
        edge_logits = preds[..., -4 * self.num_bins :]

        edge_logits = edge_logits.view(B, P, self.num_shapes, 4, self.num_bins)
        edge_probs = F.softmax(edge_logits, dim=-1)

        # Shape: (B, P, K, 4) - Residuals in range [-a, a]
        relative_edge_offsets = self.weighting_fn(edge_probs)

        shapes = self.learnable_shapes.view(1, 1, self.num_shapes, 2)
        w_k, h_k = shapes[..., 0], shapes[..., 1]

        # 1. Scale residuals by anchor dimensions
        L_res = relative_edge_offsets[..., 0] * w_k
        T_res = relative_edge_offsets[..., 1] * h_k
        R_res = relative_edge_offsets[..., 2] * w_k
        B_res = relative_edge_offsets[..., 3] * h_k

        # 2. Add base anchor distances (half width/height)
        L = (w_k / 2) + L_res
        T = (h_k / 2) + T_res
        R = (w_k / 2) + R_res
        B_edge = (h_k / 2) + B_res

        # 3. Apply center offsets to get final [x1, y1, x2, y2] relative to patch center
        cx = center_offsets[..., 0]
        cy = center_offsets[..., 1]

        x1 = cx - L
        y1 = cy - T
        x2 = cx + R
        y2 = cy + B_edge

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return classes, center_offsets, boxes, edge_logits


class OMRDetector(nn.Module):
    def __init__(
        self, vit_backbone: nn.Module, num_classes: int, num_shapes: int = 5
    ):
        super().__init__()
        self.backbone = vit_backbone

        in_dim = self.backbone.patch_embed[1].out_features
        self.backbone.pool = "none"
        self.backbone.mlp_head = None

        self.head = DFINEDenseHead(
            in_dim=in_dim, num_classes=num_classes, num_shapes=num_shapes
        )

    def forward(
        self,
        patches: torch.Tensor,
        freqs: torch.Tensor,
        patch_centers: torch.Tensor,
    ) -> dict:
        """
        patch_centers: (Batch, Num_Patches, 2) containing the normalized (x, y) center of each patch.
        Returns a dictionary ready for DFINECriterion.
        """
        features = self.backbone(patches, freqs)
        patch_tokens = features[:, 1:, :]

        classes, center_offsets, boxes, edge_logits = self.head(patch_tokens)

        B, P, K, _ = boxes.shape

        # Reshape patch_centers for broadcasting: (Batch, Num_Patches, 1, 2)
        patch_centers_expanded = patch_centers.unsqueeze(2)

        # Add absolute patch centers to the predicted center offsets
        absolute_centers = patch_centers_expanded + center_offsets

        # Shift the boxes to absolute coordinates
        boxes[..., 0] += patch_centers_expanded[..., 0]  # x1
        boxes[..., 1] += patch_centers_expanded[..., 1]  # y1
        boxes[..., 2] += patch_centers_expanded[..., 0]  # x2
        boxes[..., 3] += patch_centers_expanded[..., 1]  # y2

        # Expand learnable shapes to match (B, P, K, 2)
        raw_shapes = self.head.learnable_shapes
        expanded_shapes = raw_shapes.view(1, 1, K, 2).expand(B, P, K, 2)

        # Flatten P and K dimensions into a single "num_queries" dimension
        # and return the exact dictionary expected by the criterion
        return {
            "pred_logits": classes.view(B, P * K, -1),
            "pred_boxes": boxes.view(B, P * K, 4),
            "pred_edge_logits": edge_logits.view(B, P * K, 4, -1),
            "absolute_centers": absolute_centers.view(B, P * K, 2),
            "learnable_shapes": expanded_shapes.reshape(B, P * K, 2)
        }
