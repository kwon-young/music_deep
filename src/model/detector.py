import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DFINEWeightingFunction(nn.Module):
    """
    D-FINE's non-uniform weighting function W(n) for the FDR bins.
    Converts bin probabilities into actual edge offsets.
    """
    def __init__(self, num_bins: int = 32, a: float = 0.5, c: float = 0.25):
        super().__init__()
        self.num_bins = num_bins
        
        # Pre-compute the weighting function W(n) for all bins
        w = torch.zeros(num_bins)
        N = num_bins - 1
        
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
        # edge_probs shape: (..., 4, num_bins)
        # w shape: (num_bins,)
        # Expected offset is the weighted sum of probabilities
        return (edge_probs * self.w).sum(dim=-1)


class DFINEDenseHead(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        num_classes: int, 
        num_shapes: int = 5, 
        num_bins: int = 32
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_shapes = num_shapes
        self.num_bins = num_bins
        
        # 1 (conf) + C (classes) + 2 (center offsets) + 4*N (edge bins)
        self.preds_per_shape = 1 + num_classes + 2 + (4 * num_bins)
        self.out_dim = num_shapes * self.preds_per_shape
        
        # Shared Learnable Shapes (Dynamic Anchors): [K, 2] for (width, height)
        # Initialized to small positive values (e.g., 0.1 of the patch size)
        self.learnable_shapes = nn.Parameter(torch.full((num_shapes, 2), 0.1))
        
        # The Dense MLP applied to each patch token
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_dim)
        )
        
        self.weighting_fn = DFINEWeightingFunction(num_bins=num_bins)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patch_tokens: (Batch, Num_Patches, Dim)
        Returns:
            conf_and_classes: (Batch, Num_Patches, K, 1 + C)
            centers: (Batch, Num_Patches, K, 2)
            boxes: (Batch, Num_Patches, K, 4) - The refined edge offsets (L, T, R, B)
        """
        B, P, _ = patch_tokens.shape
        
        # Apply MLP to all patches: (B, P, K * preds_per_shape)
        raw_preds = self.mlp(patch_tokens)
        
        # Reshape to separate the K shapes: (B, P, K, preds_per_shape)
        preds = raw_preds.view(B, P, self.num_shapes, self.preds_per_shape)
        
        # Split the predictions
        conf_and_classes = preds[..., :1 + self.num_classes]
        center_offsets = preds[..., 1 + self.num_classes : 1 + self.num_classes + 2]
        edge_logits = preds[..., -4 * self.num_bins:]
        
        # Reshape edge logits to (B, P, K, 4, num_bins) and apply Softmax to get distributions
        edge_logits = edge_logits.view(B, P, self.num_shapes, 4, self.num_bins)
        edge_probs = F.softmax(edge_logits, dim=-1)
        
        # Apply D-FINE weighting function to get the expected relative offsets
        # Shape: (B, P, K, 4)
        relative_edge_offsets = self.weighting_fn(edge_probs)
        
        # Scale the relative offsets by the learnable shapes (w_k, h_k)
        # learnable_shapes is (K, 2). We need to scale L/R by w_k, and T/B by h_k.
        shapes = self.learnable_shapes.view(1, 1, self.num_shapes, 2)
        w_k, h_k = shapes[..., 0], shapes[..., 1]
        
        # relative_edge_offsets is (L, T, R, B)
        L = relative_edge_offsets[..., 0] * w_k
        T = relative_edge_offsets[..., 1] * h_k
        R = relative_edge_offsets[..., 2] * w_k
        B_edge = relative_edge_offsets[..., 3] * h_k
        
        boxes = torch.stack([L, T, R, B_edge], dim=-1)
        
        return conf_and_classes, center_offsets, boxes


class OMRDetector(nn.Module):
    def __init__(self, vit_backbone: nn.Module, num_classes: int, num_shapes: int = 5):
        super().__init__()
        self.backbone = vit_backbone
        
        # Extract the dimension from the backbone's patch embedding layer
        in_dim = self.backbone.patch_embed[1].out_features
        
        # Ensure the ViT doesn't pool the output, we need the patch sequence
        self.backbone.pool = "none" 
        self.backbone.mlp_head = nn.Identity()
        
        self.head = DFINEDenseHead(
            in_dim=in_dim, 
            num_classes=num_classes,
            num_shapes=num_shapes
        )

    def forward(self, patches: torch.Tensor, freqs: torch.Tensor):
        # 1. Extract patch features from ViT
        # Shape: (Batch, 1 + Num_Patches, Dim)
        features = self.backbone(patches, freqs)
        
        # 2. Drop the CLS token (index 0), keep only the spatial patches
        patch_tokens = features[:, 1:, :]
        
        # 3. Pass through the Dense D-FINE Head
        conf_and_classes, center_offsets, boxes = self.head(patch_tokens)
        
        return conf_and_classes, center_offsets, boxes
