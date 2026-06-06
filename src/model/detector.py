import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from functools import lru_cache
from music_types import (
    Patches,
    DetectionOutput,
    Batch,
    NumPatches,
    PatchDim,
    EmbedDim,
    Embeddings,
    ClassLogits,
    BoundingBoxes,
    EdgeLogits,
    Coordinates,
    Dimensions,
    NumQueries,
    BoxDim,
    CoordDim,
)


@lru_cache(maxsize=32)
def get_2d_patch_centers(grid_h: int, grid_w: int, device: str) -> torch.Tensor:
    """
    Computes the base normalized (x, y) centers for a full grid.
    Cached to avoid recomputing meshgrids on every forward pass.
    """
    torch_device = torch.device(device)

    y_centers = (torch.arange(grid_h, device=torch_device) + 0.5) / grid_h
    x_centers = (torch.arange(grid_w, device=torch_device) + 0.5) / grid_w
    y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")

    # Shape: (Total_Patches, 2)
    return torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)


def compute_centers(
    embeddings: Embeddings[Batch, NumPatches, EmbedDim],
) -> torch.Tensor:
    """
    Computes and gathers normalized (x, y) centers for the given patches.
    Uses the indices to ensure centers match even if patches were dropped.
    """
    c, h, w = embeddings.image_shape
    ph, pw = embeddings.patch_size
    grid_h, grid_w = h // ph, w // pw

    # Get the cached base grid of centers. Shape: (Total_Patches, 2)
    base_centers = get_2d_patch_centers(
        grid_h, grid_w, device=str(embeddings.data.device)
    )

    # Expand to (Batch, Total_Patches, 2)
    centers = base_centers.unsqueeze(0).expand(embeddings.batch_size, -1, -1)

    # Gather only the centers for the kept patches using the indices
    kept_centers = torch.gather(
        centers, 1, embeddings.indices.unsqueeze(-1).expand(-1, -1, 2)
    )

    return kept_centers


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
        base_anchor_size: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_shapes = num_shapes
        self.reg_max = reg_max
        self.num_bins = reg_max + 1

        # C (classes) + 4 (cx, cy, w, h) + 4*N (edge bins)
        self.preds_per_shape = num_classes + 4 + (4 * self.num_bins)
        self.out_dim = num_shapes * self.preds_per_shape

        # The Dense MLP applied to each patch token
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_dim),
        )

        # Initialize the bias for the width/height predictions so they start at base_anchor_size
        init_val = math.log(math.exp(base_anchor_size) - 1)
        bias_view = self.mlp[-1].bias.view(num_shapes, self.preds_per_shape)
        nn.init.constant_(bias_view[:, num_classes + 2 : num_classes + 4], init_val)

        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, _ = patch_tokens.shape

        raw_preds = self.mlp(patch_tokens)
        preds = raw_preds.view(B, P, self.num_shapes, self.preds_per_shape)

        classes = preds[..., : self.num_classes]
        center_offsets = preds[..., self.num_classes : self.num_classes + 2]
        
        # --- Dynamically predict shapes ---
        shapes_raw = preds[..., self.num_classes + 2 : self.num_classes + 4]
        shapes = F.softplus(shapes_raw) # Ensure strictly positive
        w_k, h_k = shapes[..., 0], shapes[..., 1]
        # ---------------------------------------

        edge_logits = preds[..., -4 * self.num_bins :]
        edge_logits = edge_logits.view(B, P, self.num_shapes, 4, self.num_bins)
        edge_probs = F.softmax(edge_logits, dim=-1)

        # Shape: (B, P, K, 4) - Residuals in range [-a, a]
        relative_edge_offsets = self.weighting_fn(edge_probs)

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

        return classes, center_offsets, shapes, boxes, edge_logits


class OMRDetector(nn.Module):
    def __init__(
        self, 
        vit_backbone: nn.Module, 
        num_classes: int, 
        num_shapes: int = 5,
        base_anchor_size: float = 1.0,
    ):
        super().__init__()
        self.backbone = vit_backbone

        in_dim = self.backbone.patch_embed[1].out_features

        self.head = DFINEDenseHead(
            in_dim=in_dim, 
            num_classes=num_classes, 
            num_shapes=num_shapes,
            base_anchor_size=base_anchor_size,
        )

    def forward[B: Batch](
        self,
        patches: Patches[B, NumPatches, PatchDim],
    ) -> DetectionOutput[B, NumQueries, BoxDim, CoordDim]:
        """
        Returns a DetectionOutput ready for DFINECriterion.
        """
        features = self.backbone(patches)

        # Compute centers dynamically based on the kept patches
        patch_centers = compute_centers(features)

        # Pass the actual tensor data to the dense head
        patch_tokens = features.data
        classes, center_offsets, shapes, boxes, edge_logits = self.head(patch_tokens)

        B, P, K, _ = boxes.shape

        # --- Scale from Patch Units to Image Units [0, 1] ---
        _, h, w = features.image_shape
        ph, pw = features.patch_size
        grid_h, grid_w = h // ph, w // pw
        
        scale_xy = torch.tensor(
            [1.0 / grid_w, 1.0 / grid_h], dtype=boxes.dtype, device=boxes.device
        ).view(1, 1, 1, 2)
        
        center_offsets = center_offsets * scale_xy
        
        scale_xyxy = scale_xy.repeat(1, 1, 1, 2)
        boxes = boxes * scale_xyxy
        
        # Scale dynamic shapes to Image Units for the FGL loss
        expanded_shapes = shapes * scale_xy
        # ---------------------------------------------------------

        # Reshape patch_centers for broadcasting: (Batch, Num_Patches, 1, 2)
        patch_centers_expanded = patch_centers.unsqueeze(2)

        # Add absolute patch centers to the predicted center offsets
        absolute_centers = patch_centers_expanded + center_offsets

        # Shift the boxes to absolute coordinates
        boxes[..., 0] += patch_centers_expanded[..., 0]  # x1
        boxes[..., 1] += patch_centers_expanded[..., 1]  # y1
        boxes[..., 2] += patch_centers_expanded[..., 0]  # x2
        boxes[..., 3] += patch_centers_expanded[..., 1]  # y2

        # Flatten P and K dimensions into a single "num_queries" dimension
        # and return the dataclass expected by the criterion
        return DetectionOutput(
            pred_logits=ClassLogits(classes.view(B, P * K, -1)),
            pred_boxes=BoundingBoxes(boxes.view(B, P * K, 4)),
            pred_edge_logits=EdgeLogits(edge_logits.view(B, P * K, 4, -1)),
            absolute_centers=Coordinates(absolute_centers.view(B, P * K, 2)),
            learnable_shapes=Dimensions(expanded_shapes.reshape(B, P * K, 2)),
        )
