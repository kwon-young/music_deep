import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .vit import vit_nano, vit_small, vit_base, BaseViT
from music_types import (
    Patches,
    DetectionOutput,
    SymbolOutput,
    LineOutput,
    Batch,
    NumPatches,
    PatchDim,
    EmbedDim,
    Embeddings,
    ClassLogits,
    BoundingBoxes,
    Keypoints,
    EdgeLogits,
    Coordinates,
    Dimensions,
    NumQueries,
    BoxDim,
    KeypointDim,
    CoordDim,
)


def get_2d_patch_centers(grid_h: int, grid_w: int, device: str) -> torch.Tensor:
    """
    Computes the base (x, y) centers for a full grid in Patch Units.
    """
    torch_device = torch.device(device)

    y_centers = torch.arange(grid_h, device=torch_device) + 0.5
    x_centers = torch.arange(grid_w, device=torch_device) + 0.5
    y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")

    # Shape: (Total_Patches, 2)
    return torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)


def compute_centers(
    embeddings: Embeddings[Batch, NumPatches, EmbedDim],
) -> torch.Tensor:
    """
    Computes and gathers (x, y) centers for the given patches in Patch Units.
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

    w: torch.Tensor

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


class SymbolHead(nn.Module):
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
        self.base_anchor_size = base_anchor_size

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

        bias_view = self.mlp[-1].bias.view(num_shapes, self.preds_per_shape)

        # Initialize classification bias for Focal Loss stability
        prior_prob = 0.01
        cls_bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(bias_view[:, :num_classes], cls_bias_init)

        # Initialize the bias for the width/height predictions to 0.0 for log-space
        nn.init.constant_(bias_view[:, num_classes + 2 : num_classes + 4], 0.0)

        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        B, P, _ = patch_tokens.shape

        raw_preds = self.mlp(patch_tokens)
        preds = raw_preds.reshape(B, P, self.num_shapes, self.preds_per_shape)

        classes = preds[..., : self.num_classes]
        center_offsets = preds[..., self.num_classes : self.num_classes + 2]

        # --- Dynamically predict shapes in LOG SPACE ---
        shapes_raw = preds[..., self.num_classes + 2 : self.num_classes + 4]
        shapes_raw = torch.clamp(shapes_raw, min=-10.0, max=10.0)
        shapes = self.base_anchor_size * torch.exp(shapes_raw)
        w_k, h_k = shapes[..., 0], shapes[..., 1]
        # ---------------------------------------

        edge_logits = preds[..., -4 * self.num_bins :]
        edge_logits = edge_logits.reshape(
            B, P, self.num_shapes, 4, self.num_bins
        )
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


class LineHead(nn.Module):
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
        self.base_anchor_size = base_anchor_size

        # C (classes) + 4 (raw_dx1, raw_dy1, raw_dx2, raw_dy2) + 4*N (edge bins)
        self.preds_per_shape = num_classes + 4 + (4 * self.num_bins)
        self.out_dim = num_shapes * self.preds_per_shape

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.out_dim),
        )

        bias_view = self.mlp[-1].bias.view(num_shapes, self.preds_per_shape)

        # Initialize classification bias for Focal Loss stability
        prior_prob = 0.01
        cls_bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(bias_view[:, :num_classes], cls_bias_init)

        # Initialize the bias for the raw_dirs to 0.0 (which maps to 0 distance)
        nn.init.constant_(bias_view[:, num_classes : num_classes + 4], 0.0)

        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, _ = patch_tokens.shape

        raw_preds = self.mlp(patch_tokens)
        preds = raw_preds.reshape(B, P, self.num_shapes, self.preds_per_shape)

        classes = preds[..., : self.num_classes]

        # --- Predict Signed Log Cartesian Directions ---
        raw_dirs = preds[..., self.num_classes : self.num_classes + 4]

        # Clamp to prevent exp() overflow
        raw_dirs = torch.clamp(raw_dirs, min=-10.0, max=10.0)

        # Apply Signed Log formula and scale by base anchor size
        base_dirs = torch.sign(raw_dirs) * (
            torch.exp(torch.abs(raw_dirs)) - 1.0
        )
        base_dirs = base_dirs * self.base_anchor_size

        # --- D-FINE Residuals ---
        edge_logits = preds[..., -4 * self.num_bins :]
        edge_logits = edge_logits.reshape(
            B, P, self.num_shapes, 4, self.num_bins
        )
        edge_probs = F.softmax(edge_logits, dim=-1)

        # Scale residuals by the magnitude of the base direction + base_anchor_size
        # This mirrors D-FINE box logic, allowing FGL to refine across the entire line length
        scale = torch.abs(base_dirs) + self.base_anchor_size
        res = self.weighting_fn(edge_probs) * scale

        # --- Final Endpoints (Relative to Patch Center) ---
        # No multiplication between direction and scale! Just addition.
        x1 = base_dirs[..., 0] + res[..., 0]
        y1 = base_dirs[..., 1] + res[..., 1]
        x2 = base_dirs[..., 2] + res[..., 2]
        y2 = base_dirs[..., 3] + res[..., 3]

        keypoints = torch.stack([x1, y1, x2, y2], dim=-1)

        return classes, base_dirs, keypoints, edge_logits


class OMRDetector(nn.Module):
    def __init__(
        self,
        vit_backbone: BaseViT,
        num_symbol_classes: int,
        num_line_classes: int,
        num_shapes: int = 5,
        base_anchor_size: float = 1.0,
    ):
        super().__init__()
        self.backbone = vit_backbone

        patch_embed_1 = self.backbone.patch_embed[1]
        in_dim = patch_embed_1.out_features

        self.symbol_head = SymbolHead(
            in_dim=in_dim,
            num_classes=num_symbol_classes,
            num_shapes=num_shapes,
            base_anchor_size=base_anchor_size,
        )

        self.line_head = LineHead(
            in_dim=in_dim,
            num_classes=num_line_classes,
            num_shapes=num_shapes,
            base_anchor_size=base_anchor_size,
        )

    def forward[B: Batch](
        self,
        patches: Patches[B, NumPatches, PatchDim],
    ) -> DetectionOutput[B, NumQueries, BoxDim, KeypointDim, CoordDim]:
        """
        Returns a DetectionOutput ready for DFINECriterion.
        """
        features = self.backbone(patches)

        # Compute centers dynamically based on the kept patches
        patch_centers = compute_centers(features)
        patch_tokens = features.data

        # --- Forward through both heads ---
        (
            sym_classes,
            sym_center_offsets,
            sym_shapes,
            sym_boxes,
            sym_edge_logits,
        ) = self.symbol_head(patch_tokens)
        (
            line_classes,
            line_base_dirs,
            line_keypoints,
            line_edge_logits,
        ) = self.line_head(patch_tokens)

        B, P, K, _ = sym_boxes.shape

        # Reshape patch_centers for broadcasting: (Batch, Num_Patches, 1, 2)
        patch_centers_expanded = patch_centers.unsqueeze(2)

        # --- Absolute Positioning for Symbols ---
        sym_absolute_centers = patch_centers_expanded + sym_center_offsets
        sym_boxes[..., 0] += patch_centers_expanded[..., 0]  # x1
        sym_boxes[..., 1] += patch_centers_expanded[..., 1]  # y1
        sym_boxes[..., 2] += patch_centers_expanded[..., 0]  # x2
        sym_boxes[..., 3] += patch_centers_expanded[..., 1]  # y2

        # --- Absolute Positioning for Lines ---
        # Lines are anchored exactly to the patch center (no offset)
        line_absolute_centers = patch_centers_expanded.expand(-1, -1, K, -1)
        line_keypoints[..., 0] += patch_centers_expanded[..., 0]  # x1
        line_keypoints[..., 1] += patch_centers_expanded[..., 1]  # y1
        line_keypoints[..., 2] += patch_centers_expanded[..., 0]  # x2
        line_keypoints[..., 3] += patch_centers_expanded[..., 1]  # y2

        # Flatten P and K dimensions into a single "num_queries" dimension
        # Note: We use .contiguous().flatten(1, 2) to safely merge P and K.
        # Slices of `preds` are non-contiguous and torch.compile's fake tensor
        # tracing strictly enforces stride checks, which causes reshape() to crash.
        return DetectionOutput(
            symbols=SymbolOutput(
                pred_logits=ClassLogits(sym_classes.contiguous().flatten(1, 2)),
                pred_boxes=BoundingBoxes(sym_boxes.contiguous().flatten(1, 2)),
                pred_edge_logits=EdgeLogits(
                    sym_edge_logits.contiguous().flatten(1, 2)
                ),
                absolute_centers=Coordinates(
                    sym_absolute_centers.contiguous().flatten(1, 2)
                ),
                learnable_shapes=Dimensions(
                    sym_shapes.contiguous().flatten(1, 2)
                ),
            ),
            lines=LineOutput(
                pred_logits=ClassLogits(
                    line_classes.contiguous().flatten(1, 2)
                ),
                pred_keypoints=Keypoints(
                    line_keypoints.contiguous().flatten(1, 2)
                ),
                pred_endpoint_logits=EdgeLogits(
                    line_edge_logits.contiguous().flatten(1, 2)
                ),
                absolute_centers=Coordinates(
                    line_absolute_centers.contiguous().flatten(1, 2)
                ),
                raw_directions=Coordinates(
                    line_base_dirs.contiguous().flatten(1, 2)
                ),
            ),
        )


def create_detector(
    backbone_size: str,
    patch_size: int | tuple[int, int],
    channels: int,
    use_sdpa: bool,
    num_symbol_classes: int,
    num_line_classes: int,
    num_shapes: int = 5,
    base_anchor_size: float = 1.0,
) -> OMRDetector:
    """
    Factory function to create an OMRDetector with the specified backbone size.
    """
    if backbone_size == "nano":
        vit_fn = vit_nano
    elif backbone_size == "small":
        vit_fn = vit_small
    elif backbone_size == "base":
        vit_fn = vit_base
    else:
        raise ValueError(f"Unknown backbone size: {backbone_size}")

    backbone = vit_fn(
        patch_size=patch_size,
        channels=channels,
        use_sdpa=use_sdpa,
    )

    return OMRDetector(
        backbone,
        num_symbol_classes=num_symbol_classes,
        num_line_classes=num_line_classes,
        num_shapes=num_shapes,
        base_anchor_size=base_anchor_size,
    )
