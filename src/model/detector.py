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
    PresenceLogits,
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


class BoxEmbedding(nn.Module):
    """Features of boxes relative to their patch centre (patch units): centre offset and log
    size, with sin/cos at geometric periods, plus the raw values."""

    freqs: torch.Tensor

    def __init__(self, num_freqs: int = 6, min_period: float = 0.25, max_period: float = 8.0) -> None:
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), num_freqs)
        self.register_buffer("freqs", 2 * math.pi / periods)
        self.dim = 4 * (2 * num_freqs + 1)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        centre = (boxes[..., :2] + boxes[..., 2:]) / 2
        size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-3).log()
        x = torch.cat([centre, size], dim=-1)
        angles = x.unsqueeze(-1) * self.freqs
        return torch.cat([angles.sin().flatten(-2), angles.cos().flatten(-2), x], dim=-1)


class SharedSlotClassifier(nn.Module):
    """One class classifier for all the slots of a patch: the class of the object in the slot's
    predicted box, from the patch's hidden vector and that box (detached), not from the slot
    index. The patch-set matcher gives an object to whichever slot fits its box, so per-slot
    class rows get contradictory targets and fall silent (Exp 043); a classifier shared across
    slots answers the same question for the same box whichever slot carries it.

    ``linear`` reads the hidden vector alone; ``inter_h`` / ``inter_box`` / ``out`` add the
    box-dependent part (``out`` zero-initialised, so at init every slot gets ``linear``'s scores).
    """

    def __init__(self, dim: int, num_classes: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or dim
        self.box = BoxEmbedding()
        self.linear = nn.Linear(dim, num_classes)
        self.inter_h = nn.Linear(dim, hidden)
        self.inter_box = nn.Linear(self.box.dim, hidden)
        self.out = nn.Linear(hidden, num_classes)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.constant_(self.linear.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, hidden: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """``hidden`` (B, P, D), ``boxes`` (B, P, K, 4) relative to the patch centre -> (B, P, K, C)."""
        z = F.gelu(self.inter_h(hidden).unsqueeze(2) + self.inter_box(self.box(boxes)))
        return self.linear(hidden).unsqueeze(2) + self.out(z)


class SymbolHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_shapes: int = 5,
        reg_max: int = 32,
        base_anchor_size: float = 1.0,
        presence: bool = False,
        presence_prior: float = 0.01,
        shared_class: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_shapes = num_shapes
        self.reg_max = reg_max
        self.num_bins = reg_max + 1
        self.base_anchor_size = base_anchor_size
        self.presence = presence
        self.presence_prior = presence_prior
        # the per-slot class columns stay in the output layer (same state layout) but are unused
        self.classifier = SharedSlotClassifier(in_dim, num_classes) if shared_class else None

        # C (classes) + 4 (cx, cy, w, h) + 4*N (edge bins) [+ 1 presence, last]
        self.geometry_end = num_classes + 4 + (4 * self.num_bins)
        self.preds_per_shape = self.geometry_end + int(presence)
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
        if presence:
            nn.init.constant_(bias_view[:, -1], math.log(presence_prior / (1 - presence_prior)))

        # Initialize the bias for the width/height predictions to 0.0 for log-space
        nn.init.constant_(bias_view[:, num_classes + 2 : num_classes + 4], 0.0)

        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        B, P, _ = patch_tokens.shape

        hidden = self.mlp[2](self.mlp[1](self.mlp[0](patch_tokens)))
        raw_preds = self.mlp[3](hidden)
        preds = raw_preds.reshape(B, P, self.num_shapes, self.preds_per_shape)

        classes = preds[..., : self.num_classes]
        center_offsets = preds[..., self.num_classes : self.num_classes + 2]

        # --- Dynamically predict shapes in LOG SPACE ---
        shapes_raw = preds[..., self.num_classes + 2 : self.num_classes + 4]
        shapes_raw = torch.clamp(shapes_raw, min=-10.0, max=10.0)
        shapes = self.base_anchor_size * torch.exp(shapes_raw)
        w_k, h_k = shapes[..., 0], shapes[..., 1]
        # ---------------------------------------

        edge_logits = preds[..., self.num_classes + 4 : self.geometry_end]
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
        if self.classifier is not None:
            classes = self.classifier(hidden, boxes.detach())

        presence = preds[..., -1] if self.presence else None
        return classes, center_offsets, shapes, boxes, edge_logits, presence


class LineHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_shapes: int = 5,
        reg_max: int = 32,
        base_anchor_size: float = 1.0,
        presence: bool = False,
        presence_prior: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_shapes = num_shapes
        self.reg_max = reg_max
        self.num_bins = reg_max + 1
        self.base_anchor_size = base_anchor_size
        self.presence = presence
        self.presence_prior = presence_prior

        # C (classes) + 4 (raw_dx1, raw_dy1, raw_dx2, raw_dy2) + 4*N (edge bins) [+ 1 presence]
        self.geometry_end = num_classes + 4 + (4 * self.num_bins)
        self.preds_per_shape = self.geometry_end + int(presence)
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
        if presence:
            nn.init.constant_(bias_view[:, -1], math.log(presence_prior / (1 - presence_prior)))

        # Initialize the bias for the raw_dirs to 0.0 (which maps to 0 distance)
        nn.init.constant_(bias_view[:, num_classes : num_classes + 4], 0.0)

        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        edge_logits = preds[..., self.num_classes + 4 : self.geometry_end]
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

        presence = preds[..., -1] if self.presence else None
        return classes, base_dirs, keypoints, edge_logits, presence


class OMRDetector(nn.Module):
    def __init__(
        self,
        vit_backbone: BaseViT,
        num_symbol_classes: int,
        num_line_classes: int,
        num_shapes: int = 5,
        base_anchor_size: float = 1.0,
        presence: bool = False,
        symbol_presence_prior: float = 0.08,
        line_presence_prior: float = 0.35,
        symbol_shared_class: bool = False,
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
            presence=presence,
            presence_prior=symbol_presence_prior,
            shared_class=symbol_shared_class,
        )

        self.line_head = LineHead(
            in_dim=in_dim,
            num_classes=num_line_classes,
            num_shapes=num_shapes,
            base_anchor_size=base_anchor_size,
            presence=presence,
            presence_prior=line_presence_prior,
        )

    def forward[B: Batch](
        self,
        patches: Patches[B, NumPatches, PatchDim],
    ) -> DetectionOutput[B, NumQueries, BoxDim, KeypointDim, CoordDim]:
        """
        Returns a DetectionOutput ready for DFINECriterion.
        """
        return self.forward_with_tokens(patches)[0]

    def forward_with_tokens[B: Batch](
        self,
        patches: Patches[B, NumPatches, PatchDim],
    ) -> tuple[
        DetectionOutput[B, NumQueries, BoxDim, KeypointDim, CoordDim],
        Embeddings[B, NumPatches, EmbedDim],
    ]:
        """The detection outputs and the final (normed) tokens of the kept patches they were
        predicted from, which the Exp 035 endpoint refiner reads."""
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
            sym_presence,
        ) = self.symbol_head(patch_tokens)
        (
            line_classes,
            line_base_dirs,
            line_keypoints,
            line_edge_logits,
            line_presence,
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
                pred_presence=None
                if sym_presence is None
                else PresenceLogits(sym_presence.contiguous().flatten(1, 2)),
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
                pred_presence=None
                if line_presence is None
                else PresenceLogits(line_presence.contiguous().flatten(1, 2)),
            ),
        ), features


class DetectorWithTokens(nn.Module):
    """Makes ``OMRDetector.forward_with_tokens`` the ``forward`` that DDP and torch.compile
    wrap, without changing the detector's own state-dict keys."""

    def __init__(self, detector: OMRDetector) -> None:
        super().__init__()
        self.detector = detector

    def forward[B: Batch](
        self,
        patches: Patches[B, NumPatches, PatchDim],
    ) -> tuple[
        DetectionOutput[B, NumQueries, BoxDim, KeypointDim, CoordDim],
        Embeddings[B, NumPatches, EmbedDim],
    ]:
        return self.detector.forward_with_tokens(patches)


def create_detector(
    backbone_size: str,
    patch_size: int | tuple[int, int],
    channels: int,
    use_sdpa: bool,
    num_symbol_classes: int,
    num_line_classes: int,
    num_shapes: int = 5,
    base_anchor_size: float = 1.0,
    presence: bool = False,
    symbol_presence_prior: float = 0.08,
    line_presence_prior: float = 0.35,
    symbol_shared_class: bool = False,
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
        presence=presence,
        symbol_presence_prior=symbol_presence_prior,
        line_presence_prior=line_presence_prior,
        symbol_shared_class=symbol_shared_class,
    )


def _head_layers(model: OMRDetector) -> tuple[tuple[str, SymbolHead | LineHead], ...]:
    return (("symbol_head.mlp.3", model.symbol_head), ("line_head.mlp.3", model.line_head))


def heads_need_remap(state: dict[str, torch.Tensor], model: OMRDetector) -> bool:
    """Whether the head output layers of ``state`` have another slot count or presence layout
    than ``model`` (e.g. the K = 5 Exp 037 checkpoint for a K = 8 patch-set detector)."""
    own = model.state_dict()
    return any(state[f"{key}.weight"].shape != own[f"{key}.weight"].shape for key, _ in _head_layers(model))


def remap_head_state(state: dict[str, torch.Tensor], model: OMRDetector,
                     copy_class_rows: bool = False) -> dict[str, torch.Tensor]:
    """``state`` with its head output layers rebuilt for ``model``'s slot count and presence.

    Slot k < K_old keeps its rows (classes, geometry, edge bins and presence when the source
    has one), so these slots reproduce the source outputs exactly. A new slot (k >= K_old)
    copies the geometry and edge-bin rows of slot k mod K_old and starts empty: zero class
    weights with the focal prior bias. Presence rows that the source lacks get zero weights
    and the bias logit(head.presence_prior): every slot starts at the head's positive base rate
    (patch-set smoke: lines ~35% of slots, symbols ~8%), so Adam does not spend the whole run
    moving a bias by ~4 logits at <= lr per step.
    """
    prior = -math.log((1 - 0.01) / 0.01)
    out = dict(state)
    for key, head in _head_layers(model):
        weight, bias = state[f"{key}.weight"], state[f"{key}.bias"]
        rows, dim = weight.shape
        old_presence = rows % head.geometry_end != 0
        old_pps = head.geometry_end + int(old_presence)
        if rows % old_pps != 0:
            raise ValueError(f"{key}: {rows} rows do not split into slots of {old_pps}")
        old_k = rows // old_pps
        old_w = weight.view(old_k, old_pps, dim)
        old_b = bias.view(old_k, old_pps)
        new_k, c, end = head.num_shapes, head.num_classes, head.geometry_end
        new_w = weight.new_zeros(new_k, head.preds_per_shape, dim)
        new_b = bias.new_zeros(new_k, head.preds_per_shape)
        for k in range(new_k):
            src = k % old_k
            new_w[k, c:end] = old_w[src, c:end]
            new_b[k, c:end] = old_b[src, c:end]
            if k < old_k or copy_class_rows:
                # copy_class_rows: a new slot inherits the class rows of its geometry source, so
                # it starts as a second specialist of the same classes (slot roles, Exp 043)
                new_w[k, :c] = old_w[src, :c]
                new_b[k, :c] = old_b[src, :c]
            else:
                new_b[k, :c] = prior
            if head.presence:
                if old_presence and k < old_k:
                    new_w[k, -1] = old_w[k, -1]
                    new_b[k, -1] = old_b[k, -1]
                else:
                    new_b[k, -1] = math.log(head.presence_prior / (1 - head.presence_prior))
        out[f"{key}.weight"] = new_w.reshape(-1, dim)
        out[f"{key}.bias"] = new_b.reshape(-1)
    return out


def shared_classifier_state(state: dict[str, torch.Tensor], model: OMRDetector,
                            owners: dict[int, int] | None) -> dict[str, torch.Tensor]:
    """Initial ``symbol_head.classifier`` weights for a ``state`` that has per-slot class rows
    only: the class-c row of ``linear`` is the class-c row of class c's specialist slot in
    ``state`` (``owners``: class index -> slot of the one-to-one model, Exp 043 slot roles), the
    mean over the source's slots for classes without an owner; the box-dependent part keeps the
    model's initialisation (zero output), so every slot starts with its specialist's scores."""
    head = model.symbol_head
    assert head.classifier is not None
    weight, bias = state["symbol_head.mlp.3.weight"], state["symbol_head.mlp.3.bias"]
    rows, dim = weight.shape
    old_pps = head.geometry_end + int(rows % head.geometry_end != 0)
    old_k = rows // old_pps
    c = head.num_classes
    w = weight.view(old_k, old_pps, dim)[:, :c]  # (K_old, C, D)
    b = bias.view(old_k, old_pps)[:, :c]
    lin_w, lin_b = w.mean(dim=0).clone(), b.mean(dim=0).clone()
    for cls, slot in (owners or {}).items():
        if 0 <= cls < c and 0 <= slot < old_k:
            lin_w[cls], lin_b[cls] = w[slot, cls], b[slot, cls]
    out = {f"symbol_head.classifier.{k}": v.detach().clone() for k, v in head.classifier.state_dict().items()}
    out["symbol_head.classifier.linear.weight"] = lin_w
    out["symbol_head.classifier.linear.bias"] = lin_b
    return out


def has_shared_class(state: dict[str, torch.Tensor]) -> bool:
    return "symbol_head.classifier.linear.weight" in state


def load_detector_state(model: OMRDetector, state: dict[str, torch.Tensor], copy_class_rows: bool = False,
                        owners: dict[int, int] | None = None) -> bool:
    """Load ``state`` strictly, remapping the head output layers first if their layout differs
    and building the shared symbol classifier if ``model`` has one and ``state`` does not.
    Returns True when the parameters changed shape or set (the optimizer state of ``state`` then
    no longer fits)."""
    remap = heads_need_remap(state, model)
    new_classifier = model.symbol_head.classifier is not None and not has_shared_class(state)
    loaded = remap_head_state(state, model, copy_class_rows) if remap else dict(state)
    if new_classifier:
        loaded |= shared_classifier_state(state, model, owners)
    model.load_state_dict(loaded, strict=True)
    return remap or new_classifier


def head_layout(state: dict[str, torch.Tensor], num_symbol_classes: int, reg_max: int = 32) -> tuple[int, bool]:
    """(slots per patch, presence) of a detector state dict, read from its symbol head."""
    end = num_symbol_classes + 4 + 4 * (reg_max + 1)
    rows = state["symbol_head.mlp.3.weight"].shape[0]
    return (rows // end, False) if rows % end == 0 else (rows // (end + 1), True)
