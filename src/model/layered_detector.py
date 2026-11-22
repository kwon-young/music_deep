from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import (
    DFINEWeightingFunction,
    LineHead,
    SymbolHead,
    compute_centers,
    create_backbone,
)
from .line_refiner import decode_endpoints
from .vit import BaseViT
from music_types import (
    Batch,
    BoxDim,
    ClassLogits,
    Coordinates,
    CoordDim,
    DetectionOutput,
    Dimensions,
    EdgeLogits,
    KeypointDim,
    Keypoints,
    LayeredDetectionOutput,
    LineOutput,
    NumPatches,
    NumQueries,
    PatchDim,
    Patches,
    BoundingBoxes,
    SymbolOutput,
)


def signed_log(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def decode_symbol_boxes(
    centers: torch.Tensor,
    shapes: torch.Tensor,
    edge_logits: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Same decoding as SymbolHead (D-FINE `distance2bbox`): edge = center -+ shape * (1/2 + E[W(n)])."""
    residual = (F.softmax(edge_logits, dim=-1) * bins).sum(dim=-1)
    direction = residual.new_tensor([-1.0, -1.0, 1.0, 1.0])
    return torch.cat([centers, centers], dim=-1) + direction * torch.cat(
        [shapes, shapes], dim=-1
    ) * (0.5 + residual)


def linear_at(sequential: nn.Sequential, index: int) -> nn.Linear:
    layer = sequential[index]
    assert isinstance(layer, nn.Linear)
    return layer


def layer_norm_at(sequential: nn.Sequential, index: int) -> nn.LayerNorm:
    layer = sequential[index]
    assert isinstance(layer, nn.LayerNorm)
    return layer


def dense_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """The layout of SymbolHead / LineHead, so their weights can be copied."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
    )


class LQE(nn.Module):
    """D-FINE Location Quality Estimator: a sharp edge distribution raises the score."""

    def __init__(self, k: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(4 * (k + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(linear_at(self.net, -1).weight)
        nn.init.zeros_(linear_at(self.net, -1).bias)

    def forward(
        self, class_logits: torch.Tensor, edge_logits: torch.Tensor
    ) -> torch.Tensor:
        prob = F.softmax(edge_logits, dim=-1)
        top = prob.topk(self.k, dim=-1).values
        stat = torch.cat([top, top.mean(dim=-1, keepdim=True)], dim=-1)
        return class_logits + self.net(stat.flatten(-2))


class SlotBoxConditioner(nn.Module):
    """D-FINE `query_pos_head` for a token that carries 2 x K slots.

    Slot k is always the k-th chunk of the concatenation, so the MLP keeps slot
    identity. The last layer is zero-initialized: at initialization the refinement
    blocks compute exactly what the backbone computed without conditioning.
    """

    def __init__(self, dim: int, num_shapes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_shapes * 4, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )
        nn.init.zeros_(linear_at(self.mlp, -1).weight)
        nn.init.zeros_(linear_at(self.mlp, -1).bias)

    def forward(
        self,
        patch_centers: torch.Tensor,
        symbol_boxes: torch.Tensor,
        line_keypoints: torch.Tensor,
    ) -> torch.Tensor:
        origin = patch_centers.repeat(1, 1, 1, 2)
        geometry = torch.cat(
            [symbol_boxes - origin, line_keypoints - origin], dim=2
        ).detach()
        return self.mlp(signed_log(geometry).flatten(2)).clamp(-10.0, 10.0)


class RefineHead(nn.Module):
    """Score head and FDR head of one refinement layer (D-FINE score_head[i] / bbox_head[i])."""

    def __init__(
        self, dim: int, num_classes: int, num_shapes: int, num_bins: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_shapes = num_shapes
        self.num_bins = num_bins
        self.score = dense_mlp(dim, num_shapes * num_classes)
        self.delta = dense_mlp(dim, num_shapes * 4 * num_bins)
        nn.init.zeros_(linear_at(self.delta, -1).weight)
        nn.init.zeros_(linear_at(self.delta, -1).bias)
        self.lqe = LQE()

    def forward(
        self,
        tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
        previous_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patches, _ = tokens.shape
        delta = self.delta(tokens + previous_tokens.detach())
        edge_logits = previous_logits + delta.view(
            batch, patches, self.num_shapes, 4, self.num_bins
        )
        classes = self.score(tokens).view(
            batch, patches, self.num_shapes, self.num_classes
        )
        return self.lqe(classes, edge_logits), edge_logits

    @torch.no_grad()
    def copy_classifier(self, head_mlp: nn.Sequential, preds_per_shape: int) -> None:
        """Starts the score head from the classification rows of a SymbolHead/LineHead."""
        linear_at(self.score, 0).load_state_dict(linear_at(head_mlp, 0).state_dict())
        layer_norm_at(self.score, 2).load_state_dict(
            layer_norm_at(head_mlp, 2).state_dict()
        )
        source, target = linear_at(head_mlp, 3), linear_at(self.score, 3)
        dim = source.weight.shape[-1]
        weight = source.weight.view(self.num_shapes, preds_per_shape, dim)
        bias = source.bias.view(self.num_shapes, preds_per_shape)
        target.weight.copy_(weight[:, : self.num_classes].reshape(-1, dim))
        target.bias.copy_(bias[:, : self.num_classes].reshape(-1))


class MultiLayerOMRDetector(nn.Module):
    """D-FINE Fine-grained Distribution Refinement over the last ViT blocks.

    Tap 0 (block `depth - 1 - num_refine_layers`) is D-FINE decoder layer 0: the
    dense heads give the coarse prediction (`pre`, own loss) and the first edge
    distribution. Its detached geometry is the frame of every later layer. Each
    following block is a decoder layer: its attention sees the previous boxes
    (queries and keys only), and its head adds residual logits to the previous
    distribution.
    """

    def __init__(
        self,
        backbone: BaseViT,
        num_symbol_classes: int,
        num_line_classes: int,
        num_shapes: int = 5,
        base_anchor_size: float = 1.0,
        num_refine_layers: int = 2,
        reg_max: int = 32,
        dfine_outer_bins: bool = True,
    ):
        super().__init__()
        if not 0 < num_refine_layers < backbone.depth:
            raise ValueError(
                f"num_refine_layers={num_refine_layers} must be in [1, {backbone.depth})"
            )
        self.backbone = backbone
        self.num_shapes = num_shapes
        self.base_anchor_size = base_anchor_size
        self.num_refine_layers = num_refine_layers
        self.first_tap = backbone.depth - 1 - num_refine_layers
        dim = backbone.dim
        num_bins = reg_max + 1

        self.symbol_head = SymbolHead(
            dim,
            num_symbol_classes,
            num_shapes,
            reg_max,
            base_anchor_size,
            dfine_outer_bins,
        )
        self.line_head = LineHead(
            dim,
            num_line_classes,
            num_shapes,
            reg_max,
            base_anchor_size,
            dfine_outer_bins,
        )
        self.tap_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_refine_layers)]
        )
        self.symbol_lqe = LQE()
        self.line_lqe = LQE()
        self.conditioner = SlotBoxConditioner(dim, num_shapes)
        self.symbol_refiners = nn.ModuleList(
            [
                RefineHead(dim, num_symbol_classes, num_shapes, num_bins)
                for _ in range(num_refine_layers)
            ]
        )
        self.line_refiners = nn.ModuleList(
            [
                RefineHead(dim, num_line_classes, num_shapes, num_bins)
                for _ in range(num_refine_layers)
            ]
        )
        self.weighting_fn = DFINEWeightingFunction(
            reg_max=reg_max, dfine_outer_bins=dfine_outer_bins
        )
        self.initialize_refinement()

    def tap_norm(self, tap: int) -> nn.LayerNorm:
        norm = (
            self.tap_norms[tap]
            if tap < self.num_refine_layers
            else self.backbone.transformer.norm
        )
        assert isinstance(norm, nn.LayerNorm)
        return norm

    def refiner(self, modules: nn.ModuleList, index: int) -> RefineHead:
        head = modules[index]
        assert isinstance(head, RefineHead)
        return head

    @torch.no_grad()
    def initialize_refinement(self) -> None:
        """Taps start from the final norm, refinement scores from the dense classifiers."""
        for tap in range(self.num_refine_layers):
            self.tap_norm(tap).load_state_dict(
                self.tap_norm(self.num_refine_layers).state_dict()
            )
        for index in range(self.num_refine_layers):
            self.refiner(self.symbol_refiners, index).copy_classifier(
                self.symbol_head.mlp, self.symbol_head.preds_per_shape
            )
            self.refiner(self.line_refiners, index).copy_classifier(
                self.line_head.mlp, self.line_head.preds_per_shape
            )

    def load_single_layer_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Initializes from an OMRDetector checkpoint (backbone + dense heads).

        The bin positions are not loaded: they are defined by `dfine_outer_bins`.
        """
        kept = {k: v for k, v in state.items() if not k.endswith("weighting_fn.w")}
        result = self.load_state_dict(kept, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected keys: {result.unexpected_keys}")
        fresh = (
            "tap_norms.",
            "symbol_lqe.",
            "line_lqe.",
            "conditioner.",
            "symbol_refiners.",
            "line_refiners.",
            "weighting_fn.",
            "symbol_head.weighting_fn.",
            "line_head.weighting_fn.",
        )
        missing = [k for k in result.missing_keys if not k.startswith(fresh)]
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        self.initialize_refinement()

    def forward[B: Batch](
        self, patches: Patches[B, NumPatches, PatchDim]
    ) -> LayeredDetectionOutput[B, NumQueries, BoxDim, KeypointDim, CoordDim]:
        transformer = self.backbone.transformer
        bins = self.weighting_fn.w
        anchor = self.base_anchor_size

        x, freqs = self.backbone.embed(patches)
        x = transformer.run_blocks(x, freqs, 0, self.first_tap + 1)
        tokens = self.tap_norm(0)(x)
        patch_centers = compute_centers(patches).unsqueeze(2)

        symbol_classes, offsets, shapes, _, symbol_logits = self.symbol_head(
            tokens
        )
        symbol_centers = patch_centers + offsets
        pre_boxes = torch.cat(
            [symbol_centers - shapes / 2, symbol_centers + shapes / 2], dim=-1
        )
        line_classes, base_dirs, _, line_logits = self.line_head(tokens)
        line_centers = patch_centers.expand(-1, -1, self.num_shapes, -1)
        pre_keypoints = line_centers.repeat(1, 1, 1, 2) + base_dirs

        symbol_centers, shapes = symbol_centers.detach(), shapes.detach()
        base_dirs = base_dirs.detach()

        def decode(
            symbol_logits: torch.Tensor, line_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                decode_symbol_boxes(symbol_centers, shapes, symbol_logits, bins),
                decode_endpoints(line_centers, base_dirs, line_logits, bins, anchor),
            )

        def pack(
            symbol_classes: torch.Tensor,
            boxes: torch.Tensor,
            symbol_logits: torch.Tensor,
            line_classes: torch.Tensor,
            keypoints: torch.Tensor,
            line_logits: torch.Tensor,
        ) -> DetectionOutput[B, NumQueries, BoxDim, KeypointDim, CoordDim]:
            def flat(t: torch.Tensor) -> torch.Tensor:
                return t.contiguous().flatten(1, 2)

            return DetectionOutput(
                symbols=SymbolOutput(
                    pred_logits=ClassLogits(flat(symbol_classes)),
                    pred_boxes=BoundingBoxes(flat(boxes)),
                    pred_edge_logits=EdgeLogits(flat(symbol_logits)),
                    absolute_centers=Coordinates(flat(symbol_centers)),
                    learnable_shapes=Dimensions(flat(shapes)),
                ),
                lines=LineOutput(
                    pred_logits=ClassLogits(flat(line_classes)),
                    pred_keypoints=Keypoints(flat(keypoints)),
                    pred_endpoint_logits=EdgeLogits(flat(line_logits)),
                    absolute_centers=Coordinates(flat(line_centers)),
                    raw_directions=Coordinates(flat(base_dirs)),
                ),
            )

        pre = pack(
            symbol_classes,
            pre_boxes,
            symbol_logits,
            line_classes,
            pre_keypoints,
            line_logits,
        )
        boxes, keypoints = decode(symbol_logits, line_logits)
        layers = [
            pack(
                self.symbol_lqe(symbol_classes, symbol_logits),
                boxes,
                symbol_logits,
                self.line_lqe(line_classes, line_logits),
                keypoints,
                line_logits,
            )
        ]

        for index in range(self.num_refine_layers):
            block = self.first_tap + 1 + index
            pos = self.conditioner(patch_centers, boxes, keypoints)
            x = transformer.run_blocks(x, freqs, block, block + 1, pos=pos)
            next_tokens = self.tap_norm(index + 1)(x)
            symbol_classes, symbol_logits = self.refiner(
                self.symbol_refiners, index
            )(next_tokens, tokens, symbol_logits)
            line_classes, line_logits = self.refiner(self.line_refiners, index)(
                next_tokens, tokens, line_logits
            )
            tokens = next_tokens
            boxes, keypoints = decode(symbol_logits, line_logits)
            layers.append(
                pack(
                    symbol_classes,
                    boxes,
                    symbol_logits,
                    line_classes,
                    keypoints,
                    line_logits,
                )
            )

        return LayeredDetectionOutput(pre=pre, layers=layers)


def create_layered_detector(
    backbone_size: str,
    patch_size: int | tuple[int, int],
    channels: int,
    use_sdpa: bool,
    num_symbol_classes: int,
    num_line_classes: int,
    num_shapes: int = 5,
    base_anchor_size: float = 1.0,
    num_refine_layers: int = 2,
    dfine_outer_bins: bool = True,
) -> MultiLayerOMRDetector:
    return MultiLayerOMRDetector(
        create_backbone(backbone_size, patch_size, channels, use_sdpa),
        num_symbol_classes=num_symbol_classes,
        num_line_classes=num_line_classes,
        num_shapes=num_shapes,
        base_anchor_size=base_anchor_size,
        num_refine_layers=num_refine_layers,
        dfine_outer_bins=dfine_outer_bins,
    )
