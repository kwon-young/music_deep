import unittest

import torch

from model.criterion import LineRefinerCriterion, line_fgl_loss
from model.detector import OMRDetector, create_detector
from model.line_refiner import (
    EndpointDeformableAttention,
    LineRefiner,
    LineRefinerConfig,
    decode_endpoints,
    dense_token_map,
    gather_line_queries,
)
from music_types import (
    Batch,
    ClassLabels,
    EmbedDim,
    Embeddings,
    KeypointDim,
    Keypoints,
    NumLineClasses,
    NumPatches,
    NumQueries,
    PatchDim,
    Patches,
    PatchUnit,
    TopLeft,
    X1Y1X2Y2,
)

NUM_SHAPES = 2
PATCH = 16
GRID_H, GRID_W = 4, 6


def make_detector() -> OMRDetector:
    return create_detector(
        backbone_size="nano",
        patch_size=PATCH,
        channels=3,
        use_sdpa=False,
        num_symbol_classes=3,
        num_line_classes=2,
        num_shapes=NUM_SHAPES,
    ).eval()


def make_features(
    detector: OMRDetector,
) -> Embeddings[Batch, NumPatches, EmbedDim]:
    kept = torch.tensor([[0, 1, 3, 4, 7, 8, 9, 12, 13, 15, 17, 18, 20, 22, 23]])
    patches: Patches[Batch, NumPatches, PatchDim] = Embeddings(
        data=torch.randn(1, kept.shape[1], 3 * PATCH * PATCH),
        indices=kept,
        image_shape=(3, GRID_H * PATCH, GRID_W * PATCH),
        patch_size=(PATCH, PATCH),
    )
    with torch.no_grad():
        return detector.backbone(patches)


def make_config(num_layers: int = 2) -> LineRefinerConfig:
    return LineRefinerConfig(
        dim=192, num_layers=num_layers, heads=6, points=4, num_shapes=NUM_SHAPES
    )


class TestLineRefiner(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.detector = make_detector()
        self.features = make_features(self.detector)
        with torch.no_grad():
            self.lines = self.detector.predict_lines(self.features)
        num_queries = self.lines.pred_keypoints.data.shape[1]
        self.query_indices = torch.randperm(num_queries)[:10].unsqueeze(0)
        self.queries = gather_line_queries(
            self.features, self.lines, self.query_indices, NUM_SHAPES
        )
        self.stage0 = self.lines.pred_keypoints.data[0, self.query_indices[0]]

    def test_decode_matches_line_head(self) -> None:
        decoded = decode_endpoints(
            self.queries.absolute_centers.data,
            self.queries.raw_directions.data,
            self.queries.endpoint_logits.data,
            self.detector.line_head.weighting_fn.w,
            self.detector.line_head.base_anchor_size,
        )
        torch.testing.assert_close(decoded[0], self.stage0)

    def test_dense_token_map_places_tokens(self) -> None:
        dense = dense_token_map(self.features).data
        self.assertEqual(dense.shape, (1, 192, GRID_H, GRID_W))
        kept = set(self.features.indices[0].tolist())
        for j, index in enumerate(self.features.indices[0].tolist()):
            torch.testing.assert_close(
                dense[0, :, index // GRID_W, index % GRID_W],
                self.features.data[0, j],
            )
        for index in set(range(GRID_H * GRID_W)) - kept:
            self.assertEqual(
                dense[0, :, index // GRID_W, index % GRID_W].abs().sum().item(),
                0.0,
            )

    def test_sample_at_patch_center_reads_that_token(self) -> None:
        attention = EndpointDeformableAttention(
            dim=192, heads=1, points=1, offset_scale=0.5
        )
        with torch.no_grad():
            attention.sampling_offsets.bias.zero_()
            attention.output_proj.weight.copy_(torch.eye(192))
        value = dense_token_map(self.features)
        row, col = 2, 3
        center = torch.tensor([[[col + 0.5, row + 0.5, col + 0.5, row + 0.5]]])
        out = attention(
            torch.zeros(1, 1, 192), center, torch.ones(1, 1, 4), value
        )
        torch.testing.assert_close(
            out[0, 0], value.data[0, :, row, col], atol=1e-5, rtol=1e-5
        )

    def test_untrained_refiner_reproduces_stage0(self) -> None:
        refiner = LineRefiner(make_config(num_layers=3))
        refined = refiner(self.features, self.queries)
        self.assertEqual(len(refined.keypoints), 3)
        for keypoints in refined.keypoints:
            torch.testing.assert_close(keypoints.data[0], self.stage0)

    def test_every_layer_receives_gradient(self) -> None:
        refiner = LineRefiner(make_config())
        refined = refiner(self.features, self.queries)
        torch.stack([k.data.sum() for k in refined.keypoints]).sum().backward()
        for layer in refiner.layers:
            grad = layer.delta_out.weight.grad
            assert grad is not None
            self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_fgl_is_minimal_for_matching_distribution(self) -> None:
        bins = self.detector.line_head.weighting_fn.w
        centers = torch.zeros(1, 2)
        base_dirs = torch.tensor([[3.0, 0.0, -3.0, 0.0]])
        scale = base_dirs.abs() + 1.0
        target_bins = torch.tensor([20, 16, 9, 16])
        targets = torch.cat([centers, centers], -1) + base_dirs + bins[target_bins] * scale
        sharp = torch.full((1, 4, len(bins)), -50.0)
        sharp[0, torch.arange(4), target_bins] = 50.0
        flat = torch.zeros(1, 4, len(bins))
        loss_sharp = line_fgl_loss(
            sharp, centers, base_dirs, targets, torch.ones(1), bins, 1.0, 1.0
        )
        loss_flat = line_fgl_loss(
            flat, centers, base_dirs, targets, torch.ones(1), bins, 1.0, 1.0
        )
        self.assertLess(loss_sharp.item(), 1e-3)
        self.assertGreater(loss_flat.item(), 1.0)

    def test_refiner_learns_endpoint_shift(self) -> None:
        refiner = LineRefiner(make_config())
        criterion = LineRefinerCriterion(
            line_weights=[1.0, 1.0], loss_line_l1=5.0, loss_line_fgl=0.05
        )
        targets: Keypoints[
            tuple[Batch, NumQueries, KeypointDim], X1Y1X2Y2, PatchUnit, TopLeft
        ] = Keypoints(
            (self.stage0 + torch.tensor([0.3, -0.2, -0.3, 0.2])).unsqueeze(0)
        )
        labels: ClassLabels[tuple[Batch, NumQueries], NumLineClasses] = (
            ClassLabels(torch.zeros(1, 10, dtype=torch.long))
        )
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-3)

        def final_error() -> float:
            with torch.no_grad():
                refined = refiner(self.features, self.queries)
            return (refined.keypoints[-1].data - targets.data).abs().mean().item()

        before = final_error()
        for _ in range(150):
            refined = refiner(self.features, self.queries)
            losses = criterion(refined, self.queries, targets, labels)
            total = torch.stack(losses.loss_line_l1 + losses.loss_line_fgl).sum()
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
        self.assertLess(final_error(), 0.3 * before)


if __name__ == "__main__":
    unittest.main()
