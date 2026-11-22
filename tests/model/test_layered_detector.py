import unittest

import torch
import torch.nn.functional as F

from model.criterion import DFINECriterion, DetectionTargets
from model.detector import DFINEWeightingFunction, OMRDetector, create_detector
from model.layered_criterion import DFINELayeredCriterion, ddf_loss, go_indices
from model.layered_detector import MultiLayerOMRDetector, create_layered_detector
from model.matcher import HungarianMatcher
from model.vit import Attention
from music_types import (
    BoundingBoxes,
    ClassLabels,
    DetectionLossWeights,
    DetectionTarget,
    Embeddings,
    FlattenedIndices,
    Keypoints,
    MatchIndices,
    Patches,
    Batch,
    NumPatches,
    PatchDim,
)

NUM_SHAPES = 2
PATCH = 16
GRID_H, GRID_W = 4, 6
SYMBOL_CLASSES, LINE_CLASSES = 3, 2


def make_layered() -> MultiLayerOMRDetector:
    return create_layered_detector(
        backbone_size="nano",
        patch_size=PATCH,
        channels=3,
        use_sdpa=False,
        num_symbol_classes=SYMBOL_CLASSES,
        num_line_classes=LINE_CLASSES,
        num_shapes=NUM_SHAPES,
    )


def make_single() -> OMRDetector:
    return create_detector(
        backbone_size="nano",
        patch_size=PATCH,
        channels=3,
        use_sdpa=False,
        num_symbol_classes=SYMBOL_CLASSES,
        num_line_classes=LINE_CLASSES,
        num_shapes=NUM_SHAPES,
    )


def make_patches(batch: int = 1) -> Patches[Batch, NumPatches, PatchDim]:
    kept = torch.tensor([[0, 1, 3, 4, 7, 8, 9, 12, 13, 15, 17, 18, 20, 22, 23]])
    return Embeddings(
        data=torch.randn(batch, kept.shape[1], 3 * PATCH * PATCH),
        indices=kept.expand(batch, -1),
        image_shape=(3, GRID_H * PATCH, GRID_W * PATCH),
        patch_size=(PATCH, PATCH),
    )


def dfine_weighting_function(reg_max: int, up: float, reg_scale: float) -> torch.Tensor:
    """dfine_utils.weighting_function, expressed in anchor units (divided by reg_scale)."""
    upper_bound1 = up * reg_scale
    upper_bound2 = up * reg_scale * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2, *left, 0.0, *right, upper_bound2]
    return torch.tensor(values) / reg_scale


def dfine_go_indices(
    indices: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """dfine_criterion._get_go_indices for one image (the reference loop)."""
    ind = torch.cat([torch.stack([rows, cols], dim=1) for rows, cols in indices])
    unique, counts = torch.unique(ind, return_counts=True, dim=0)
    unique_sorted = unique[torch.argsort(counts, descending=True, stable=True)]
    column_to_row: dict[int, int] = {}
    for idx in unique_sorted:
        row, col = int(idx[0].item()), int(idx[1].item())
        if row not in column_to_row:
            column_to_row[row] = col
    return torch.tensor(list(column_to_row.keys())), torch.tensor(
        list(column_to_row.values())
    )


class WeightingFunctionTest(unittest.TestCase):
    def test_dfine_outer_bins_match_dfine(self) -> None:
        ours = DFINEWeightingFunction(reg_max=32, dfine_outer_bins=True).w
        reference = dfine_weighting_function(32, up=0.5, reg_scale=4.0)
        torch.testing.assert_close(ours, reference)

    def test_legacy_bins_are_unchanged(self) -> None:
        legacy = DFINEWeightingFunction(reg_max=32).w
        self.assertEqual(legacy[0].item(), legacy[1].item())
        self.assertEqual(legacy[0].item(), -0.5)


class AttentionPositionTest(unittest.TestCase):
    def test_position_enters_queries_and_keys_only(self) -> None:
        torch.manual_seed(0)
        attention = Attention(dim=32, heads=2, dim_head=16, use_sdpa=False).eval()
        x = torch.randn(1, 5, 32)
        pos = torch.randn(1, 5, 32)

        normed = attention.norm(x)
        q, k, _ = attention.to_qkv(normed + pos).chunk(3, dim=-1)
        _, _, v = attention.to_qkv(normed).chunk(3, dim=-1)
        q, k, v = (t.unflatten(-1, (2, -1)).transpose(1, 2) for t in (q, k, v))
        weights = (q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(dim=-1)
        expected = attention.to_out((weights @ v).transpose(1, 2).flatten(2))

        torch.testing.assert_close(attention(x, pos=pos), expected)
        torch.testing.assert_close(
            attention(x, pos=torch.zeros_like(pos)), attention(x)
        )


class LayeredDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_first_tap_is_block_nine_of_nano(self) -> None:
        self.assertEqual(make_layered().first_tap, 9)

    def test_refinement_starts_as_identity(self) -> None:
        model = make_layered().eval()
        with torch.no_grad():
            out = model(make_patches())
        self.assertEqual(len(out.layers), 3)
        for layer in out.layers[1:]:
            torch.testing.assert_close(
                layer.symbols.pred_boxes.data, out.layers[0].symbols.pred_boxes.data
            )
            torch.testing.assert_close(
                layer.lines.pred_keypoints.data,
                out.layers[0].lines.pred_keypoints.data,
            )

    def test_first_layer_decodes_like_the_dense_heads(self) -> None:
        model = make_layered().eval()
        patches = make_patches()
        with torch.no_grad():
            out = model(patches)
            x, freqs = model.backbone.embed(patches)
            tokens = model.tap_norm(0)(
                model.backbone.transformer.run_blocks(x, freqs, 0, 10)
            )
            _, offsets, _, boxes, _ = model.symbol_head(tokens)
            _, _, keypoints, _ = model.line_head(tokens)
        centers = out.layers[0].lines.absolute_centers.data.view(1, -1, NUM_SHAPES, 2)
        centers = centers.repeat(1, 1, 1, 2)
        torch.testing.assert_close(
            out.layers[0].symbols.pred_boxes.data, (boxes + centers).flatten(1, 2)
        )
        torch.testing.assert_close(
            out.layers[0].lines.pred_keypoints.data,
            (keypoints + centers).flatten(1, 2),
        )
        self.assertTrue(offsets.abs().sum() > 0)

    def test_init_from_single_layer_keeps_the_final_classifier(self) -> None:
        single = make_single().eval()
        layered = make_layered().eval()
        layered.load_single_layer_state(single.state_dict())
        self.assertEqual(layered.weighting_fn.w[0].item(), -1.0)
        self.assertEqual(layered.symbol_head.weighting_fn.w[0].item(), -1.0)

        patches = make_patches()
        with torch.no_grad():
            expected = single(patches)
            final = layered(patches).final
        torch.testing.assert_close(
            final.symbols.pred_logits.data, expected.symbols.pred_logits.data
        )
        torch.testing.assert_close(
            final.lines.pred_logits.data, expected.lines.pred_logits.data
        )

    def test_refined_boxes_do_not_train_the_frame(self) -> None:
        model = make_layered()
        out = model(make_patches())
        (
            out.final.symbols.pred_boxes.data.sum()
            + out.final.lines.pred_keypoints.data.sum()
        ).backward()

        for head in (model.symbol_head, model.line_head):
            last = head.mlp[-1]
            assert isinstance(last, torch.nn.Linear) and last.bias.grad is not None
            grad = last.bias.grad.view(NUM_SHAPES, head.preds_per_shape)
            frame = grad[:, head.num_classes : head.num_classes + 4]
            bins = grad[:, head.num_classes + 4 :]
            self.assertEqual(frame.abs().sum().item(), 0.0)
            self.assertGreater(bins.abs().sum().item(), 0.0)


class GoIndicesTest(unittest.TestCase):
    def test_matches_dfine_reference(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            layers = []
            for _ in range(4):
                preds = torch.randperm(12, generator=generator)[:6]
                targets = torch.randint(0, 5, (6,), generator=generator)
                layers.append((preds, targets))
            # layer 0 pairs reach count >= 4 > 3: no tie for the rows it matches
            layers.extend([layers[0]] * 3)
            rows, cols = dfine_go_indices(layers)
            go = go_indices(
                [MatchIndices(pred_indices=p, target_indices=t) for p, t in layers]
            )
            reference = dict(zip(rows.tolist(), cols.tolist()))
            ours = dict(zip(go.pred_indices.tolist(), go.target_indices.tolist()))
            self.assertEqual(ours.keys(), reference.keys())
            for row in layers[0][0].tolist():
                self.assertEqual(ours[row], reference[row])

    def test_empty(self) -> None:
        empty = torch.empty(0, dtype=torch.int64)
        go = go_indices([MatchIndices(pred_indices=empty, target_indices=empty)])
        self.assertEqual(len(go.pred_indices), 0)


class ChunkedMatcherTest(unittest.TestCase):
    def test_chunking_does_not_change_the_pairs(self) -> None:
        generator = torch.Generator().manual_seed(0)
        centers = torch.rand(600, 2, generator=generator) * 20
        boxes = torch.cat([centers, centers + 0.5], dim=1)
        lines = torch.cat([centers, centers + torch.randn(600, 2, generator=generator)], 1)
        targets = torch.rand(70, 2, generator=generator) * 20
        target_boxes = torch.cat([targets, targets + 1.0], dim=1)
        target_lines = torch.cat([targets, targets + 5.0], dim=1)

        dense = HungarianMatcher(max_pair_elements=1 << 30)
        chunked = HungarianMatcher(max_pair_elements=600 * 8)
        for expected, actual in (
            (
                dense._get_valid_pairs_boxes(boxes, target_boxes),
                chunked._get_valid_pairs_boxes(boxes, target_boxes),
            ),
            (
                dense._get_valid_pairs_keypoints(lines, target_lines),
                chunked._get_valid_pairs_keypoints(lines, target_lines),
            ),
        ):
            for e, a in zip(expected, actual):
                self.assertTrue(torch.equal(e, a))


class DDFTest(unittest.TestCase):
    def test_zero_when_student_equals_teacher(self) -> None:
        logits = torch.randn(2, 7, 4, 33)
        scores = torch.randn(2, 7, 3)
        matched = FlattenedIndices(batch=torch.tensor([0, 1]), src=torch.tensor([3, 5]))
        loss = ddf_loss(logits, logits, scores, matched, torch.tensor([0.5, 0.9]))
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_gradient_only_reaches_student(self) -> None:
        student = torch.randn(1, 7, 4, 33, requires_grad=True)
        teacher = torch.randn(1, 7, 4, 33, requires_grad=True)
        matched = FlattenedIndices(batch=torch.tensor([0]), src=torch.tensor([2]))
        loss = ddf_loss(student, teacher, torch.randn(1, 7, 3), matched, torch.ones(1))
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)


class LayeredCriterionTest(unittest.TestCase):
    def test_losses_are_finite_and_train_every_head(self) -> None:
        torch.manual_seed(0)
        model = make_layered()
        base = DFINECriterion(
            HungarianMatcher(calc_device=torch.device("cpu")),
            num_symbol_classes=SYMBOL_CLASSES,
            num_line_classes=LINE_CLASSES,
            weights=DetectionLossWeights(),
            symbol_weights=[1.0] * SYMBOL_CLASSES,
            line_weights=[1.0] * LINE_CLASSES,
            dfine_outer_bins=True,
        )
        criterion = DFINELayeredCriterion(base)
        targets: DetectionTargets = [
            DetectionTarget(
                boxes=BoundingBoxes(
                    torch.tensor([[0.2, 0.3, 0.9, 1.1], [2.1, 1.2, 2.8, 2.0]])
                ),
                box_labels=ClassLabels(torch.tensor([0, 2])),
                keypoints=Keypoints(torch.tensor([[0.5, 1.5, 4.5, 1.5]])),
                keypoint_labels=ClassLabels(torch.tensor([1])),
            )
        ]
        losses = criterion(model(make_patches()), targets)
        self.assertEqual(len(losses.layers), 3)
        assert losses.pre is not None and losses.loss_ddf is not None
        self.assertEqual(losses.pre.loss_fgl.item(), 0.0)
        total = losses.total()
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for name in ("conditioner", "symbol_refiners", "line_refiners", "symbol_lqe"):
            grads = [
                p.grad
                for n, p in model.named_parameters()
                if n.startswith(name) and p.grad is not None
            ]
            self.assertTrue(grads, name)


if __name__ == "__main__":
    unittest.main()
