import math
import unittest

import torch

from dataset.coco import SizeHistogram, size_histogram
from model.criterion import (
    DFINECriterion,
    line_oks,
    quality_focal_loss,
    sigmoid_focal_loss,
)
from model.detector import create_detector
from model.matcher import HungarianMatcher
from model.size_weighting import SizeWeighting, SizeWeightingConfig, bin_weights
from music_types import (
    Batch,
    BoxShape,
    KeypointShape,
    LabelShape,
    NumLineClasses,
    NumPatches,
    NumSymbolClasses,
    PatchDim,
    Patches,
    PatchUnit,
    TopLeft,
    X1Y1X2Y2,
    XYXY,
    BoundingBoxes,
    ClassLabels,
    DetectionLossWeights,
    DetectionTarget,
    Embeddings,
    FlattenedIndices,
    Keypoints,
)


def flat(batch: list[int], src: list[int]) -> FlattenedIndices:
    return FlattenedIndices(batch=torch.tensor(batch), src=torch.tensor(src))


class TestQualityFocalLoss(unittest.TestCase):
    def test_reduces_to_focal_loss_at_unit_quality(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 7, 4)
        idx = flat([0, 1, 1], [3, 0, 5])
        classes = torch.tensor([2, 0, 3])
        alpha = torch.tensor([0.05, 0.3, 0.85, 0.5])
        target = torch.zeros_like(logits)
        target[idx.batch, idx.src, classes] = 1.0
        expected = sigmoid_focal_loss(logits, target, alpha=alpha, gamma=2.0)
        actual = quality_focal_loss(
            logits, idx, classes, torch.ones(3), torch.ones(3), alpha, 2.0
        )
        torch.testing.assert_close(actual, expected)

    def test_positive_is_minimal_at_its_quality(self) -> None:
        idx = flat([0], [0])
        classes = torch.tensor([0])
        alpha = torch.tensor([0.25])
        quality = torch.tensor([0.7])

        def positive_loss(logit: float) -> float:
            logits = torch.tensor([[[logit]]])
            loss = quality_focal_loss(
                logits, idx, classes, quality, torch.ones(1), alpha, 2.0
            )
            return loss.item()

        at_quality = positive_loss(math.log(0.7 / 0.3))
        self.assertLess(at_quality, positive_loss(math.log(0.6 / 0.4)))
        self.assertLess(at_quality, positive_loss(math.log(0.8 / 0.2)))

    def test_positive_weight_scales_only_the_positive(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(1, 5, 2)
        idx = flat([0], [2])
        classes = torch.tensor([1])
        alpha = torch.tensor([0.25, 0.25])
        base = quality_focal_loss(
            logits, idx, classes, torch.ones(1), torch.ones(1), alpha, 2.0
        )
        scaled = quality_focal_loss(
            logits, idx, classes, torch.ones(1), torch.full((1,), 3.0), alpha, 2.0
        )
        torch.testing.assert_close(scaled[0, 2, 1], 3.0 * base[0, 2, 1])
        mask = torch.ones_like(base, dtype=torch.bool)
        mask[0, 2, 1] = False
        torch.testing.assert_close(scaled[mask], base[mask])


class TestLineOks(unittest.TestCase):
    def test_exact_match_is_one(self) -> None:
        target = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
        self.assertAlmostEqual(
            line_oks(target, target, torch.tensor([0.1])).item(), 1.0
        )

    def test_matches_coco_formula(self) -> None:
        target = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
        pred = torch.tensor([[0.0, 0.3, 10.4, 0.0]])
        thickness = torch.tensor([0.0625])
        area = 10.0 * 0.0625
        variance = (2 * 0.1) ** 2
        expected = 0.5 * (
            math.exp(-(0.3**2) / variance / area / 2)
            + math.exp(-(0.4**2) / variance / area / 2)
        )
        self.assertAlmostEqual(
            line_oks(pred, target, thickness).item(), expected, places=6
        )


def histogram_from(sizes: list[float]) -> SizeHistogram:
    return size_histogram(sizes, num_bins=10)


class TestSizeWeighting(unittest.TestCase):
    def setUp(self) -> None:
        common = [100.0 + i % 7 for i in range(900)]
        rare = [1000.0 + i for i in range(30)]
        self.wide = histogram_from(common + rare)
        self.narrow = histogram_from([50.0 + 0.01 * i for i in range(500)])
        self.config = SizeWeightingConfig(alpha=0.5, min_cv=0.3, min_count=100)

    def test_histogram_counts_every_instance(self) -> None:
        self.assertEqual(sum(self.wide.counts), self.wide.n)
        self.assertGreater(self.wide.cv, 0.3)
        self.assertLess(self.narrow.cv, 0.3)

    def test_narrow_or_disabled_classes_keep_unit_weights(self) -> None:
        self.assertEqual(bin_weights(self.narrow, self.config), [1.0] * 10)
        disabled = SizeWeightingConfig(alpha=0.0)
        self.assertEqual(bin_weights(self.wide, disabled), [1.0] * 10)

    def test_weights_have_unit_instance_mean_and_favor_rare_sizes(self) -> None:
        weights = bin_weights(self.wide, self.config)
        mean = sum(c * w for c, w in zip(self.wide.counts, weights)) / self.wide.n
        self.assertAlmostEqual(mean, 1.0, places=6)
        self.assertGreater(weights[-1], weights[0])
        self.assertGreater(weights[0], 0.0)

    def test_lookup_interpolates_between_bin_centers(self) -> None:
        weighting = SizeWeighting([self.wide, self.narrow], self.config)
        table = weighting.weights[0]
        h = self.wide
        centers = torch.tensor(
            [math.exp(h.log_min + (i + 0.5) * h.bin_width) for i in range(10)]
        )
        classes = torch.zeros(10, dtype=torch.long)
        torch.testing.assert_close(weighting(classes, centers), table)
        midpoint = torch.tensor([math.exp(h.log_min + 1.0 * h.bin_width)])
        torch.testing.assert_close(
            weighting(torch.zeros(1, dtype=torch.long), midpoint),
            0.5 * (table[0:1] + table[1:2]),
        )
        tiny = weighting(torch.zeros(1, dtype=torch.long), torch.tensor([0.5]))
        torch.testing.assert_close(tiny, table[0:1])
        narrow = weighting(torch.ones(3, dtype=torch.long), torch.tensor([10.0, 50.0, 1e4]))
        torch.testing.assert_close(narrow, torch.ones(3))


class TestCriterionForward(unittest.TestCase):
    def test_weighted_quality_criterion_runs_and_backpropagates(self) -> None:
        torch.manual_seed(0)
        detector = create_detector(
            backbone_size="nano",
            patch_size=16,
            channels=3,
            use_sdpa=False,
            num_symbol_classes=3,
            num_line_classes=2,
            num_shapes=2,
        )
        patches: Patches[Batch, NumPatches, PatchDim] = Embeddings(
            data=torch.randn(1, 24, 3 * 16 * 16),
            indices=torch.arange(24).unsqueeze(0),
            image_shape=(3, 64, 96),
            patch_size=(16, 16),
        )
        outputs = detector(patches)
        target: DetectionTarget[
            BoundingBoxes[BoxShape, XYXY, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumSymbolClasses],
            Keypoints[KeypointShape, X1Y1X2Y2, PatchUnit, TopLeft],
            ClassLabels[LabelShape, NumLineClasses],
        ] = DetectionTarget(
            boxes=BoundingBoxes(torch.tensor([[1.0, 1.0, 1.5, 1.6], [3.0, 2.0, 3.4, 2.5]])),
            box_labels=ClassLabels(torch.tensor([0, 2])),
            keypoints=Keypoints(torch.tensor([[0.5, 1.0, 5.0, 1.0], [2.0, 0.5, 2.0, 3.0]])),
            keypoint_labels=ClassLabels(torch.tensor([0, 1])),
        )
        wide = histogram_from([100.0 + i % 7 for i in range(900)] + [1000.0] * 30)
        config = SizeWeightingConfig(alpha=0.5, min_cv=0.3, min_count=100)
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        losses: dict[bool, torch.Tensor] = {}
        for enabled in (False, True):
            criterion = DFINECriterion(
                matcher,
                num_symbol_classes=3,
                num_line_classes=2,
                weights=DetectionLossWeights(),
                symbol_weights=[0.25, 0.25, 0.25],
                line_weights=[0.25, 0.25],
                symbol_size_weighting=SizeWeighting([wide] * 3, config) if enabled else None,
                line_size_weighting=SizeWeighting([wide] * 2, config) if enabled else None,
                line_thickness=[4.0, 5.5],
                line_quality_target=enabled,
                nominal_patch_size=16.0,
            )
            result = criterion(outputs, [target])
            total = (
                result.loss_ce + result.loss_bbox + result.loss_giou + result.loss_fgl
                + result.loss_line_l1 + result.loss_line_fgl
            )
            self.assertTrue(torch.isfinite(total))
            losses[enabled] = total
        detector.zero_grad()
        losses[True].backward()
        grads = [p.grad for p in detector.line_head.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertNotAlmostEqual(losses[True].item(), losses[False].item(), places=4)


if __name__ == "__main__":
    unittest.main()
