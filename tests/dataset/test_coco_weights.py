import unittest

from dataset.coco import (
    ClassSizeStats,
    _apply_size_difficulty,
    _collect_class_size_stats,
    _compute_difficulty_balanced_weights,
    CocoLineAnnotation,
    CocoSymbolAnnotation,
)


class TestSizeDifficultyWeights(unittest.TestCase):
    def test_cv_definition(self):
        self.assertAlmostEqual(
            ClassSizeStats(n=2, mean=10.0, variance=4.0).cv, 0.2
        )
        self.assertEqual(ClassSizeStats(n=1, mean=10.0, variance=0.0).cv, 0.0)
        self.assertEqual(ClassSizeStats(n=0, mean=0.0, variance=0.0).cv, 0.0)

    def test_apply_no_change_for_zero_variance(self):
        weights = [0.05, 0.25, 0.85]
        stats = {
            i: ClassSizeStats(n=5, mean=10.0, variance=0.0)
            for i in range(3)
        }
        out = _apply_size_difficulty(
            weights, stats, variance_lambda=0.5, max_diff=8.0
        )
        self.assertEqual(out, weights)

    def test_apply_boosts_high_cv_class_after_clamping(self):
        weights = [0.05, 0.25, 0.85]
        stats = {
            0: ClassSizeStats(n=100, mean=500.0, variance=299022.0),
            1: ClassSizeStats(n=100, mean=10.0, variance=0.0),
            2: ClassSizeStats(n=100, mean=10.0, variance=0.0),
        }
        out = _apply_size_difficulty(
            weights, stats, variance_lambda=0.5, max_diff=8.0
        )
        # slur-like class was clamped at the floor but must rise.
        self.assertGreater(out[0], weights[0])
        # zero-variance classes stay put.
        self.assertEqual(out[1], weights[1])

    def test_diff_capped_by_max(self):
        weights = [0.1]
        stats = {0: ClassSizeStats(n=100, mean=1.0, variance=1.0e6)}
        out = _apply_size_difficulty(
            weights, stats, variance_lambda=1.0, max_diff=3.0
        )
        self.assertAlmostEqual(out[0], min(0.1 * 3.0, 0.85))

    def test_compute_difficulty_balanced_preserves_layout(self):
        counts = {0: 100, 1: 1, 2: 50}
        stats = {
            0: ClassSizeStats(n=100, mean=500.0, variance=299022.0),
            1: ClassSizeStats(n=1, mean=100.0, variance=0.0),
            2: ClassSizeStats(n=50, mean=10.0, variance=0.0),
        }
        out = _compute_difficulty_balanced_weights(
            counts, 3, stats, variance_lambda=0.5, size_difficulty_max=8.0
        )
        self.assertEqual(len(out), 3)
        self.assertGreater(out[0], 0.05)
        self.assertLessEqual(out[0], 0.85)

    def test_collect_class_size_stats(self):
        annotations = {
            1: [
                CocoSymbolAnnotation(bbox=[0, 0, 3, 4], category_id=10),
                CocoSymbolAnnotation(bbox=[0, 0, 6, 8], category_id=10),
                CocoLineAnnotation(keypoints=[0, 0, 3, 4], category_id=20),
            ],
            2: [
                CocoSymbolAnnotation(bbox=[0, 0, 9, 12], category_id=10),
                CocoLineAnnotation(keypoints=[0, 0, 6, 8], category_id=20),
            ],
        }
        sym_stats, line_stats = _collect_class_size_stats(
            annotations,
            {10: 0},
            {20: 0},
            num_symbol_classes=1,
            num_line_classes=1,
        )
        self.assertEqual(sym_stats[0].n, 3)
        # 3-4-5, 6-8-10, 9-12-15 triangles.
        self.assertAlmostEqual(sym_stats[0].mean, (5 + 10 + 15) / 3)
        self.assertEqual(line_stats[0].n, 2)


if __name__ == "__main__":
    unittest.main()