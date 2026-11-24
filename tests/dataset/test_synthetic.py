"""Tests for the synthetic music-score page generator (dataset.synthetic)."""

import random
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from dataset.synthetic import (
    LINE_NAMES,
    SYMBOL_NAMES,
    SyntheticConfig,
    SyntheticDataset,
    create_synthetic_dataset,
    load_synthetic_sample,
)

VEROVIO_ROOT = Path("/home/kwon-young/prog/verovio")


class TestSyntheticDataset(unittest.TestCase):
    config: ClassVar[SyntheticConfig]
    dataset: ClassVar[SyntheticDataset]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = SyntheticConfig(
            page_size=1024,
            objects_per_page=40,
            lines_per_page=16,
            noise=0.0,
            seed=1234,
        )
        cls.dataset = create_synthetic_dataset(cls.config, VEROVIO_ROOT)

    def test_atlas_covers_glyph_symbols(self) -> None:
        atlas = self.dataset.generator.atlas
        for name in SYMBOL_NAMES:
            if name in ("dots", "slur", "tie"):
                continue
            sprite = atlas.sprite(name, 0)
            self.assertGreater(sprite.ink_width, 0)
            self.assertGreater(sprite.ink_height, 0)

    def test_line_endpoints_follow_trompa_order(self) -> None:
        for _ in range(5):
            page = self.dataset.generator.generate()
            self.assertEqual(len(page.keypoint_thicknesses), len(page.keypoints))
            for (x1, y1, x2, y2), thickness in zip(
                page.keypoints, page.keypoint_thicknesses
            ):
                self.assertGreaterEqual(thickness, 1.0)
                if abs(y2 - y1) > abs(x2 - x1):
                    self.assertLess(y1, y2)
                else:
                    self.assertLessEqual(x1, x2)

    def test_generate_shapes_and_bounds(self) -> None:
        page = self.dataset.generator.generate()
        self.assertEqual(page.image.shape, (1024, 1024, 3))
        self.assertEqual(page.image.dtype, np.uint8)
        self.assertEqual(len(page.box_labels), len(page.boxes))
        self.assertEqual(len(page.keypoint_labels), len(page.keypoints))
        self.assertGreater(len(page.boxes), 0)
        self.assertGreater(len(page.keypoints), 0)

        for x1, y1, x2, y2 in page.boxes:
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, 1024.0)
            self.assertLessEqual(y2, 1024.0)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

        for x1, y1, x2, y2 in page.keypoints:
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, 1024.0)
            self.assertLessEqual(y2, 1024.0)

        for label in page.box_labels:
            self.assertLess(label, len(SYMBOL_NAMES))
        for label in page.keypoint_labels:
            self.assertLess(label, len(LINE_NAMES))

    def test_reproducible_with_seed(self) -> None:
        a = create_synthetic_dataset(self.config, VEROVIO_ROOT).generator.generate()
        b = create_synthetic_dataset(self.config, VEROVIO_ROOT).generator.generate()
        self.assertEqual(a.boxes, b.boxes)
        self.assertEqual(a.box_labels, b.box_labels)
        self.assertEqual(a.keypoints, b.keypoints)
        self.assertEqual(a.keypoint_labels, b.keypoint_labels)

    def test_uniform_class_coverage(self) -> None:
        rng = random.Random(7)
        seen: set[int] = set()
        gen = self.dataset.generator
        # small page to keep it fast, several pages to hit most classes
        for _ in range(100):
            page = gen.generate()
            seen.update(page.box_labels)
            if len(seen) >= len(SYMBOL_NAMES):
                break
        self.assertGreater(len(seen), len(SYMBOL_NAMES) // 2)

    def test_load_synthetic_sample(self) -> None:
        data = load_synthetic_sample(self.dataset, 0, torch.device("cpu"))
        img = data.sample.image.data
        self.assertEqual(img.shape[0], 3)
        self.assertEqual(img.shape[1], 1024)
        self.assertEqual(img.shape[2], 1024)
        self.assertEqual(img.dtype, torch.uint8)
        boxes = data.sample.boxes.data
        kps = data.sample.keypoints.data
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(kps.shape[1], 4)
        self.assertEqual(data.metadata.width, 1024)
        self.assertEqual(data.metadata.height, 1024)

    def test_sizes_decoupled_within_page(self) -> None:
        """One page must mix small and large symbols (no page-level zoom)."""
        page = self.dataset.generator.generate()
        heights = sorted(b[3] - b[1] for b in page.boxes)
        self.assertGreater(len(heights), 5)
        ratio = heights[-1] / max(heights[0], 1.0)
        self.assertGreater(ratio, 3.0)

    def test_symbol_pair_overlap_bounded(self) -> None:
        page = self.dataset.generator.generate()
        boxes = page.boxes
        max_pair = self.config.symbol_max_pair_overlap
        worst = 0.0
        for i in range(len(boxes)):
            a = boxes[i]
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            for b in boxes[i + 1 :]:
                w = min(a[2], b[2]) - max(a[0], b[0])
                h = min(a[3], b[3]) - max(a[1], b[1])
                if w <= 0 or h <= 0:
                    continue
                inter = w * h
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                ratio = inter / min(area_a, area_b)
                worst = max(worst, ratio)
        self.assertLessEqual(worst, max_pair + 1e-6)

    def test_bezier_and_dots_render(self) -> None:
        gen = self.dataset.generator
        rng = random.Random(9)
        ss = 64.0
        tile, iw, ih = gen._bezier_tile(rng, "slur", ss)
        self.assertGreater(iw, 0)
        self.assertGreater(ih, 0)
        tile2, iw2, ih2 = gen._bezier_tile(rng, "tie", ss)
        self.assertGreater(iw2, 0)
        self.assertGreater(ih2, 0)


if __name__ == "__main__":
    unittest.main()
