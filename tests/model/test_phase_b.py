import unittest

import torch

from dataset.coco import CocoMetadata
from dataset.split import piece_name, split_by_piece
from model.criterion import DFINECriterion
from model.detector import DetectorWithTokens, create_detector
from model.matcher import HungarianMatcher
from music_types import (
    BoundingBoxes,
    ClassLabels,
    DetectionLossWeights,
    DetectionTarget,
    Keypoints,
    Embeddings,
    Patches,
)


def small_detector() -> tuple[DetectorWithTokens, Patches]:
    torch.manual_seed(0)
    detector = create_detector(
        backbone_size="nano", patch_size=8, channels=3, use_sdpa=False,
        num_symbol_classes=3, num_line_classes=2,
    )
    detector.eval()
    grid_h, grid_w = 3, 4
    cells = torch.tensor([[0, 1, 2, 5, 6, 9, 11]])
    patches: Patches = Embeddings(
        data=torch.rand(1, cells.shape[1], 3 * 8 * 8),
        indices=cells,
        image_shape=(3, grid_h * 8, grid_w * 8),
        patch_size=(8, 8),
    )
    return DetectorWithTokens(detector), patches


class TestForwardWithTokens(unittest.TestCase):
    def test_tokens_are_the_final_normed_backbone_output(self) -> None:
        wrapper, patches = small_detector()
        captured: list[torch.Tensor] = []
        handle = wrapper.detector.backbone.transformer.norm.register_forward_hook(
            lambda module, args, output: captured.append(output)
        )
        with torch.no_grad():
            outputs, features = wrapper(patches)
            plain = wrapper.detector(patches)
        handle.remove()
        torch.testing.assert_close(features.data, captured[0])
        torch.testing.assert_close(features.indices, patches.indices)
        torch.testing.assert_close(outputs.lines.pred_keypoints.data, plain.lines.pred_keypoints.data)
        self.assertEqual(set(wrapper.detector.state_dict()), set(create_detector(
            backbone_size="nano", patch_size=8, channels=3, use_sdpa=False,
            num_symbol_classes=3, num_line_classes=2,
        ).state_dict()))


class TestCriterionSharedMatching(unittest.TestCase):
    def test_precomputed_indices_give_the_same_losses(self) -> None:
        wrapper, patches = small_detector()
        matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        criterion = DFINECriterion(
            matcher, num_symbol_classes=3, num_line_classes=2, weights=DetectionLossWeights(),
            symbol_weights=[1.0, 1.0, 1.0], line_weights=[1.0, 1.0],
        )
        targets = [
            DetectionTarget(
                boxes=BoundingBoxes(torch.tensor([[0.2, 0.3, 0.9, 1.1]])),
                box_labels=ClassLabels(torch.tensor([1])),
                keypoints=Keypoints(torch.tensor([[0.5, 0.5, 3.5, 0.6], [1.5, 1.0, 1.5, 2.8]])),
                keypoint_labels=ClassLabels(torch.tensor([0, 1])),
            )
        ]
        with torch.no_grad():
            outputs, _ = wrapper(patches)
            shared = criterion(outputs, targets, matcher(outputs, targets))
            own = criterion(outputs, targets)
        for name in ("loss_ce", "loss_bbox", "loss_giou", "loss_fgl", "loss_line_l1", "loss_line_fgl"):
            torch.testing.assert_close(getattr(shared, name), getattr(own, name))


class TestSplitByPiece(unittest.TestCase):
    def test_pieces_stay_on_one_side(self) -> None:
        images = [
            CocoMetadata(img_id=10 * p + k, file_name=f"piece{p}_{k}.png", width=10, height=10)
            for p in range(10)
            for k in range(3)
        ]
        train, held_out = split_by_piece(images, seed=0, fraction=0.2)
        train_pieces = {piece_name(i.file_name) for i in train}
        held_pieces = {piece_name(i.file_name) for i in held_out}
        self.assertEqual(len(held_pieces), 2)
        self.assertFalse(train_pieces & held_pieces)
        self.assertEqual(len(train) + len(held_out), len(images))
        self.assertEqual(piece_name("a_b_12.png"), "a_b")


if __name__ == "__main__":
    unittest.main()
