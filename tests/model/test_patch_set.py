import unittest

import torch

from model.assignment import (
    PatchSetAssigner,
    PatchSetConfig,
    SlotLabels,
    assign_image,
    greedy_slot_assignment,
    on_line_quality,
    pair_quality,
    responsible_gt,
)
from model.criterion import PatchSetCriterion, quality_focal_loss, sigmoid_focal_loss, voter_weights
from model.box_ops import generalized_box_iou, paired_generalized_box_iou
from model.refiner import spread_voters
from model.detector import OMRDetector, create_detector, heads_need_remap, load_detector_state
from music_types import (
    BoundingBoxes,
    ClassLabels,
    DetectionLossWeights,
    DetectionTarget,
    Embeddings,
    Keypoints,
    Patches,
)


def detector(num_shapes: int, presence: bool) -> OMRDetector:
    model = create_detector(
        backbone_size="nano", patch_size=8, channels=3, use_sdpa=False,
        num_symbol_classes=3, num_line_classes=2, num_shapes=num_shapes, presence=presence,
    )
    model.eval()
    return model


def patches() -> Patches:
    cells = torch.tensor([[0, 1, 2, 5, 6, 9, 11]])
    return Embeddings(
        data=torch.rand(1, cells.shape[1], 3 * 8 * 8),
        indices=cells,
        image_shape=(3, 3 * 8, 4 * 8),
        patch_size=(8, 8),
    )


class TestHeadRemap(unittest.TestCase):
    def test_k5_checkpoint_into_k8_presence_detector(self) -> None:
        torch.manual_seed(0)
        old = detector(5, presence=False)
        new = detector(8, presence=True)
        self.assertTrue(heads_need_remap(old.state_dict(), new))
        self.assertTrue(load_detector_state(new, old.state_dict()))
        x = patches()
        with torch.no_grad():
            a, b = old(x), new(x)
        p = x.data.shape[1]

        def slots(t: torch.Tensor, k: int) -> torch.Tensor:
            return t.view(1, p, k, *t.shape[2:])

        # the five original slots reproduce Exp 037 exactly
        for name in ("pred_logits", "pred_boxes", "pred_edge_logits"):
            torch.testing.assert_close(
                slots(getattr(b.symbols, name).data, 8)[:, :, :5], slots(getattr(a.symbols, name).data, 5)
            )
        for name in ("pred_logits", "pred_keypoints", "pred_endpoint_logits"):
            torch.testing.assert_close(
                slots(getattr(b.lines, name).data, 8)[:, :, :5], slots(getattr(a.lines, name).data, 5)
            )
        # new slots copy the geometry of slots 0-2 and start empty
        torch.testing.assert_close(
            slots(b.lines.pred_keypoints.data, 8)[:, :, 5:], slots(a.lines.pred_keypoints.data, 5)[:, :, :3]
        )
        new_scores = slots(b.symbols.pred_logits.data, 8)[:, :, 5:].sigmoid()
        torch.testing.assert_close(new_scores, torch.full_like(new_scores, 0.01))
        for presence, prior in ((b.symbols.pred_presence, 0.08), (b.lines.pred_presence, 0.35)):
            assert presence is not None
            torch.testing.assert_close(presence.data.sigmoid(), torch.full_like(presence.data, prior))
        self.assertIsNone(a.symbols.pred_presence)

    def test_same_layout_is_not_remapped(self) -> None:
        model = detector(8, presence=True)
        self.assertFalse(load_detector_state(detector(8, presence=True), model.state_dict()))


def grid_centers(h: int, w: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return torch.stack([xs.flatten() + 0.5, ys.flatten() + 0.5], dim=-1)


def reference_greedy(cost: torch.Tensor) -> torch.Tensor:
    k, r = cost.shape
    out = torch.full((k,), -1, dtype=torch.long)
    pairs = sorted((float(cost[i, j]), i, j) for i in range(k) for j in range(r) if torch.isfinite(cost[i, j]))
    rows: set[int] = set()
    cols: set[int] = set()
    for _, i, j in pairs:
        if i not in rows and j not in cols:
            out[i] = j
            rows.add(i)
            cols.add(j)
    return out


class TestAssignment(unittest.TestCase):
    def test_greedy_matches_reference_per_patch(self) -> None:
        torch.manual_seed(1)
        cost = torch.rand(50, 8, 6)
        cost[torch.rand(50, 8, 6) < 0.2] = torch.inf
        cost[:, :, 4:][torch.rand(50, 8, 2) < 0.5] = torch.inf
        got = greedy_slot_assignment(cost)
        for p in range(50):
            self.assertEqual(got[p].tolist(), reference_greedy(cost[p]).tolist())

    def test_every_valid_column_gets_one_slot(self) -> None:
        cost = torch.rand(10, 8, 8)
        cost[:, :, 5:] = torch.inf
        got = greedy_slot_assignment(cost)
        for p in range(10):
            self.assertEqual(sorted(c for c in got[p].tolist() if c >= 0), [0, 1, 2, 3, 4])

    def test_overflow_keeps_the_gt_with_fewest_voters(self) -> None:
        # patch 0 is responsible for 3 GT but has 2 slots; GT 2 is only seen by patch 0
        distance = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0], [0.0, 0.0, 9.0]])
        center = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 9.0], [0.0, 0.0, 9.0]])
        kept, responsible, overflow = responsible_gt(distance, center, 0.5, 2)
        self.assertEqual(sorted(kept[0].tolist()), [0, 2])
        self.assertEqual(overflow[0].tolist(), [False, True, False])
        self.assertEqual(int(overflow.sum()), 1)
        self.assertEqual(responsible.sum(dim=0).tolist(), [3, 3, 1])

    def assign(self, gt: list[list[float]], labels: list[int], pred: torch.Tensor,
               is_line: bool = False, k: int = 2) -> SlotLabels:
        centers = grid_centers(1, 4)
        probs = torch.full((centers.shape[0] * k, 2), 0.5)
        return assign_image(centers, probs, pred, torch.tensor(gt), torch.tensor(labels), is_line, k,
                            PatchSetConfig())

    def test_every_responsible_patch_votes_once(self) -> None:
        # a box in cell 1: patches 0, 1, 2 are within the margin, patch 3 is 1.4 away
        box = [1.3, 0.3, 1.7, 0.7]
        pred = torch.tensor([box] * 8)
        labels = self.assign([box], [0], pred)
        self.assertEqual(labels.voters.tolist(), [3])
        self.assertEqual(sorted((labels.match.pred_indices // 2).tolist()), [0, 1, 2])

    def test_coincident_gt_take_two_slots_of_each_patch(self) -> None:
        box = [1.3, 0.3, 1.7, 0.7]
        labels = self.assign([box, box], [0, 0], torch.tensor([box] * 8))
        self.assertEqual(labels.voters.tolist(), [3, 3])
        for patch in (0, 1, 2):
            got = labels.match.target_indices[(labels.match.pred_indices // 2) == patch]
            self.assertEqual(sorted(got.tolist()), [0, 1])

    def test_in_patch_duplicate_is_negative_and_band_prediction_is_ignored(self) -> None:
        # GT A in cell 0 (patch 0, 1 responsible); GT B in cell 3 is in patch 1's ignore band
        a = [0.3, 0.3, 0.7, 0.7]
        b = [3.6, 0.3, 3.9, 0.7]
        pred = torch.tensor([a, [9, 9, 9.5, 9.5], a, a, b, b, b, b])
        labels = self.assign([a, b], [0, 1], pred)
        ignored = labels.ignored.view(4, 2)
        positive = torch.zeros(8, dtype=torch.bool)
        positive[labels.match.pred_indices] = True
        positive = positive.view(4, 2)
        # patch 1: one slot is A's voter, the other copies A -> negative, never ignored
        self.assertEqual(positive[1].tolist(), [True, False])
        self.assertFalse(bool(ignored[1].any()))
        # patch 0 predicts nothing useful in slot 1 -> negative
        self.assertEqual(ignored[0].tolist(), [False, False])
        # patch 2: B is in its band (cell distance 0.6) and A is out of reach; both slots
        # predict B well -> ignored, not negative
        self.assertEqual(positive[2].tolist(), [False, False])
        self.assertEqual(ignored[2].tolist(), [True, True])
        # patch 3 is responsible for B
        self.assertEqual(int(positive[3].sum()), 1)
        self.assertEqual(ignored[3].tolist(), [False, False])

    def test_on_line_slots_are_extra_line_positives(self) -> None:
        # a staff line along y = 0.5 through 4 cells; patch 1's slots: one matched-like copy, one
        # on the line with a wrong far end, one on the parallel line 0.84 below
        line = [0.0, 0.5, 4.0, 0.5]
        pred = torch.tensor([line, line, line, [1.0, 0.52, 9.0, 0.5], line, [0.5, 1.34, 3.5, 1.34], line, line])
        labels = self.assign([line], [0], pred, is_line=True)
        positive = torch.zeros(8, dtype=torch.bool)
        positive[labels.match.pred_indices] = True
        self.assertEqual(positive.view(4, 2).tolist(), [[True, True], [True, True], [True, False], [True, True]])
        self.assertEqual(labels.voters.tolist(), [7])
        self.assertAlmostEqual(float(on_line_quality(torch.tensor([1.0, 0.6, 3.0, 0.6]), torch.tensor(line), 0.1)),
                               float(torch.exp(torch.tensor(-0.5))), places=5)
        # no overlap along the line -> 0
        self.assertEqual(float(on_line_quality(torch.tensor([5.0, 0.5, 6.0, 0.5]), torch.tensor(line), 0.1)), 0.0)

    def test_line_quality_is_oks(self) -> None:
        gt = torch.tensor([[0.0, 0.0, 16.0, 0.0]])
        self.assertAlmostEqual(float(pair_quality(gt, gt, True)[0]), 1.0)
        off = gt + torch.tensor([[0.0, 0.0, 0.0, 0.5]])
        area = 16.0 / 16
        expected = (1 + torch.exp(torch.tensor(-0.25 / 0.04 / area / 2))) / 2
        self.assertAlmostEqual(float(pair_quality(off, gt, True)[0]), float(expected), places=5)


class TestPatchSetLosses(unittest.TestCase):
    def test_quality_focal_equals_focal_on_hard_targets(self) -> None:
        torch.manual_seed(2)
        logits = torch.randn(6, 4)
        target = (torch.rand(6, 4) < 0.3).float()
        alpha = torch.tensor([0.1, 0.25, 0.5, 0.9])
        torch.testing.assert_close(
            quality_focal_loss(logits, target, target.bool(), alpha, 2.0),
            sigmoid_focal_loss(logits, target, alpha=alpha, gamma=2.0),
        )
        # a soft target is a fixed point: zero gradient when p == q on the positive entry
        q = torch.tensor([0.3])
        logit = torch.logit(q).requires_grad_()
        quality_focal_loss(logit, q, torch.tensor([True]), 0.25, 2.0).sum().backward()
        assert logit.grad is not None
        self.assertAlmostEqual(float(logit.grad[0]), 0.0, places=6)

    def test_line_class_target_ignores_extent(self) -> None:
        criterion = PatchSetCriterion(
            num_symbol_classes=3, num_line_classes=2, weights=DetectionLossWeights(),
            symbol_weights=[0.3] * 3, line_weights=[0.3] * 2,
        )
        gt = torch.tensor([[0.0, 0.5, 16.0, 0.5]] * 2)
        pred = torch.tensor([[1.0, 0.5, 9.0, 0.5], [1.0, 0.5 + 7 / 64, 9.0, 0.5 + 7 / 64]])
        q = criterion.line_quality(pred, gt)
        self.assertAlmostEqual(float(q[0]), 1.0, places=5)  # wrong far end, on the line
        self.assertAlmostEqual(float(q[1]), float(torch.exp(torch.tensor(-((7 / 64) ** 2) / 0.02))), places=5)

    def test_paired_giou_is_the_diagonal_of_the_pairwise_matrix(self) -> None:
        torch.manual_seed(5)
        xy = torch.rand(50, 2)
        b1 = torch.cat([xy, xy + torch.rand(50, 2) + 0.01], dim=-1)
        xy = torch.rand(50, 2)
        b2 = torch.cat([xy, xy + torch.rand(50, 2) + 0.01], dim=-1)
        torch.testing.assert_close(paired_generalized_box_iou(b1, b2), generalized_box_iou(b1, b2).diag())

    def test_voter_weights_sum_to_one_per_gt(self) -> None:
        voters = torch.tensor([4, 4, 4, 4, 1])
        self.assertAlmostEqual(float(voter_weights(voters[:4], "inverse").sum()), 1.0)
        self.assertEqual(voter_weights(voters, "none").tolist(), [1.0] * 5)

    def test_ignored_slots_get_no_score_gradient(self) -> None:
        torch.manual_seed(3)
        model = detector(4, presence=True)
        x = patches()
        targets: list[DetectionTarget] = [DetectionTarget(
            boxes=BoundingBoxes(torch.tensor([[0.2, 0.3, 0.9, 1.1], [2.2, 1.2, 2.8, 1.7]])),
            box_labels=ClassLabels(torch.tensor([1, 2])),
            keypoints=Keypoints(torch.tensor([[0.5, 0.5, 3.5, 0.6]])),
            keypoint_labels=ClassLabels(torch.tensor([0])),
        )]
        outputs = model(x)
        assignment = PatchSetAssigner(PatchSetConfig(ignore_quality=0.0), num_slots=4)(outputs, targets)
        criterion = PatchSetCriterion(
            num_symbol_classes=3, num_line_classes=2, weights=DetectionLossWeights(),
            symbol_weights=[0.3, 0.3, 0.3], line_weights=[0.3, 0.3],
        )
        logits = outputs.symbols.pred_logits.data.detach().requires_grad_()
        presence_data = outputs.symbols.pred_presence
        assert presence_data is not None
        presence = presence_data.data.detach().requires_grad_()
        outputs.symbols.pred_logits.data = logits
        presence_data.data = presence
        losses = criterion(outputs, targets, assignment=assignment)
        (losses.loss_ce + losses.loss_presence).backward()
        assert logits.grad is not None and presence.grad is not None
        ignored = assignment.symbols[0].ignored
        positive = torch.zeros_like(ignored)
        positive[assignment.symbols[0].match.pred_indices] = True
        negative = ~ignored & ~positive
        self.assertGreater(int(ignored.sum()), 0)
        self.assertGreater(int(negative.sum()), 0)
        self.assertEqual(float(logits.grad[0, ignored].abs().sum()), 0.0)
        self.assertEqual(float(presence.grad[0, ignored].abs().sum()), 0.0)
        self.assertGreater(float(presence.grad[0, negative].abs().min()), 0.0)
        self.assertGreater(float(presence.grad[0, positive].abs().min()), 0.0)
        for name in ("loss_bbox", "loss_giou", "loss_fgl", "loss_line_l1", "loss_line_fgl"):
            self.assertTrue(bool(torch.isfinite(getattr(losses, name))))


class TestVoterSubsampling(unittest.TestCase):
    def test_spread_voters_evenly_along_the_segment(self) -> None:
        # GT 0 has 10 voters at positions 0.0 .. 0.9 (shuffled), GT 1 has 2
        torch.manual_seed(4)
        perm = torch.randperm(10)
        truth = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(2, dtype=torch.long)])
        position = torch.cat([perm.float() / 10, torch.tensor([0.7, 0.2])])
        kept, first = spread_voters(truth, position, 4)
        got0 = sorted(round(float(position[i]) * 10) for i in kept if truth[i] == 0)
        self.assertEqual(got0, [1, 3, 6, 8])
        self.assertEqual(sorted(int(i) for i in kept if truth[i] == 1), [10, 11])
        self.assertAlmostEqual(float(position[first[0]]), 0.0)
        self.assertEqual(int(first[1]), 11)


if __name__ == "__main__":
    unittest.main()
