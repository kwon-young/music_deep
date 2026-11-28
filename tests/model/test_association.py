import unittest

import torch

from model.association import (
    AssociationConfig,
    Detections,
    associate,
    associate_candidates,
    select_candidates,
    support_ratio,
)
from model.regions import cell_segment_distance, expected_voters


def dets(
    geometry: list[list[float]],
    patch: list[int],
    score: list[float] | None = None,
    label: list[int] | None = None,
    std: list[list[float]] | None = None,
) -> Detections:
    n = len(geometry)
    s = torch.tensor(score if score is not None else [0.9] * n)
    return Detections(
        geometry=torch.tensor(geometry),
        std=torch.tensor(std) if std is not None else torch.full((n, 4), 0.05),
        label=torch.tensor(label if label is not None else [0] * n),
        score=s,
        presence=s,
        patch=torch.tensor(patch),
    )


PLAIN = AssociationConfig(radius=0.1, normalized=False)


class TestAssociate(unittest.TestCase):
    def test_duplicates_from_different_patches_merge(self) -> None:
        c = associate(dets([[0, 0, 4, 0], [0.02, 0, 4.02, 0]], [0, 1], [0.8, 0.2]), PLAIN)
        self.assertEqual(c.geometry.shape[0], 1)
        self.assertEqual(c.member.tolist(), [0, 0])
        # plain mode: the fused geometry is the score-weighted mean
        self.assertAlmostEqual(float(c.geometry[0, 0]), 0.2 * 0.02, places=5)
        self.assertAlmostEqual(float(c.quality[0]), 0.8, places=5)
        self.assertAlmostEqual(float(c.evidence[0]), 1.0, places=5)
        self.assertAlmostEqual(float(c.noisy_or[0]), 1 - 0.2 * 0.8, places=5)
        self.assertEqual(int(c.representative[0]), 0)

    def test_cannot_link_keeps_coincident_objects_apart(self) -> None:
        # two coincident noteheads predicted by patches 0 and 1, each in two slots
        box = [1.0, 1.0, 1.2, 1.2]
        c = associate(dets([box, box, box, box], [0, 0, 1, 1], [0.9, 0.8, 0.7, 0.6]), PLAIN)
        self.assertEqual(c.geometry.shape[0], 2)
        self.assertEqual(c.size.tolist(), [2, 2])
        # the two predictions of one patch always land in different clusters
        self.assertNotEqual(int(c.member[0]), int(c.member[1]))
        self.assertNotEqual(int(c.member[2]), int(c.member[3]))

    def test_without_cannot_link_coincident_objects_merge(self) -> None:
        box = [1.0, 1.0, 1.2, 1.2]
        cfg = AssociationConfig(radius=0.1, normalized=False, cannot_link=False)
        c = associate(dets([box, box], [0, 0]), cfg)
        self.assertEqual(c.geometry.shape[0], 1)

    def test_labels_never_merge(self) -> None:
        c = associate(dets([[0, 0, 4, 0], [0, 0, 4, 0]], [0, 1], label=[0, 1]), PLAIN)
        self.assertEqual(c.geometry.shape[0], 2)
        self.assertEqual(sorted(c.label.tolist()), [0, 1])

    def test_far_apart_stay_apart(self) -> None:
        c = associate(dets([[0, 0, 4, 0], [0, 0.84, 4, 0.84]], [0, 1]), PLAIN)
        self.assertEqual(c.geometry.shape[0], 2)

    def test_uncertain_far_endpoint_joins_and_is_outvoted(self) -> None:
        # patch 0 sees both ends; patch 1 sits at the start and guesses the far end 2 patches off
        sure = [0.0, 0.0, 10.0, 0.0]
        guess = [0.0, 0.0, 8.0, 0.0]
        stds = [[0.02] * 4, [0.02, 0.02, 1.5, 0.02]]
        plain = associate(dets([sure, guess], [0, 1], [0.9, 0.9], std=stds), PLAIN)
        self.assertEqual(plain.geometry.shape[0], 2)
        cfg = AssociationConfig(radius=3.0, normalized=True, variance_weights=True)
        c = associate(dets([sure, guess], [0, 1], [0.9, 0.9], std=stds), cfg)
        self.assertEqual(c.geometry.shape[0], 1)
        self.assertGreater(float(c.geometry[0, 2]), 9.99)

    def test_nearest_open_cluster_is_joined(self) -> None:
        # patch 2's candidate is close to both clusters but patch 0 is in the nearer one
        c = associate(
            dets(
                [[0, 0, 1, 0], [0, 0.05, 1, 0.05], [0, 0.03, 1, 0.03]],
                [0, 0, 1],
                [0.9, 0.8, 0.7],
            ),
            PLAIN,
        )
        self.assertEqual(c.geometry.shape[0], 2)
        self.assertEqual(int(c.member[2]), int(c.member[1]))

    def test_empty(self) -> None:
        empty = Detections(
            geometry=torch.zeros(0, 4), std=torch.zeros(0, 4), label=torch.zeros(0, dtype=torch.long),
            score=torch.zeros(0), presence=torch.zeros(0), patch=torch.zeros(0, dtype=torch.long),
        )
        c = associate(empty, PLAIN)
        self.assertEqual(c.geometry.shape, (0, 4))
        self.assertEqual(c.member.shape, (0,))

    def test_support_ratio(self) -> None:
        c = associate(dets([[0, 0, 4, 0], [0, 0, 4, 0]], [0, 1], [0.5, 0.5]), PLAIN)
        ratio = support_ratio(c, torch.tensor([4]))
        self.assertAlmostEqual(float(ratio[0]), 0.25, places=5)


class TestInferenceGlue(unittest.TestCase):
    def test_presence_floor_and_support_quality_ranking(self) -> None:
        # 2 patches x 2 slots; patch centres (0.5, 0.5) and (1.5, 0.5)
        logits = torch.logit(torch.tensor([[0.9, 0.1], [0.2, 0.1], [0.8, 0.1], [0.1, 0.7]]))
        presence = torch.logit(torch.tensor([0.95, 0.1, 0.9, 0.6]))
        cands = select_candidates(logits, presence, 0.5)
        self.assertEqual(cands.index.tolist(), [0, 2, 3])
        box = [0.8, 0.3, 1.2, 0.7]  # straddles both cells: 2 expected voters
        geometry = torch.tensor([box, box, [1.3, 0.2, 1.7, 0.4]])
        centers = torch.tensor([[0.5, 0.5], [1.5, 0.5]])
        out = associate_candidates(
            cands, geometry, torch.full((3, 4), 0.05), 2, centers, False,
            AssociationConfig(radius=0.1, normalized=False), 0.0, "support_quality", 10,
        )
        self.assertEqual(out.label.tolist(), [0, 1])
        # cluster 0: presence 0.95 + 0.9 over 2 voters, quality 0.9
        self.assertAlmostEqual(float(out.score[0]), (0.95 + 0.9) / 2 * 0.9, places=5)
        # cluster 1: one voter out of 1 expected, presence 0.6, quality 0.7
        self.assertAlmostEqual(float(out.score[1]), 0.6 * 0.7, places=5)


class TestRegions(unittest.TestCase):
    def test_crossing_segment_is_at_distance_zero(self) -> None:
        centers = torch.tensor([[0.5, 0.5], [0.5, 2.5]])
        d = cell_segment_distance(centers, torch.tensor([[-1.0, 0.5, 3.0, 0.5]]))
        self.assertAlmostEqual(float(d[0, 0]), 0.0)
        self.assertAlmostEqual(float(d[1, 0]), 1.5, places=5)

    def test_expected_voters(self) -> None:
        # a 3x3 grid of kept patches, a horizontal line through the middle row
        ys, xs = torch.meshgrid(torch.arange(3.0), torch.arange(3.0), indexing="ij")
        centers = torch.stack([xs.flatten() + 0.5, ys.flatten() + 0.5], dim=-1)
        line = torch.tensor([[0.0, 1.5, 3.0, 1.5]])
        self.assertEqual(expected_voters(centers, line, True, 0.0).tolist(), [3])
        self.assertEqual(expected_voters(centers, line, True, 0.5).tolist(), [9])
        box = torch.tensor([[1.2, 1.2, 1.8, 1.8]])
        self.assertEqual(expected_voters(centers, box, False, 0.0).tolist(), [1])


if __name__ == "__main__":
    unittest.main()
