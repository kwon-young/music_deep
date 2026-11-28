import unittest

import torch

from model.line_association import LineAssociationConfig, LineObjects, associate_lines, vote_modes

SURE, GUESS = 2 / 64, 40 / 64


def staff_voters(x0: float, x1: float, y: float) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Voters along a horizontal segment, one per patch: each sees the end of its half and
    guesses the other one 4 patches away with a large spread."""
    geometry, std, centers = [], [], []
    for cx in [x0 + 0.5 + i for i in range(int(x1 - x0))]:
        left_sure = cx < (x0 + x1) / 2
        left = x0 if left_sure else cx - 4
        right = cx + 4 if left_sure else x1
        geometry.append([left, y, right, y])
        std.append([SURE, SURE, GUESS, GUESS] if left_sure else [GUESS, GUESS, SURE, SURE])
        centers.append([cx, y])
    return geometry, std, centers


def run(segments: list[tuple[float, float, float]], cfg: LineAssociationConfig = LineAssociationConfig()) -> LineObjects:
    geometry, std, centers = [], [], []
    for x0, x1, y in segments:
        g, s, c = staff_voters(x0, x1, y)
        geometry += g
        std += s
        centers += c
    n = len(geometry)
    return associate_lines(torch.tensor(geometry), torch.tensor(std), torch.zeros(n, dtype=torch.long),
                           torch.full((n,), 0.5), torch.full((n,), 0.9), torch.tensor(centers), cfg)


class TestLineAssociation(unittest.TestCase):
    def test_abutting_segments_split_at_their_shared_barline(self) -> None:
        out = run([(0.0, 10.0, 1.0), (10.0, 20.0, 1.0)])
        geometry = sorted(out.geometry.tolist())
        self.assertEqual(len(geometry), 2)
        for got, want in zip(geometry, [[0.0, 1.0, 10.0, 1.0], [10.0, 1.0, 20.0, 1.0]]):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, places=4)
        self.assertEqual(sorted(out.size.tolist()), [10, 10])

    def test_parallel_line_one_staff_space_away_is_another_chain(self) -> None:
        out = run([(0.0, 10.0, 1.0), (0.0, 10.0, 1.84)])
        self.assertEqual(sorted(round(r[1], 2) for r in out.geometry.tolist()), [1.0, 1.84])

    def test_chain_without_two_cuts_keeps_its_most_certain_voter(self) -> None:
        geometry = torch.tensor([[0.0, 1.0, 3.0, 1.0], [0.2, 1.0, 2.0, 1.0]])
        std = torch.tensor([[GUESS] * 4, [0.1] * 4])
        out = associate_lines(geometry, std, torch.zeros(2, dtype=torch.long), torch.tensor([0.3, 0.6]),
                              torch.tensor([0.9, 0.9]), torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
                              LineAssociationConfig())
        torch.testing.assert_close(out.geometry, torch.tensor([[0.2, 1.0, 2.0, 1.0]]))
        self.assertAlmostEqual(float(out.quality[0]), 0.6, places=5)

    def test_vote_modes(self) -> None:
        t = torch.tensor([0.0, 0.01, -0.01, 5.0, 5.02, 9.0], dtype=torch.float64)
        w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=torch.float64)
        modes = vote_modes(t, w, 0.0625, 1.5)
        self.assertEqual(len(modes), 2)
        self.assertAlmostEqual(modes[0], 0.0, places=6)
        self.assertAlmostEqual(modes[1], 5.01, places=6)


if __name__ == "__main__":
    unittest.main()
