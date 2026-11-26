import unittest

import torch

import numpy as np

from model.refiner import (
    EndpointRefiner,
    LineQueries,
    QualityHead,
    RefineStep,
    TokenGrid,
    deduplicate_lines,
    fgl_loss,
    line_oks,
    refiner_loss,
)


def grid_with_drops(grid_h: int, grid_w: int, dropped: list[int], dim: int) -> TokenGrid:
    cells = torch.tensor([c for c in range(grid_h * grid_w) if c not in dropped])
    tokens = torch.randn(1, cells.shape[0], dim)
    return TokenGrid.from_batch(tokens, cells.unsqueeze(0), grid_h, grid_w)


class TestTokenGrid(unittest.TestCase):
    def test_window_cells_and_masking(self) -> None:
        # 3 rows x 4 cols, cell 6 = (row 1, col 2) dropped.
        grid = grid_with_drops(3, 4, dropped=[6], dim=8)
        token, valid, centers = grid.window(torch.tensor([0]), torch.tensor([[1.3, 1.7]]))

        expected_cells = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        kept = [c for c in range(12) if c != 6]
        for k, cell in enumerate(expected_cells):
            self.assertEqual(bool(valid[0, k]), cell != 6)
            if cell != 6:
                self.assertEqual(int(token[0, k]), kept.index(cell))
            torch.testing.assert_close(
                centers[0, k], torch.tensor([cell % 4 + 0.5, cell // 4 + 0.5])
            )

    def test_window_outside_image_is_invalid(self) -> None:
        grid = grid_with_drops(3, 4, dropped=[], dim=8)
        _, valid, _ = grid.window(torch.tensor([0, 0]), torch.tensor([[0.2, 0.2], [50.0, -9.0]]))
        self.assertEqual(valid[0].tolist(), [False] * 4 + [True, True, False, True, True])
        self.assertFalse(bool(valid[1].any()))

    def test_concat_matches_batched_grid(self) -> None:
        torch.manual_seed(0)
        tokens = torch.randn(2, 5, 8)
        cells = torch.tensor([[0, 2, 3, 7, 11], [1, 4, 5, 6, 9]])
        batched = TokenGrid.from_batch(tokens, cells, 3, 4)
        concat = TokenGrid.concat(
            [TokenGrid.from_batch(tokens[i : i + 1], cells[i : i + 1], 3, 4) for i in range(2)]
        )
        for name in ("tokens", "lookup", "offset", "height", "width"):
            torch.testing.assert_close(getattr(concat, name), getattr(batched, name))
        token = torch.tensor([3, 5, 9])
        image = torch.tensor([0, 1, 1])
        torch.testing.assert_close(
            concat.centers(token, image), torch.tensor([[3.5, 1.5], [1.5, 0.5], [1.5, 2.5]])
        )


class TestRefiner(unittest.TestCase):
    def test_identity_at_init_and_dropped_window(self) -> None:
        torch.manual_seed(0)
        grid = grid_with_drops(4, 4, dropped=[0, 1, 4, 5], dim=12)
        refiner = EndpointRefiner(dim=12, num_classes=3, heads=3)
        # The first endpoint sits in a fully dropped 3x3 neighbourhood corner.
        endpoints = torch.tensor([[0.1, 0.1, 3.5, 3.5], [1.2, 2.5, 2.8, 0.6]])
        queries = LineQueries(
            image=torch.tensor([0, 0]),
            source=torch.tensor([3, 7]),
            label=torch.tensor([0, 2]),
            endpoints=endpoints,
        )
        outputs = refiner(grid, queries)
        self.assertEqual(len(outputs), 2)
        for out in outputs:
            self.assertTrue(bool(torch.isfinite(out.endpoints).all()))
            torch.testing.assert_close(out.endpoints, endpoints)

    def test_num_steps_override_reuses_last_step(self) -> None:
        torch.manual_seed(0)
        grid = grid_with_drops(6, 6, dropped=[7, 20], dim=12)
        refiner = EndpointRefiner(dim=12, num_classes=3, num_steps=2, heads=3)
        for step in refiner.steps:
            assert isinstance(step, RefineStep)
            # Non-zero bins so every step actually moves the endpoints.
            torch.nn.init.normal_(step.to_bins.weight, std=0.5)
        refiner.eval()
        queries = LineQueries(
            image=torch.tensor([0, 0, 0]),
            source=torch.tensor([3, 10, 25]),
            label=torch.tensor([0, 2, 1]),
            endpoints=torch.tensor(
                [[1.2, 1.4, 4.6, 1.5], [2.5, 0.3, 2.4, 5.2], [0.1, 5.8, 5.9, 0.2]]
            ),
        )
        with torch.no_grad():
            default = refiner(grid, queries)
            explicit = refiner(grid, queries, num_steps=2)
            short = refiner(grid, queries, num_steps=1)
            long = refiner(grid, queries, num_steps=4)
            none = refiner(grid, queries, num_steps=0)

        self.assertEqual([len(default), len(short), len(long), len(none)], [2, 1, 4, 0])
        for a, b in zip(default, explicit):
            self.assertTrue(torch.equal(a.endpoints, b.endpoints))
        self.assertTrue(torch.equal(short[0].endpoints, default[0].endpoints))
        for s in range(2):
            self.assertTrue(torch.equal(long[s].endpoints, default[s].endpoints))
        # Steps 3 and 4 are the last trained step applied again, from the previous estimate.
        self.assertTrue(torch.equal(long[2].start, long[1].endpoints))
        self.assertFalse(torch.equal(long[3].endpoints, long[1].endpoints))
        self.assertIs(refiner.schedule(4)[3], refiner.steps[1])
        with self.assertRaises(ValueError):
            refiner.schedule(-1)

    def test_fgl_loss_is_minimal_at_target_bin(self) -> None:
        weights = torch.linspace(-0.5, 0.5, 9)
        logits = torch.full((1, 9), -20.0)
        logits[0, 6] = 20.0
        good = fgl_loss(logits, weights[6:7], weights)
        bad = fgl_loss(logits, weights[2:3], weights)
        self.assertLess(float(good), 1e-3)
        self.assertGreater(float(bad), 10.0)

    def test_learns_to_snap_to_endpoints(self) -> None:
        """Tokens encode where 'their' endpoint is; the refiner must learn to read them."""
        torch.manual_seed(0)
        grid_h = grid_w = 8
        dim = 24
        true_offset = torch.rand(grid_h * grid_w, 2) * 0.8 + 0.1
        cells = torch.arange(grid_h * grid_w)
        centers = torch.stack([cells % grid_w, cells // grid_w], dim=-1).float()
        true_points = centers + true_offset
        code = torch.randn(4, dim)
        features = torch.cat(
            [(true_offset * 6.28).sin(), (true_offset * 6.28).cos()], dim=-1
        ) @ code
        grid = TokenGrid.from_batch(features.unsqueeze(0), cells.unsqueeze(0), grid_h, grid_w)

        refiner = EndpointRefiner(dim=dim, num_classes=1, num_steps=1, heads=3)
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-3)

        def batch(n: int) -> tuple[LineQueries, torch.Tensor]:
            ends = torch.randint(0, grid_h * grid_w, (n, 2))
            target = true_points[ends].reshape(n, 4)
            coarse = target + (torch.rand(n, 4) - 0.5) * 0.6
            return (
                LineQueries(
                    image=torch.zeros(n, dtype=torch.long),
                    source=ends[:, 0],
                    label=torch.zeros(n, dtype=torch.long),
                    endpoints=coarse,
                ),
                target,
            )

        test_queries, test_target = batch(256)
        before = (test_queries.endpoints - test_target).abs().mean()
        for _ in range(300):
            queries, target = batch(128)
            loss, _ = refiner_loss(
                refiner, refiner(grid, queries), target, torch.ones_like(target), 0.1
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            after = (refiner(grid, test_queries)[-1].endpoints - test_target).abs().mean()
        self.assertLess(float(after), 0.5 * float(before))


class TestQuality(unittest.TestCase):
    def test_line_oks_matches_evaluator_formula(self) -> None:
        torch.manual_seed(0)
        pred = torch.rand(5, 4, dtype=torch.float64) * 100
        gt = torch.rand(3, 4, dtype=torch.float64) * 100
        area = torch.rand(3, dtype=torch.float64) * 500 + 10
        variance = (np.array([0.1, 0.1]) * 2) ** 2
        p, g, a = pred.numpy(), gt.numpy(), area.numpy()
        dx = p[:, None, 0::2] - g[None, :, 0::2]
        dy = p[:, None, 1::2] - g[None, :, 1::2]
        e = (dx**2 + dy**2) / variance / (a[None, :, None] + np.spacing(1)) / 2
        expected = np.exp(-e).mean(axis=2)
        np.testing.assert_allclose(line_oks(pred, gt, area).numpy(), expected, rtol=1e-6)

    def test_quality_head_reads_both_endpoint_states(self) -> None:
        head = QualityHead(dim=12)
        state = torch.randn(7, 2, 12)
        self.assertEqual(head(state).shape, (7,))


class TestDeduplicateLines(unittest.TestCase):
    def test_suppresses_coinciding_lines_only(self) -> None:
        endpoints = torch.tensor(
            [
                [0.0, 0.0, 100.0, 0.0],  # staff segment
                [1.0, 1.0, 101.0, 0.0],  # its duplicate (both endpoints within 2)
                [100.0, 0.0, 200.0, 0.0],  # next segment: shares one endpoint only
                [0.0, 0.0, 100.0, 0.0],  # same geometry, other class
            ]
        )
        scores = torch.tensor([0.4, 0.5, 0.3, 0.2])
        labels = torch.tensor([0, 0, 0, 1])

        plain = deduplicate_lines(endpoints, scores, labels, radius=2.0, merge=False)
        self.assertEqual(sorted(plain.keep.tolist()), [1, 2, 3])
        kept = dict(zip(plain.keep.tolist(), plain.score.tolist()))
        self.assertAlmostEqual(kept[1], 0.5)
        sizes = dict(zip(plain.keep.tolist(), plain.group_size.tolist()))
        self.assertEqual(sizes, {1: 2, 2: 1, 3: 1})
        self.assertEqual(plain.score.tolist(), sorted(plain.score.tolist(), reverse=True))

        merged = deduplicate_lines(endpoints, scores, labels, radius=2.0, merge=True)
        kept = dict(zip(merged.keep.tolist(), merged.score.tolist()))
        self.assertAlmostEqual(kept[1], 1 - (1 - 0.5) * (1 - 0.4), places=6)
        self.assertAlmostEqual(kept[2], 0.3, places=6)

    def test_empty_input(self) -> None:
        out = deduplicate_lines(
            torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long), 4.0, True
        )
        self.assertEqual(out.keep.numel(), 0)


if __name__ == "__main__":
    unittest.main()
