import unittest

import torch

from model.refiner import (
    EndpointRefiner,
    LineQueries,
    TokenGrid,
    fgl_loss,
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


if __name__ == "__main__":
    unittest.main()
