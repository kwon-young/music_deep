import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from dataset.coco import SizeHistogram


@dataclass(frozen=True)
class SizeWeightingConfig:
    """Intra-class size rebalancing: rare sizes of a class weigh more than common ones.

    The weight of a size bin is `density ** -alpha`, normalized so that the mean weight
    over the class's instances is 1 (the class balance is left to the class weights),
    then clamped to [min_weight, max_weight]. Only classes whose size distribution is
    wide (`cv >= min_cv`) and well sampled (`n >= min_count`) are rebalanced.
    """

    alpha: float = 0.5
    min_cv: float = 0.3
    min_count: int = 100
    min_weight: float = 0.25
    max_weight: float = 4.0


def bin_weights(histogram: SizeHistogram, config: SizeWeightingConfig) -> list[float]:
    counts = histogram.counts
    num_bins = len(counts)
    if (
        config.alpha == 0.0
        or histogram.n < config.min_count
        or histogram.cv < config.min_cv
    ):
        return [1.0] * num_bins

    padded = [counts[0], *counts, counts[-1]]
    smoothed = [
        max(0.25 * padded[i] + 0.5 * padded[i + 1] + 0.25 * padded[i + 2], 1.0)
        for i in range(num_bins)
    ]
    weights = [math.pow(density, -config.alpha) for density in smoothed]

    def normalize(values: list[float]) -> list[float]:
        mean = sum(c * w for c, w in zip(counts, values)) / histogram.n
        return [w / mean for w in values]

    weights = normalize(weights)
    weights = [min(max(w, config.min_weight), config.max_weight) for w in weights]
    return normalize(weights)


class SizeWeighting(nn.Module):
    """Looks up the weight of each instance from its class and size in pixels.

    Weights are linearly interpolated between bin centers in log-size, so small size
    jitter (patch-size and scale augmentation) only moves the weight slightly.
    """

    log_min: torch.Tensor
    bin_width: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self, histograms: list[SizeHistogram], config: SizeWeightingConfig
    ):
        super().__init__()
        self.config = config
        self.register_buffer(
            "log_min",
            torch.tensor([h.log_min for h in histograms], dtype=torch.float32),
        )
        self.register_buffer(
            "bin_width",
            torch.tensor([h.bin_width for h in histograms], dtype=torch.float32),
        )
        self.register_buffer(
            "weights",
            torch.tensor(
                [bin_weights(h, config) for h in histograms], dtype=torch.float32
            ).reshape(len(histograms), -1),
        )

    def forward(self, classes: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        num_bins = self.weights.shape[1]
        position = (
            sizes.clamp(min=1.0).log() - self.log_min[classes]
        ) / self.bin_width[classes] - 0.5
        position = position.clamp(0.0, num_bins - 1.0)
        left = position.floor().long()
        right = (left + 1).clamp(max=num_bins - 1)
        frac = position - left.to(position.dtype)
        return (
            self.weights[classes, left] * (1.0 - frac)
            + self.weights[classes, right] * frac
        )
