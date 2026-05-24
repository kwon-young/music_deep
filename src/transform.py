from typing import Iterator, Generator
from pathlib import Path
import random
import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from dataset.imslp import Image, Layout, Mode


NpImage = Image[np.ndarray, Layout, Mode]
TensorImage = Image[torch.Tensor, Layout, Mode]


def to_numpy(
    image: Image[PILImage.Image, Layout, Mode],
) -> Image[np.ndarray, Layout, Mode]:
    return Image(image.metadata, np.array(image.image))


def to_tensor(image: NpImage) -> TensorImage:
    return Image(image.metadata, torch.as_tensor(image.image))


def shuffle[T](it: Iterator[T]) -> Generator[T]:
    l = list(it)
    random.shuffle(l)
    yield from l


def to(image: TensorImage, device: torch.device) -> TensorImage:
    return Image(image.metadata, image.image.to(device))


def gpu_random_affine(
    x: torch.Tensor, max_angle_deg: float = 3.0, max_translate: float = 0.05
) -> torch.Tensor:
    N = x.size(0)
    device = x.device

    max_angle_rad = max_angle_deg * math.pi / 180.0
    angles = (torch.rand(N, device=device) * 2 - 1) * max_angle_rad

    tx = (torch.rand(N, device=device) * 2 - 1) * max_translate
    ty = (torch.rand(N, device=device) * 2 - 1) * max_translate

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    matrix = torch.zeros(N, 2, 3, device=device)
    matrix[:, 0, 0] = cos_a
    matrix[:, 0, 1] = -sin_a
    matrix[:, 0, 2] = tx
    matrix[:, 1, 0] = sin_a
    matrix[:, 1, 1] = cos_a
    matrix[:, 1, 2] = ty

    grid = F.affine_grid(matrix, x.size(), align_corners=False)

    # Assuming normalized image where white is 1.0.
    # Shift so white is 0.0, apply grid_sample (pads with 0.0), then shift back to 1.0
    x_shifted = x - 1.0
    x_transformed = F.grid_sample(
        x_shifted, grid, padding_mode="zeros", align_corners=False
    )
    return x_transformed + 1.0
