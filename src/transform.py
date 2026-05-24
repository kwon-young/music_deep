from typing import Iterator, Generator, Callable, Concatenate, cast
from functools import wraps
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from dataset.imslp import (
    Data,
    BatchedData,
    Layout,
    Mode,
    PILImage,
    ArrayImage,
    TensorImage,
    Image,
    BatchedImage,
)


def image_transform[T: Image, U: Image, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[Data[T], P], Data[U]]:
    @wraps(func)
    def wrapper(img: Data[T], *args: P.args, **kwargs: P.kwargs) -> Data[U]:
        return Data(img.metadata, func(img.image, *args, **kwargs))

    return wrapper


def batched_image_transform[T: BatchedImage, U: BatchedImage, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[BatchedData[T], P], BatchedData[U]]:
    @wraps(func)
    def wrapper(
        img: BatchedData[T], *args: P.args, **kwargs: P.kwargs
    ) -> BatchedData[U]:
        return BatchedData(img.metadata, func(img.image, *args, **kwargs))

    return wrapper


@image_transform
def to_numpy[L: Layout, M: Mode](image: PILImage[L, M]) -> ArrayImage[L, M]:
    return cast(ArrayImage[L, M], np.array(image))


@image_transform
def to_tensor[L: Layout, M: Mode](image: ArrayImage[L, M]) -> TensorImage[L, M]:
    return cast(TensorImage[L, M], torch.as_tensor(image))


def shuffle[T](it: Iterator[T]) -> Generator[T]:
    l = list(it)
    random.shuffle(l)
    yield from l


@image_transform
def to[L: Layout, M: Mode](
    image: TensorImage[L, M], device: torch.device
) -> TensorImage[L, M]:
    return cast(TensorImage[L, M], image.to(device))


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
