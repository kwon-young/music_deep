from typing import Iterable, Generator, Callable, Concatenate, cast
from functools import wraps
from itertools import batched
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
    Batch,
    HWC,
    CHW,
    VCHW,
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


def shuffle[T](it: Iterable[T]) -> Generator[T]:
    l = list(it)
    random.shuffle(l)
    yield from l


@image_transform
def to[I: TensorImage](image: I, device: torch.device) -> I:
    return cast(I, image.to(device))


@batched_image_transform
def random_affine[L: BatchedLayouts, M: Mode](
    x: TensorImage[L, M],
    max_angle_deg: float = 3.0,
    max_translate: float = 0.05,
) -> TensorImage[L, M]:
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
    return cast(TensorImage[L, M], x_transformed + 1.0)


def random_crops[M: Mode](
    x: TensorImage[HWC, M], crop_size: int
) -> TensorImage[tuple[Batch, *HWC], M]:
    # random crop n time where n*crop_size**2 will in average == h*w
    (h, w, c) = x.shape
    num_crop_frac = (h / crop_size) * (w / crop_size)
    num_crop = int(num_crop_frac)
    frac = num_crop_frac - num_crop
    last_crop = random.binomialvariate(n=1, p=frac)
    num_crop += last_crop
    x_max = w - crop_size + 1
    y_max = h - crop_size + 1
    xs = torch.randint(0, x_max, size=(num_crop,))
    ys = torch.randint(0, y_max, size=(num_crop,))
    crops = [
        x[y:y + crop_size, x_val:x_val + crop_size, :]
        for y, x_val in zip(ys, xs)
    ]
    return cast(TensorImage[tuple[Batch, *HWC], M], torch.stack(crops))


@image_transform
def random_crop[M: Mode](
    image: TensorImage[HWC, M], crop_size: int
) -> TensorImage[HWC, M]:
    (h, w, c) = image.shape
    x_max = w - crop_size + 1
    y_max = h - crop_size + 1
    x = torch.randint(0, x_max, size=(1,))[0]
    y = torch.randint(0, y_max, size=(1,))[0]
    image = image[y:y + crop_size, x:x + crop_size, :]
    return image


@image_transform
def make_views[M: Mode](image: TensorImage[CHW, M], n: int) -> TensorImage[VCHW, M]:
    return cast(TensorImage[VCHW, M], torch.stack([image] * n))


def collate[M: Mode](it: Iterable[Data[TensorImage[VCHW, M]]], batch_size: int
                     ) -> Iterable[BatchedData[TensorImage[tuple[Batch, *VCHW], M]]]:
    for batch in batched(it, n=batch_size):
        m = [b.metadata for b in batch]
        i = [b.image for b in batch]
        yield BatchedData(m, torch.stack(i))
