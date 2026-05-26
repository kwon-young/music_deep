from typing import Iterable, Generator, Callable, Concatenate, cast
from functools import wraps
from itertools import batched
from dataclasses import dataclass
from model.vit import get_2d_pope_frequencies
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from dataset.imslp import (
    Data,
    BatchedData,
    Metadata,
    Layout,
    AnyLayouts,
    BatchedLayouts,
    Mode,
    Range,
    Int255,
    Float1,
    PILImage,
    ArrayImage,
    TensorImage,
    Image,
    BatchedImage,
    Batch,
    HW,
    HWC,
    CHW,
    VCHW,
    BCHW,
    BVCHW,
    RGB,
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
def to_numpy[R: Range](
    image: PILImage[HWC, RGB, R],
) -> ArrayImage[CHW, RGB, R]:
    arr = np.array(image)
    arr = np.transpose(arr, (2, 0, 1))
    return cast(ArrayImage[CHW, RGB, R], arr)


@image_transform
def to_tensor[L: Layout, M: Mode, R: Range](
    image: ArrayImage[L, M, R],
) -> TensorImage[L, M, R]:
    return cast(TensorImage[L, M, R], torch.as_tensor(image))


@image_transform
def to_float1[L: AnyLayouts, M: Mode](
    image: TensorImage[L, M, Int255],
) -> TensorImage[L, M, Float1]:
    return cast(TensorImage[L, M, Float1], image.float() / 255.0)


def shuffle[T](it: Iterable[T]) -> Generator[T]:
    l = list(it)
    random.shuffle(l)
    yield from l


@image_transform
def to[I: TensorImage](image: I, device: torch.device) -> I:
    return cast(I, image.to(device))


@image_transform
def random_affine[L: BCHW | BVCHW | VCHW, M: Mode](
    x: TensorImage[L, M, Float1],
    max_angle_deg: float = 3.0,
    max_translate: float = 0.05,
) -> TensorImage[L, M, Float1]:
    original_shape = x.shape
    x_flat = x.view(-1, *original_shape[-3:])
    N = x_flat.size(0)
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

    grid = F.affine_grid(matrix, x_flat.size(), align_corners=False)

    # Assuming normalized image where white is 1.0.
    # Shift so white is 0.0, apply grid_sample (pads with 0.0), then shift back to 1.0
    x_shifted = x_flat - 1.0
    x_transformed = F.grid_sample(
        x_shifted, grid, padding_mode="zeros", align_corners=False
    )
    x_out = x_transformed + 1.0
    return cast(TensorImage[L, M, Float1], x_out.view(original_shape))


def random_crops[M: Mode, R: Range](
    x: TensorImage[CHW, M, R], crop_size: int
) -> TensorImage[tuple[Batch, *CHW], M, R]:
    # random crop n time where n*crop_size**2 will in average == h*w
    (c, h, w) = x.shape
    num_crop_frac = (h / crop_size) * (w / crop_size)
    num_crop = int(num_crop_frac)
    frac = num_crop_frac - num_crop
    last_crop = random.binomialvariate(n=1, p=frac)
    num_crop += last_crop
    x_max = w - crop_size + 1
    y_max = h - crop_size + 1
    xs = torch.randint(0, x_max, size=(num_crop,), device=x.device)
    ys = torch.randint(0, y_max, size=(num_crop,), device=x.device)
    crops = [
        x[:, y : y + crop_size, x_val : x_val + crop_size]
        for y, x_val in zip(ys, xs)
    ]
    return cast(TensorImage[tuple[Batch, *CHW], M, R], torch.stack(crops))


@image_transform
def random_crop[M: Mode, R: Range](
    image: TensorImage[CHW, M, R], crop_size: int
) -> TensorImage[CHW, M, R]:
    (c, h, w) = image.shape
    x_max = w - crop_size + 1
    y_max = h - crop_size + 1
    x = torch.randint(0, x_max, size=(1,), device=image.device)[0]
    y = torch.randint(0, y_max, size=(1,), device=image.device)[0]
    image = image[:, y : y + crop_size, x : x + crop_size]
    return image


@image_transform
def make_views[M: Mode, R: Range](
    image: TensorImage[CHW, M, R], n: int
) -> TensorImage[VCHW, M, R]:
    return cast(TensorImage[VCHW, M, R], torch.stack([image] * n))


def collate[M: Mode, R: Range](
    it: Iterable[Data[TensorImage[VCHW, M, R]]], batch_size: int
) -> Iterable[BatchedData[TensorImage[tuple[Batch, *VCHW], M, R]]]:
    for batch in batched(it, n=batch_size):
        m = [b.metadata for b in batch]
        i = [b.image for b in batch]
        yield BatchedData(m, torch.stack(i))


@dataclass
class PatchSequence:
    patches: torch.Tensor
    freqs: torch.Tensor


@dataclass
class BatchedPatchData:
    metadata: list[Metadata]
    sequence: PatchSequence


def extract_patches(
    image: TensorImage[AnyLayouts, Mode, Range],
    patch_size: tuple[int, int],
    dim_head: int,
) -> PatchSequence:
    b, c, h, w = image.shape
    ph, pw = patch_size

    x = image.unflatten(2, (h // ph, ph)).unflatten(4, (w // pw, pw))
    x = x.permute(0, 2, 4, 3, 5, 1)
    patches = x.reshape(b, -1, ph * pw * c)

    grid_h = h // ph
    grid_w = w // pw
    freqs = get_2d_pope_frequencies(
        grid_h, grid_w, dim_head, device=image.device
    )
    freqs = freqs.unsqueeze(0).expand(b, -1, -1)

    return PatchSequence(patches=patches, freqs=freqs)


def random_patch_drop(
    patch_seq: PatchSequence, drop_rate: float
) -> PatchSequence:
    if drop_rate <= 0.0:
        return patch_seq

    b, num_patches, _ = patch_seq.patches.shape
    num_keep = int(num_patches * (1.0 - drop_rate))

    if num_keep >= num_patches:
        return patch_seq

    rand_indices = torch.rand(b, num_patches, device=patch_seq.patches.device)
    indices, _ = rand_indices.argsort(dim=-1)[:, :num_keep].sort(dim=-1)

    kept_patches = torch.gather(
        patch_seq.patches,
        1,
        indices.unsqueeze(-1).expand(-1, -1, patch_seq.patches.shape[-1]),
    )
    kept_freqs = torch.gather(
        patch_seq.freqs,
        1,
        indices.unsqueeze(-1).expand(-1, -1, patch_seq.freqs.shape[-1]),
    )

    return PatchSequence(patches=kept_patches, freqs=kept_freqs)


def variance_patch_drop(
    patch_seq: PatchSequence, drop_rate: float
) -> PatchSequence:
    if drop_rate <= 0.0:
        return patch_seq

    b, num_patches, _ = patch_seq.patches.shape
    num_keep = int(num_patches * (1.0 - drop_rate))

    if num_keep >= num_patches:
        return patch_seq

    variances = patch_seq.patches.var(dim=-1)
    _, top_indices = variances.topk(num_keep, dim=-1)
    indices, _ = top_indices.sort(dim=-1)

    kept_patches = torch.gather(
        patch_seq.patches,
        1,
        indices.unsqueeze(-1).expand(-1, -1, patch_seq.patches.shape[-1]),
    )
    kept_freqs = torch.gather(
        patch_seq.freqs,
        1,
        indices.unsqueeze(-1).expand(-1, -1, patch_seq.freqs.shape[-1]),
    )

    return PatchSequence(patches=kept_patches, freqs=kept_freqs)
