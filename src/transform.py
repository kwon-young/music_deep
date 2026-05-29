from typing import Iterable, Generator, Callable, Concatenate
from functools import wraps
from itertools import batched
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from music_types import (
    Data,
    BatchedData,
    Layout,
    AnyLayouts,
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
    Height,
    Width,
    Channel,
    View,
    CHW,
    VCHW,
    BCHW,
    BVCHW,
    RGB,
    Embeddings,
    Patches,
    NumPatches,
    PatchDim,
    FlatViewEmbeddings,
    ViewEmbeddings,
    BatchView,
    EmbedDim,
)


def image_transform[Meta, T: Image, U: Image, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[Data[Meta, T], P], Data[Meta, U]]:
    @wraps(func)
    def wrapper(
        img: Data[Meta, T], *args: P.args, **kwargs: P.kwargs
    ) -> Data[Meta, U]:
        return Data(img.metadata, func(img.image, *args, **kwargs))

    return wrapper


def batched_image_transform[Meta, T: BatchedImage, U: BatchedImage, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[BatchedData[Meta, T], P], BatchedData[Meta, U]]:
    @wraps(func)
    def wrapper(
        img: BatchedData[Meta, T], *args: P.args, **kwargs: P.kwargs
    ) -> BatchedData[Meta, U]:
        return BatchedData(img.metadata, func(img.image, *args, **kwargs))

    return wrapper


@image_transform
def to_numpy[H: Height, W: Width, C: Channel, R: Range](
    image: PILImage[tuple[H, W, C], RGB, R],
) -> ArrayImage[tuple[C, H, W], RGB, R]:
    arr = np.array(image.data)
    arr = np.transpose(arr, (2, 0, 1))
    return ArrayImage(arr)


@image_transform
def to_tensor[L: Layout, M: Mode, R: Range](
    image: ArrayImage[L, M, R],
) -> TensorImage[L, M, R]:
    return TensorImage(torch.as_tensor(image.data))


@image_transform
def to_float1[L: AnyLayouts, M: Mode](
    image: TensorImage[L, M, Int255],
) -> TensorImage[L, M, Float1]:
    return TensorImage(image.data.float() / 255.0)


def shuffle[T](it: Iterable[T]) -> Generator[T, None, None]:
    l = list(it)
    random.shuffle(l)
    yield from l


@image_transform
def to[I: TensorImage](image: I, device: torch.device) -> I:
    return type(image)(image.data.to(device))


@image_transform
def random_affine[L: BCHW | BVCHW | VCHW, M: Mode](
    image: TensorImage[L, M, Float1],
    max_angle_deg: float = 3.0,
    max_translate: float = 0.05,
) -> TensorImage[L, M, Float1]:
    x = image.data
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
    return TensorImage(x_out.view(original_shape))


def random_crops[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R], crop_size: int
) -> TensorImage[tuple[Batch, C, Height, Width], M, R]:
    x = image.data
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
    return TensorImage(torch.stack(crops))


@image_transform
def random_crop[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R], crop_size: int
) -> TensorImage[tuple[C, Height, Width], M, R]:
    x_data = image.data
    (c, h, w) = x_data.shape
    x_max = w - crop_size + 1
    y_max = h - crop_size + 1
    x = torch.randint(0, x_max, size=(1,), device=x_data.device)[0]
    y = torch.randint(0, y_max, size=(1,), device=x_data.device)[0]
    cropped = x_data[:, y : y + crop_size, x : x + crop_size]
    return TensorImage(cropped)


@image_transform
def make_views[C: Channel, H: Height, W: Width, M: Mode, R: Range](
    image: TensorImage[tuple[C, H, W], M, R], n: int
) -> TensorImage[tuple[View, C, H, W], M, R]:
    return TensorImage(torch.stack([image.data] * n))


def collate[Meta, M: Mode, R: Range](
    it: Iterable[Data[Meta, TensorImage[VCHW, M, R]]], batch_size: int
) -> Iterable[BatchedData[Meta, TensorImage[tuple[Batch, *VCHW], M, R]]]:
    for batch in batched(it, n=batch_size):
        m = [b.metadata for b in batch]
        i = [b.image.data for b in batch]
        yield BatchedData(m, TensorImage(torch.stack(i)))


def extract_patches[B: Batch](
    image: TensorImage[tuple[B, *CHW], Mode, Range],
    patch_size: tuple[int, int],
) -> Patches[B, NumPatches, PatchDim]:
    x_data = image.data
    b, c, h, w = x_data.shape
    ph, pw = patch_size

    x = x_data.unflatten(2, (h // ph, ph)).unflatten(4, (w // pw, pw))
    x = x.permute(0, 2, 4, 3, 5, 1)
    patches = x.reshape(b, -1, ph * pw * c)

    num_patches = patches.shape[1]
    indices = (
        torch.arange(num_patches, device=x_data.device)
        .unsqueeze(0)
        .expand(b, -1)
    )

    return Embeddings(
        data=patches,
        indices=indices,
        image_shape=(c, h, w),
        patch_size=(ph, pw),
    )


def random_patch_drop[B: Batch, P: PatchDim](
    patches: Patches[B, NumPatches, P], drop_rate: float
) -> Patches[B, NumPatches, P]:
    if drop_rate <= 0.0:
        return patches

    b, num_patches, _ = patches.data.shape
    num_keep = int(num_patches * (1.0 - drop_rate))

    if num_keep >= num_patches:
        return patches

    rand_indices = torch.rand(b, num_patches, device=patches.data.device)
    indices_sort, _ = rand_indices.argsort(dim=-1)[:, :num_keep].sort(dim=-1)

    kept_data = torch.gather(
        patches.data,
        1,
        indices_sort.unsqueeze(-1).expand(-1, -1, patches.data.shape[-1]),
    )
    kept_indices = torch.gather(
        patches.indices,
        1,
        indices_sort,
    )

    return Embeddings(
        data=kept_data,
        indices=kept_indices,
        image_shape=patches.image_shape,
        patch_size=patches.patch_size,
    )


def variance_patch_drop[B: Batch, P: PatchDim](
    patches: Patches[B, NumPatches, P], drop_rate: float
) -> Patches[B, NumPatches, P]:
    if drop_rate <= 0.0:
        return patches

    b, num_patches, _ = patches.data.shape
    num_keep = int(num_patches * (1.0 - drop_rate))

    if num_keep >= num_patches:
        return patches

    variances = patches.data.var(dim=-1)
    _, top_indices = variances.topk(num_keep, dim=-1)
    indices_sort, _ = top_indices.sort(dim=-1)

    kept_data = torch.gather(
        patches.data,
        1,
        indices_sort.unsqueeze(-1).expand(-1, -1, patches.data.shape[-1]),
    )
    kept_indices = torch.gather(
        patches.indices,
        1,
        indices_sort,
    )

    return Embeddings(
        data=kept_data,
        indices=kept_indices,
        image_shape=patches.image_shape,
        patch_size=patches.patch_size,
    )


def unflatten_views[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](emb: FlatViewEmbeddings[B, BV, V, N, D]) -> ViewEmbeddings[B, V, N, D]:
    b = emb.original_batch_size
    v = emb.num_views
    _, n, d = emb.data.shape

    return ViewEmbeddings(
        data=emb.data.view(b, v, n, d),
        indices=emb.indices.view(b, v, n),
        image_shape=emb.image_shape,
        patch_size=emb.patch_size,
    )
