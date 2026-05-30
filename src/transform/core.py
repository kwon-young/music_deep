from typing import Iterable, Generator, Callable, Concatenate, Any, Literal
from functools import wraps
from itertools import batched
from dataclasses import replace
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
    FlatViewTensorImage,
    Image,
    BatchedImage,
    Batch,
    Height,
    Width,
    Channel,
    View,
    CHW,
    BCHW,
    BVCHW,
    RGB,
    Embeddings,
    Patches,
    NumPatches,
    PatchDim,
    FlatViewEmbeddings,
    FlatViewPatches,
    ViewEmbeddings,
    BatchView,
    EmbedDim,
    SSLSample,
    ClassificationSample,
    DetectionSample,
    BoundingBoxes,
    ClassLabels,
)


def transform[Meta, T, U, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[Data[Meta, T], P], Data[Meta, U]]:
    @wraps(func)
    def wrapper(
        item: Data[Meta, T], *args: P.args, **kwargs: P.kwargs
    ) -> Data[Meta, U]:
        return Data(item.metadata, func(item.data, *args, **kwargs))

    return wrapper


def batched_transform[Meta, T, U, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[BatchedData[Meta, T], P], BatchedData[Meta, U]]:
    @wraps(func)
    def wrapper(
        batch: BatchedData[Meta, T], *args: P.args, **kwargs: P.kwargs
    ) -> BatchedData[Meta, U]:
        return BatchedData(batch.metadata, func(batch.data, *args, **kwargs))

    return wrapper


# --- Core Math ---


def _to_numpy_img[H: Height, W: Width, C: Channel, R: Range](
    image: PILImage[tuple[H, W, C], RGB, R],
) -> ArrayImage[tuple[C, H, W], RGB, R]:
    arr = np.array(image.data)
    arr = np.transpose(arr, (2, 0, 1))
    return ArrayImage(arr)


def _to_tensor_img[L: Layout, M: Mode, R: Range](
    image: ArrayImage[L, M, R],
) -> TensorImage[L, M, R]:
    return TensorImage(torch.as_tensor(image.data))


def _to_float1_img[L: AnyLayouts, M: Mode](
    image: TensorImage[L, M, Int255],
) -> TensorImage[L, M, Float1]:
    return TensorImage(image.data.float() / 255.0)


def _to_device_img[I: TensorImage](image: I, device: torch.device) -> I:
    return replace(image, data=image.data.to(device))


def _extract_patches_img[B: Batch, M: Mode, R: Range](
    image: TensorImage[tuple[B, *CHW], M, R],
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


def shuffle[T](
    iterator: Iterable[T], buffer_size: int = 1000
) -> Generator[T, None, None]:
    buffer = []
    for item in iterator:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)
    random.shuffle(buffer)
    yield from buffer


def _random_crop_img[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R], crop_size: int
) -> TensorImage[tuple[C, Height, Width], M, R]:
    c, h, w = image.data.shape
    x = torch.randint(
        0, w - crop_size + 1, size=(1,), device=image.data.device
    )[0]
    y = torch.randint(
        0, h - crop_size + 1, size=(1,), device=image.data.device
    )[0]
    return TensorImage(image.data[:, y : y + crop_size, x : x + crop_size])


def _make_views_img[L: CHW, M: Mode, R: Range](
    image: TensorImage[L, M, R], n: int
) -> FlatViewTensorImage[Literal[1], View, tuple[BatchView, *CHW], M, R]:
    data = image.data.unsqueeze(0).expand(n, -1, -1, -1)
    return FlatViewTensorImage(data, num_views=n, original_batch_size=1)


def _create_affine_matrix(
    angle_deg: float, translate: tuple[float, float], device: torch.device
) -> torch.Tensor:
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    tx, ty = translate
    return torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
        dtype=torch.float32,
        device=device,
    )


def _random_flatview_affine_img[B: Batch, V: View, M: Mode, R: Range](
    image: FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R],
    max_angle_deg: float,
    max_translate: float,
) -> FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R]:
    bv, c, h, w = image.data.shape
    device = image.data.device

    angles = [random.uniform(-max_angle_deg, max_angle_deg) for _ in range(bv)]
    translates = [
        (
            random.uniform(-max_translate, max_translate),
            random.uniform(-max_translate, max_translate),
        )
        for _ in range(bv)
    ]

    matrices = torch.stack(
        [
            _create_affine_matrix(a, t, device)
            for a, t in zip(angles, translates)
        ]
    )

    grid = F.affine_grid(matrices, [bv, c, h, w], align_corners=False)
    transformed_data = F.grid_sample(
        image.data,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    return replace(image, data=transformed_data)


def _extract_flatviewpatches_img[B: Batch, V: View, M: Mode, R: Range](
    image: FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R],
    patch_size: tuple[int, int],
) -> FlatViewPatches[B, BatchView, V, NumPatches, PatchDim]:
    patches = _extract_patches_img(image, patch_size)
    return FlatViewEmbeddings(
        data=patches.data,
        indices=patches.indices,
        image_shape=patches.image_shape,
        patch_size=patches.patch_size,
        num_views=image.num_views,
        original_batch_size=image.original_batch_size,
    )


def _random_flatview_patch_drop_img[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](
    patches: FlatViewEmbeddings[B, BV, V, N, D], drop_rate: float
) -> FlatViewEmbeddings[B, BV, V, NumPatches, D]:
    bv, n, d = patches.data.shape
    device = patches.data.device

    num_keep = max(1, int(n * (1.0 - drop_rate)))
    noise = torch.rand(bv, n, device=device)
    ids_keep = torch.argsort(noise, dim=1)[:, :num_keep]

    kept_data = torch.gather(
        patches.data, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d)
    )
    kept_indices = torch.gather(patches.indices, 1, ids_keep)

    return replace(patches, data=kept_data, indices=kept_indices)


def _unflatten_views_img[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](patches: FlatViewEmbeddings[B, BV, V, N, D]) -> ViewEmbeddings[B, V, N, D]:
    b = patches.original_batch_size
    v = patches.num_views
    _, n, d = patches.data.shape

    view_data = patches.data.view(b, v, n, d)
    view_indices = patches.indices.view(b, v, n)

    return ViewEmbeddings(
        data=view_data,
        indices=view_indices,
        image_shape=patches.image_shape,
        patch_size=patches.patch_size,
    )
