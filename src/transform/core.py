from typing import Iterable, Generator, Callable, Concatenate, Literal
from functools import wraps
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
    Batch,
    Height,
    Width,
    Channel,
    View,
    CHW,
    Embeddings,
    Patches,
    NumPatches,
    PatchDim,
    FlatViewEmbeddings,
    FlatViewPatches,
    ViewEmbeddings,
    BatchView,
    EmbedDim,
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


def _to_numpy_img[H: Height, W: Width, C: Channel, M: Mode, R: Range](
    image: PILImage[tuple[H, W, C], M, R],
) -> ArrayImage[tuple[C, H, W], M, R]:
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


# --- Crop ---


def get_random_crop_params(
    h: int, w: int, crop_size: int, device: torch.device
) -> tuple[int, int]:
    x = torch.randint(0, w - crop_size + 1, size=(1,), device=device).item()
    y = torch.randint(0, h - crop_size + 1, size=(1,), device=device).item()
    return int(x), int(y)


def apply_crop_img[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R],
    x: int,
    y: int,
    crop_size: int,
) -> TensorImage[tuple[C, Height, Width], M, R]:
    return TensorImage(image.data[:, y : y + crop_size, x : x + crop_size])


def apply_crop_boxes(boxes: BoundingBoxes, x: int, y: int) -> BoundingBoxes:
    new_data = boxes.data.clone()
    if boxes.format == "xyxy":
        new_data[:, [0, 2]] -= x
        new_data[:, [1, 3]] -= y
    # TODO: handle cxcywh if needed
    return BoundingBoxes(new_data, boxes.format)


# --- Affine ---


def get_affine_matrices(
    bv: int, max_angle_deg: float, max_translate: float, device: torch.device
) -> torch.Tensor:
    angles = [random.uniform(-max_angle_deg, max_angle_deg) for _ in range(bv)]
    translates = [
        (
            random.uniform(-max_translate, max_translate),
            random.uniform(-max_translate, max_translate),
        )
        for _ in range(bv)
    ]
    return torch.stack(
        [
            _create_affine_matrix(a, t, device)
            for a, t in zip(angles, translates)
        ]
    )


def apply_flatview_affine_img[B: Batch, V: View, M: Mode, R: Range](
    image: FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R],
    matrices: torch.Tensor,
) -> FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R]:
    bv, c, h, w = image.data.shape
    grid = F.affine_grid(matrices, [bv, c, h, w], align_corners=False)
    transformed_data = F.grid_sample(
        image.data,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return replace(image, data=transformed_data)


def apply_flatview_affine_boxes(
    boxes: BoundingBoxes, matrices: torch.Tensor
) -> BoundingBoxes:
    # TODO: Implement bounding box rotation/translation using the affine matrices
    return boxes


# --- Patch Drop ---


def get_random_patch_drop_indices(
    bv: int, n: int, drop_rate: float, device: torch.device
) -> torch.Tensor:
    num_keep = max(1, int(n * (1.0 - drop_rate)))
    noise = torch.rand(bv, n, device=device)
    ids_keep = torch.argsort(noise, dim=1)[:, :num_keep]
    return ids_keep


def apply_flatview_patch_drop_img[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](
    patches: FlatViewEmbeddings[B, BV, V, N, D], ids_keep: torch.Tensor
) -> FlatViewEmbeddings[B, BV, V, NumPatches, D]:
    bv, n, d = patches.data.shape
    kept_data = torch.gather(
        patches.data, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d)
    )
    kept_indices = torch.gather(patches.indices, 1, ids_keep)
    return replace(patches, data=kept_data, indices=kept_indices)


def apply_flatview_patch_drop_labels(
    labels: ClassLabels, ids_keep: torch.Tensor
) -> ClassLabels:
    # TODO: Implement label dropping if doing dense patch-level prediction
    return labels
