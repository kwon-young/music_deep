import torch
from typing import Literal
from .core import (
    transform,
    batched_transform,
    _to_numpy_img,
    _to_tensor_img,
    _to_float1_img,
    _to_device_img,
    _make_views_img,
    _extract_flatviewpatches_img,
    _unflatten_views_img,
    get_random_crop_params,
    apply_crop_img,
    get_affine_matrices,
    apply_flatview_affine_img,
    get_random_patch_drop_indices,
    apply_flatview_patch_drop_img,
)
from music_types import (
    SSLSample,
    PILImage,
    ArrayImage,
    TensorImage,
    FlatViewTensorImage,
    FlatViewPatches,
    FlatViewEmbeddings,
    ViewEmbeddings,
    Height,
    Width,
    Channel,
    Range,
    Mode,
    Layout,
    AnyLayouts,
    Batch,
    View,
    BatchView,
    CHW,
    NumPatches,
    PatchDim,
    EmbedDim,
    Data,
    BatchedData,
    RGB,
    Int255,
    Float1,
)


@transform
def to_numpy[H: Height, W: Width, C: Channel, R: Range](
    sample: SSLSample[PILImage[tuple[H, W, C], RGB, R]],
) -> SSLSample[ArrayImage[tuple[C, H, W], RGB, R]]:
    return SSLSample(image=_to_numpy_img(sample.image))


@transform
def to_tensor[L: Layout, M: Mode, R: Range](
    sample: SSLSample[ArrayImage[L, M, R]],
) -> SSLSample[TensorImage[L, M, R]]:
    return SSLSample(image=_to_tensor_img(sample.image))


@transform
def to_float1[L: AnyLayouts, M: Mode](
    sample: SSLSample[TensorImage[L, M, Int255]],
) -> SSLSample[TensorImage[L, M, Float1]]:
    return SSLSample(image=_to_float1_img(sample.image))


@transform
def to[I: TensorImage](
    sample: SSLSample[I], device: torch.device
) -> SSLSample[I]:
    return SSLSample(image=_to_device_img(sample.image, device))


@transform
def random_crop[C: Channel, M: Mode, R: Range](
    sample: SSLSample[TensorImage[tuple[C, Height, Width], M, R]],
    crop_size: int,
) -> SSLSample[TensorImage[tuple[C, Height, Width], M, R]]:
    c, h, w = sample.image.data.shape
    x, y = get_random_crop_params(h, w, crop_size, sample.image.data.device)
    return SSLSample(image=apply_crop_img(sample.image, x, y, crop_size))


@transform
def make_views[L: CHW, M: Mode, R: Range](
    sample: SSLSample[TensorImage[L, M, R]], n: int
) -> SSLSample[
    FlatViewTensorImage[Literal[1], View, tuple[BatchView, *CHW], M, R]
]:
    return SSLSample(image=_make_views_img(sample.image, n))


@transform
def random_flatview_affine[B: Batch, V: View, M: Mode, R: Range](
    sample: SSLSample[FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R]],
    max_angle_deg: float,
    max_translate: float,
) -> SSLSample[FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R]]:
    bv = sample.image.data.shape[0]
    matrices = get_affine_matrices(
        bv, max_angle_deg, max_translate, sample.image.data.device
    )
    return SSLSample(image=apply_flatview_affine_img(sample.image, matrices))


@batched_transform
def extract_flatviewpatches[B: Batch, V: View, M: Mode, R: Range](
    sample: SSLSample[FlatViewTensorImage[B, V, tuple[BatchView, *CHW], M, R]],
    patch_size: tuple[int, int],
) -> SSLSample[FlatViewPatches[B, BatchView, V, NumPatches, PatchDim]]:
    return SSLSample(
        image=_extract_flatviewpatches_img(sample.image, patch_size)
    )


@batched_transform
def random_flatview_patch_drop[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](
    sample: SSLSample[FlatViewEmbeddings[B, BV, V, N, D]], drop_rate: float
) -> SSLSample[FlatViewEmbeddings[B, BV, V, NumPatches, D]]:
    bv, n, _ = sample.image.data.shape
    ids_keep = get_random_patch_drop_indices(
        bv, n, drop_rate, sample.image.data.device
    )
    return SSLSample(
        image=apply_flatview_patch_drop_img(sample.image, ids_keep)
    )


def unflatten_views[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](patches: FlatViewEmbeddings[B, BV, V, N, D]) -> ViewEmbeddings[B, V, N, D]:
    # This operates directly on the embeddings (used after the model forward pass)
    return _unflatten_views_img(patches)


def collate[Meta, V: View, C: Channel, H: Height, W: Width, M: Mode, R: Range](
    batch: tuple[
        Data[
            Meta,
            SSLSample[
                FlatViewTensorImage[
                    Literal[1], V, tuple[BatchView, C, H, W], M, R
                ]
            ],
        ],
        ...,
    ],
    batch_size: int,
) -> BatchedData[
    Meta,
    SSLSample[FlatViewTensorImage[Batch, V, tuple[BatchView, C, H, W], M, R]],
]:
    m = [b.metadata for b in batch]
    i = [b.data.image.data for b in batch]

    stacked_data = torch.cat(i, dim=0)
    num_views = batch[0].data.image.num_views

    return BatchedData(
        metadata=m,
        data=SSLSample(
            image=FlatViewTensorImage(
                stacked_data,
                num_views=num_views,
                original_batch_size=len(batch),
            )
        ),
    )
