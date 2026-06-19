import torch
from typing import Literal
from dataclasses import replace
from .core import (
    transform,
    batched_transform,
    decode_nvimgcodec_img,
    decode_pyvips_img,
    decode_and_crop_pyvips_img,
    to_numpy_img,
    to_tensor_img,
    to_float1_img,
    to_device,
    to_device_embeddings,
    make_views_img,
    extract_flatviewpatches_img,
    unflatten_views_img,
    random_crop_params,
    crop_img,
    affine_matrix_params,
    random_affine_img,
    get_random_affine_params,
    get_affine_matrices,
    affine_img,
    pad_to_patch_size_img,
    extract_patches_img,
    random_patch_drop_indices,
    variance_patch_drop_indices,
    spatial_mask_drop_indices,
    patch_drop_img,
    stack_tensor_img,
)
from music_types import (
    SSLSample,
    MaskedPair,
    LazyImage,
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
    Patches,
)


@transform
def to_numpy[H: Height, W: Width, C: Channel, R: Range](
    sample: SSLSample[PILImage[tuple[H, W, C], RGB, R]],
) -> SSLSample[ArrayImage[tuple[C, H, W], RGB, R]]:
    return SSLSample(image=to_numpy_img(sample.image))


@transform
def to_tensor[L: Layout, M: Mode, R: Range](
    sample: SSLSample[ArrayImage[L, M, R]],
) -> SSLSample[TensorImage[L, M, R]]:
    return SSLSample(image=to_tensor_img(sample.image))


@transform
def to_float1[L: AnyLayouts, M: Mode](
    sample: SSLSample[TensorImage[L, M, Int255]],
) -> SSLSample[TensorImage[L, M, Float1]]:
    return SSLSample(image=to_float1_img(sample.image))


@transform
def to[I: TensorImage](
    sample: SSLSample[I], device: torch.device
) -> SSLSample[I]:
    return SSLSample(image=to_device(sample.image, device))


@transform
def decode_nvimgcodec(sample: SSLSample[LazyImage], device: torch.device) -> SSLSample[TensorImage[CHW, RGB, Int255]]:
    return SSLSample(image=decode_nvimgcodec_img(sample.image, device))


@transform
def decode_pyvips(sample: SSLSample[LazyImage], device: torch.device) -> SSLSample[TensorImage[CHW, RGB, Int255]]:
    return SSLSample(image=decode_pyvips_img(sample.image, device))


@transform
def decode_and_crop_pyvips(sample: SSLSample[LazyImage], crop_size: int, device: torch.device) -> SSLSample[TensorImage[CHW, RGB, Int255]]:
    h, w = sample.image.height, sample.image.width
    x, y = random_crop_params(h, w, crop_size, torch.device("cpu"))
    return SSLSample(image=decode_and_crop_pyvips_img(sample.image, x, y, crop_size, device))


@transform
def random_affine(
    sample: SSLSample[TensorImage[tuple[Channel, Height, Width], Mode, Float1]], 
    max_translate_frac: float, 
    max_angle_deg: float, 
    max_shear_deg: float, 
    max_scale: float
) -> SSLSample[TensorImage[tuple[Channel, Height, Width], Mode, Float1]]:
    c, h, w = sample.image.data.shape
    device = sample.image.data.device
    tx, ty, angle, shear, scale = get_random_affine_params(max_translate_frac, max_angle_deg, max_shear_deg, max_scale)
    fwd_matrix, theta_grid = get_affine_matrices(img_h=h, img_w=w, tx_frac=tx, ty_frac=ty, angle_deg=angle, shear_deg=shear, scale=scale, device=device)
    return SSLSample(image=affine_img(sample.image, theta_grid))


@transform
def pad_to_patch_size[C: Channel, H: Height, W: Width, M: Mode, R: Range](
    sample: SSLSample[TensorImage[tuple[C, H, W], M, R]], patch_size: tuple[int, int]
) -> SSLSample[TensorImage[tuple[C, Height, Width], M, R]]:
    return SSLSample(image=pad_to_patch_size_img(sample.image, patch_size))


@batched_transform
def extract_patches[B: Batch](
    sample: SSLSample[TensorImage[tuple[B, *CHW], RGB, Float1]], patch_size: tuple[int, int]
) -> SSLSample[Patches[B, NumPatches, PatchDim]]:
    return SSLSample(image=extract_patches_img(sample.image, patch_size))


@batched_transform
def variance_patch_drop[I: Patches](
    sample: SSLSample[I], var_threshold: float | None = None, drop_rate: float | None = None
) -> SSLSample[I]:
    ids_keep = variance_patch_drop_indices(sample.image.data, var_threshold=var_threshold, drop_rate=drop_rate)
    new_img_base = patch_drop_img(sample.image, ids_keep)
    new_img = replace(sample.image, data=new_img_base.data, indices=new_img_base.indices)
    return SSLSample(image=new_img)


@batched_transform
def random_spatial_mask[B: Batch, N: NumPatches, P: PatchDim](
    sample: SSLSample[Patches[B, N, P]], drop_ratio: float
) -> SSLSample[MaskedPair[B, N, P]]:
    target_patches = sample.image
    _, _, w = target_patches.image_shape
    _, pw = target_patches.patch_size
    grid_w = w // pw
    
    ids_keep = spatial_mask_drop_indices(target_patches.indices, grid_w=grid_w, drop_ratio=drop_ratio)
    context_patches = patch_drop_img(target_patches, ids_keep)
    
    return SSLSample(image=MaskedPair(target=target_patches, context=context_patches))


def collate_images[Meta, C: Channel, H: Height, W: Width, M: Mode, R: Range](
    batch: tuple[Data[Meta, SSLSample[TensorImage[tuple[C, H, W], M, R]]], ...]
) -> BatchedData[Meta, SSLSample[TensorImage[tuple[Batch, C, H, W], M, R]]]:
    m = [b.metadata for b in batch]
    stacked_image = stack_tensor_img([b.sample.image for b in batch])
    return BatchedData(metadata=m, sample=SSLSample(image=stacked_image))


@transform
def to_masked_patches[B: Batch, N: NumPatches, P: PatchDim](
    sample: SSLSample[MaskedPair[B, N, P]], device: torch.device
) -> SSLSample[MaskedPair[B, N, P]]:
    return SSLSample(
        image=MaskedPair(
            target=to_device_embeddings(sample.image.target, device),
            context=to_device_embeddings(sample.image.context, device)
        )
    )


@transform
def random_crop[C: Channel, M: Mode, R: Range](
    sample: SSLSample[TensorImage[tuple[C, Height, Width], M, R]],
    crop_size: int,
) -> SSLSample[TensorImage[tuple[C, Height, Width], M, R]]:
    c, h, w = sample.image.data.shape
    x, y = random_crop_params(h, w, crop_size, sample.image.data.device)
    return SSLSample(image=crop_img(sample.image, x, y, crop_size))


@transform
def make_views[L: CHW, M: Mode, R: Range](
    sample: SSLSample[TensorImage[L, M, R]], n: int
) -> SSLSample[
    FlatViewTensorImage[Literal[1], View, tuple[BatchView, *CHW], M, R]
]:
    return SSLSample(image=make_views_img(sample.image, n))


@transform
def random_flatview_affine[I: FlatViewTensorImage](
    sample: SSLSample[I],
    max_angle_deg: float,
    max_translate: float,
) -> SSLSample[I]:
    bv = sample.image.data.shape[0]
    matrices = affine_matrix_params(
        bv, max_angle_deg, max_translate, sample.image.data.device
    )

    new_img_base = random_affine_img(sample.image, matrices)

    return SSLSample(image=replace(sample.image, data=new_img_base.data))


@batched_transform
def extract_flatviewpatches[B: Batch, V: View, BV: BatchView](
    sample: SSLSample[FlatViewTensorImage[B, V, tuple[BV, *CHW], Mode, Range]],
    patch_size: tuple[int, int],
) -> SSLSample[FlatViewPatches[B, BV, V, NumPatches, PatchDim]]:
    return SSLSample(
        image=extract_flatviewpatches_img(sample.image, patch_size)
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
    ids_keep = random_patch_drop_indices(
        bv, n, drop_rate, sample.image.data.device
    )

    new_img_base = patch_drop_img(sample.image, ids_keep)

    return SSLSample(
        image=replace(
            sample.image,
            data=new_img_base.data,
            indices=new_img_base.indices,
        )
    )


def unflatten_views[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](patches: FlatViewEmbeddings[B, BV, V, N, D]) -> ViewEmbeddings[B, V, N, D]:
    # This operates directly on the embeddings (used after the model forward pass)
    return unflatten_views_img(patches)


def collate[Meta, V: View, C: Channel, H: Height, W: Width, M: Mode, R: Range](
    batch: tuple[
        Data[
            Meta,
            SSLSample[
                FlatViewTensorImage[Batch, V, tuple[BatchView, C, H, W], M, R]
            ],
        ],
        ...,
    ],
) -> BatchedData[
    Meta,
    SSLSample[FlatViewTensorImage[Batch, V, tuple[BatchView, C, H, W], M, R]],
]:
    b0 = batch[0]
    m, i = [b0.metadata], [b0.sample.image.data]
    for b in batch[1:]:
        m.append(b.metadata)
        i.append(b.sample.image.data)
    v = b0.sample.image.num_views
    ob = len(batch) * b0.sample.image.original_batch_size

    stacked_data = stack_tensor_img(i)

    return BatchedData(
        metadata=m,
        sample=SSLSample(
            image=FlatViewTensorImage(
                stacked_data,
                num_views=v,
                original_batch_size=ob,
            )
        ),
    )
