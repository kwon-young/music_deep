import torch
from dataclasses import replace
from .core import (
    transform,
    batched_transform,
    to_numpy_img,
    to_tensor_img,
    to_float1_img,
    to_device,
    extract_patches_img,
    random_crop_params,
    crop_img,
    crop_boxes,
    affine_matrix_params,
    random_affine_img,
    affine_boxes,
    random_patch_drop_indices,
    patch_drop_img,
    patch_drop_labels,
)
from music_types import (
    DetectionSample,
    PILImage,
    ArrayImage,
    TensorImage,
    BoundingBoxes,
    ClassLabels,
    Height,
    Width,
    Channel,
    Range,
    Mode,
    Layout,
    AnyLayouts,
    Batch,
    CHW,
    NumPatches,
    PatchDim,
    Patches,
    Data,
    BatchedData,
    Int255,
    Float1,
    View,
    BatchView,
    FlatViewTensorImage,
    FlatViewEmbeddings,
    EmbedDim,
)


@transform
def to_numpy[H: Height, W: Width, C: Channel, M: Mode, R: Range, B, L](
    sample: DetectionSample[PILImage[tuple[H, W, C], M, R], B, L],
) -> DetectionSample[ArrayImage[tuple[C, H, W], M, R], B, L]:
    return DetectionSample(
        image=to_numpy_img(sample.image),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def to_tensor[L: Layout, M: Mode, R: Range, B, Lbl](
    sample: DetectionSample[ArrayImage[L, M, R], B, Lbl],
) -> DetectionSample[TensorImage[L, M, R], B, Lbl]:
    return DetectionSample(
        image=to_tensor_img(sample.image),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def to_float1[L: AnyLayouts, M: Mode, B, Lbl](
    sample: DetectionSample[TensorImage[L, M, Int255], B, Lbl],
) -> DetectionSample[TensorImage[L, M, Float1], B, Lbl]:
    return DetectionSample(
        image=to_float1_img(sample.image),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def to[I: TensorImage, B: BoundingBoxes, L: ClassLabels](
    sample: DetectionSample[I, B, L], device: torch.device
) -> DetectionSample[I, B, L]:
    return DetectionSample(
        image=to_device(sample.image, device),
        boxes=to_device(sample.boxes, device),
        labels=to_device(sample.labels, device),
    )


@batched_transform
def extract_patches[B: Batch, Bx, L](
    sample: DetectionSample[TensorImage[tuple[B, *CHW], Mode, Range], Bx, L],
    patch_size: tuple[int, int],
) -> DetectionSample[Patches[B, NumPatches, PatchDim], Bx, L]:
    return DetectionSample(
        image=extract_patches_img(sample.image, patch_size),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def random_crop[C: Channel, M: Mode, R: Range, L](
    sample: DetectionSample[
        TensorImage[tuple[C, Height, Width], M, R], BoundingBoxes, L
    ],
    crop_size: int,
) -> DetectionSample[
    TensorImage[tuple[C, Height, Width], M, R], BoundingBoxes, L
]:
    c, h, w = sample.image.data.shape
    x, y = random_crop_params(h, w, crop_size, sample.image.data.device)

    new_img = crop_img(sample.image, x, y, crop_size)
    new_boxes = crop_boxes(sample.boxes, x, y)

    return DetectionSample(image=new_img, boxes=new_boxes, labels=sample.labels)


@transform
def random_flatview_affine[I: FlatViewTensorImage, L](
    sample: DetectionSample[I, BoundingBoxes, L],
    max_angle_deg: float,
    max_translate: float,
) -> DetectionSample[I, BoundingBoxes, L]:
    bv = sample.image.data.shape[0]
    matrices = affine_matrix_params(
        bv, max_angle_deg, max_translate, sample.image.data.device
    )

    new_img_base = random_affine_img(sample.image, matrices)
    new_boxes = affine_boxes(sample.boxes, matrices)

    return DetectionSample(
        image=replace(sample.image, data=new_img_base.data),
        boxes=new_boxes,
        labels=sample.labels
    )


@batched_transform
def random_flatview_patch_drop[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim | PatchDim,
    Bx,
](
    sample: DetectionSample[
        FlatViewEmbeddings[B, BV, V, N, D], Bx, ClassLabels
    ],
    drop_rate: float,
) -> DetectionSample[
    FlatViewEmbeddings[B, BV, V, NumPatches, D], Bx, ClassLabels
]:
    bv, n, _ = sample.image.data.shape
    ids_keep = random_patch_drop_indices(
        bv, n, drop_rate, sample.image.data.device
    )

    new_img_base = patch_drop_img(sample.image, ids_keep)
    new_labels = patch_drop_labels(sample.labels, ids_keep)

    new_img = replace(
        sample.image,
        data=new_img_base.data,
        indices=new_img_base.indices,
    )

    return DetectionSample(image=new_img, boxes=sample.boxes, labels=new_labels)


def collate_tensors[
    Meta,
    C: Channel,
    H: Height,
    W: Width,
    M: Mode,
    R: Range,
    B,
    L,
](
    batch: tuple[
        Data[Meta, DetectionSample[TensorImage[tuple[C, H, W], M, R], B, L]],
        ...,
    ],
) -> BatchedData[
    Meta,
    DetectionSample[TensorImage[tuple[Batch, C, H, W], M, R], list[B], list[L]],
]:
    """Collates a tuple of Data[DetectionSample] into a BatchedData[DetectionSample]."""
    m = [b.metadata for b in batch]
    i = [b.data.image.data for b in batch]
    boxes = [b.data.boxes for b in batch]
    labels = [b.data.labels for b in batch]

    return BatchedData(
        metadata=m,
        data=DetectionSample(
            image=TensorImage(torch.stack(i, dim=0)), boxes=boxes, labels=labels
        ),
    )
