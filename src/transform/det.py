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
    random_patch_drop_indices,
    patch_drop_img,
    stack_tensor_img,
    pad_to_patch_size_img,
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
    RGB,
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


@transform
def pad_to_patch_size[C: Channel, H: Height, W: Width, M: Mode, R: Range, Bx, Lbl](
    sample: DetectionSample[TensorImage[tuple[C, H, W], M, R], Bx, Lbl],
    patch_size: tuple[int, int],
) -> DetectionSample[TensorImage[tuple[C, Height, Width], M, R], Bx, Lbl]:
    return DetectionSample(
        image=pad_to_patch_size_img(sample.image, patch_size),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@batched_transform
def extract_patches[B: Batch, Bx, L](
    sample: DetectionSample[TensorImage[tuple[B, *CHW], RGB, Float1], Bx, L],
    patch_size: tuple[int, int],
) -> DetectionSample[Patches[B, NumPatches, PatchDim], Bx, L]:
    return DetectionSample(
        image=extract_patches_img(sample.image, patch_size),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@batched_transform
def random_patch_drop[I: Patches, B, L](
    sample: DetectionSample[I, B, L],
    drop_rate: float,
) -> DetectionSample[I, B, L]:
    b, n, _ = sample.image.data.shape
    ids_keep = random_patch_drop_indices(
        b, n, drop_rate, sample.image.data.device
    )

    new_img_base = patch_drop_img(sample.image, ids_keep)

    new_img = replace(
        sample.image,
        data=new_img_base.data,
        indices=new_img_base.indices,
    )

    return DetectionSample(
        image=new_img, boxes=sample.boxes, labels=sample.labels
    )


def collate[
    Meta,
    C: Channel,
    H: Height,
    W: Width,
    M: Mode,
    R: Range,
    Bx,
    Lbl,
](
    batch: tuple[
        Data[
            Meta,
            DetectionSample[
                TensorImage[tuple[C, H, W], M, R],
                Bx,
                Lbl,
            ],
        ],
        ...,
    ],
) -> BatchedData[
    Meta,
    DetectionSample[
        TensorImage[tuple[Batch, C, H, W], M, R],
        list[Bx],
        list[Lbl],
    ],
]:
    m = [b.metadata for b in batch]

    stacked_image = stack_tensor_img([b.sample.image for b in batch])
    boxes_list = [b.sample.boxes for b in batch]
    labels_list = [b.sample.labels for b in batch]

    return BatchedData(
        metadata=m,
        sample=DetectionSample(
            image=stacked_image,
            boxes=boxes_list,
            labels=labels_list,
        ),
    )
