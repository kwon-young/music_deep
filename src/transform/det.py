import torch
from .core import (
    transform,
    batched_transform,
    _to_numpy_img,
    _to_tensor_img,
    _to_float1_img,
    _to_device_img,
    _extract_patches_img,
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
    RGB,
    Int255,
    Float1,
)


@transform
def to_numpy[H: Height, W: Width, C: Channel, R: Range, B, L](
    sample: DetectionSample[PILImage[tuple[H, W, C], RGB, R], B, L],
) -> DetectionSample[ArrayImage[tuple[C, H, W], RGB, R], B, L]:
    return DetectionSample(
        image=_to_numpy_img(sample.image),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def to_tensor[L: Layout, M: Mode, R: Range, B, Lbl](
    sample: DetectionSample[ArrayImage[L, M, R], B, Lbl],
) -> DetectionSample[TensorImage[L, M, R], B, Lbl]:
    return DetectionSample(
        image=_to_tensor_img(sample.image),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def to_float1[L: AnyLayouts, M: Mode, B, Lbl](
    sample: DetectionSample[TensorImage[L, M, Int255], B, Lbl],
) -> DetectionSample[TensorImage[L, M, Float1], B, Lbl]:
    return DetectionSample(
        image=_to_float1_img(sample.image),
        boxes=sample.boxes,
        labels=sample.labels,
    )


@transform
def to[I: TensorImage, B, L](
    sample: DetectionSample[I, B, L], device: torch.device
) -> DetectionSample[I, B, L]:
    new_boxes = sample.boxes
    if isinstance(sample.boxes, BoundingBoxes):
        new_boxes = BoundingBoxes(
            sample.boxes.data.to(device), sample.boxes.format
        )
    new_labels = sample.labels
    if isinstance(sample.labels, ClassLabels):
        new_labels = ClassLabels(sample.labels.data.to(device))

    return DetectionSample(
        image=_to_device_img(sample.image, device),
        boxes=new_boxes,
        labels=new_labels,
    )


@batched_transform
def extract_patches[B: Batch, M: Mode, R: Range, Bx, L](
    sample: DetectionSample[TensorImage[tuple[B, *CHW], M, R], Bx, L],
    patch_size: tuple[int, int],
) -> DetectionSample[Patches[B, NumPatches, PatchDim], Bx, L]:
    return DetectionSample(
        image=_extract_patches_img(sample.image, patch_size),
        boxes=sample.boxes,
        labels=sample.labels,
    )


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
