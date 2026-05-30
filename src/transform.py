from typing import Iterable, Generator, Callable, Concatenate, Any
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


# --- Namespaces ---

class det:
    @staticmethod
    @transform
    def to_numpy[H: Height, W: Width, C: Channel, R: Range, B, L](
        sample: DetectionSample[PILImage[tuple[H, W, C], RGB, R], B, L],
    ) -> DetectionSample[ArrayImage[tuple[C, H, W], RGB, R], B, L]:
        return DetectionSample(
            image=_to_numpy_img(sample.image),
            boxes=sample.boxes,
            labels=sample.labels
        )

    @staticmethod
    @transform
    def to_tensor[L: Layout, M: Mode, R: Range, B, Lbl](
        sample: DetectionSample[ArrayImage[L, M, R], B, Lbl],
    ) -> DetectionSample[TensorImage[L, M, R], B, Lbl]:
        return DetectionSample(
            image=_to_tensor_img(sample.image),
            boxes=sample.boxes,
            labels=sample.labels
        )

    @staticmethod
    @transform
    def to_float1[L: AnyLayouts, M: Mode, B, Lbl](
        sample: DetectionSample[TensorImage[L, M, Int255], B, Lbl],
    ) -> DetectionSample[TensorImage[L, M, Float1], B, Lbl]:
        return DetectionSample(
            image=_to_float1_img(sample.image),
            boxes=sample.boxes,
            labels=sample.labels
        )

    @staticmethod
    @transform
    def to[I: TensorImage, B, L](
        sample: DetectionSample[I, B, L], device: torch.device
    ) -> DetectionSample[I, B, L]:
        new_boxes = sample.boxes
        if isinstance(sample.boxes, BoundingBoxes):
            new_boxes = BoundingBoxes(sample.boxes.data.to(device), sample.boxes.format)
        new_labels = sample.labels
        if isinstance(sample.labels, ClassLabels):
            new_labels = ClassLabels(sample.labels.data.to(device))
            
        return DetectionSample(
            image=_to_device_img(sample.image, device),
            boxes=new_boxes,
            labels=new_labels
        )

    @staticmethod
    @batched_transform
    def extract_patches[B: Batch, M: Mode, R: Range, Bx, L](
        sample: DetectionSample[TensorImage[tuple[B, *CHW], M, R], Bx, L],
        patch_size: tuple[int, int],
    ) -> DetectionSample[Patches[B, NumPatches, PatchDim], Bx, L]:
        return DetectionSample(
            image=_extract_patches_img(sample.image, patch_size),
            boxes=sample.boxes,
            labels=sample.labels
        )


def collate_tensors[Meta, C: Channel, H: Height, W: Width, M: Mode, R: Range, B, L](
    batch: tuple[Data[Meta, DetectionSample[TensorImage[tuple[C, H, W], M, R], B, L]], ...],
) -> BatchedData[Meta, DetectionSample[TensorImage[tuple[Batch, C, H, W], M, R], list[B], list[L]]]:
    """Collates a tuple of Data[DetectionSample] into a BatchedData[DetectionSample]."""
    m = [b.metadata for b in batch]
    i = [b.data.image.data for b in batch]
    boxes = [b.data.boxes for b in batch]
    labels = [b.data.labels for b in batch]
    
    return BatchedData(
        metadata=m,
        data=DetectionSample(
            image=TensorImage(torch.stack(i, dim=0)),
            boxes=boxes,
            labels=labels
        ),
    )
