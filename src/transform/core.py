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
