from typing import (
    Iterable,
    Generator,
    Callable,
    Concatenate,
    Literal,
    TypeVar,
    Any,
)
from functools import wraps, cache
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
    LazyImage,
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
    NumBoxes,
    BoxDim,
    BoxRange,
    Origin,
    NumClasses,
    XYXY,
    Absolute,
    RGB,
)


def transform[Meta, T, U, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[Data[Meta, T], P], Data[Meta, U]]:
    @wraps(func)
    def wrapper(
        item: Data[Meta, T], *args: P.args, **kwargs: P.kwargs
    ) -> Data[Meta, U]:
        return Data(item.metadata, func(item.sample, *args, **kwargs))

    return wrapper


def batched_transform[Meta, T, U, **P](
    func: Callable[Concatenate[T, P], U],
) -> Callable[Concatenate[BatchedData[Meta, T], P], BatchedData[Meta, U]]:
    @wraps(func)
    def wrapper(
        batch: BatchedData[Meta, T], *args: P.args, **kwargs: P.kwargs
    ) -> BatchedData[Meta, U]:
        return BatchedData(batch.metadata, func(batch.sample, *args, **kwargs))

    return wrapper


def stack_tensor_img[C: Channel, H: Height, W: Width, M: Mode, R: Range](
    items: list[TensorImage[tuple[C, H, W], M, R]],
) -> TensorImage[tuple[Batch, C, H, W], M, R]:
    stacked_tensor = torch.stack([item.data for item in items], dim=0)
    return TensorImage(stacked_tensor)


@cache
def get_nv_decoder(device_id: int = 0):
    try:
        from nvidia.nvimgcodec import Decoder

        return Decoder(device_id=device_id)
    except ImportError as e:
        raise RuntimeError(
            "nvimgcodec is not installed but a CUDA prep_device was requested."
        ) from e


def decode_nvimgcodec_img(
    image: LazyImage,
    device: torch.device,
) -> TensorImage[CHW, RGB, Int255]:
    """Decodes a LazyImage directly to GPU VRAM and formats it as a CHW RGB tensor."""
    device_id = device.index if device.index is not None else 0
    decoder = get_nv_decoder(device_id)
    nv_img = decoder.read(str(image.path))
    t_gpu = torch.from_dlpack(nv_img).to(device)

    # nvimagecodec might return (H, W, 1) for grayscale. Squeeze to (H, W)
    if t_gpu.ndim == 3 and t_gpu.shape[-1] == 1:
        t_gpu = t_gpu.squeeze(-1)

    if t_gpu.ndim == 2:  # Grayscale (H, W)
        t_rgb = t_gpu.unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_rgb = t_gpu.permute(2, 0, 1)

    return TensorImage(t_rgb)


def decode_pyvips_img(
    image: LazyImage,
    device: torch.device,
) -> TensorImage[CHW, RGB, Int255]:
    """Decodes a full LazyImage using pyvips and formats it as a CHW RGB tensor."""
    import pyvips

    vips_img = pyvips.Image.new_from_file(str(image.path))

    arr = np.ndarray(
        buffer=vips_img.write_to_memory(),
        dtype=np.uint8,
        shape=(vips_img.height, vips_img.width, vips_img.bands),
    )
    t = torch.from_numpy(arr).to(device)

    if t.shape[-1] == 1:  # Grayscale (H, W, 1)
        t_rgb = t.squeeze(-1).unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_rgb = t.permute(2, 0, 1)

    return TensorImage(t_rgb)


def decode_and_crop_pyvips_img(
    image: LazyImage,
    x: int,
    y: int,
    crop_size: int,
    device: torch.device,
) -> TensorImage[CHW, RGB, Int255]:
    """Lazily crops a LazyImage using pyvips and formats it as a CHW RGB tensor."""
    import pyvips

    # Open lazily and crop
    vips_img = pyvips.Image.new_from_file(str(image.path))
    crop = vips_img.crop(x, y, crop_size, crop_size)

    # Decode to memory and wrap in numpy (zero-copy from the buffer)
    arr = np.ndarray(
        buffer=crop.write_to_memory(),
        dtype=np.uint8,
        shape=(crop.height, crop.width, crop.bands),
    )
    t = torch.from_numpy(arr).to(device)

    if t.shape[-1] == 1:  # Grayscale (H, W, 1)
        t_rgb = t.squeeze(-1).unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_rgb = t.permute(2, 0, 1)

    return TensorImage(t_rgb)


def to_numpy_img[H: Height, W: Width, C: Channel, M: Mode, R: Range](
    image: PILImage[tuple[H, W, C], M, R],
) -> ArrayImage[tuple[C, H, W], M, R]:
    arr = np.array(image.data)
    arr = np.transpose(arr, (2, 0, 1))
    return ArrayImage(arr)


def to_tensor_img[L: Layout, M: Mode, R: Range](
    image: ArrayImage[L, M, R],
) -> TensorImage[L, M, R]:
    return TensorImage(torch.as_tensor(image.data))


def to_float1_img[L: AnyLayouts, M: Mode](
    image: TensorImage[L, M, Int255],
) -> TensorImage[L, M, Float1]:
    return TensorImage(image.data.float() / 255.0)


def to_device[I: TensorImage | BoundingBoxes | ClassLabels](
    image: I, device: torch.device
) -> I:
    return replace(image, data=image.data.to(device))


def to_device_embeddings[E: Embeddings](
    embeddings: E, device: torch.device
) -> E:
    """Specialized device move for Embeddings/Patches."""
    return replace(
        embeddings,
        data=embeddings.data.to(device),
        indices=embeddings.indices.to(device),
    )


def pad_to_patch_size_img[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R],
    patch_size: tuple[int, int],
) -> TensorImage[tuple[C, Height, Width], M, R]:
    c, h, w = image.data.shape
    ph, pw = patch_size

    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw

    if pad_h > 0 or pad_w > 0:
        # F.pad with mode="replicate" for 2D padding requires 4D input (N, C, H, W)
        padded_data = F.pad(
            image.data.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
        ).squeeze(0)
        return replace(image, data=padded_data)

    return image


def extract_patches_img[B: Batch](
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


def make_views_img[L: CHW, M: Mode, R: Range](
    image: TensorImage[L, M, R], n: int
) -> FlatViewTensorImage[Literal[1], View, tuple[BatchView, *CHW], M, R]:
    data = image.data.unsqueeze(0).expand(n, -1, -1, -1)
    return FlatViewTensorImage(data, num_views=n, original_batch_size=1)


def extract_flatviewpatches_img[B: Batch, V: View, BV: BatchView](
    image: FlatViewTensorImage[B, V, tuple[BV, *CHW], Mode, Range],
    patch_size: tuple[int, int],
) -> FlatViewPatches[B, BV, V, NumPatches, PatchDim]:
    patches = extract_patches_img(image, patch_size)
    return FlatViewEmbeddings(
        data=patches.data,
        indices=patches.indices,
        image_shape=patches.image_shape,
        patch_size=patches.patch_size,
        num_views=image.num_views,
        original_batch_size=image.original_batch_size,
    )


def unflatten_views_img[
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


def random_crop_params(
    h: int, w: int, crop_size: int, device: torch.device
) -> tuple[int, int]:
    x_max = max(1, w - crop_size + 1)
    y_max = max(1, h - crop_size + 1)
    x = torch.randint(0, x_max, size=(1,), device=device).item()
    y = torch.randint(0, y_max, size=(1,), device=device).item()
    return int(x), int(y)


def crop_img[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R],
    x: int,
    y: int,
    crop_size: int,
) -> TensorImage[tuple[C, Height, Width], M, R]:
    return TensorImage(image.data[:, y : y + crop_size, x : x + crop_size])


def crop_boxes_xyxy[
    B: NumBoxes,
    D: BoxDim,
    R: BoxRange,
    O: Origin,
    C: NumClasses,
](
    boxes: BoundingBoxes[tuple[B, D], XYXY, R, O],
    labels: ClassLabels[tuple[B], C],
    x: int,
    y: int,
    crop_size: int,
) -> tuple[
    BoundingBoxes[tuple[NumBoxes, D], XYXY, R, O],
    ClassLabels[tuple[NumBoxes], C],
]:
    """Shifts boxes by (x, y), clips them to crop_size, and removes invalid ones."""
    if len(boxes.data) == 0:
        return boxes, labels

    new_boxes_data = boxes.data.clone()
    new_boxes_data[:, 0] -= x
    new_boxes_data[:, 1] -= y
    new_boxes_data[:, 2] -= x
    new_boxes_data[:, 3] -= y

    # Clip to crop boundaries
    new_boxes_data[:, 0] = new_boxes_data[:, 0].clamp(min=0, max=crop_size)
    new_boxes_data[:, 1] = new_boxes_data[:, 1].clamp(min=0, max=crop_size)
    new_boxes_data[:, 2] = new_boxes_data[:, 2].clamp(min=0, max=crop_size)
    new_boxes_data[:, 3] = new_boxes_data[:, 3].clamp(min=0, max=crop_size)

    # Keep only boxes with positive area
    valid = (new_boxes_data[:, 2] > new_boxes_data[:, 0]) & (
        new_boxes_data[:, 3] > new_boxes_data[:, 1]
    )

    return (
        replace(boxes, data=new_boxes_data[valid]),
        replace(labels, data=labels.data[valid]),
    )


def affine_matrix_params(
    bv: BatchView,
    max_angle_deg: float,
    max_translate: float,
    device: torch.device,
) -> torch.Tensor:
    matrices = []
    for _ in range(bv):
        angle_deg = random.uniform(-max_angle_deg, max_angle_deg)
        tx = random.uniform(-max_translate, max_translate)
        ty = random.uniform(-max_translate, max_translate)

        angle_rad = angle_deg * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        matrices.append(
            torch.tensor(
                [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
                dtype=torch.float32,
                device=device,
            )
        )
    return torch.stack(matrices)


def random_affine_img[B: Batch, M: Mode, R: Range](
    image: TensorImage[tuple[B, *CHW], M, R],
    matrices: torch.Tensor,
) -> TensorImage[tuple[B, *CHW], M, R]:
    b, c, h, w = image.data.shape
    grid = F.affine_grid(matrices, [b, c, h, w], align_corners=False)
    transformed_data = F.grid_sample(
        image.data,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return replace(image, data=transformed_data)


def random_patch_drop_indices(
    bv: int, n: int, drop_rate: float, device: torch.device
) -> torch.Tensor:
    num_keep = max(1, int(n * (1.0 - drop_rate)))
    noise = torch.rand(bv, n, device=device)
    ids_keep = torch.argsort(noise, dim=1)[:, :num_keep]
    return ids_keep


def variance_patch_drop_indices(
    patches_data: torch.Tensor, var_threshold: float
) -> torch.Tensor:
    b, n, d = patches_data.shape
    # For data in [0, 1], the maximum possible variance is 0.25.
    # We normalize the variance to [0, 1] for clean threshold bounds.
    normalized_vars = patches_data.var(dim=-1) / 0.25

    # Find how many patches pass the threshold for each image
    passing_counts = (normalized_vars > var_threshold).sum(dim=-1)

    # The number of patches to keep is the max passing across the batch (at least 1)
    max_keep = max(1, passing_counts.max().item())

    # Get the indices of the top `max_keep` patches with the highest variance
    _, topk_indices = torch.topk(normalized_vars, k=max_keep, dim=-1)

    # Sort the indices so they remain in their original spatial order
    sorted_indices, _ = torch.sort(topk_indices, dim=-1)

    return sorted_indices


def patch_drop_img[
    B: Batch,
    N: NumPatches,
    D: EmbedDim | PatchDim,
](
    patches: Embeddings[B, N, D], ids_keep: torch.Tensor
) -> Embeddings[B, NumPatches, D]:
    b, n, d = patches.data.shape
    kept_data = torch.gather(
        patches.data, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d)
    )
    kept_indices = torch.gather(patches.indices, 1, ids_keep)
    return replace(patches, data=kept_data, indices=kept_indices)


def normalize_boxes_img[B: NumBoxes, D: BoxDim, O: Origin](
    boxes: BoundingBoxes[tuple[B, D], XYXY, Absolute, O],
    image_shape: tuple[int, int],
) -> BoundingBoxes[tuple[B, D], XYXY, Float1, O]:
    """Normalizes absolute pixel coordinates to [0, 1] based on image shape."""
    h, w = image_shape
    new_data = boxes.data.clone()
    if len(new_data) > 0:
        new_data[:, [0, 2]] /= w
        new_data[:, [1, 3]] /= h
    return BoundingBoxes(data=new_data)
