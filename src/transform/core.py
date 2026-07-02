from typing import (
    Iterable,
    Generator,
    Callable,
    Concatenate,
    Literal,
    TypeVar,
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
    PatchUnit,
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
    Keypoints,
    NumBoxes,
    NumKeypoints,
    BoxDim,
    KeypointDim,
    BoxRange,
    KeypointRange,
    Origin,
    NumLineClasses,
    NumSymbolClasses,
    XYXY,
    X1Y1X2Y2,
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
    import time

    device_id = device.index if device.index is not None else 0
    decoder = get_nv_decoder(device_id)

    max_retries = 10
    for attempt in range(max_retries):
        try:
            nv_img = decoder.read(str(image.path))
            t_gpu = torch.from_dlpack(nv_img).to(device)
            break  # Success
        except Exception as e:
            if attempt < max_retries - 1:
                # VRAM is likely full. Clear cache and wait for the training thread
                # to finish its backward pass and free intermediate activations.
                torch.cuda.empty_cache()
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    f"Failed to decode {image.path} with nvimgcodec after {max_retries} attempts. "
                    f"Last error: {e}"
                ) from e

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


def to_device[I: TensorImage | BoundingBoxes | ClassLabels | Keypoints](
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


def pad_to_size_img[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R],
    target_size: tuple[int, int],
) -> TensorImage[tuple[C, Height, Width], M, R]:
    """Pads an image to an exact target (height, width) using replicate padding."""
    c, h, w = image.data.shape
    target_h, target_w = target_size

    pad_bottom = target_h - h
    pad_right = target_w - w

    if pad_bottom > 0 or pad_right > 0:
        padded_data = F.pad(
            image.data.unsqueeze(0),
            (0, pad_right, 0, pad_bottom),
            mode="replicate",
        ).squeeze(0)
        return replace(image, data=padded_data)

    return image


def pad_to_patch_size_img[C: Channel, M: Mode, R: Range](
    image: TensorImage[tuple[C, Height, Width], M, R],
    patch_size: tuple[int, int],
) -> TensorImage[tuple[C, Height, Width], M, R]:
    """Pads an image to the nearest multiple of the patch size."""
    c, h, w = image.data.shape
    ph, pw = patch_size

    target_h = h + (ph - h % ph) % ph
    target_w = w + (pw - w % pw) % pw

    return pad_to_size_img(image, (target_h, target_w))


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


def resize_patches_img[B: Batch, N: NumPatches, D: PatchDim](
    patches: Embeddings[B, N, D], target_patch_size: tuple[int, int]
) -> Embeddings[B, N, PatchDim]:
    b, n, dim = patches.data.shape
    c, h, w = patches.image_shape
    ph, pw = patches.patch_size
    target_ph, target_pw = target_patch_size

    if ph == target_ph and pw == target_pw:
        return patches

    # Reshape to (B*N, C, ph, pw) for 2D interpolation
    patches_2d = patches.data.view(b * n, c, ph, pw)

    resized_2d = F.interpolate(
        patches_2d,
        size=(target_ph, target_pw),
        mode="bilinear",
        align_corners=False,
    )

    # Flatten back to (B, N, C * target_ph * target_pw)
    resized_flat = resized_2d.view(b, n, c * target_ph * target_pw)

    # CRITICAL: Update image_shape so the ViT's RoPE grid dimensions remain identical
    grid_h, grid_w = h // ph, w // pw
    new_image_shape = (c, grid_h * target_ph, grid_w * target_pw)

    return replace(
        patches,
        data=resized_flat,
        patch_size=(target_ph, target_pw),
        image_shape=new_image_shape,
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


def random_patch_size_params(min_size: int, max_size: int) -> int:
    return random.randint(min_size, max_size)


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
    C: NumSymbolClasses,
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


def crop_keypoints[
    K: NumKeypoints,
    D: KeypointDim,
    R: KeypointRange,
    O: Origin,
    C: NumLineClasses,
](
    keypoints: Keypoints[tuple[K, D], X1Y1X2Y2, R, O],
    labels: ClassLabels[tuple[K], C],
    x: int,
    y: int,
    crop_size: int,
) -> tuple[
    Keypoints[tuple[NumKeypoints, D], X1Y1X2Y2, R, O],
    ClassLabels[tuple[NumKeypoints], C],
]:
    """Shifts keypoints by (x, y), clips them to crop_size, and removes invalid ones."""
    if len(keypoints.data) == 0:
        return keypoints, labels

    new_kp_data = keypoints.data.clone()
    new_kp_data[:, 0] -= x
    new_kp_data[:, 1] -= y
    new_kp_data[:, 2] -= x
    new_kp_data[:, 3] -= y

    # Clip to crop boundaries
    new_kp_data[:, 0] = new_kp_data[:, 0].clamp(min=0, max=crop_size)
    new_kp_data[:, 1] = new_kp_data[:, 1].clamp(min=0, max=crop_size)
    new_kp_data[:, 2] = new_kp_data[:, 2].clamp(min=0, max=crop_size)
    new_kp_data[:, 3] = new_kp_data[:, 3].clamp(min=0, max=crop_size)

    # Keep only keypoints where at least one endpoint is inside the crop
    valid = (
        ~((new_kp_data[:, 0] == 0) & (new_kp_data[:, 2] == 0))
        & ~((new_kp_data[:, 0] == crop_size) & (new_kp_data[:, 2] == crop_size))
        & ~((new_kp_data[:, 1] == 0) & (new_kp_data[:, 3] == 0))
        & ~((new_kp_data[:, 1] == crop_size) & (new_kp_data[:, 3] == crop_size))
    )

    return (
        replace(keypoints, data=new_kp_data[valid]),
        replace(labels, data=labels.data[valid]),
    )


def morphological_downscale_img[C: Channel, H: Height, W: Width, M: Mode](
    image: TensorImage[tuple[C, H, W], M, Float1],
    h_out: int,
    w_out: int,
) -> TensorImage[tuple[C, Height, Width], M, Float1]:
    """Downscales the image using adaptive min pooling to preserve thin lines."""
    c, h, w = image.data.shape
    if h_out == h and w_out == w:
        return image

    # Invert: black (0.0) becomes 1.0, white (1.0) becomes 0.0
    inverted = 1.0 - image.data

    # Adaptive max pool over inverted image.
    pooled = F.adaptive_max_pool2d(
        inverted.unsqueeze(0), (h_out, w_out)
    ).squeeze(0)

    # Invert back
    downsampled = 1.0 - pooled

    return replace(image, data=downsampled)


def scale_boxes_xyxy[
    B: NumBoxes,
    D: BoxDim,
    R: BoxRange,
    O: Origin,
    C: NumSymbolClasses,
](
    boxes: BoundingBoxes[tuple[B, D], XYXY, R, O],
    labels: ClassLabels[tuple[B], C],
    scale_h: float,
    scale_w: float,
) -> tuple[BoundingBoxes[tuple[B, D], XYXY, R, O], ClassLabels[tuple[B], C]]:
    """Scales absolute box coordinates by 1/scale_factor."""
    if len(boxes.data) == 0 or (scale_h == 1.0 and scale_w == 1.0):
        return boxes, labels

    new_boxes = boxes.data.clone()
    new_boxes[:, [0, 2]] /= scale_w
    new_boxes[:, [1, 3]] /= scale_h
    return replace(boxes, data=new_boxes), labels


def scale_keypoints[
    K: NumKeypoints,
    D: KeypointDim,
    R: KeypointRange,
    O: Origin,
    C: NumLineClasses,
](
    keypoints: Keypoints[tuple[K, D], X1Y1X2Y2, R, O],
    labels: ClassLabels[tuple[K], C],
    scale_h: float,
    scale_w: float,
) -> tuple[Keypoints[tuple[K, D], X1Y1X2Y2, R, O], ClassLabels[tuple[K], C]]:
    """Scales absolute keypoint coordinates by 1/scale_factor."""
    if len(keypoints.data) == 0 or (scale_h == 1.0 and scale_w == 1.0):
        return keypoints, labels

    new_kps = keypoints.data.clone()
    new_kps[:, [0, 2]] /= scale_w
    new_kps[:, [1, 3]] /= scale_h
    return replace(keypoints, data=new_kps), labels


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


def get_random_affine_params(
    max_translate_frac: float,
    max_angle_deg: float,
    max_shear_deg: float,
    max_scale: float,
) -> tuple[float, float, float, float, float]:
    """Generates random parameters for affine augmentation."""
    apply_trans = random.random() > 0.5
    tx_frac = (
        random.uniform(-max_translate_frac, max_translate_frac)
        if apply_trans
        else 0.0
    )
    ty_frac = (
        random.uniform(-max_translate_frac, max_translate_frac)
        if apply_trans
        else 0.0
    )

    apply_rot = random.random() > 0.5
    angle_deg = (
        random.uniform(-max_angle_deg, max_angle_deg) if apply_rot else 0.0
    )

    apply_shear = random.random() > 0.5
    shear_deg = (
        random.uniform(-max_shear_deg, max_shear_deg) if apply_shear else 0.0
    )

    apply_scale = random.random() > 0.5
    scale = random.uniform(1.0 / max_scale, max_scale) if apply_scale else 1.0

    return tx_frac, ty_frac, angle_deg, shear_deg, scale


def get_affine_matrices(
    img_h: int,
    img_w: int,
    tx_frac: float,
    ty_frac: float,
    angle_deg: float,
    shear_deg: float,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the forward (for targets) and inverse (for image) affine matrices."""
    tx_px = tx_frac * img_w
    ty_px = ty_frac * img_h
    angle_rad = math.radians(angle_deg)
    shear_rad = math.radians(shear_deg)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    tan_s = math.tan(shear_rad)

    cx, cy = img_w / 2.0, img_h / 2.0

    t_center = torch.tensor(
        [[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=torch.float32, device=device
    )
    t_center_inv = torch.tensor(
        [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    s_mat = torch.tensor(
        [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    r_mat = torch.tensor(
        [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    sh_mat = torch.tensor(
        [[1, tan_s, 0], [0, 1, 0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    t_offset = torch.tensor(
        [[1, 0, tx_px], [0, 1, ty_px], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )

    # Forward Matrix: Center -> Scale -> Shear -> Rotate -> Uncenter -> Translate
    fwd_matrix = t_offset @ t_center @ r_mat @ sh_mat @ s_mat @ t_center_inv

    # Inverse Matrix for grid_sample
    inv_pixel = torch.inverse(fwd_matrix)

    # Convert to normalized coordinates [-1, 1] for grid_sample
    n2p = torch.tensor(
        [
            [img_w / 2.0, 0, img_w / 2.0],
            [0, img_h / 2.0, img_h / 2.0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    p2n = torch.tensor(
        [[2.0 / img_w, 0, -1.0], [0, 2.0 / img_h, -1.0], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )

    theta_norm = p2n @ inv_pixel @ n2p
    theta_grid = theta_norm[:2, :].unsqueeze(0)

    return fwd_matrix, theta_grid


def affine_img[C: Channel, M: Mode](
    image: TensorImage[tuple[C, Height, Width], M, Float1],
    theta_grid: torch.Tensor,
) -> TensorImage[tuple[C, Height, Width], M, Float1]:
    """Applies affine transformation, padding with white (1.0) using in-place ops to save VRAM."""
    c, h, w = image.data.shape
    grid = F.affine_grid(theta_grid, [1, c, h, w], align_corners=False)

    # In-place invert input: 0.0 (black) becomes 1.0, 1.0 (white) becomes 0.0
    image.data.mul_(-1.0).add_(1.0)

    # grid_sample allocates the output tensor and pads out-of-bounds with 0.0
    transformed_data = F.grid_sample(
        image.data.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    # In-place invert output back to normal: 0.0 padding becomes 1.0 (white)
    transformed_data.mul_(-1.0).add_(1.0)

    return replace(image, data=transformed_data.squeeze(0))


def affine_boxes_xyxy[
    B: NumBoxes,
    D: BoxDim,
    R: BoxRange,
    O: Origin,
    C: NumSymbolClasses,
](
    boxes: BoundingBoxes[tuple[B, D], XYXY, R, O],
    labels: ClassLabels[tuple[B], C],
    fwd_matrix: torch.Tensor,
    img_w: int,
    img_h: int,
) -> tuple[
    BoundingBoxes[tuple[NumBoxes, D], XYXY, R, O],
    ClassLabels[tuple[NumBoxes], C],
]:
    """Applies affine transformation, clips to image, and filters invalid boxes."""
    if len(boxes.data) == 0:
        return boxes, labels

    x1, y1 = boxes.data[:, 0], boxes.data[:, 1]
    x2, y2 = boxes.data[:, 2], boxes.data[:, 3]

    corners = torch.stack(
        [
            torch.stack([x1, y1, torch.ones_like(x1)], dim=-1),
            torch.stack([x2, y1, torch.ones_like(x1)], dim=-1),
            torch.stack([x1, y2, torch.ones_like(x1)], dim=-1),
            torch.stack([x2, y2, torch.ones_like(x1)], dim=-1),
        ],
        dim=1,
    )

    transformed_corners = torch.einsum("ij,nkj->nki", fwd_matrix, corners)

    new_x = transformed_corners[:, :, 0]
    new_y = transformed_corners[:, :, 1]

    new_x1 = new_x.min(dim=1).values.clamp(min=0, max=img_w)
    new_y1 = new_y.min(dim=1).values.clamp(min=0, max=img_h)
    new_x2 = new_x.max(dim=1).values.clamp(min=0, max=img_w)
    new_y2 = new_y.max(dim=1).values.clamp(min=0, max=img_h)

    valid = (new_x2 > new_x1) & (new_y2 > new_y1)

    new_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

    return replace(boxes, data=new_boxes[valid]), replace(
        labels, data=labels.data[valid]
    )


def affine_keypoints[
    K: NumKeypoints,
    D: KeypointDim,
    R: KeypointRange,
    O: Origin,
    C: NumLineClasses,
](
    keypoints: Keypoints[tuple[K, D], X1Y1X2Y2, R, O],
    labels: ClassLabels[tuple[K], C],
    fwd_matrix: torch.Tensor,
    img_w: int,
    img_h: int,
) -> tuple[
    Keypoints[tuple[NumKeypoints, D], X1Y1X2Y2, R, O],
    ClassLabels[tuple[NumKeypoints], C],
]:
    """Applies affine transformation, clips to image, and filters invalid keypoints."""
    if len(keypoints.data) == 0:
        return keypoints, labels

    x1, y1 = keypoints.data[:, 0], keypoints.data[:, 1]
    x2, y2 = keypoints.data[:, 2], keypoints.data[:, 3]

    points = torch.stack(
        [
            torch.stack([x1, y1, torch.ones_like(x1)], dim=-1),
            torch.stack([x2, y2, torch.ones_like(x2)], dim=-1),
        ],
        dim=1,
    )

    transformed_points = torch.einsum("ij,nkj->nki", fwd_matrix, points)

    new_x1 = transformed_points[:, 0, 0].clamp(min=0, max=img_w)
    new_y1 = transformed_points[:, 0, 1].clamp(min=0, max=img_h)
    new_x2 = transformed_points[:, 1, 0].clamp(min=0, max=img_w)
    new_y2 = transformed_points[:, 1, 1].clamp(min=0, max=img_h)

    valid = ~(
        ((new_x1 == 0) & (new_x2 == 0))
        | ((new_x1 == img_w) & (new_x2 == img_w))
        | ((new_y1 == 0) & (new_y2 == 0))
        | ((new_y1 == img_h) & (new_y2 == img_h))
    )

    new_kps = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

    return replace(keypoints, data=new_kps[valid]), replace(
        labels, data=labels.data[valid]
    )


def spatial_mask_drop_indices(
    indices: torch.Tensor, grid_w: int, drop_ratio: float
) -> torch.Tensor:
    """
    Drops a contiguous spatial region of patches from a sparse set of indices.
    Returns the indices of the patches to KEEP.
    """
    b, n = indices.shape
    device = indices.device

    drop_count = int(n * drop_ratio)
    keep_count = n - drop_count

    if drop_count == 0:
        return torch.arange(n, device=device).unsqueeze(0).expand(b, -1)

    # 1. Convert 1D patch indices back to 2D grid coordinates
    py = indices // grid_w
    px = indices % grid_w

    # 2. Pick a random existing patch as the center of the mask for each batch item
    center_idx = torch.randint(0, n, (b, 1), device=device)
    cx = torch.gather(px, 1, center_idx)
    cy = torch.gather(py, 1, center_idx)

    # 3. Compute squared spatial distance from the center to all patches
    dist_sq = (px - cx) ** 2 + (py - cy) ** 2

    # 4. We want to DROP the closest patches, so we KEEP the ones with the LARGEST distances
    _, keep_idx = torch.topk(dist_sq, k=keep_count, dim=1, largest=True)

    # 5. Sort the kept indices to maintain the original sequence order
    keep_idx, _ = torch.sort(keep_idx, dim=1)

    return keep_idx


def random_patch_drop_indices(
    bv: int, n: int, drop_rate: float, device: torch.device
) -> torch.Tensor:
    num_keep = max(1, int(n * (1.0 - drop_rate)))
    noise = torch.rand(bv, n, device=device)
    ids_keep = torch.argsort(noise, dim=1)[:, :num_keep]
    return ids_keep


def variance_patch_drop_indices(
    patches_data: torch.Tensor,
    var_threshold: float | None = None,
    drop_rate: float | None = None,
) -> torch.Tensor:
    if (var_threshold is None) == (drop_rate is None):
        raise ValueError(
            "Must provide exactly one of var_threshold or drop_rate"
        )

    b, n, d = patches_data.shape

    if drop_rate is not None:
        # Fraction-based drop
        num_keep = max(1, int(n * (1.0 - drop_rate)))
        patch_vars = patches_data.var(dim=-1)
        _, topk_indices = torch.topk(patch_vars, k=num_keep, dim=-1)
    else:
        # Threshold-based drop
        assert var_threshold is not None
        normalized_vars = patches_data.var(dim=-1) / 0.25
        passing_counts = (normalized_vars > var_threshold).sum(dim=-1)
        max_keep = max(1, int(passing_counts.max().item()))
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
    patch_size: tuple[int, int],
) -> BoundingBoxes[tuple[B, D], XYXY, PatchUnit, O]:
    """Normalizes absolute pixel coordinates to Patch Units."""
    ph, pw = patch_size
    new_data = boxes.data.clone()
    if len(new_data) > 0:
        new_data[:, [0, 2]] /= pw
        new_data[:, [1, 3]] /= ph
    return BoundingBoxes(data=new_data)


def normalize_keypoints_img[K: NumKeypoints, D: KeypointDim, O: Origin](
    keypoints: Keypoints[tuple[K, D], X1Y1X2Y2, Absolute, O],
    patch_size: tuple[int, int],
) -> Keypoints[tuple[K, D], X1Y1X2Y2, PatchUnit, O]:
    """Normalizes absolute pixel coordinates to Patch Units."""
    ph, pw = patch_size
    new_data = keypoints.data.clone()
    if len(new_data) > 0:
        new_data[:, [0, 2]] /= pw
        new_data[:, [1, 3]] /= ph
    return Keypoints(data=new_data)
