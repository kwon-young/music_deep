import random
import torch
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
    extract_patches_img,
    random_patch_drop_indices,
    variance_patch_drop_indices,
    patch_drop_img,
    stack_tensor_img,
    pad_to_size_img,
    pad_to_patch_size_img,
    random_crop_params,
    random_patch_size_params,
    resize_patches_img,
    crop_img,
    crop_boxes_xyxy,
    crop_keypoints,
    normalize_boxes_img,
    normalize_keypoints_img,
    get_random_affine_params,
    get_affine_matrices,
    affine_img,
    affine_boxes_xyxy,
    affine_keypoints,
    morphological_downscale_img,
    scale_boxes_xyxy,
    scale_keypoints,
)
from music_types import (
    DetectionSample,
    LazyImage,
    PILImage,
    ArrayImage,
    TensorImage,
    BoundingBoxes,
    ClassLabels,
    Keypoints,
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
    PatchUnit,
    RGB,
    NumBoxes,
    NumKeypoints,
    BoxDim,
    KeypointDim,
    BoxRange,
    KeypointRange,
    Origin,
    NumSymbolClasses,
    NumLineClasses,
    XYXY,
    X1Y1X2Y2,
    Absolute,
)


@transform
def decode_nvimgcodec[Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[LazyImage, Bx, BxLbl, Kp, KpLbl],
    device: torch.device,
) -> DetectionSample[TensorImage[CHW, RGB, Int255], Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=decode_nvimgcodec_img(sample.image, device),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@transform
def decode_pyvips[Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[LazyImage, Bx, BxLbl, Kp, KpLbl],
    device: torch.device,
) -> DetectionSample[TensorImage[CHW, RGB, Int255], Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=decode_pyvips_img(sample.image, device),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@transform
def decode_and_crop_pyvips[
    B: NumBoxes,
    K: NumKeypoints,
    D: BoxDim,
    KD: KeypointDim,
    BR: BoxRange,
    KR: KeypointRange,
    O: Origin,
    SC: NumSymbolClasses,
    LC: NumLineClasses,
](
    sample: DetectionSample[
        LazyImage,
        BoundingBoxes[tuple[B, D], XYXY, BR, O],
        ClassLabels[tuple[B], SC],
        Keypoints[tuple[K, KD], X1Y1X2Y2, KR, O],
        ClassLabels[tuple[K], LC],
    ],
    crop_size: int,
    device: torch.device,
) -> DetectionSample[
    TensorImage[CHW, RGB, Int255],
    BoundingBoxes[tuple[NumBoxes, D], XYXY, BR, O],
    ClassLabels[tuple[NumBoxes], SC],
    Keypoints[tuple[NumKeypoints, KD], X1Y1X2Y2, KR, O],
    ClassLabels[tuple[NumKeypoints], LC],
]:
    h, w = sample.image.height, sample.image.width
    x, y = random_crop_params(h, w, crop_size, torch.device("cpu"))

    new_img = decode_and_crop_pyvips_img(sample.image, x, y, crop_size, device)
    new_boxes, new_box_labels = crop_boxes_xyxy(
        sample.boxes, sample.box_labels, x, y, crop_size
    )
    new_keypoints, new_keypoint_labels = crop_keypoints(
        sample.keypoints, sample.keypoint_labels, x, y, crop_size
    )

    return DetectionSample(
        image=new_img,
        boxes=new_boxes,
        box_labels=new_box_labels,
        keypoints=new_keypoints,
        keypoint_labels=new_keypoint_labels,
    )


@transform
def to_numpy[
    H: Height,
    W: Width,
    C: Channel,
    M: Mode,
    R: Range,
    Bx,
    BxLbl,
    Kp,
    KpLbl,
](
    sample: DetectionSample[
        PILImage[tuple[H, W, C], M, R], Bx, BxLbl, Kp, KpLbl
    ],
) -> DetectionSample[ArrayImage[tuple[C, H, W], M, R], Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=to_numpy_img(sample.image),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@transform
def to_tensor[L: Layout, M: Mode, R: Range, Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[ArrayImage[L, M, R], Bx, BxLbl, Kp, KpLbl],
) -> DetectionSample[TensorImage[L, M, R], Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=to_tensor_img(sample.image),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@transform
def to_float1[L: AnyLayouts, M: Mode, Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[TensorImage[L, M, Int255], Bx, BxLbl, Kp, KpLbl],
) -> DetectionSample[TensorImage[L, M, Float1], Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=to_float1_img(sample.image),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@transform
def pad_to_patch_size[
    C: Channel,
    H: Height,
    W: Width,
    M: Mode,
    R: Range,
    Bx,
    BxLbl,
    Kp,
    KpLbl,
](
    sample: DetectionSample[
        TensorImage[tuple[C, Height, Width], M, R], Bx, BxLbl, Kp, KpLbl
    ],
    patch_size: tuple[int, int],
) -> DetectionSample[
    TensorImage[tuple[C, Height, Width], M, R], Bx, BxLbl, Kp, KpLbl
]:
    return DetectionSample(
        image=pad_to_patch_size_img(sample.image, patch_size),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@transform
def to[
    I: TensorImage,
    Bx: BoundingBoxes,
    BxLbl: ClassLabels,
    Kp: Keypoints,
    KpLbl: ClassLabels,
](
    sample: DetectionSample[I, Bx, BxLbl, Kp, KpLbl], device: torch.device
) -> DetectionSample[I, Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=to_device(sample.image, device),
        boxes=to_device(sample.boxes, device),
        box_labels=to_device(sample.box_labels, device),
        keypoints=to_device(sample.keypoints, device),
        keypoint_labels=to_device(sample.keypoint_labels, device),
    )


@transform
def to_patches[
    E: Patches,
    Bx: BoundingBoxes,
    BxLbl: ClassLabels,
    Kp: Keypoints,
    KpLbl: ClassLabels,
](
    sample: DetectionSample[E, list[Bx], list[BxLbl], list[Kp], list[KpLbl]],
    device: torch.device,
) -> DetectionSample[E, list[Bx], list[BxLbl], list[Kp], list[KpLbl]]:
    """Specialized device move for batched patches and lists of boxes/labels/keypoints."""
    boxes = [to_device(b, device) for b in sample.boxes]
    box_labels = [to_device(l, device) for l in sample.box_labels]
    keypoints = [to_device(k, device) for k in sample.keypoints]
    keypoint_labels = [to_device(l, device) for l in sample.keypoint_labels]
    return DetectionSample(
        image=to_device_embeddings(sample.image, device),
        boxes=boxes,
        box_labels=box_labels,
        keypoints=keypoints,
        keypoint_labels=keypoint_labels,
    )


@transform
def random_crop[
    C: Channel,
    M: Mode,
    R: Range,
    B: NumBoxes,
    K: NumKeypoints,
    D: BoxDim,
    KD: KeypointDim,
    BR: BoxRange,
    KR: KeypointRange,
    O: Origin,
    SC: NumSymbolClasses,
    LC: NumLineClasses,
](
    sample: DetectionSample[
        TensorImage[tuple[C, Height, Width], M, R],
        BoundingBoxes[tuple[B, D], XYXY, BR, O],
        ClassLabels[tuple[B], SC],
        Keypoints[tuple[K, KD], X1Y1X2Y2, KR, O],
        ClassLabels[tuple[K], LC],
    ],
    crop_size: int,
) -> DetectionSample[
    TensorImage[tuple[C, Height, Width], M, R],
    BoundingBoxes[tuple[NumBoxes, D], XYXY, BR, O],
    ClassLabels[tuple[NumBoxes], SC],
    Keypoints[tuple[NumKeypoints, KD], X1Y1X2Y2, KR, O],
    ClassLabels[tuple[NumKeypoints], LC],
]:
    c, h, w = sample.image.data.shape
    x, y = random_crop_params(h, w, crop_size, sample.image.data.device)

    new_img = crop_img(sample.image, x, y, crop_size)
    new_boxes, new_box_labels = crop_boxes_xyxy(
        sample.boxes, sample.box_labels, x, y, crop_size
    )
    new_keypoints, new_keypoint_labels = crop_keypoints(
        sample.keypoints, sample.keypoint_labels, x, y, crop_size
    )

    return DetectionSample(
        image=new_img,
        boxes=new_boxes,
        box_labels=new_box_labels,
        keypoints=new_keypoints,
        keypoint_labels=new_keypoint_labels,
    )


@transform
def random_affine[T: DetectionSample](
    sample: T,
    max_translate_frac: float,
    max_angle_deg: float,
    max_shear_deg: float,
    max_scale: float,
) -> T:
    """Applies random affine augmentation to image and targets."""
    _, h, w = sample.image.data.shape
    device = sample.image.data.device

    tx, ty, angle, shear, scale = get_random_affine_params(
        max_translate_frac, max_angle_deg, max_shear_deg, max_scale
    )

    fwd_matrix, theta_grid = get_affine_matrices(
        img_h=h,
        img_w=w,
        tx_frac=tx,
        ty_frac=ty,
        angle_deg=angle,
        shear_deg=shear,
        scale=scale,
        device=device,
    )

    new_img = affine_img(sample.image, theta_grid)
    new_boxes, new_box_labels = affine_boxes_xyxy(
        sample.boxes, sample.box_labels, fwd_matrix, w, h
    )
    new_kps, new_kp_labels = affine_keypoints(
        sample.keypoints, sample.keypoint_labels, fwd_matrix, w, h
    )

    return replace(
        sample,
        image=new_img,
        boxes=new_boxes,
        box_labels=new_box_labels,
        keypoints=new_kps,
        keypoint_labels=new_kp_labels,
    )


@transform
def random_morphological_downscale[
    C: Channel,
    H: Height,
    W: Width,
    M: Mode,
    B: NumBoxes,
    K: NumKeypoints,
    D: BoxDim,
    KD: KeypointDim,
    BR: BoxRange,
    KR: KeypointRange,
    O: Origin,
    SC: NumSymbolClasses,
    LC: NumLineClasses,
](
    sample: DetectionSample[
        TensorImage[tuple[C, H, W], M, Float1],
        BoundingBoxes[tuple[B, D], XYXY, BR, O],
        ClassLabels[tuple[B], SC],
        Keypoints[tuple[K, KD], X1Y1X2Y2, KR, O],
        ClassLabels[tuple[K], LC],
    ],
    max_scale_factor: float,
) -> DetectionSample[
    TensorImage[tuple[C, Height, Width], M, Float1],
    BoundingBoxes[tuple[B, D], XYXY, BR, O],
    ClassLabels[tuple[B], SC],
    Keypoints[tuple[K, KD], X1Y1X2Y2, KR, O],
    ClassLabels[tuple[K], LC],
]:
    """Applies random fractional morphological downscaling to simulate different DPIs."""
    _, h, w = sample.image.data.shape
    scale_factor = random.uniform(1.0, max_scale_factor)

    h_out = max(1, int(h / scale_factor))
    w_out = max(1, int(w / scale_factor))

    # Calculate exact scale factors to perfectly align targets
    exact_scale_h = h / h_out
    exact_scale_w = w / w_out

    if exact_scale_h == 1.0 and exact_scale_w == 1.0:
        return sample

    new_img = morphological_downscale_img(sample.image, h_out, w_out)
    new_boxes, new_box_labels = scale_boxes_xyxy(
        sample.boxes, sample.box_labels, exact_scale_h, exact_scale_w
    )
    new_keypoints, new_keypoint_labels = scale_keypoints(
        sample.keypoints, sample.keypoint_labels, exact_scale_h, exact_scale_w
    )

    return DetectionSample(
        image=new_img,
        boxes=new_boxes,
        box_labels=new_box_labels,
        keypoints=new_keypoints,
        keypoint_labels=new_keypoint_labels,
    )


def random_extract_patches_and_collate[
    Meta,
    C: Channel,
    M: Mode,
    R: Range,
    Bx,
    BxLbl,
    Kp,
    KpLbl,
](
    batch: tuple[
        Data[
            Meta,
            DetectionSample[
                TensorImage[tuple[C, Height, Width], M, R],
                Bx,
                BxLbl,
                Kp,
                KpLbl,
            ],
        ],
        ...,
    ],
    min_patch_size: int,
    max_patch_size: int,
) -> BatchedData[
    Meta,
    DetectionSample[
        Patches[Batch, NumPatches, PatchDim],
        list[Bx],
        list[BxLbl],
        list[Kp],
        list[KpLbl],
    ],
]:
    eps = random_patch_size_params(min_patch_size, max_patch_size)
    patch_size = (eps, eps)

    # 1. Find the maximum dimensions across the batch
    max_h = max(b.sample.image.data.shape[1] for b in batch)
    max_w = max(b.sample.image.data.shape[2] for b in batch)

    # 2. Round up to the nearest multiple of the random patch size
    target_h = max_h + (eps - max_h % eps) % eps
    target_w = max_w + (eps - max_w % eps) % eps

    # 3. Pad images to the target dimensions
    padded_batch_items = []
    for b in batch:
        new_img = pad_to_size_img(b.sample.image, (target_h, target_w))
        new_sample = replace(b.sample, image=new_img)
        padded_batch_items.append(replace(b, sample=new_sample))

    # 4. Reuse existing collate to stack images and aggregate targets
    batched_data = collate(tuple(padded_batch_items))

    # 5. Reuse existing extract_patches_img
    patches = extract_patches_img(batched_data.sample.image, patch_size)

    return BatchedData(
        metadata=batched_data.metadata,
        sample=DetectionSample(
            image=patches,
            boxes=batched_data.sample.boxes,
            box_labels=batched_data.sample.box_labels,
            keypoints=batched_data.sample.keypoints,
            keypoint_labels=batched_data.sample.keypoint_labels,
        ),
    )


@batched_transform
def normalize_batched_targets[
    I: Patches,
    B: NumBoxes,
    K: NumKeypoints,
    D: BoxDim,
    KD: KeypointDim,
    O: Origin,
    BxLbl,
    KpLbl,
](
    sample: DetectionSample[
        I,
        list[BoundingBoxes[tuple[B, D], XYXY, Absolute, O]],
        list[BxLbl],
        list[Keypoints[tuple[K, KD], X1Y1X2Y2, Absolute, O]],
        list[KpLbl],
    ],
) -> DetectionSample[
    I,
    list[BoundingBoxes[tuple[B, D], XYXY, PatchUnit, O]],
    list[BxLbl],
    list[Keypoints[tuple[K, KD], X1Y1X2Y2, PatchUnit, O]],
    list[KpLbl],
]:
    patch_size = sample.image.patch_size

    norm_boxes = [normalize_boxes_img(b, patch_size) for b in sample.boxes]
    norm_keypoints = [
        normalize_keypoints_img(k, patch_size) for k in sample.keypoints
    ]

    return DetectionSample(
        image=sample.image,
        boxes=norm_boxes,
        box_labels=sample.box_labels,
        keypoints=norm_keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@batched_transform
def resize_patches[I: Patches, Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[I, Bx, BxLbl, Kp, KpLbl],
    target_patch_size: tuple[int, int],
) -> DetectionSample[I, Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=resize_patches_img(sample.image, target_patch_size),
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@batched_transform
def random_patch_drop[I: Patches, Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[I, Bx, BxLbl, Kp, KpLbl],
    drop_rate: float,
) -> DetectionSample[I, Bx, BxLbl, Kp, KpLbl]:
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
        image=new_img,
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


@batched_transform
def variance_patch_drop[I: Patches, Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[I, Bx, BxLbl, Kp, KpLbl],
    var_threshold: float | None = None,
    drop_rate: float | None = None,
    topk: int | None = None,
) -> DetectionSample[I, Bx, BxLbl, Kp, KpLbl]:
    ids_keep = variance_patch_drop_indices(
        sample.image.data, 
        var_threshold=var_threshold, 
        drop_rate=drop_rate,
        topk=topk
    )

    new_img_base = patch_drop_img(sample.image, ids_keep)

    new_img = replace(
        sample.image,
        data=new_img_base.data,
        indices=new_img_base.indices,
    )

    return DetectionSample(
        image=new_img,
        boxes=sample.boxes,
        box_labels=sample.box_labels,
        keypoints=sample.keypoints,
        keypoint_labels=sample.keypoint_labels,
    )


def collate[
    Meta,
    C: Channel,
    H: Height,
    W: Width,
    M: Mode,
    R: Range,
    Bx,
    BxLbl,
    Kp,
    KpLbl,
](
    batch: tuple[
        Data[
            Meta,
            DetectionSample[
                TensorImage[tuple[C, H, W], M, R],
                Bx,
                BxLbl,
                Kp,
                KpLbl,
            ],
        ],
        ...,
    ],
) -> BatchedData[
    Meta,
    DetectionSample[
        TensorImage[tuple[Batch, C, H, W], M, R],
        list[Bx],
        list[BxLbl],
        list[Kp],
        list[KpLbl],
    ],
]:
    m = [b.metadata for b in batch]

    stacked_image = stack_tensor_img([b.sample.image for b in batch])
    boxes_list = [b.sample.boxes for b in batch]
    box_labels_list = [b.sample.box_labels for b in batch]
    keypoints_list = [b.sample.keypoints for b in batch]
    keypoint_labels_list = [b.sample.keypoint_labels for b in batch]

    return BatchedData(
        metadata=m,
        sample=DetectionSample(
            image=stacked_image,
            boxes=boxes_list,
            box_labels=box_labels_list,
            keypoints=keypoints_list,
            keypoint_labels=keypoint_labels_list,
        ),
    )
