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
    pad_to_patch_size_img,
    random_crop_params,
    crop_img,
    crop_boxes_xyxy,
    crop_keypoints,
    normalize_boxes_img,
    normalize_keypoints_img,
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
        TensorImage[tuple[C, H, W], M, R], Bx, BxLbl, Kp, KpLbl
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
def normalize_targets[
    I: TensorImage,
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
        BoundingBoxes[tuple[B, D], XYXY, Absolute, O],
        BxLbl,
        Keypoints[tuple[K, KD], X1Y1X2Y2, Absolute, O],
        KpLbl,
    ],
) -> DetectionSample[
    I,
    BoundingBoxes[tuple[B, D], XYXY, Float1, O],
    BxLbl,
    Keypoints[tuple[K, KD], X1Y1X2Y2, Float1, O],
    KpLbl,
]:
    h, w = sample.image.data.shape[-2:]
    return DetectionSample(
        image=sample.image,
        boxes=normalize_boxes_img(sample.boxes, (h, w)),
        box_labels=sample.box_labels,
        keypoints=normalize_keypoints_img(sample.keypoints, (h, w)),
        keypoint_labels=sample.keypoint_labels,
    )


@batched_transform
def extract_patches[B: Batch, Bx, BxLbl, Kp, KpLbl](
    sample: DetectionSample[
        TensorImage[tuple[B, *CHW], RGB, Float1], Bx, BxLbl, Kp, KpLbl
    ],
    patch_size: tuple[int, int],
) -> DetectionSample[Patches[B, NumPatches, PatchDim], Bx, BxLbl, Kp, KpLbl]:
    return DetectionSample(
        image=extract_patches_img(sample.image, patch_size),
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
    var_threshold: float,
) -> DetectionSample[I, Bx, BxLbl, Kp, KpLbl]:
    ids_keep = variance_patch_drop_indices(sample.image.data, var_threshold)

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
