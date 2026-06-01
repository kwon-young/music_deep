from pathlib import Path
from typing import Generator
from dataclasses import dataclass
import torch
from PIL import Image as Image_

from music_types import (
    Data,
    PILImage,
    HWC,
    RGB,
    Int255,
    DetectionSample,
    BoundingBoxes,
    ClassLabels,
    NumBoxes,
    NumClasses,
    BoxDim,
    XYXY,
    Float1,
    TopLeft,
)


@dataclass
class YOLOMetadata:
    img_path: Path
    lbl_path: Path
    img_w: int
    img_h: int


def load_yolo_metadata(
    img_dir: Path, lbl_dir: Path, img_w: int, img_h: int
) -> Generator[YOLOMetadata, None, None]:
    """Yields metadata for all images found in the directory."""
    for img_path in img_dir.glob("*.jpg"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        yield YOLOMetadata(img_path, lbl_path, img_w, img_h)


def load_sample(
    metadata: YOLOMetadata,
) -> Data[
    YOLOMetadata,
    DetectionSample[
        PILImage[HWC, RGB, Int255],
        BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Float1, TopLeft],
        ClassLabels[tuple[NumBoxes], NumClasses],
    ],
]:
    pil_img = (
        Image_.open(metadata.img_path)
        .convert("RGB")
        .resize((metadata.img_w, metadata.img_h))
    )

    labels: list[int] = []
    boxes: list[list[float]] = []
    if metadata.lbl_path.exists():
        with open(metadata.lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                labels.append(class_id)
                boxes.append([x1, y1, x2, y2])

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    return Data(
        metadata=metadata,
        sample=DetectionSample(
            image=PILImage(pil_img),
            boxes=BoundingBoxes(boxes_tensor),
            labels=ClassLabels(labels_tensor),
        ),
    )
