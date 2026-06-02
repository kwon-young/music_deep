import json
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
class CocoMetadata:
    img_id: int
    file_name: str
    width: int
    height: int


def load_coco(
    anno_path: Path, img_dir: Path
) -> Generator[
    Data[
        CocoMetadata,
        DetectionSample[
            PILImage[HWC, RGB, Int255],
            BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Float1, TopLeft],
            ClassLabels[tuple[NumBoxes], NumClasses],
        ],
    ],
    None,
    None,
]:
    """Yields metadata and detection samples for all images in the COCO dataset."""
    with open(anno_path, "r") as f:
        coco_data = json.load(f)

    img_dict = {img["id"]: img for img in coco_data["images"]}
    anno_dict: dict[int, list[dict]] = {}
    for ann in coco_data["annotations"]:
        anno_dict.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in img_dict.items():
        file_name = img_info["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        pil_img = Image_.open(img_path).convert("RGB")
        anns = anno_dict.get(img_id, [])

        labels: list[int] = []
        boxes: list[list[float]] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            labels.append(ann["category_id"])
            # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        metadata = CocoMetadata(
            img_id=img_id,
            file_name=file_name,
            width=img_info["width"],
            height=img_info["height"],
        )

        yield Data(
            metadata=metadata,
            sample=DetectionSample(
                image=PILImage(pil_img),
                boxes=BoundingBoxes(boxes_tensor),
                labels=ClassLabels(labels_tensor),
            ),
        )
