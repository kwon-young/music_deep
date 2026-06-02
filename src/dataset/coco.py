import json
import pickle
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
    Absolute,
    TopLeft,
)


@dataclass
class CocoMetadata:
    img_id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoParsedAnnotation:
    bbox: list[float]
    category_id: int


@dataclass
class CocoDataset:
    num_classes: int
    cat_id_to_idx: dict[int, int]
    images: list[CocoMetadata]
    annotations: dict[int, list[CocoParsedAnnotation]]


def parse_coco(anno_path: Path) -> CocoDataset:
    """Parses the COCO JSON, builds class mappings, and caches the result."""
    cache_path = anno_path.with_suffix(".pkl")
    
    # Load from cache if it exists and is newer than the JSON file
    if cache_path.exists() and cache_path.stat().st_mtime > anno_path.stat().st_mtime:
        print(f"Loading cached COCO dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Parsing COCO JSON from {anno_path} (this might take a while)...")
    with open(anno_path, "r") as f:
        coco_data = json.load(f)

    # Extract categories and create a 0-indexed contiguous mapping
    categories = coco_data.get("categories", [])
    categories.sort(key=lambda x: x["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    num_classes = len(categories)

    images = [
        CocoMetadata(
            img_id=img["id"],
            file_name=img["file_name"],
            width=img["width"],
            height=img["height"],
        )
        for img in coco_data["images"]
    ]

    annotations: dict[int, list[CocoParsedAnnotation]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        parsed_ann = CocoParsedAnnotation(
            bbox=ann["bbox"],
            category_id=ann["category_id"]
        )
        annotations.setdefault(img_id, []).append(parsed_ann)

    dataset = CocoDataset(
        num_classes=num_classes,
        cat_id_to_idx=cat_id_to_idx,
        images=images,
        annotations=annotations,
    )

    print(f"Caching parsed dataset to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def iter_coco(
    dataset: CocoDataset, img_dir: Path
) -> Generator[
    Data[
        CocoMetadata,
        DetectionSample[
            PILImage[HWC, RGB, Int255],
            BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft],
            ClassLabels[tuple[NumBoxes], NumClasses],
        ],
    ],
    None,
    None,
]:
    """Yields metadata and detection samples using the pre-parsed dataset."""
    for img_meta in dataset.images:
        img_path = img_dir / img_meta.file_name
        if not img_path.exists():
            continue

        pil_img = Image_.open(img_path).convert("RGB")
        anns = dataset.annotations.get(img_meta.img_id, [])

        labels: list[int] = []
        boxes: list[list[float]] = []
        for ann in anns:
            x, y, w, h = ann.bbox
            # Map the raw category_id to the 0-indexed contiguous ID
            mapped_label = dataset.cat_id_to_idx[ann.category_id]
            labels.append(mapped_label)
            # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        yield Data(
            metadata=img_meta,
            sample=DetectionSample(
                image=PILImage(pil_img),
                boxes=BoundingBoxes(boxes_tensor),
                labels=ClassLabels(labels_tensor),
            ),
        )
