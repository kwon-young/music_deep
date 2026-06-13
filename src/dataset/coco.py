import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
import torch
from PIL import Image as Image_

from music_types import (
    Data,
    LazyImage,
    RGB,
    Int255,
    DetectionSample,
    BoundingBoxes,
    ClassLabels,
    NumBoxes,
    NumClasses,
    BoxDim,
    XYXY,
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

    @property
    def num_symbols(self) -> int:
        return sum(len(anns) for anns in self.annotations.values())


def parse_coco(anno_path: Path, cache_dir: Path | None = None) -> CocoDataset:
    """Parses the COCO JSON, builds class mappings, and caches the result."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / anno_path.with_suffix(".pkl").name
    else:
        cache_path = anno_path.with_suffix(".pkl")

    # Load from cache if it exists and is newer than the JSON file
    if (
        cache_path.exists()
        and cache_path.stat().st_mtime > anno_path.stat().st_mtime
    ):
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
            bbox=ann["bbox"], category_id=ann["category_id"]
        )
        annotations.setdefault(img_id, []).append(parsed_ann)

    dataset = CocoDataset(
        num_classes=num_classes,
        cat_id_to_idx=cat_id_to_idx,
        images=images,
        annotations=annotations,
    )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"Caching parsed dataset to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


def load_coco_sample(
    dataset: CocoDataset, img_dir: Path, index: int
) -> Data[
    CocoMetadata,
    DetectionSample[
        LazyImage,
        BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft],
        ClassLabels[tuple[NumBoxes], NumClasses],
    ],
]:
    """Loads a single COCO sample by its index in the dataset."""
    img_meta = dataset.images[index]
    img_path = img_dir / img_meta.file_name

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

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

    # Use reshape to ensure correct dimensions even if empty
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    return Data(
        metadata=img_meta,
        sample=DetectionSample(
            image=LazyImage(
                path=img_path, width=img_meta.width, height=img_meta.height
            ),
            boxes=BoundingBoxes(boxes_tensor),
            labels=ClassLabels(labels_tensor),
        ),
    )
