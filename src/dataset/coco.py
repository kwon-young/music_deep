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
    Keypoints,
    NumBoxes,
    NumKeypoints,
    NumSymbolClasses,
    NumLineClasses,
    BoxDim,
    KeypointDim,
    XYXY,
    X1Y1X2Y2,
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
class CocoSymbolAnnotation:
    bbox: list[float]
    category_id: int


@dataclass
class CocoLineAnnotation:
    keypoints: list[float]
    category_id: int


type CocoParsedAnnotation = CocoSymbolAnnotation | CocoLineAnnotation


@dataclass
class CocoDataset:
    num_symbol_classes: int
    num_line_classes: int
    symbol_cat_id_to_idx: dict[int, int]
    line_cat_id_to_idx: dict[int, int]
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

    line_category_ids = set()
    for cat in categories:
        if (
            "keypoints" in cat
            and "start" in cat["keypoints"]
            and "end" in cat["keypoints"]
        ):
            line_category_ids.add(cat["id"])

    # Split categories and create separate 0-indexed contiguous mappings
    symbol_categories = [
        cat for cat in categories if cat["id"] not in line_category_ids
    ]
    line_categories = [
        cat for cat in categories if cat["id"] in line_category_ids
    ]

    symbol_cat_id_to_idx = {
        cat["id"]: idx for idx, cat in enumerate(symbol_categories)
    }
    line_cat_id_to_idx = {
        cat["id"]: idx for idx, cat in enumerate(line_categories)
    }

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
        cat_id = ann["category_id"]

        parsed_ann: CocoParsedAnnotation
        if (
            cat_id in line_category_ids
            and "keypoints" in ann
            and len(ann["keypoints"]) >= 6
        ):
            # Extract [x1, y1, v1, x2, y2, v2] -> [x1, y1, x2, y2]
            x1, y1, _, x2, y2, _ = ann["keypoints"][:6]
            parsed_ann = CocoLineAnnotation(
                keypoints=[x1, y1, x2, y2], category_id=cat_id
            )
        else:
            parsed_ann = CocoSymbolAnnotation(
                bbox=ann["bbox"], category_id=cat_id
            )

        annotations.setdefault(img_id, []).append(parsed_ann)

    dataset = CocoDataset(
        num_symbol_classes=len(symbol_categories),
        num_line_classes=len(line_categories),
        symbol_cat_id_to_idx=symbol_cat_id_to_idx,
        line_cat_id_to_idx=line_cat_id_to_idx,
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
        ClassLabels[tuple[NumBoxes], NumSymbolClasses],
        Keypoints[
            tuple[NumKeypoints, KeypointDim], X1Y1X2Y2, Absolute, TopLeft
        ],
        ClassLabels[tuple[NumKeypoints], NumLineClasses],
    ],
]:
    """Loads a single COCO sample by its index in the dataset."""
    img_meta = dataset.images[index]
    img_path = img_dir / img_meta.file_name

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    anns = dataset.annotations.get(img_meta.img_id, [])

    box_labels: list[int] = []
    boxes: list[list[float]] = []
    keypoint_labels: list[int] = []
    keypoints: list[list[float]] = []

    for ann in anns:
        if isinstance(ann, CocoLineAnnotation):
            mapped_label = dataset.line_cat_id_to_idx[ann.category_id]
            keypoints.append(ann.keypoints)
            keypoint_labels.append(mapped_label)
        else:
            mapped_label = dataset.symbol_cat_id_to_idx[ann.category_id]
            x, y, w, h = ann.bbox
            boxes.append([x, y, x + w, y + h])
            box_labels.append(mapped_label)

    # Use reshape to ensure correct dimensions even if empty
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    box_labels_tensor = torch.tensor(box_labels, dtype=torch.int64)

    keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32).reshape(
        -1, 4
    )
    keypoint_labels_tensor = torch.tensor(keypoint_labels, dtype=torch.int64)

    return Data(
        metadata=img_meta,
        sample=DetectionSample(
            image=LazyImage(
                path=img_path, width=img_meta.width, height=img_meta.height
            ),
            boxes=BoundingBoxes(boxes_tensor),
            box_labels=ClassLabels(box_labels_tensor),
            keypoints=Keypoints(keypoints_tensor),
            keypoint_labels=ClassLabels(keypoint_labels_tensor),
        ),
    )
