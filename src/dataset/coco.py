import os
import json
import pickle
import math
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import torch
from PIL import Image as Image_

from music_types import (
    Data,
    LazyImage,
    RGB,
    Int255,
    SSLSample,
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
    symbol_weights: list[float]
    line_weights: list[float]
    images: list[CocoMetadata]
    annotations: dict[int, list[CocoParsedAnnotation]]
    symbol_categories: list[dict]
    line_categories: list[dict]

    @property
    def num_symbols(self) -> int:
        return sum(len(anns) for anns in self.annotations.values())

    def restrict_labels(self, keep_names: list[str]) -> None:
        """
        Filters the dataset to keep only the specified class names.
        Updates mappings, weights, and class counts in-place.
        """
        if not keep_names:
            return

        # 1. Identify COCO IDs to keep and create new contiguous mappings
        new_sym_id_to_idx = {}
        new_line_id_to_idx = {}
        keep_sym_ids = set()
        keep_line_ids = set()

        # Build sets of IDs to keep
        for name in keep_names:
            # Check symbols
            for cat in self.symbol_categories:
                if cat["name"] == name:
                    keep_sym_ids.add(cat["id"])
            # Check lines
            for cat in self.line_categories:
                if cat["name"] == name:
                    keep_line_ids.add(cat["id"])

        # Build new mappings (re-indexing to 0..N)
        # Symbols
        new_idx = 0
        for cat in self.symbol_categories:
            if cat["id"] in keep_sym_ids:
                new_sym_id_to_idx[cat["id"]] = new_idx
                new_idx += 1

        # Lines
        new_idx = 0
        for cat in self.line_categories:
            if cat["id"] in keep_line_ids:
                new_line_id_to_idx[cat["id"]] = new_idx
                new_idx += 1

        # 2. Filter Annotations
        new_annotations: dict[int, list[CocoParsedAnnotation]] = {}

        for img_id, anns in self.annotations.items():
            filtered_anns: list[CocoParsedAnnotation] = []
            for ann in anns:
                if isinstance(ann, CocoSymbolAnnotation):
                    if ann.category_id in keep_sym_ids:
                        filtered_anns.append(ann)
                elif isinstance(ann, CocoLineAnnotation):
                    if ann.category_id in keep_line_ids:
                        filtered_anns.append(ann)

            if filtered_anns:
                new_annotations[img_id] = filtered_anns

        # 3. Update Class Counts
        self.num_symbol_classes = len(new_sym_id_to_idx)
        self.num_line_classes = len(new_line_id_to_idx)

        # 4. Recalculate Weights
        final_sym_counts = {idx: 0 for idx in range(self.num_symbol_classes)}
        final_line_counts = {idx: 0 for idx in range(self.num_line_classes)}

        for anns in new_annotations.values():
            for ann in anns:
                if isinstance(ann, CocoSymbolAnnotation):
                    if ann.category_id in new_sym_id_to_idx:
                        final_sym_counts[
                            new_sym_id_to_idx[ann.category_id]
                        ] += 1
                elif isinstance(ann, CocoLineAnnotation):
                    if ann.category_id in new_line_id_to_idx:
                        final_line_counts[
                            new_line_id_to_idx[ann.category_id]
                        ] += 1

        self.symbol_weights = _compute_smoothed_weights(
            final_sym_counts, self.num_symbol_classes
        )
        self.line_weights = _compute_smoothed_weights(
            final_line_counts, self.num_line_classes
        )

        # 5. Update Mappings and Categories
        self.symbol_cat_id_to_idx = new_sym_id_to_idx
        self.line_cat_id_to_idx = new_line_id_to_idx
        self.symbol_categories = [
            c for c in self.symbol_categories if c["id"] in keep_sym_ids
        ]
        self.line_categories = [
            c for c in self.line_categories if c["id"] in keep_line_ids
        ]
        self.annotations = new_annotations

        print(
            f"Restricted dataset to {self.num_symbol_classes} symbol classes and {self.num_line_classes} line classes."
        )


def _compute_smoothed_weights(
    counts: dict[int, int],
    num_classes: int,
    beta: float = 0.5,
    target_mean: float = 0.25,
    min_val: float = 0.05,
    max_val: float = 0.85,
) -> list[float]:
    """
    Computes smoothed inverse frequency weights for class balancing.
    """
    if num_classes == 0:
        return []

    # Extract counts in order of class index
    freqs = [counts.get(i, 0) for i in range(num_classes)]

    # Handle zero frequencies by setting them to 1 to avoid division by zero.
    # They will naturally get the maximum clamped weight.
    freqs_safe = [f if f > 0 else 1 for f in freqs]

    # Smooth: f^beta
    smoothed = [math.pow(f, beta) for f in freqs_safe]

    # Inverse
    inverse = [1.0 / s for s in smoothed]

    # Normalize so the mean of the weights equals the target_mean (e.g., 0.25 for Focal Loss alpha)
    current_mean = sum(inverse) / num_classes
    scale = target_mean / current_mean if current_mean > 0 else 1.0
    normalized = [i * scale for i in inverse]

    # Clamp to prevent gradient explosions or vanishing gradients
    clamped = [max(min_val, min(max_val, n)) for n in normalized]

    return clamped


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

    annotations: defaultdict[int, list[CocoParsedAnnotation]] = defaultdict(list)
    symbol_counts: dict[int, int] = {
        idx: 0 for idx in range(len(symbol_categories))
    }
    line_counts: dict[int, int] = {
        idx: 0 for idx in range(len(line_categories))
    }

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
            idx = line_cat_id_to_idx[cat_id]
            line_counts[idx] += 1
        else:
            parsed_ann = CocoSymbolAnnotation(
                bbox=ann["bbox"], category_id=cat_id
            )
            idx = symbol_cat_id_to_idx[cat_id]
            symbol_counts[idx] += 1

        annotations[img_id].append(parsed_ann)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n--- Symbol Class Frequencies ---")
        for idx, cat in enumerate(symbol_categories):
            print(f"  {cat['name']:<30}: {symbol_counts[idx]}")

        print("\n--- Line Class Frequencies ---")
        for idx, cat in enumerate(line_categories):
            print(f"  {cat['name']:<30}: {line_counts[idx]}")
        print("--------------------------------\n")

    symbol_weights = _compute_smoothed_weights(
        symbol_counts, len(symbol_categories)
    )
    line_weights = _compute_smoothed_weights(line_counts, len(line_categories))

    dataset = CocoDataset(
        num_symbol_classes=len(symbol_categories),
        num_line_classes=len(line_categories),
        symbol_cat_id_to_idx=symbol_cat_id_to_idx,
        line_cat_id_to_idx=line_cat_id_to_idx,
        symbol_weights=symbol_weights,
        line_weights=line_weights,
        images=images,
        annotations=dict(annotations),
        symbol_categories=symbol_categories,
        line_categories=line_categories,
    )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"Caching parsed dataset to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


def load_coco_ssl_sample(
    dataset: CocoDataset, img_dir: Path, index: int
) -> Data[CocoMetadata, SSLSample[LazyImage]]:
    """Loads a single COCO sample for SSL (ignoring annotations)."""
    img_meta = dataset.images[index]
    img_path = img_dir / img_meta.file_name

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    return Data(
        metadata=img_meta,
        sample=SSLSample(
            image=LazyImage(
                path=img_path, width=img_meta.width, height=img_meta.height
            )
        ),
    )


def load_coco_sample(
    dataset: CocoDataset, img_dir: Path, index: int, device: torch.device
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

    # Create tensors directly on the target device
    boxes_tensor = torch.tensor(
        boxes, dtype=torch.float32, device=device
    ).reshape(-1, 4)
    box_labels_tensor = torch.tensor(
        box_labels, dtype=torch.int64, device=device
    )

    keypoints_tensor = torch.tensor(
        keypoints, dtype=torch.float32, device=device
    ).reshape(-1, 4)
    keypoint_labels_tensor = torch.tensor(
        keypoint_labels, dtype=torch.int64, device=device
    )

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
