"""Renders synthetic pages to disk as a COCO dataset in the Trompa-COCO format.

The output can be read by ``inference_coco.py`` and ``evaluate_coco.py`` unchanged:
grayscale PNG pages, categories copied from a reference Trompa-COCO JSON (same ids
and order), symbol boxes, and line keypoints (start, end) whose ``area`` is
length × thickness, the convention OKS uses to normalize line distances.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from dataset.synthetic import (
    DEFAULT_VEROVIO_ROOT,
    LINE_CATEGORIES,
    SYMBOL_CATEGORIES,
    SyntheticConfig,
    SyntheticPage,
    create_synthetic_dataset,
)

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


@dataclass(frozen=True)
class RenderArgs:
    out_dir: Path
    reference_anno: Path
    num_pages: int
    config: SyntheticConfig
    verovio_root: Path


def symbol_annotation(
    ann_id: int, image_id: int, box: list[float], label: int
) -> dict[str, JsonValue]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": SYMBOL_CATEGORIES[label][0],
        "bbox": [x1, y1, w, h],
        "area": w * h,
        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
        "keypoints": [x1, (y1 + y2) / 2, 2],
        "num_keypoints": 1,
        "iscrowd": 0,
    }


def line_annotation(
    ann_id: int, image_id: int, kp: list[float], thickness: float, label: int
) -> dict[str, JsonValue]:
    x1, y1, x2, y2 = kp
    length = math.hypot(x2 - x1, y2 - y1)
    nx, ny = (
        (-(y2 - y1) / length, (x2 - x1) / length) if length > 0 else (0.0, 1.0)
    )
    hx, hy = nx * thickness / 2, ny * thickness / 2
    polygon = [x1 + hx, y1 + hy, x2 + hx, y2 + hy, x2 - hx, y2 - hy, x1 - hx, y1 - hy]
    xs, ys = polygon[0::2], polygon[1::2]
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": LINE_CATEGORIES[label][0],
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "area": max(1.0, length * thickness),
        "segmentation": [list[JsonValue](polygon)],
        "keypoints": [x1, y1, 2, x2, y2, 2],
        "num_keypoints": 2,
        "iscrowd": 0,
    }


def page_annotations(
    page: SyntheticPage, image_id: int, first_id: int
) -> list[dict[str, JsonValue]]:
    symbols = [
        symbol_annotation(first_id + i, image_id, box, label)
        for i, (box, label) in enumerate(zip(page.boxes, page.box_labels))
    ]
    offset = first_id + len(symbols)
    lines = [
        line_annotation(offset + i, image_id, kp, thickness, label)
        for i, (kp, thickness, label) in enumerate(
            zip(page.keypoints, page.keypoint_thicknesses, page.keypoint_labels)
        )
    ]
    return symbols + lines


def render(args: RenderArgs) -> None:
    reference = json.loads(args.reference_anno.read_text())
    assert isinstance(reference, dict)
    dataset = create_synthetic_dataset(args.config, args.verovio_root)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    images: list[JsonValue] = []
    annotations: list[JsonValue] = []
    for image_id in range(1, args.num_pages + 1):
        page = dataset.generator.generate()
        file_name = f"synthetic_{image_id:06d}.png"
        Image.fromarray(page.image[..., 0]).save(image_dir / file_name)
        images.append(
            {
                "id": image_id,
                "width": page.width,
                "height": page.height,
                "file_name": file_name,
            }
        )
        annotations.extend(page_annotations(page, image_id, len(annotations) + 1))
        print(
            f"page {image_id}/{args.num_pages}: "
            f"{len(page.boxes)} symbols, {len(page.keypoints)} lines"
        )

    coco: dict[str, JsonValue] = {
        "info": {"description": "Synthetic pages", "config": repr(args.config)},
        "images": images,
        "annotations": annotations,
        "categories": reference["categories"],
    }
    (args.out_dir / "instances.json").write_text(json.dumps(coco))


def parse_args() -> RenderArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_anno",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument("--num_pages", type=int, default=100)
    parser.add_argument("--page_size", type=int, default=2048)
    parser.add_argument("--staff_space_min", type=float, default=24.0)
    parser.add_argument("--staff_space_max", type=float, default=96.0)
    parser.add_argument("--objects_per_page", type=int, default=240)
    parser.add_argument("--lines_per_page", type=int, default=200)
    parser.add_argument("--line_max_pair_overlap", type=float, default=0.5)
    parser.add_argument("--font", type=str, default="Bravura")
    parser.add_argument("--background", type=int, default=247)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--ink_min", type=int, default=5)
    parser.add_argument("--ink_max", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verovio_root", type=Path, default=DEFAULT_VEROVIO_ROOT)
    a = parser.parse_args()
    config = SyntheticConfig(
        page_size=a.page_size,
        staff_space_min=a.staff_space_min,
        staff_space_max=a.staff_space_max,
        objects_per_page=a.objects_per_page,
        lines_per_page=a.lines_per_page,
        line_max_pair_overlap=a.line_max_pair_overlap,
        fonts=(a.font,),
        background=a.background,
        noise=a.noise,
        ink_min=a.ink_min,
        ink_max=a.ink_max,
        seed=a.seed,
    )
    return RenderArgs(
        out_dir=a.out_dir,
        reference_anno=a.reference_anno,
        num_pages=a.num_pages,
        config=config,
        verovio_root=a.verovio_root,
    )


if __name__ == "__main__":
    render(parse_args())
