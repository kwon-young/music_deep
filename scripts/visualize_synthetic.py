#!/usr/bin/env python3
"""Generate synthetic pages and save preview images with ground-truth overlays.

Usage:
    mamba run -n pytorch python scripts/visualize_synthetic.py --num_pages 3

Writes ``page_XXX.png`` (raw) and ``page_XXX_gt.png`` (green symbol boxes,
blue keypoint lines) into --out_dir, and prints a per-page summary.
"""

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from dataset.synthetic import (
    LINE_NAMES,
    SYMBOL_NAMES,
    SyntheticConfig,
    create_synthetic_dataset,
)


def draw_ground_truth(page, draw: ImageDraw.ImageDraw) -> None:
    for (x1, y1, x2, y2), label in zip(page.boxes, page.box_labels):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
    for (x1, y1, x2, y2), label in zip(page.keypoints, page.keypoint_labels):
        draw.line([x1, y1, x2, y2], fill=(0, 0, 255), width=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview synthetic pages")
    parser.add_argument("--num_pages", type=int, default=3)
    parser.add_argument("--out_dir", type=Path, default=Path("synthetic_preview"))
    parser.add_argument("--page_size", type=int, default=2048)
    parser.add_argument("--objects_per_page", type=int, default=240)
    parser.add_argument("--lines_per_page", type=int, default=200)
    parser.add_argument("--staff_space_min", type=float, default=24.0)
    parser.add_argument("--staff_space_max", type=float, default=96.0)
    parser.add_argument("--font", type=str, default="Bravura")
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verovio_root", type=Path, default=Path("/home/kwon-young/prog/verovio"))
    args = parser.parse_args()

    config = SyntheticConfig(
        page_size=args.page_size,
        staff_space_min=args.staff_space_min,
        staff_space_max=args.staff_space_max,
        objects_per_page=args.objects_per_page,
        lines_per_page=args.lines_per_page,
        fonts=(args.font,),
        noise=args.noise,
        seed=args.seed,
    )
    dataset = create_synthetic_dataset(config, verovio_root=args.verovio_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_pages):
        page = dataset.generator.generate()
        raw = Image.fromarray(page.image)
        raw.save(args.out_dir / f"page_{i:03d}.png")

        gt = raw.copy()
        draw_ground_truth(page, ImageDraw.Draw(gt))
        gt.save(args.out_dir / f"page_{i:03d}_gt.png")

        sym_counts = Counter(SYMBOL_NAMES[l] for l in page.box_labels)
        line_counts = Counter(LINE_NAMES[l] for l in page.keypoint_labels)
        top = ", ".join(f"{n}:{c}" for n, c in sym_counts.most_common(5))
        if page.boxes:
            heights = [b[3] - b[1] for b in page.boxes]
            size_range = f"sym_h={min(heights):.0f}-{max(heights):.0f}px"
        else:
            size_range = "sym_h=-"
        print(
            f"page_{i:03d}: {size_range} symbols={len(page.boxes)} "
            f"lines={len(page.keypoints)} | top: {top}"
        )

    print(f"Wrote {args.num_pages} pages to {args.out_dir}")


if __name__ == "__main__":
    main()
