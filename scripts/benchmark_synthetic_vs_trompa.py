#!/usr/bin/env python3
"""Benchmark synthetic page generation vs loading/decoding a real Trompa page.

Measures the per-sample cost of the data thread for both sources:
  * real Trompa:  load_coco_sample -> full-page decode (pyvips CPU / nvimgcodec GPU)
                  -> to_float1  -> patch extraction tail
  * synthetic:    generate() (Pillow) -> to tensor -> to_float1 -> same tail

The shared patch-extraction tail is reported for one full real page and one
synthetic page so the total data-thread cost is comparable.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from dataset.coco import parse_coco, load_coco_sample, CocoMetadata
from dataset.synthetic import SyntheticConfig, create_synthetic_dataset
from music_types import (
    Absolute,
    BoxDim,
    BoundingBoxes,
    CHW,
    ClassLabels,
    Data,
    DetectionSample,
    Float1,
    Int255,
    KeypointDim,
    Keypoints,
    NumBoxes,
    NumKeypoints,
    NumLineClasses,
    NumSymbolClasses,
    RGB,
    TensorImage,
    TopLeft,
    X1Y1X2Y2,
    XYXY,
)
from transform.core import (
    decode_nvimgcodec_img,
    decode_pyvips_img,
    to_float1_img,
)
import transform.det as det_tf


def timed(fn, iters: int, device: torch.device) -> float:
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def synthetic_float_data(generator, page_size: int) -> Data[
    CocoMetadata,
    DetectionSample[
        TensorImage[CHW, RGB, Float1],
        BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft],
        ClassLabels[tuple[NumBoxes], NumSymbolClasses],
        Keypoints[tuple[NumKeypoints, KeypointDim], X1Y1X2Y2, Absolute, TopLeft],
        ClassLabels[tuple[NumKeypoints], NumLineClasses],
    ],
]:
    page = generator.generate()
    arr = np.ascontiguousarray(page.image.transpose(2, 0, 1))
    img: TensorImage[CHW, RGB, Int255] = TensorImage(torch.from_numpy(arr))
    boxes: BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft] = (
        BoundingBoxes(torch.zeros((0, 4), dtype=torch.float32))
    )
    kps: Keypoints[tuple[NumKeypoints, KeypointDim], X1Y1X2Y2, Absolute, TopLeft] = (
        Keypoints(torch.zeros((0, 4), dtype=torch.float32))
    )
    meta = CocoMetadata(img_id=0, file_name="synthetic.png", width=page_size, height=page_size)
    return Data(
        metadata=meta,
        sample=DetectionSample(
            image=to_float1_img(img),
            boxes=boxes,
            box_labels=ClassLabels(torch.zeros((0,), dtype=torch.int64)),
            keypoints=kps,
            keypoint_labels=ClassLabels(torch.zeros((0,), dtype=torch.int64)),
        ),
    )


def tail_cost(data: Data[
    CocoMetadata,
    DetectionSample[
        TensorImage[CHW, RGB, Float1],
        BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft],
        ClassLabels[tuple[NumBoxes], NumSymbolClasses],
        Keypoints[tuple[NumKeypoints, KeypointDim], X1Y1X2Y2, Absolute, TopLeft],
        ClassLabels[tuple[NumKeypoints], NumLineClasses],
    ],
], device: torch.device) -> None:
    patched = det_tf.random_extract_patches_and_collate((data,), 54, 64)
    norm = det_tf.normalize_batched_targets(patched)
    dropped = det_tf.variance_patch_drop(norm, var_threshold=0.001)
    det_tf.resize_patches(dropped, target_patch_size=(64, 64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", type=Path,
                        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"))
    parser.add_argument("--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--page_sizes", type=str, default="2048,4096")
    parser.add_argument("--objects_per_page", type=int, default=240)
    parser.add_argument("--verovio_root", type=Path, default=Path("/home/kwon-young/prog/verovio"))
    args = parser.parse_args()

    cpu = torch.device("cpu")
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading cached COCO dataset (pickle)...")
    dataset = parse_coco(args.anno_path)

    data = load_coco_sample(dataset, args.img_dir, args.index, cpu)
    lazy = data.sample.image
    w, h = lazy.width, lazy.height
    print(f"Trompa page {lazy.path.name}: {w}x{h} ({w * h / 1e6:.1f} MP)")

    iters = args.iters

    print("\n--- Real Trompa per-sample load cost ---")
    t = timed(lambda: load_coco_sample(dataset, args.img_dir, args.index, cpu), iters, cpu)
    print(f"load_coco_sample (annotations):     {t * 1000:7.0f} ms")

    t = timed(lambda: to_float1_img(decode_pyvips_img(lazy, cpu)), iters, cpu)
    print(f"pyvips full decode + to_float1:     {t * 1000:7.0f} ms  ({w * h / 1e6 / t:.1f} MP/s)")

    try:
        def nv() -> None:
            to_float1_img(decode_nvimgcodec_img(lazy, cuda))
        t = timed(nv, iters, cuda)
        print(f"nvimgcodec full decode + to_float1: {t * 1000:7.0f} ms  ({w * h / 1e6 / t:.1f} MP/s)")
    except Exception as e:
        print(f"nvimgcodec full decode + to_float1: FAILED ({type(e).__name__}: {e})")

    print("\n--- Synthetic per-sample generation cost ---")
    for size_str in args.page_sizes.split(","):
        size = int(size_str)
        cfg = SyntheticConfig(page_size=size, objects_per_page=args.objects_per_page,
                              lines_per_page=args.objects_per_page // 3)
        ds = create_synthetic_dataset(cfg, args.verovio_root)
        gen = ds.generator
        page = gen.generate()
        print(f"  page {size}x{size} ({size * size / 1e6:.1f} MP): "
              f"{len(page.boxes)} symbols, {len(page.keypoints)} keypoint lines")
        t = timed(lambda: gen.generate(), iters, cpu)
        print(f"  generate():                        {t * 1000:7.0f} ms  ({size * size / 1e6 / t:.1f} MP/s)")
        t = timed(lambda: synthetic_float_data(gen, size), iters, cpu)
        print(f"  generate + tensor + to_float1:     {t * 1000:7.0f} ms")

    print("\n--- Shared patch-extraction tail (single full page) ---")
    real_float = det_tf.to_float1(det_tf.decode_pyvips(data, cpu))
    t = timed(lambda: tail_cost(real_float, cuda), iters, cuda)
    print(f"real 6300x8910 page tail:           {t * 1000:7.0f} ms")

    cfg = SyntheticConfig(page_size=2048, objects_per_page=args.objects_per_page,
                          lines_per_page=args.objects_per_page // 3)
    ds = create_synthetic_dataset(cfg, args.verovio_root)
    syn_float = synthetic_float_data(ds.generator, 2048)
    t = timed(lambda: tail_cost(syn_float, cuda), iters, cuda)
    print(f"synthetic 2048x2048 page tail:       {t * 1000:7.0f} ms")


if __name__ == "__main__":
    main()
