import argparse
from pathlib import Path
import torch
import numpy as np
import pyvips

from dataset.coco import parse_coco, load_coco_sample
from transform.core import (
    morphological_downscale_img,
    decode_pyvips_img,
    to_float1_img,
    scale_boxes_xyxy,
    scale_keypoints,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visually test morphological downscaling on a single image with ground truth overlay."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
        help="Path to the COCO annotations file.",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=Path("data/trompa-coco/trainval2017"),
        help="Path to the image directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("downscale_test"),
        help="Directory to save the output images.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset and find the matching image index
    dataset = parse_coco(args.anno_path)
    img_name = args.image_path.name
    try:
        index = next(i for i, img in enumerate(dataset.images) if img.file_name == img_name)
    except StopIteration:
        raise ValueError(f"Image {img_name} not found in dataset annotations.")

    # Load sample (image + annotations)
    item = load_coco_sample(dataset, args.img_dir, index, torch.device("cpu"))

    # Decode image and convert to float1
    img = decode_pyvips_img(item.sample.image, torch.device("cpu"))
    img = to_float1_img(img)

    boxes = item.sample.boxes
    box_labels = item.sample.box_labels
    keypoints = item.sample.keypoints
    keypoint_labels = item.sample.keypoint_labels

    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for scale in scales:
        c, h, w = img.data.shape
        h_out = max(1, int(h / scale))
        w_out = max(1, int(w / scale))

        exact_scale_h = h / h_out
        exact_scale_w = w / w_out

        downscaled = morphological_downscale_img(img, h_out, w_out)

        if exact_scale_h == 1.0 and exact_scale_w == 1.0:
            new_boxes, _ = boxes, box_labels
            new_kps, _ = keypoints, keypoint_labels
        else:
            new_boxes, _ = scale_boxes_xyxy(boxes, box_labels, exact_scale_h, exact_scale_w)
            new_kps, _ = scale_keypoints(keypoints, keypoint_labels, exact_scale_h, exact_scale_w)

        # Draw overlay directly on the image tensor to preserve exact resolution
        img_np = downscaled.data.permute(1, 2, 0).numpy()
        img_uint8 = (img_np * 255).astype(np.uint8).copy()
        h, w, _ = img_uint8.shape

        # Draw boxes (Red, 50% opacity)
        for i in range(len(new_boxes.data)):
            x1, y1, x2, y2 = map(int, new_boxes.data[i].tolist())
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
            if x2 >= x1 and y2 >= y1:
                # Top & Bottom
                img_uint8[y1, x1:x2+1] = (0.5 * img_uint8[y1, x1:x2+1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                img_uint8[y2, x1:x2+1] = (0.5 * img_uint8[y2, x1:x2+1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                # Left & Right
                img_uint8[y1:y2+1, x1] = (0.5 * img_uint8[y1:y2+1, x1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                img_uint8[y1:y2+1, x2] = (0.5 * img_uint8[y1:y2+1, x2] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

        # Draw keypoints (Green, 50% opacity)
        for i in range(len(new_kps.data)):
            x1, y1, x2, y2 = new_kps.data[i].tolist()
            length = max(abs(x2 - x1), abs(y2 - y1))
            if length > 0:
                xs = np.linspace(x1, x2, int(length) + 1).astype(int)
                ys = np.linspace(y1, y2, int(length) + 1).astype(int)
                valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                xs, ys = xs[valid], ys[valid]
                img_uint8[ys, xs] = (0.5 * img_uint8[ys, xs] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)

        # Save using pyvips to avoid matplotlib resampling
        height, width, bands = img_uint8.shape
        vips_img = pyvips.Image.new_from_memory(
            img_uint8.tobytes(), width, height, bands, "uchar"
        )
        out_path = args.output_dir / f"downscale_{scale:.1f}x_gt.png"
        vips_img.write_to_file(str(out_path))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
