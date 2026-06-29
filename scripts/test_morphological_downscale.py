import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

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

        # Plotting
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(downscaled.data.permute(1, 2, 0).numpy(), cmap="gray")

        # Draw boxes with transparency
        for i in range(len(new_boxes.data)):
            x1, y1, x2, y2 = new_boxes.data[i].tolist()
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="red", alpha=0.3
            )
            ax.add_patch(rect)

        # Draw keypoints (lines) with transparency
        for i in range(len(new_kps.data)):
            x1, y1, x2, y2 = new_kps.data[i].tolist()
            ax.plot([x1, x2], [y1, y2], color="lime", linewidth=1, alpha=0.5)

        ax.set_title(f"Scale {scale:.1f}x")
        ax.axis("off")

        out_path = args.output_dir / f"downscale_{scale:.1f}x_gt.png"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
