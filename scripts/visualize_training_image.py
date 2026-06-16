import argparse
from pathlib import Path
from unittest.mock import patch
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset.imslp import load_imslp, load_image
import transform.ssl as ssl_tf
from transform.core import random_patch_drop_indices


def visualize(
    manifest_path: Path,
    image_dir: Path,
    image_size: int = 224,
    patch_size: int = 16,
    drop_rate: float = 0.5,
):
    metadata_gen = load_imslp(manifest_path)
    metadata = next(metadata_gen)

    # 1. Reuse the exact preprocessing pipeline up to random_crop
    data = load_image(metadata, image_dir=image_dir)
    data_np = ssl_tf.to_numpy(data)
    data_t = ssl_tf.to_tensor(data_np)
    data_t = ssl_tf.to_float1(data_t)

    full_img = data_t.sample.image.data.clone()

    # Intercept torch.randint inside transform.core to get the crop coordinates
    coords = []
    original_randint = torch.randint

    def mock_randint(*args, **kwargs):
        val = original_randint(*args, **kwargs)
        coords.append(val.item())
        return val

    with patch("transform.core.torch.randint", side_effect=mock_randint):
        data_t = ssl_tf.random_crop(data_t, crop_size=image_size)

    x, y = coords[0], coords[1]

    # 2. Reuse the actual patch dropping code
    img_batch = data_t.sample.image.data.unsqueeze(0)

    num_patches_x = image_size // patch_size
    num_patches_y = image_size // patch_size
    num_patches = num_patches_x * num_patches_y

    # Intercept torch.rand inside transform.core to get the patch drop indices
    captured_rands = []
    original_rand = torch.rand

    def mock_rand(*args, **kwargs):
        val = original_rand(*args, **kwargs)
        if len(args) >= 2 and args[1] == num_patches:
            captured_rands.append(val)
        return val

    with patch("transform.core.torch.rand", side_effect=mock_rand):
        keep_indices = random_patch_drop_indices(
            1, num_patches, drop_rate, img_batch.device
        )[0]

    num_keep_patches = len(keep_indices)

    keep_mask = torch.zeros(num_patches, dtype=torch.bool)
    keep_mask[keep_indices] = True
    keep_mask = keep_mask.view(num_patches_y, num_patches_x)

    # 3. Create the visualization image
    dim_factor = 0.3
    vis_img = full_img.clone() * dim_factor  # Dim the entire image initially

    # Undim only the kept patches inside the cropped region
    for py in range(num_patches_y):
        for px in range(num_patches_x):
            if keep_mask[py, px]:
                start_y = y + py * patch_size
                end_y = start_y + patch_size
                start_x = x + px * patch_size
                end_x = start_x + patch_size

                vis_img[:, start_y:end_y, start_x:end_x] = full_img[
                    :, start_y:end_y, start_x:end_x
                ]

    vis_img_np = vis_img.permute(1, 2, 0).numpy()

    # 4. Plot using matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(vis_img_np)

    # Add the red bounding box around the cropped region
    rect = patches.Rectangle(
        (x, y),
        image_size,
        image_size,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)

    plt.title(
        f"Crop Visualization\nKeep {num_keep_patches}/{num_patches} patches"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path", type=Path, default=Path("data/imslp/imslp.jsonl")
    )
    parser.add_argument(
        "--image_dir", type=Path, default=Path("data/imslp/images")
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    args = parser.parse_args()

    visualize(
        manifest_path=args.manifest_path,
        image_dir=args.image_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
        drop_rate=args.drop_rate,
    )
