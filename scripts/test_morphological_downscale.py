import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import pyvips

from transform.core import morphological_downscale_img
from music_types import TensorImage, CHW, RGB, Float1


def load_image(path: Path) -> TensorImage[CHW, RGB, Float1]:
    """Loads an image using pyvips and converts it to a float1 CHW TensorImage."""
    vips_img = pyvips.Image.new_from_file(str(path))
    arr = np.ndarray(
        buffer=vips_img.write_to_memory(),
        dtype=np.uint8,
        shape=(vips_img.height, vips_img.width, vips_img.bands),
    )
    t = torch.from_numpy(arr)
    if t.shape[-1] == 1:
        t = t.squeeze(-1).unsqueeze(0).expand(3, -1, -1)
    else:
        t = t.permute(2, 0, 1)
    return TensorImage(t.float() / 255.0)


def main():
    parser = argparse.ArgumentParser(
        description="Visually test morphological downscaling on a single image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("downscale_test.png"),
        help="Path to save the output visualization.",
    )
    args = parser.parse_args()

    img = load_image(args.image_path)

    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    fig, axes = plt.subplots(1, len(scales), figsize=(20, 10))
    for ax, scale in zip(axes, scales):
        c, h, w = img.data.shape
        h_out = max(1, int(h / scale))
        w_out = max(1, int(w / scale))

        downscaled = morphological_downscale_img(img, h_out, w_out)

        plot_img = downscaled.data.permute(1, 2, 0).numpy()
        ax.imshow(plot_img)
        ax.set_title(f"Scale {scale:.1f}x\n{downscaled.data.shape[1]}x{downscaled.data.shape[2]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
