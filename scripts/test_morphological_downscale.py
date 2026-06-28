import argparse
from pathlib import Path
import torch
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


def save_image(img: TensorImage[CHW, RGB, Float1], path: Path) -> None:
    """Saves a TensorImage directly to a file using pyvips to avoid matplotlib resampling."""
    arr = (img.data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    height, width, bands = arr.shape
    vips_img = pyvips.Image.new_from_memory(
        arr.tobytes(), width, height, bands, "uchar"
    )
    vips_img.write_to_file(str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Visually test morphological downscaling on a single image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("downscale_test"),
        help="Directory to save the output images.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    img = load_image(args.image_path)

    scales = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for scale in scales:
        c, h, w = img.data.shape
        h_out = max(1, int(h / scale))
        w_out = max(1, int(w / scale))

        downscaled = morphological_downscale_img(img, h_out, w_out)
        
        out_path = args.output_dir / f"downscale_{scale:.1f}x.png"
        save_image(downscaled, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
