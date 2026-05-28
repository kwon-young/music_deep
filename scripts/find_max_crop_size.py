from pathlib import Path
from PIL import Image


def find_maximum_crop_size(image_dir: Path):
    min_width = float("inf")
    min_height = float("inf")

    # Assuming images are .tiff based on the README
    for img_path in image_dir.glob("*.tiff"):
        with Image.open(img_path) as img:
            w, h = img.size
            if w < min_width:
                min_width = w
            if h < min_height:
                min_height = h

    print(f"Minimum width across dataset: {min_width}")
    print(f"Minimum height across dataset: {min_height}")
    print(f"Maximum safe square crop_size: {min(min_width, min_height)}")


if __name__ == "__main__":
    find_maximum_crop_size(Path("data/imslp/images"))
