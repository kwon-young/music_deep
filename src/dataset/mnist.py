from dataclasses import dataclass
from pathlib import Path
from typing import Generator, cast
from PIL import Image as Image_

from dataset.imslp import Data, PILImage, HWC, RGB, Int255


@dataclass
class MNISTMetadata:
    path: Path
    label: int


def load_mnist(
    root_dir: Path, split: str = "train"
) -> Generator[MNISTMetadata, None, None]:
    split_dir = root_dir / split
    for label_dir in split_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = int(label_dir.name)
        for img_path in label_dir.glob("*.png"):
            yield MNISTMetadata(path=img_path, label=label)


def load_image(metadata: MNISTMetadata) -> Data[PILImage[HWC, RGB, Int255]]:
    pil_img = Image_.open(metadata.path).convert("RGB")
    return Data(metadata, cast(PILImage[HWC, RGB, Int255], pil_img))
