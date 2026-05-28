import json
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
from PIL import Image as Image_

from types import Data, PILImage, HWC, RGB, Int255


@dataclass
class Metadata:
    score: str
    page: int
    name: str


def load_imslp(manifest: Path) -> Generator[Metadata, None, None]:
    with manifest.open("r") as f:
        for line in f:
            yield Metadata(**json.loads(line))


def load_image(
    metadata: Metadata,
    image_dir: Path,
) -> Data[Metadata, PILImage[HWC, RGB, Int255]]:
    pil_img = Image_.open(image_dir / metadata.name).convert("RGB")
    return Data(metadata, PILImage(pil_img))
