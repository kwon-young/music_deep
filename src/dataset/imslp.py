from typing import Literal, cast
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
from PIL import Image as PILImage
from threaded_generator import partial_generator


@dataclass
class Metadata:
    score: str
    page: int
    name: str


HWC = Literal["HWC"]
CHW = Literal["CHW"]
Layout = HWC | CHW
Binary = Literal["1"]
Gray = Literal["L"]
RGB = Literal["RGB"]
Mode = Binary | Gray | RGB


class TypedImage[L: Layout, M: Mode](PILImage.Image):
    pass


@dataclass
class Image[T]:
    metadata: Metadata
    image: T


@partial_generator
def load_imslp(manifest: Path, image_dir: Path) -> Generator[Metadata]:
    with manifest.open("r") as f:
        for line in f:
            yield Metadata(**json.loads(line))


def load_image[T: Mode](
    metadata: Metadata,
    image_dir: Path,
    mode: Mode,
) -> Image[TypedImage[HWC, T]]:
    pil_img = PILImage.open(image_dir / metadata.name).convert(mode)
    return Image(metadata, cast(TypedImage[HWC, T], pil_img))
