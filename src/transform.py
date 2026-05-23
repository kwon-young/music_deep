from typing import Literal, Iterator, Generator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage
from dataset.imslp import Metadata
import random


HWC = Literal["HWC"]
CHW = Literal["CHW"]
Layout = HWC | CHW
Binary = Literal["1"]
Gray = Literal["L"]
RGB = Literal["RGB"]
Mode = Binary | Gray | RGB


@dataclass
class Image[T, Layout, Mode]:
    metadata: Metadata
    image: T


NpImage = Image[np.ndarray, Layout, Mode]
TensorImage = Image[torch.Tensor, Layout, Mode]


def load_image[T: Mode](
    metadata: Metadata,
    image_dir: Path,
    mode: Mode,
) -> Image[PILImage.Image, HWC, T]:
    return Image(
        metadata, PILImage.open(image_dir / metadata.name).convert(mode)
    )


def to_numpy(
    image: Image[PILImage.Image, Layout, Mode],
) -> Image[np.ndarray, Layout, Mode]:
    return Image(image.metadata, np.array(image.image))


def to_tensor(image: NpImage) -> TensorImage:
    return Image(image.metadata, torch.as_tensor(image.image))


def shuffle[T](it: Iterator[T]) -> Generator[T]:
    l = list(it)
    random.shuffle(l)
    yield from l


def to(image: TensorImage, device: torch.device) -> TensorImage:
    return Image(image.metadata, image.image.to(device))
