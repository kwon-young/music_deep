from typing import Literal, cast, Concatenate
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
from PIL import Image as Image_
import numpy as np
import torch


@dataclass
class Metadata:
    score: str
    page: int
    name: str


type Batch = int
type Height = int
type Width = int
type Channel = int
type View = int
type Dim = Batch | View | Height | Width | Channel
type Shape = tuple[Dim, ...]
HWC = tuple[Height, Width, Channel]
CHW = tuple[Channel, Height, Width]
type Layout = HWC | CHW
VCHW = tuple[View, *CHW]
type ViewLayout = tuple[View, *HWC] | VCHW
type Layouts = Layout | ViewLayout
type BCHW = tuple[Batch, *CHW]
type BatchedLayout = tuple[Batch, *HWC] | BCHW
type BVCHW = tuple[Batch, *VCHW]
type BatchedViewLayout = tuple[Batch, View, *HWC] | BVCHW
type BatchedLayouts = BatchedLayout | BatchedViewLayout
type AnyLayouts = Layouts | BatchedLayouts
type Binary = Literal["1"]
type Gray = Literal["L"]
type RGB = Literal["RGB"]
type Mode = Binary | Gray | RGB

type Int255 = Literal["Int255"]
type Float1 = Literal["Float1"]
type Range = Int255 | Float1


type PILImage[L: HWC, M: Mode, R: Range] = Image_.Image
type ArrayImage[L: AnyLayouts, M: Mode, R: Range] = np.ndarray
type TensorImage[L: AnyLayouts, M: Mode, R: Range] = torch.Tensor


type Image[L: Layouts, M, R] = (
    PILImage[L, M, R] | ArrayImage[L, M, R] | TensorImage[L, M, R]
)
type BatchedImage[L: BatchedLayouts, M, R] = (
    ArrayImage[L, M, R] | TensorImage[L, M, R]
)


@dataclass
class Data[I: Image]:
    metadata: Metadata
    image: I


@dataclass
class BatchedData[I: BatchedImage]:
    metadata: list[Metadata]
    image: I


def load_imslp(manifest: Path) -> Generator[Metadata]:
    with manifest.open("r") as f:
        for line in f:
            yield Metadata(**json.loads(line))


def load_image[M: Mode](
    metadata: Metadata,
    image_dir: Path,
    mode: M,
) -> Data[PILImage[HWC, M, Int255]]:
    pil_img = Image_.open(image_dir / metadata.name).convert(mode)
    return Data(metadata, cast(PILImage[HWC, M, Int255], pil_img))
