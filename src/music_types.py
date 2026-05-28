from typing import Literal
from dataclasses import dataclass
from PIL import Image as Image_
import numpy as np
import torch

type Batch = int
type Height = int
type Width = int
type Channel = int
type View = int
type Dim = Batch | View | Height | Width | Channel
type Shape = tuple[Dim, ...]
HW = tuple[Height, Width]
HWC = tuple[Height, Width, Channel]
CHW = tuple[Channel, Height, Width]
type Layout = HW | HWC | CHW
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


@dataclass
class PILImage[L: HWC | HW, M: Mode, R: Range]:
    data: Image_.Image


@dataclass
class ArrayImage[L: AnyLayouts, M: Mode, R: Range]:
    data: np.ndarray


@dataclass
class TensorImage[L: AnyLayouts, M: Mode, R: Range]:
    data: torch.Tensor


type Image[L: Layouts, M, R] = (
    PILImage[L, M, R] | ArrayImage[L, M, R] | TensorImage[L, M, R]
)
type BatchedImage[L: BatchedLayouts, M, R] = (
    ArrayImage[L, M, R] | TensorImage[L, M, R]
)


@dataclass
class Data[Meta, I: Image]:
    metadata: Meta
    image: I


@dataclass
class BatchedData[Meta, I: BatchedImage]:
    metadata: list[Meta]
    image: I


type NumPatches = int
type PatchDim = int


@dataclass
class Patches[B: Batch, N: NumPatches, P: PatchDim]:
    data: torch.Tensor
    indices: torch.Tensor
    image_shape: CHW
    patch_size: HW

    @property
    def batch_size(self) -> B:
        return self.data.shape[0]


@dataclass
class BatchedPatchData[Meta, PT: Patches]:
    metadata: list[Meta]
    patches: PT
