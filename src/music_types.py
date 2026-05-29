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


@dataclass
class DetectionTarget:
    """
    Holds the ground truth bounding boxes and labels for an image.
    """
    labels: torch.Tensor  # 1D tensor of shape (N,), dtype: torch.int64
    boxes: torch.Tensor   # 2D tensor of shape (N, 4), dtype: torch.float32


@dataclass
class DetectionOutput:
    """
    Holds the predictions from the OMRDetector.
    """
    pred_logits: torch.Tensor       # (B, P*K, C)
    pred_boxes: torch.Tensor        # (B, P*K, 4)
    pred_edge_logits: torch.Tensor  # (B, P*K, 4, reg_max+1)
    absolute_centers: torch.Tensor  # (B, P*K, 2)
    learnable_shapes: torch.Tensor  # (B, P*K, 2)


@dataclass
class DetectionLossWeights:
    """Weights for the different components of the detection loss."""
    loss_ce: float = 2.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_fgl: float = 0.15


@dataclass
class DetectionLosses:
    """Holds the weighted losses from the DFINECriterion."""
    loss_ce: torch.Tensor
    loss_bbox: torch.Tensor
    loss_giou: torch.Tensor
    loss_fgl: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.loss_ce + self.loss_bbox + self.loss_giou + self.loss_fgl


@dataclass
class MatchIndices:
    """Holds the bipartite matching indices for a single image."""
    pred_indices: torch.Tensor    # 1D tensor of matched prediction indices
    target_indices: torch.Tensor  # 1D tensor of matched ground truth indices
