from typing import Literal, Any
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

type BCHW = tuple[Batch, *CHW]
type BHWC = tuple[Batch, *HWC]
type BatchView = int
type BVCHW = tuple[BatchView, *CHW]

type BatchedLayout = BHWC | BCHW | BVCHW
type AnyLayouts = Layout | BatchedLayout

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
class TensorData[Shape]:
    """Base class for any dataclass that wraps a PyTorch tensor."""

    data: torch.Tensor


@dataclass
class TensorImage[L: AnyLayouts, M: Mode, R: Range](TensorData[L]):
    pass


@dataclass
class BatchedTensorImage[
    B: Batch,
    M: Mode,
    R: Range,
](TensorImage[tuple[B, *CHW], M, R]):
    @property
    def batch_size(self) -> B:
        return self.data.shape[0]


@dataclass
class FlatViewTensorImage[
    B: Batch,
    V: View,
    L: BVCHW,
    M: Mode,
    R: Range,
](TensorImage[L, M, R]):
    num_views: V
    original_batch_size: B

    @property
    def batch_size(self) -> B:
        return self.data.shape[0]


type Image[L: Layout, M, R] = (
    PILImage[L, M, R] | ArrayImage[L, M, R] | TensorImage[L, M, R]
)
type BatchedImage[L: BatchedLayout, M, R] = (
    ArrayImage[L, M, R] | TensorImage[L, M, R]
)


@dataclass
class Data[Meta, T]:
    metadata: Meta
    data: T


@dataclass
class BatchedData[Meta, T]:
    metadata: list[Meta]
    data: T


type NumPatches = int
type PatchDim = int
type EmbedDim = int


@dataclass
class Embeddings[B: Batch, N: NumPatches, D: EmbedDim](
    TensorData[tuple[B, N, D]]
):
    indices: torch.Tensor
    image_shape: CHW
    patch_size: HW

    @property
    def batch_size(self) -> B:
        return self.data.shape[0]


type Patches[B: Batch, N: NumPatches, P: PatchDim] = Embeddings[B, N, P]


@dataclass
class FlatViewEmbeddings[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    D: EmbedDim,
](Embeddings[BV, N, D]):
    num_views: V
    original_batch_size: B


type FlatViewPatches[
    B: Batch,
    BV: BatchView,
    V: View,
    N: NumPatches,
    P: PatchDim,
] = FlatViewEmbeddings[B, BV, V, N, P]


@dataclass
class ViewEmbeddings[B: Batch, V: View, N: NumPatches, D: EmbedDim | PatchDim](
    Embeddings[B, N, D]
):
    @property
    def num_views(self) -> V:
        return self.data.shape[1]


type ViewPatches[B: Batch, V: View, N: NumPatches, P: PatchDim] = (
    ViewEmbeddings[B, V, N, P]
)


@dataclass
class DetectionTarget[Bx, Lbl]:
    """
    Holds the ground truth bounding boxes and labels for an image.
    """

    labels: Lbl
    boxes: Bx


@dataclass
class DetectionOutput:
    """
    Holds the predictions from the OMRDetector.
    """

    pred_logits: torch.Tensor  # (B, P*K, C)
    pred_boxes: torch.Tensor  # (B, P*K, 4)
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

    pred_indices: torch.Tensor  # 1D tensor of matched prediction indices
    target_indices: torch.Tensor  # 1D tensor of matched ground truth indices


@dataclass
class FlattenedIndices:
    """Holds the flattened batch and prediction indices for advanced indexing."""

    batch: torch.Tensor
    src: torch.Tensor


@dataclass
class MatchedTargets:
    """Holds the ground truth data for matched predictions."""

    labels: torch.Tensor
    boxes: torch.Tensor


@dataclass
class MatchedOutputs:
    """Holds the model predictions for matched indices."""

    boxes: torch.Tensor
    edge_logits: torch.Tensor
    centers: torch.Tensor
    shapes: torch.Tensor


# --- Modalities ---

type NumBoxes = int
type BoxDim = int

type BoxShape = tuple[NumBoxes, BoxDim]
type BatchedBoxShape = tuple[Batch, NumBoxes, BoxDim]
type AnyBoxShape = BoxShape | BatchedBoxShape
type XYXY = Literal["xyxy"]
type CXCYWH = Literal["cxcywh"]
type LTRB = Literal["ltrb"]  # Left Top Right Bottom
type BoxFormat = XYXY | CXCYWH | LTRB
type Absolute = Literal["Absolute"]
type ShapeNormalized = Literal["ShapeNormalized"]
type BoxRange = Float1 | Absolute | ShapeNormalized
type TopLeft = Literal["TopLeft"]
type Center = Literal["Center"]
type Origin = TopLeft | Center

type LabelShape = tuple[NumBoxes]
type BatchedLabelShape = tuple[Batch, NumBoxes]
type AnyLabelShape = LabelShape | BatchedLabelShape


@dataclass
class BoundingBoxes[L: AnyBoxShape, F: BoxFormat, R: BoxRange, O: Origin](
    TensorData[L]
):
    pass


@dataclass
class ClassLabels[L: AnyLabelShape](TensorData[L]):
    pass


# --- Task-Specific Samples ---


@dataclass
class SSLSample[I]:
    image: I


@dataclass
class ClassificationSample[I, L]:
    image: I
    labels: L


@dataclass
class DetectionSample[I, B, L]:
    image: I
    boxes: B
    labels: L
