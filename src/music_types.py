from typing import Literal, Self
from dataclasses import dataclass, replace, fields
from pathlib import Path
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
class DetachMixin:
    """Mixin that detaches any fields supporting the detach() method."""

    def detach(self) -> Self:
        changes = {}
        for f in fields(self):
            val = getattr(self, f.name)

            if hasattr(val, "detach") and callable(val.detach):
                changes[f.name] = val.detach()
            elif isinstance(val, list):
                # Handle lists of detachable objects (e.g., list[BoundingBoxes])
                changes[f.name] = [
                    v.detach()
                    if hasattr(v, "detach") and callable(v.detach)
                    else v
                    for v in val
                ]

        return replace(self, **changes)


@dataclass
class LazyImage[M: Mode, R: Range]:
    path: Path
    width: int
    height: int


@dataclass
class PILImage[L: HWC | HW, M: Mode, R: Range]:
    data: Image_.Image


@dataclass
class ArrayImage[L: AnyLayouts, M: Mode, R: Range]:
    data: np.ndarray


@dataclass
class TensorData[Shape](DetachMixin):
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
    sample: T


@dataclass
class BatchedData[Meta, T]:
    metadata: list[Meta]
    sample: T


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
class DetectionLossWeights:
    """Weights for the different components of the detection loss."""

    loss_ce: float = 2.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_fgl: float = 0.15


@dataclass
class DetectionLosses(DetachMixin):
    """Holds the weighted losses from the DFINECriterion."""

    loss_ce: torch.Tensor
    loss_bbox: torch.Tensor
    loss_giou: torch.Tensor
    loss_fgl: torch.Tensor


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


type NumClasses = int


@dataclass
class BoundingBoxes[L: AnyBoxShape, F: BoxFormat, R: BoxRange, O: Origin](
    TensorData[L]
):
    pass


@dataclass
class ClassLabels[L: AnyLabelShape, C: NumClasses](TensorData[L]):
    pass


# --- Detection Output Types ---

type NumShapes = int
type NumQueries = int  # NumQueries = NumPatches * NumShapes
type NumBins = int
type CoordDim = int

type LogitsShape = tuple[NumQueries, NumClasses]
type BatchedLogitsShape = tuple[Batch, NumQueries, NumClasses]
type AnyLogitsShape = LogitsShape | BatchedLogitsShape


@dataclass
class ClassLogits[L: AnyLogitsShape](TensorData[L]):
    pass


type EdgeLogitsShape = tuple[NumQueries, BoxDim, NumBins]
type BatchedEdgeLogitsShape = tuple[Batch, NumQueries, BoxDim, NumBins]
type AnyEdgeLogitsShape = EdgeLogitsShape | BatchedEdgeLogitsShape


@dataclass
class EdgeLogits[L: AnyEdgeLogitsShape](TensorData[L]):
    pass


type CoordShape = tuple[NumQueries, CoordDim]
type BatchedCoordShape = tuple[Batch, NumQueries, CoordDim]
type AnyCoordShape = CoordShape | BatchedCoordShape


@dataclass
class Coordinates[L: AnyCoordShape, R: Range](TensorData[L]):
    pass


@dataclass
class Dimensions[L: AnyCoordShape, R: Range](TensorData[L]):
    pass


@dataclass
class DetectionTarget[Bx, Lbl]:
    """
    Holds the ground truth bounding boxes and labels for an image.
    """

    labels: Lbl
    boxes: Bx


@dataclass
class DetectionOutput[B: Batch, Q: NumQueries, BD: BoxDim, CD: CoordDim](
    DetachMixin
):
    """
    Holds the predictions from the OMRDetector.
    """

    pred_logits: ClassLogits[tuple[B, Q, NumClasses]]
    pred_boxes: BoundingBoxes[tuple[B, Q, BD], XYXY, Float1, TopLeft]
    pred_edge_logits: EdgeLogits[tuple[B, Q, BD, NumBins]]
    absolute_centers: Coordinates[tuple[B, Q, CD], Float1]
    learnable_shapes: Dimensions[tuple[B, Q, CD], Float1]


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

    @property
    def num_symbols(self) -> int:
        if isinstance(self.labels, list):
            return sum(
                len(l.data) for l in self.labels if isinstance(l, TensorData)
            )
        if isinstance(self.labels, TensorData):
            return len(self.labels.data)
        return 0
