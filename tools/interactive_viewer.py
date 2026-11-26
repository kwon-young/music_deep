import math
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QProgressBar,
    QCheckBox,
    QSpinBox,
)
from PySide6.QtGui import QPixmap, QPen, QColor, QPainter, QImage
from PySide6.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal

from model.detector import create_detector
from model.refiner import EndpointRefiner, LineQueries, RefinedEndpoints, TokenGrid
from dataset.coco import parse_coco
import transform.core as core_tf
from music_types import TensorImage, CHW, RGB, Int255, Float1, Batch


class ZoomPanGraphicsView(QGraphicsView):
    """A custom QGraphicsView that supports zooming with the mouse wheel and panning."""

    measurement_made = Signal(float)

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.measure_mode = False
        self.start_point = None
        self.temp_line = None

    def set_measure_mode(self, enabled: bool):
        self.measure_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        """Zoom in and out with the mouse wheel."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if self.measure_mode and event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.position().toPoint())
            self.temp_line = QGraphicsLineItem(
                self.start_point.x(), self.start_point.y(), self.start_point.x(), self.start_point.y()
            )
            pen = QPen(QColor(0, 255, 0, 200), 2)
            pen.setCosmetic(True)
            self.temp_line.setPen(pen)
            self.scene().addItem(self.temp_line)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.measure_mode and self.temp_line:
            end_point = self.mapToScene(event.position().toPoint())
            # Lock the X coordinate to the start point to ensure a strictly vertical line
            self.temp_line.setLine(
                self.start_point.x(), self.start_point.y(), self.start_point.x(), end_point.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.measure_mode and self.temp_line:
            end_point = self.mapToScene(event.position().toPoint())
            # Calculate distance only along the vertical axis
            dist = abs(end_point.y() - self.start_point.y())
            self.scene().removeItem(self.temp_line)
            self.temp_line = None
            self.start_point = None
            if dist > 0:
                self.measurement_made.emit(dist)
        else:
            super().mouseReleaseEvent(event)


LineResult = tuple[float, float, float, float, float, int]

MAX_REFINE_STEPS = 8


@dataclass(frozen=True)
class LineCache:
    """What the refiner needs from one detector pass, so it can rerun without the backbone:
    the kept final tokens and the kept coarse lines (patch units) with their scores."""

    grid: TokenGrid
    queries: LineQueries
    scores: torch.Tensor


@dataclass(frozen=True)
class RefinedLines:
    """Lines of ``cache`` after ``num_steps`` refiner applications, (N, 4) in patch units."""

    cache: LineCache
    num_steps: int
    endpoints: torch.Tensor
    lines: list[LineResult]


def load_refiner(
    path: Path, dim: int, num_classes: int, device: torch.device
) -> EndpointRefiner:
    saved: dict[str, object] = torch.load(path, map_location=device, weights_only=True)
    config = saved["config"]
    state = saved["refiner"]
    assert isinstance(config, dict) and isinstance(state, dict)
    refiner = EndpointRefiner(
        dim=dim,
        num_classes=num_classes,
        num_steps=int(str(config["num_steps"])),
        heads=int(str(config["heads"])),
        scale=float(str(config["scale"])),
    ).to(device)
    refiner.load_state_dict(state)
    refiner.eval()
    return refiner


def pixel_lines(
    endpoints: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    patch_size: int,
    img_w: int,
    img_h: int,
) -> list[LineResult]:
    """(N, 4) patch-unit endpoints -> pixel lines clipped to the image."""
    px = (endpoints * patch_size).float()
    px[:, 0::2] = px[:, 0::2].clamp(0.0, float(img_w))
    px[:, 1::2] = px[:, 1::2].clamp(0.0, float(img_h))
    return [
        (float(x1), float(y1), float(x2), float(y2), float(score), int(label))
        for (x1, y1, x2, y2), score, label in zip(
            px.tolist(), scores.tolist(), labels.tolist()
        )
    ]


def refine_lines(
    refiner: EndpointRefiner,
    cache: LineCache,
    num_steps: int,
    patch_size: int,
    img_w: int,
    img_h: int,
) -> RefinedLines:
    endpoints = cache.queries.endpoints
    if num_steps > 0 and endpoints.shape[0] > 0:
        with torch.no_grad():
            outputs: list[RefinedEndpoints] = refiner(
                cache.grid, cache.queries, num_steps=num_steps
            )
        endpoints = outputs[-1].endpoints
    lines = pixel_lines(
        endpoints, cache.scores, cache.queries.label, patch_size, img_w, img_h
    )
    return RefinedLines(cache=cache, num_steps=num_steps, endpoints=endpoints, lines=lines)


class InferenceSignals(QObject):
    # task id, symbols, coarse lines, dropped patches, LineCache | None, RefinedLines | None
    inference_done = Signal(int, list, list, list, object, object)


class RefineSignals(QObject):
    refine_done = Signal(int, object)


class RefineTask(QRunnable):
    """Reruns only the refiner on cached tokens: cheap enough for every UI change."""

    def __init__(
        self,
        refiner: EndpointRefiner,
        cache: LineCache,
        num_steps: int,
        patch_size: int,
        img_w: int,
        img_h: int,
        refine_id: int,
    ) -> None:
        super().__init__()
        self.refiner = refiner
        self.cache = cache
        self.num_steps = num_steps
        self.patch_size = patch_size
        self.img_w = img_w
        self.img_h = img_h
        self.refine_id = refine_id
        self.signals = RefineSignals()

    def run(self) -> None:
        try:
            refined = refine_lines(
                self.refiner,
                self.cache,
                self.num_steps,
                self.patch_size,
                self.img_w,
                self.img_h,
            )
        except Exception as e:
            print(f"Refinement error: {e}")
            self.signals.refine_done.emit(self.refine_id, None)
            return
        self.signals.refine_done.emit(self.refine_id, refined)


class InferenceTask(QRunnable):
    def __init__(
        self,
        model,
        patched_img,
        patch_size,
        img_w,
        img_h,
        var_threshold,
        threshold,
        device,
        use_amp,
        task_id,
        viewport_indices=None,
        refiner: EndpointRefiner | None = None,
        refine_steps: int | None = None,
        num_shapes: int = 5,
    ):
        super().__init__()
        self.refiner = refiner
        self.refine_steps = refine_steps
        self.num_shapes = num_shapes
        self.model = model
        self.patched_img = patched_img
        self.patch_size = patch_size
        self.img_w = img_w
        self.img_h = img_h
        self.var_threshold = var_threshold
        self.threshold = threshold
        self.device = device
        self.use_amp = use_amp
        self.task_id = task_id
        self.viewport_indices = viewport_indices
        self.signals = InferenceSignals()

    def run(self):
        try:
            with torch.no_grad():
                keep_indices = core_tf.variance_patch_drop_indices(
                    self.patched_img.data,
                    var_threshold=self.var_threshold,
                    drop_rate=None,
                )

                if self.viewport_indices is not None:
                    mask = torch.isin(keep_indices[0], self.viewport_indices)
                    keep_indices = keep_indices[:, mask]

                if keep_indices.shape[1] == 0:
                    self.signals.inference_done.emit(self.task_id, [], [], [], None, None)
                    return

                # --- Compute dropped patch coordinates for visualization ---
                _, h, w = self.patched_img.image_shape
                ph, pw = self.patch_size, self.patch_size
                grid_h = h // ph
                grid_w = w // pw

                num_patches = self.patched_img.data.shape[1]
                all_indices = torch.arange(num_patches, device=keep_indices.device)
                dropped_mask = ~torch.isin(all_indices, keep_indices[0])
                dropped_indices = all_indices[dropped_mask]

                dropped_patches = []
                for idx in dropped_indices.tolist():
                    row = idx // grid_w
                    col = idx % grid_w
                    x1 = col * pw
                    y1 = row * ph
                    dropped_patches.append((x1, y1, pw, ph))

                dropped_img = core_tf.patch_drop_img(
                    self.patched_img, keep_indices
                )
                # Move the patched image to the target device before inference
                dropped_img = core_tf.to_device_embeddings(dropped_img, self.device)
                final: list[torch.Tensor] = []

                def keep_final(
                    module: torch.nn.Module,
                    args: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                ) -> None:
                    final.append(output)

                norm = self.model.backbone.transformer.norm
                handle = (
                    norm.register_forward_hook(keep_final)
                    if self.refiner is not None
                    else None
                )
                try:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        outputs = self.model(dropped_img)
                finally:
                    if handle is not None:
                        handle.remove()

                sym_results = []
                sym_logits = outputs.symbols.pred_logits.data[0]
                sym_boxes = outputs.symbols.pred_boxes.data[0]
                sym_probs = torch.sigmoid(sym_logits)
                sym_max_probs, sym_labels = sym_probs.max(dim=-1)

                sym_keep = sym_max_probs >= self.threshold
                sym_boxes_kept = sym_boxes[sym_keep]
                sym_labels_kept = sym_labels[sym_keep]
                sym_scores_kept = sym_max_probs[sym_keep]

                sym_boxes_kept[:, [0, 1, 2, 3]] *= self.patch_size

                for box, score, label in zip(
                    sym_boxes_kept, sym_scores_kept, sym_labels_kept
                ):
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0.0, min(x1, float(self.img_w)))
                    y1 = max(0.0, min(y1, float(self.img_h)))
                    x2 = max(0.0, min(x2, float(self.img_w)))
                    y2 = max(0.0, min(y2, float(self.img_h)))
                    sym_results.append((x1, y1, x2, y2, score.item(), label.item()))

                line_logits = outputs.lines.pred_logits.data[0]
                line_kps = outputs.lines.pred_keypoints.data[0]
                line_probs = torch.sigmoid(line_logits)
                line_max_probs, line_labels = line_probs.max(dim=-1)

                line_kept = (line_max_probs >= self.threshold).nonzero().squeeze(1)
                coarse = line_kps[line_kept].float()
                line_scores_kept = line_max_probs[line_kept]
                line_labels_kept = line_labels[line_kept]
                line_results = pixel_lines(
                    coarse,
                    line_scores_kept,
                    line_labels_kept,
                    self.patch_size,
                    self.img_w,
                    self.img_h,
                )

                cache: LineCache | None = None
                refined: RefinedLines | None = None
                if self.refiner is not None and final:
                    # Each kept line query q comes from token q // num_shapes.
                    cache = LineCache(
                        grid=TokenGrid.from_batch(
                            final[0], dropped_img.indices, grid_h, grid_w
                        ),
                        queries=LineQueries(
                            image=torch.zeros_like(line_kept),
                            source=line_kept // self.num_shapes,
                            label=line_labels_kept,
                            endpoints=coarse,
                        ),
                        scores=line_scores_kept,
                    )
                    if self.refine_steps is not None:
                        refined = refine_lines(
                            self.refiner,
                            cache,
                            self.refine_steps,
                            self.patch_size,
                            self.img_w,
                            self.img_h,
                        )

                self.signals.inference_done.emit(
                    self.task_id,
                    sym_results,
                    line_results,
                    dropped_patches,
                    cache,
                    refined,
                )
        except Exception as e:
            print(f"Inference error: {e}")
            self.signals.inference_done.emit(self.task_id, [], [], [], None, None)


class InteractiveViewer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.patched_img = None
        self.img_pil = None
        self.img_np = None
        self.current_pixmap = None
        self.pixmap_item = None
        self.var_threshold = 0.001
        self.conf_threshold = 0.5
        self.task_id = 0
        self.scale_factor = 1.0
        self.target_interline = 54.0
        self.current_path = None
        self.img_w = 0
        self.img_h = 0
        self.refiner: EndpointRefiner | None = None
        self.refine_id = 0
        self.line_cache: LineCache | None = None
        self.refined: RefinedLines | None = None
        self.sym_results: list[tuple[float, float, float, float, float, int]] = []
        self.coarse_lines: list[LineResult] = []
        self.dropped_patches: list[tuple[int, int, int, int]] = []

        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

        self.setup_model()
        self.init_ui()

    def setup_model(self):
        print(f"Loading dataset metadata from {self.args.anno_path}...")
        dataset = parse_coco(self.args.anno_path)

        # Create mappings from class index to class name
        self.sym_idx_to_name = {
            dataset.symbol_cat_id_to_idx[cat["id"]]: cat["name"]
            for cat in dataset.symbol_categories
        }
        self.line_idx_to_name = {
            dataset.line_cat_id_to_idx[cat["id"]]: cat["name"]
            for cat in dataset.line_categories
        }

        print("Creating model...")
        self.model = create_detector(
            backbone_size=self.args.backbone_size,
            patch_size=self.args.patch_size,
            channels=self.args.channels,
            use_sdpa=self.args.use_sdpa,
            num_symbol_classes=dataset.num_symbol_classes,
            num_line_classes=dataset.num_line_classes,
            num_shapes=self.args.num_shapes,
            base_anchor_size=self.args.base_anchor_size,
        ).to(self.device)

        print(f"Loading checkpoint from {self.args.checkpoint}")
        checkpoint = torch.load(
            self.args.checkpoint, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        print("Model loaded successfully.")

        if self.args.refiner_checkpoint is not None:
            print(f"Loading endpoint refiner from {self.args.refiner_checkpoint}")
            self.refiner = load_refiner(
                self.args.refiner_checkpoint,
                dim=self.model.backbone.transformer.norm.normalized_shape[0],
                num_classes=dataset.num_line_classes,
                device=self.device,
            )
            print(f"Refiner loaded ({len(self.refiner.steps)} trained steps).")

    def init_ui(self):
        self.setWindowTitle("Interactive OMR Viewer")
        self.resize(1280, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls
        controls_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.on_load_image_clicked)
        controls_layout.addWidget(self.load_btn)

        controls_layout.addSpacing(20)

        self.viewport_crop_checkbox = QCheckBox("Drop patches outside viewport")
        self.viewport_crop_checkbox.setChecked(False)
        controls_layout.addWidget(self.viewport_crop_checkbox)

        controls_layout.addSpacing(20)

        self.var_label = QLabel(f"Var Threshold: {self.var_threshold:.4f}")
        controls_layout.addWidget(self.var_label)

        self.var_slider = QSlider(Qt.Horizontal)
        self.var_slider.setRange(0, 100)
        self.var_slider.setValue(int(self.var_threshold * 10000))
        self.var_slider.valueChanged.connect(self.on_var_changed)
        controls_layout.addWidget(self.var_slider)

        controls_layout.addSpacing(20)

        self.conf_label = QLabel(f"Confidence: {self.conf_threshold:.2f}")
        controls_layout.addWidget(self.conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        controls_layout.addWidget(self.conf_slider)

        if self.refiner is not None:
            controls_layout.addSpacing(20)
            trained_steps = len(self.refiner.steps)

            self.refine_checkbox = QCheckBox("Refine lines")
            self.refine_checkbox.setChecked(True)
            self.refine_checkbox.toggled.connect(self.sync_refinement)
            controls_layout.addWidget(self.refine_checkbox)

            self.refine_steps_spin = QSpinBox()
            self.refine_steps_spin.setRange(0, MAX_REFINE_STEPS)
            self.refine_steps_spin.setValue(trained_steps)
            self.refine_steps_spin.setPrefix("steps: ")
            self.refine_steps_spin.setToolTip(
                f"Refiner applications (trained: {trained_steps}). 0 = coarse lines; "
                "beyond the trained count the last step is applied again."
            )
            self.refine_steps_spin.valueChanged.connect(self.sync_refinement)
            controls_layout.addWidget(self.refine_steps_spin)

            self.show_coarse_checkbox = QCheckBox("Show coarse")
            self.show_coarse_checkbox.setChecked(False)
            self.show_coarse_checkbox.setToolTip(
                "Also draw the detector's coarse lines (thin orange) under the refined ones"
            )
            self.show_coarse_checkbox.toggled.connect(self.redraw)
            controls_layout.addWidget(self.show_coarse_checkbox)

        controls_layout.addSpacing(20)

        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self.run_inference)
        controls_layout.addWidget(self.run_btn)

        controls_layout.addSpacing(20)

        self.measure_btn = QPushButton("Measure")
        self.measure_btn.setCheckable(True)
        self.measure_btn.toggled.connect(self.on_measure_toggled)
        controls_layout.addWidget(self.measure_btn)

        self.measure_label = QLabel("Scale: 1.00x (Interline: 54px)")
        controls_layout.addWidget(self.measure_label)

        # Spinner to indicate inference is running
        self.spinner = QProgressBar()
        self.spinner.setRange(0, 0)  # Indeterminate mode
        self.spinner.setTextVisible(False)
        self.spinner.setMaximumWidth(30)
        self.spinner.hide()
        controls_layout.addWidget(self.spinner)

        main_layout.addLayout(controls_layout)

        # Graphics View
        self.scene = QGraphicsScene()
        self.view = ZoomPanGraphicsView(self.scene)
        self.view.measurement_made.connect(self.update_measurement)
        main_layout.addWidget(self.view)

    def on_load_image_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.load_image(file_path)

    def on_var_changed(self, value):
        self.var_threshold = value / 10000.0
        self.var_label.setText(f"Var Threshold: {self.var_threshold:.4f}")

    def on_conf_changed(self, value):
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"Confidence: {self.conf_threshold:.2f}")

    def on_measure_toggled(self, checked):
        self.view.set_measure_mode(checked)

    def update_measurement(self, dist):
        # The measurement is done on the currently displayed image, which is scaled by self.scale_factor.
        # To get the scale factor relative to the original image, we must account for the current scale.
        self.scale_factor = self.scale_factor * self.target_interline / dist
        self.measure_label.setText(f"Scale: {self.scale_factor:.2f}x (Measured: {dist:.1f}px)")
        self.measure_btn.setChecked(False)
        if self.current_path:
            self.load_image(self.current_path)

    def load_image(self, path: str):
        self.current_path = path
        img = Image.open(path).convert("RGB")
        self.img_pil = img

        # Convert to QPixmap
        self.img_np = np.array(img)

        # Preprocess for model
        img_tensor = torch.from_numpy(self.img_np).permute(2, 0, 1)
        tensor_img: TensorImage[CHW, RGB, Int255] = TensorImage(data=img_tensor)
        float1_img = core_tf.to_float1_img(tensor_img)

        if self.scale_factor != 1.0:
            data = float1_img.data.unsqueeze(0)
            new_h = int(data.shape[2] * self.scale_factor)
            new_w = int(data.shape[3] * self.scale_factor)
            resized_data = F.interpolate(data, size=(new_h, new_w), mode='bilinear', align_corners=False)
            float1_img = TensorImage(data=resized_data.squeeze(0))
            # .contiguous() is required because permute() creates a non-contiguous view,
            # which causes a BufferError when passed to QImage.
            self.img_np = (resized_data.squeeze(0).permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)

        self.img_w = self.img_np.shape[1]
        self.img_h = self.img_np.shape[0]

        h, w, c = self.img_np.shape
        q_img = QImage(self.img_np.data, w, h, w * c, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)

        padded_img = core_tf.pad_to_patch_size_img(
            float1_img, (self.args.patch_size, self.args.patch_size)
        )
        batched_img: TensorImage[tuple[Batch, *CHW], RGB, Float1] = TensorImage(
            data=padded_img.data.unsqueeze(0)
        )
        self.patched_img = core_tf.extract_patches_img(
            batched_img, (self.args.patch_size, self.args.patch_size)
        )

        self.scene.clear()
        if self.current_pixmap:
            self.pixmap_item = QGraphicsPixmapItem(self.current_pixmap)
            self.scene.addItem(self.pixmap_item)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.run_inference()

    def get_viewport_patch_indices(self):
        if self.patched_img is None:
            return None
        
        # Get the visible rectangle in scene coordinates
        viewport_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        
        # Get image dimensions
        _, h, w = self.patched_img.image_shape
        ph, pw = self.args.patch_size, self.args.patch_size
        grid_h = h // ph
        grid_w = w // pw
        
        # Calculate the range of rows and cols that intersect the viewport
        start_col = max(0, int(viewport_rect.x() // pw))
        end_col = min(grid_w, int((viewport_rect.x() + viewport_rect.width()) // pw) + 1)
        
        start_row = max(0, int(viewport_rect.y() // ph))
        end_row = min(grid_h, int((viewport_rect.y() + viewport_rect.height()) // ph) + 1)
        
        # Create a list of indices
        indices = []
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                indices.append(r * grid_w + c)
            
        if not indices:
            return None
            
        return torch.tensor(indices, dtype=torch.long, device=self.patched_img.data.device)

    def run_inference(self):
        if self.patched_img is None:
            return

        self.spinner.show()
        self.task_id += 1
        # Any refinement still pending belongs to the previous inference.
        self.refine_id += 1
        
        viewport_indices = None
        if self.viewport_crop_checkbox.isChecked():
            viewport_indices = self.get_viewport_patch_indices()

        task = InferenceTask(
            self.model,
            self.patched_img,
            self.args.patch_size,
            self.img_w,
            self.img_h,
            self.var_threshold,
            self.conf_threshold,
            self.device,
            self.args.use_amp,
            self.task_id,
            viewport_indices=viewport_indices,
            refiner=self.refiner,
            refine_steps=self.requested_refine_steps(),
            num_shapes=self.args.num_shapes,
        )
        task.signals.inference_done.connect(self.display_results)
        self.thread_pool.start(task)

    def requested_refine_steps(self) -> int | None:
        """Refiner applications wanted by the UI, None when refinement is off."""
        if self.refiner is None or not self.refine_checkbox.isChecked():
            return None
        return self.refine_steps_spin.value()

    def sync_refinement(self) -> None:
        """Brings the refined lines in line with the UI, rerunning only the refiner on the
        cached tokens of the last inference (off the GUI thread) when they are stale."""
        steps = self.requested_refine_steps()
        if steps is None or self.refiner is None or self.line_cache is None:
            self.refine_id += 1
            self.refined = None
            self.redraw()
            return
        current = self.refined
        if (
            current is not None
            and current.cache is self.line_cache
            and current.num_steps == steps
        ):
            self.redraw()
            return
        self.refine_id += 1
        self.spinner.show()
        task = RefineTask(
            self.refiner,
            self.line_cache,
            steps,
            self.args.patch_size,
            self.img_w,
            self.img_h,
            self.refine_id,
        )
        task.signals.refine_done.connect(self.on_refine_done)
        self.thread_pool.start(task)

    def on_refine_done(self, refine_id: int, refined: RefinedLines | None) -> None:
        if refine_id != self.refine_id:
            return
        self.spinner.hide()
        self.refined = refined
        self.redraw()

    def display_results(
        self,
        task_id: int,
        sym_results: list[tuple[float, float, float, float, float, int]],
        line_results: list[LineResult],
        dropped_patches: list[tuple[int, int, int, int]],
        line_cache: LineCache | None,
        refined: RefinedLines | None,
    ) -> None:
        if task_id != self.task_id:
            return

        self.spinner.hide()
        self.sym_results = sym_results
        self.coarse_lines = line_results
        self.dropped_patches = dropped_patches
        self.line_cache = line_cache
        self.refined = refined
        # The controls may have changed while the backbone was running.
        self.sync_refinement()

    def redraw(self) -> None:
        sym_results = self.sym_results
        dropped_patches = self.dropped_patches

        # Clear previous overlays, keeping only the base pixmap
        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)

        # Draw dropped patches (semi-transparent gray)
        drop_brush = QColor(128, 128, 128, 100)
        drop_pen = QPen(Qt.NoPen)
        for px, py, pw, ph in dropped_patches:
            rect = QGraphicsRectItem(px, py, pw, ph)
            rect.setBrush(drop_brush)
            rect.setPen(drop_pen)
            self.scene.addItem(rect)

        sym_pen = QPen(QColor(255, 0, 0, 200), 2)
        sym_pen.setCosmetic(True)

        for x1, y1, x2, y2, score, label in sym_results:
            rect = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            rect.setPen(sym_pen)
            name = self.sym_idx_to_name.get(label, str(label))
            rect.setToolTip(f"Sym: {name} ({score:.2f})")
            self.scene.addItem(rect)

        line_pen = QPen(QColor(0, 0, 255, 200), 2)
        line_pen.setCosmetic(True)

        refined = self.refined
        if refined is None:
            for x1, y1, x2, y2, score, label in self.coarse_lines:
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(line_pen)
                name = self.line_idx_to_name.get(label, str(label))
                line.setToolTip(f"Line: {name} ({score:.2f})")
                self.scene.addItem(line)
            return

        if self.show_coarse_checkbox.isChecked():
            coarse_pen = QPen(QColor(255, 140, 0, 160), 1)
            coarse_pen.setCosmetic(True)
            for x1, y1, x2, y2, score, label in self.coarse_lines:
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(coarse_pen)
                line.setZValue(1)
                name = self.line_idx_to_name.get(label, str(label))
                line.setToolTip(f"Coarse line: {name} ({score:.2f})")
                self.scene.addItem(line)

        moves = (
            torch.linalg.vector_norm(
                (refined.endpoints - refined.cache.queries.endpoints).view(-1, 2, 2), dim=-1
            )
            * self.args.patch_size
        ).tolist()
        for (x1, y1, x2, y2, score, label), (d1, d2) in zip(refined.lines, moves):
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(line_pen)
            line.setZValue(2)
            name = self.line_idx_to_name.get(label, str(label))
            line.setToolTip(
                f"Line: {name} ({score:.2f}), refined x{refined.num_steps}: "
                f"endpoints moved {d1:.1f} / {d2:.1f} px"
            )
            self.scene.addItem(line)


def main():
    parser = argparse.ArgumentParser(description="Interactive OMR Viewer")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument(
        "--backbone_size",
        type=str,
        choices=["nano", "small", "base"],
        default="nano",
    )
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument(
        "--refiner_checkpoint",
        type=Path,
        default=None,
        help="Optional Exp 035 line endpoint refiner (refiner.pt); enables the refine controls",
    )
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = InteractiveViewer(args)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
