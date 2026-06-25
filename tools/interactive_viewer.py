import sys
import argparse
from pathlib import Path
import numpy as np
import torch
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
)
from PySide6.QtGui import QPixmap, QPen, QColor, QPainter, QImage
from PySide6.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal

from model.detector import create_detector
from dataset.coco import parse_coco
import transform.core as core_tf
from music_types import TensorImage, CHW, RGB, Int255, Float1, Batch


class ZoomPanGraphicsView(QGraphicsView):
    """A custom QGraphicsView that supports zooming with the mouse wheel and panning."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        """Zoom in and out with the mouse wheel."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)


class InferenceSignals(QObject):
    inference_done = Signal(int, list, list)


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
    ):
        super().__init__()
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
        self.signals = InferenceSignals()

    def run(self):
        try:
            with torch.no_grad():
                keep_indices = core_tf.variance_patch_drop_indices(
                    self.patched_img.data,
                    var_threshold=self.var_threshold,
                    drop_rate=None,
                )
                dropped_img = core_tf.patch_drop_img(
                    self.patched_img, keep_indices
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(dropped_img)

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

                line_results = []
                line_logits = outputs.lines.pred_logits.data[0]
                line_kps = outputs.lines.pred_keypoints.data[0]
                line_probs = torch.sigmoid(line_logits)
                line_max_probs, line_labels = line_probs.max(dim=-1)

                line_keep = line_max_probs >= self.threshold
                line_kps_kept = line_kps[line_keep]
                line_labels_kept = line_labels[line_keep]
                line_scores_kept = line_max_probs[line_keep]

                line_kps_kept[:, [0, 1, 2, 3]] *= self.patch_size

                for kp, score, label in zip(
                    line_kps_kept, line_scores_kept, line_labels_kept
                ):
                    x1, y1, x2, y2 = kp.tolist()
                    x1 = max(0.0, min(x1, float(self.img_w)))
                    y1 = max(0.0, min(y1, float(self.img_h)))
                    x2 = max(0.0, min(x2, float(self.img_w)))
                    y2 = max(0.0, min(y2, float(self.img_h)))
                    line_results.append((x1, y1, x2, y2, score.item(), label.item()))

                self.signals.inference_done.emit(
                    self.task_id, sym_results, line_results
                )
        except Exception as e:
            print(f"Inference error: {e}")
            self.signals.inference_done.emit(self.task_id, [], [])


class InteractiveViewer(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.patched_img = None
        self.img_pil = None
        self.img_np = None
        self.current_pixmap = None
        self.var_threshold = 0.001
        self.conf_threshold = 0.5
        self.task_id = 0

        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

        self.setup_model()
        self.init_ui()

    def setup_model(self):
        print(f"Loading dataset metadata from {self.args.anno_path}...")
        dataset = parse_coco(self.args.anno_path)
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

        main_layout.addLayout(controls_layout)

        # Graphics View
        self.scene = QGraphicsScene()
        self.view = ZoomPanGraphicsView(self.scene)
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
        self.run_inference()

    def on_conf_changed(self, value):
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"Confidence: {self.conf_threshold:.2f}")
        self.run_inference()

    def load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        self.img_pil = img

        # Convert to QPixmap
        self.img_np = np.array(img)
        h, w, c = self.img_np.shape
        q_img = QImage(self.img_np.data, w, h, w * c, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)

        # Preprocess for model
        img_tensor = torch.from_numpy(self.img_np).permute(2, 0, 1)
        tensor_img: TensorImage[CHW, RGB, Int255] = TensorImage(data=img_tensor)
        float1_img = core_tf.to_float1_img(tensor_img)
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
            pixmap_item = QGraphicsPixmapItem(self.current_pixmap)
            self.scene.addItem(pixmap_item)
            self.scene.setSceneRect(pixmap_item.boundingRect())
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        self.run_inference()

    def run_inference(self):
        if self.patched_img is None:
            return

        self.task_id += 1
        task = InferenceTask(
            self.model,
            self.patched_img,
            self.args.patch_size,
            self.img_pil.width,
            self.img_pil.height,
            self.var_threshold,
            self.conf_threshold,
            self.device,
            self.args.use_amp,
            self.task_id,
        )
        task.signals.inference_done.connect(self.display_results)
        self.thread_pool.start(task)

    def display_results(self, task_id, sym_results, line_results):
        if task_id != self.task_id:
            return

        # Clear previous overlays (pixmap is item 0)
        for item in self.scene.items()[1:]:
            self.scene.removeItem(item)

        sym_pen = QPen(QColor(255, 0, 0, 200), 2)
        sym_pen.setCosmetic(True)

        for x1, y1, x2, y2, score, label in sym_results:
            rect = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            rect.setPen(sym_pen)
            rect.setToolTip(f"Sym: {label} ({score:.2f})")
            self.scene.addItem(rect)

        line_pen = QPen(QColor(0, 0, 255, 200), 2)
        line_pen.setCosmetic(True)

        for x1, y1, x2, y2, score, label in line_results:
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(line_pen)
            line.setToolTip(f"Line: {label} ({score:.2f})")
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
