import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

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
    QListWidget,
    QSlider,
    QLabel,
    QCheckBox,
    QSplitter,
    QMessageBox,
)
from PySide6.QtGui import QPixmap, QPen, QColor, QPainter
from PySide6.QtCore import Qt


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


class CocoViewer(QMainWindow):
    def __init__(self, anno_path: Path, pred_path: Path, img_dir: Path):
        super().__init__()
        self.anno_path = anno_path
        self.pred_path = pred_path
        self.img_dir = img_dir

        self.images = {}
        self.categories = {}
        self.gt_anns = defaultdict(list)
        self.pred_anns = defaultdict(list)

        self.current_pred_items = []  # List of (QGraphicsItem, score)
        self.current_gt_items = []    # List of QGraphicsItem

        self.setWindowTitle("COCO OMR Viewer")
        self.resize(1280, 720)

        self.init_ui()
        self.load_data()

    def init_ui(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter to separate list and viewer
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left Panel: Image List
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        left_layout.addWidget(QLabel("Images:"))
        left_layout.addWidget(self.image_list)
        
        splitter.addWidget(left_panel)

        # Right Panel: Controls + Viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.gt_checkbox = QCheckBox("Show Ground Truth (Green)")
        self.gt_checkbox.setChecked(True)
        self.gt_checkbox.stateChanged.connect(self.update_visibility)
        controls_layout.addWidget(self.gt_checkbox)

        self.pred_checkbox = QCheckBox("Show Predictions (Red)")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.stateChanged.connect(self.update_visibility)
        controls_layout.addWidget(self.pred_checkbox)

        controls_layout.addSpacing(20)

        self.thresh_label = QLabel("Threshold: 0.50")
        controls_layout.addWidget(self.thresh_label)

        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(50)
        self.thresh_slider.valueChanged.connect(self.on_threshold_changed)
        controls_layout.addWidget(self.thresh_slider)

        right_layout.addLayout(controls_layout)

        # Graphics View
        self.scene = QGraphicsScene()
        self.view = ZoomPanGraphicsView(self.scene)
        right_layout.addWidget(self.view)

        splitter.addWidget(right_panel)
        splitter.setSizes([250, 1030])

    def load_data(self):
        print(f"Loading Ground Truth from {self.anno_path}...")
        try:
            with open(self.anno_path, "r") as f:
                coco = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load GT JSON:\n{e}")
            sys.exit(1)

        for cat in coco.get("categories", []):
            self.categories[cat["id"]] = cat["name"]

        for img in coco.get("images", []):
            self.images[img["id"]] = img

        for ann in coco.get("annotations", []):
            self.gt_anns[ann["image_id"]].append(ann)

        if self.pred_path and self.pred_path.exists():
            print(f"Loading Predictions from {self.pred_path}...")
            try:
                # Support loading a directory of JSONs (e.g., preds_symbols.json and preds_lines.json)
                pred_files = [self.pred_path] if self.pred_path.is_file() else list(self.pred_path.glob("*.json"))
                for pf in pred_files:
                    with open(pf, "r") as f:
                        preds = json.load(f)
                    for p in preds:
                        self.pred_anns[p["image_id"]].append(p)
            except Exception as e:
                print(f"Warning: Failed to load predictions:\n{e}")

        # Populate list
        sorted_images = sorted(self.images.values(), key=lambda x: x["file_name"])
        for img in sorted_images:
            self.image_list.addItem(img["file_name"])
            # Store image_id in the item's UserRole data for easy retrieval
            self.image_list.item(self.image_list.count() - 1).setData(Qt.UserRole, img["id"])

        print("Data loaded successfully.")

    def on_image_selected(self, current, previous):
        if not current:
            return

        img_id = current.data(Qt.UserRole)
        img_meta = self.images[img_id]
        img_path = self.img_dir / img_meta["file_name"]

        self.scene.clear()
        self.current_pred_items.clear()
        self.current_gt_items.clear()

        if not img_path.exists():
            self.scene.addText(f"Image not found:\n{img_path}")
            return

        # Load Image
        pixmap = QPixmap(str(img_path))
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)
        self.scene.setSceneRect(pixmap_item.boundingRect())

        # Draw Ground Truth
        gt_pen = QPen(QColor(0, 255, 0, 200), 2)
        gt_pen.setCosmetic(True)  # Keeps line width constant regardless of zoom
        
        for ann in self.gt_anns.get(img_id, []):
            cat_name = self.categories.get(ann["category_id"], "Unknown")
            
            if "keypoints" in ann and len(ann["keypoints"]) >= 4:
                kps = ann["keypoints"]
                if len(kps) >= 6:
                    x1, y1, x2, y2 = kps[0], kps[1], kps[3], kps[4]
                else:
                    x1, y1, x2, y2 = kps[0], kps[1], kps[2], kps[3]
                item = QGraphicsLineItem(x1, y1, x2, y2)
            elif "bbox" in ann:
                x, y, w, h = ann["bbox"]
                item = QGraphicsRectItem(x, y, w, h)
            else:
                continue

            item.setPen(gt_pen)
            item.setToolTip(f"GT: {cat_name}")
            self.scene.addItem(item)
            self.current_gt_items.append(item)

        # Draw Predictions
        pred_pen = QPen(QColor(255, 0, 0, 200), 2, Qt.DashLine)
        pred_pen.setCosmetic(True)

        for ann in self.pred_anns.get(img_id, []):
            score = ann.get("score", 0.0)
            cat_name = self.categories.get(ann["category_id"], "Unknown")
            
            if "keypoints" in ann and len(ann["keypoints"]) >= 4:
                kps = ann["keypoints"]
                if len(kps) >= 6:
                    x1, y1, x2, y2 = kps[0], kps[1], kps[3], kps[4]
                else:
                    x1, y1, x2, y2 = kps[0], kps[1], kps[2], kps[3]
                item = QGraphicsLineItem(x1, y1, x2, y2)
            elif "bbox" in ann:
                x, y, w, h = ann["bbox"]
                item = QGraphicsRectItem(x, y, w, h)
            else:
                continue
            
            item.setPen(pred_pen)
            item.setToolTip(f"Pred: {cat_name}\nScore: {score:.3f}")
            self.scene.addItem(item)
            self.current_pred_items.append((item, score))

        # Fit view to image
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # Apply current visibility and threshold settings
        self.update_visibility()

    def on_threshold_changed(self, value):
        threshold = value / 100.0
        self.thresh_label.setText(f"Threshold: {threshold:.2f}")
        self.update_visibility()

    def update_visibility(self):
        show_gt = self.gt_checkbox.isChecked()
        show_pred = self.pred_checkbox.isChecked()
        threshold = self.thresh_slider.value() / 100.0

        for item in self.current_gt_items:
            item.setVisible(show_gt)

        for item, score in self.current_pred_items:
            item.setVisible(show_pred and score >= threshold)


def main():
    parser = argparse.ArgumentParser(description="Fast COCO Viewer with PySide6")
    parser.add_argument("--anno_path", type=Path, required=True, help="Path to ground truth JSON")
    parser.add_argument("--pred_path", type=Path, default=None, help="Path to predictions JSON or directory containing JSONs")
    parser.add_argument("--img_dir", type=Path, required=True, help="Path to image directory")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = CocoViewer(args.anno_path, args.pred_path, args.img_dir)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
