import sys
import torch
import torch.optim as optim
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model.vit import vit_nano
from model.detector import OMRDetector
from model.matcher import HungarianMatcher
from model.criterion import DFINECriterion
from transform import extract_patches


def load_yolo_label(txt_path: Path, img_w: int, img_h: int):
    """Converts YOLO normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]"""
    labels = []
    boxes = []
    if txt_path.exists():
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # Keep normalized coordinates
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                labels.append(class_id)
                boxes.append([x1, y1, x2, y2])

    return {
        "labels": torch.tensor(labels, dtype=torch.int64),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Model (using vit_nano for speed)
    num_classes = 80  # COCO has 80 classes
    backbone = vit_nano(num_classes=0, patch_size=16, channels=3)
    model = OMRDetector(backbone, num_classes=num_classes, num_shapes=5).to(
        device
    )

    # 2. Setup Matcher and Criterion
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_fgl": 0.15,
    }
    criterion = DFINECriterion(
        matcher, num_classes=num_classes, weight_dict=weight_dict
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Load a single image from COCO128
    img_dir = Path("data/coco128/images/train2017")
    lbl_dir = Path("data/coco128/labels/train2017")

    if not img_dir.exists():
        print(
            f"Error: {img_dir} does not exist. Please ensure COCO128 is downloaded."
        )
        return

    # Get the first image
    img_path = next(img_dir.glob("*.jpg"))
    lbl_path = lbl_dir / (img_path.stem + ".txt")

    print(f"Loading image: {img_path.name}")

    # Resize to a fixed size that is a multiple of 16 (e.g., 256x256) for simplicity
    img_h, img_w = 256, 256
    img = Image.open(img_path).convert("RGB").resize((img_w, img_h))

    # Convert to tensor without torchvision
    img_np = np.array(img).transpose(2, 0, 1)  # HWC to CHW
    image = (
        torch.from_numpy(img_np).float().unsqueeze(0).to(device) / 255.0
    )  # (1, 3, H, W)

    # Load targets
    target_dict = load_yolo_label(lbl_path, img_w, img_h)
    target_dict["labels"] = target_dict["labels"].to(device)
    target_dict["boxes"] = target_dict["boxes"].to(device)
    targets = [target_dict]

    print(f"Found {len(target_dict['labels'])} objects in the image.")

    # 4. Prepare Patches and Centers
    patch_seq = extract_patches(image, patch_size=(16, 16), dim_head=64)
    patches = patch_seq.patches.to(device)
    freqs = patch_seq.freqs.to(device)

    # Generate normalized patch centers for the detector
    grid_h, grid_w = img_h // 16, img_w // 16
    y_centers = (torch.arange(grid_h, device=device) + 0.5) / grid_h
    x_centers = (torch.arange(grid_w, device=device) + 0.5) / grid_w
    y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")
    patch_centers = torch.stack(
        [x_grid.flatten(), y_grid.flatten()], dim=-1
    ).unsqueeze(0)  # (1, P, 2)

    # 5. Overfit Loop
    print("Starting sanity check (overfitting a single batch)...")
    model.train()
    for epoch in range(3001):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(patches, freqs, patch_centers)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict.values())

        # Backward pass
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} | "
                f"CE: {loss_dict.get('loss_ce', torch.tensor(0)).item():.4f} | "
                f"BBox: {loss_dict.get('loss_bbox', torch.tensor(0)).item():.4f} | "
                f"GIoU: {loss_dict.get('loss_giou', torch.tensor(0)).item():.4f} | "
                f"FGL: {loss_dict.get('loss_fgl', torch.tensor(0)).item():.4f}"
            )

    print(
        "Sanity check complete. If the total loss dropped significantly (near 0), the architecture is learning!"
    )


if __name__ == "__main__":
    main()
