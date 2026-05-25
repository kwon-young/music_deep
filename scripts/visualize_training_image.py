import argparse
import sys
from pathlib import Path
from unittest.mock import patch
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path to reuse existing codebase
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset.imslp import load_imslp, load_image
from transform import to_numpy, to_tensor, to_float1, random_crop
from model.vit import vit_nano

def visualize(
    manifest_path: Path, 
    image_dir: Path, 
    image_size: int = 224, 
    patch_size: int = 16, 
    num_keep_patches: int = 128
):
    metadata_gen = load_imslp(manifest_path)
    metadata = next(metadata_gen)
    
    # 1. Reuse the exact preprocessing pipeline up to random_crop
    data = load_image(metadata, image_dir=image_dir)
    data = to_numpy(data)
    data = to_tensor(data)
    data = to_float1(data)
    
    full_img = data.image.clone()
    
    # Intercept torch.randint inside transform.py to get the crop coordinates
    coords = []
    original_randint = torch.randint
    
    def mock_randint(*args, **kwargs):
        val = original_randint(*args, **kwargs)
        coords.append(val.item())
        return val
    
    with patch('transform.torch.randint', side_effect=mock_randint):
        data = random_crop(data, crop_size=image_size)
        
    x, y = coords[0], coords[1]
    
    # 2. Reuse the actual patch dropping code by running the model
    model = vit_nano(
        image_size=image_size, 
        patch_size=patch_size, 
        num_classes=0, 
        num_keep_patches=num_keep_patches
    )
    
    # Intercept torch.rand inside model/vit.py to get the patch drop indices
    captured_rands = []
    original_rand = torch.rand
    
    def mock_rand(*args, **kwargs):
        val = original_rand(*args, **kwargs)
        if len(args) >= 2 and args[1] == (image_size // patch_size) ** 2:
            captured_rands.append(val)
        return val

    img_batch = data.image.unsqueeze(0)
    with patch('model.vit.torch.rand', side_effect=mock_rand):
        # Execute the actual forward pass containing the patch drop logic
        model(img_batch, random_drop=True)
    
    # Reconstruct the keep mask from the exact random values used by the model
    num_patches_x = image_size // patch_size
    num_patches_y = image_size // patch_size
    num_patches = num_patches_x * num_patches_y
    
    rand_val = captured_rands[0]
    keep_indices = rand_val.argsort(dim=-1)[:, :num_keep_patches][0]
    
    keep_mask = torch.zeros(num_patches, dtype=torch.bool)
    keep_mask[keep_indices] = True
    keep_mask = keep_mask.view(num_patches_y, num_patches_x)
    
    # 3. Create the visualization image
    dim_factor = 0.3
    vis_img = full_img.clone() * dim_factor  # Dim the entire image initially
    
    # Undim only the kept patches inside the cropped region
    for py in range(num_patches_y):
        for px in range(num_patches_x):
            if keep_mask[py, px]:
                start_y = y + py * patch_size
                end_y = start_y + patch_size
                start_x = x + px * patch_size
                end_x = start_x + patch_size
                
                vis_img[:, start_y:end_y, start_x:end_x] = full_img[:, start_y:end_y, start_x:end_x]
                
    vis_img_np = vis_img.permute(1, 2, 0).numpy()
    
    # 4. Plot using matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(vis_img_np)
    
    # Add the red bounding box around the cropped region
    rect = patches.Rectangle(
        (x, y), image_size, image_size, 
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    plt.title(f"Crop Visualization\nKeep {num_keep_patches}/{num_patches} patches")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=Path, default=Path("data/imslp/imslp.jsonl"))
    parser.add_argument("--image_dir", type=Path, default=Path("data/imslp/images"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_keep_patches", type=int, default=128)
    args = parser.parse_args()

    visualize(
        manifest_path=args.manifest_path,
        image_dir=args.image_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_keep_patches=args.num_keep_patches
    )
