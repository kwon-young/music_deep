from typing import Literal, Iterator, Generator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage
from dataset.imslp import Metadata
import random
import itertools
import math
import torch.nn.functional as F


HWC = Literal["HWC"]
CHW = Literal["CHW"]
Layout = HWC | CHW
Binary = Literal["1"]
Gray = Literal["L"]
RGB = Literal["RGB"]
Mode = Binary | Gray | RGB


@dataclass
class Image[T, Layout, Mode]:
    metadata: Metadata
    image: T


NpImage = Image[np.ndarray, Layout, Mode]
TensorImage = Image[torch.Tensor, Layout, Mode]


def load_image[T: Mode](
    metadata: Metadata,
    image_dir: Path,
    mode: Mode,
) -> Image[PILImage.Image, HWC, T]:
    return Image(
        metadata, PILImage.open(image_dir / metadata.name).convert(mode)
    )


def to_numpy(
    image: Image[PILImage.Image, Layout, Mode],
) -> Image[np.ndarray, Layout, Mode]:
    return Image(image.metadata, np.array(image.image))


def to_tensor(image: NpImage) -> TensorImage:
    return Image(image.metadata, torch.as_tensor(image.image))


def shuffle[T](it: Iterator[T]) -> Generator[T]:
    l = list(it)
    random.shuffle(l)
    yield from l


def to(image: TensorImage, device: torch.device) -> TensorImage:
    return Image(image.metadata, image.image.to(device))


def gpu_random_affine(x: torch.Tensor, max_angle_deg: float = 3.0, max_translate: float = 0.05) -> torch.Tensor:
    N = x.size(0)
    device = x.device
    
    max_angle_rad = max_angle_deg * math.pi / 180.0
    angles = (torch.rand(N, device=device) * 2 - 1) * max_angle_rad
    
    tx = (torch.rand(N, device=device) * 2 - 1) * max_translate
    ty = (torch.rand(N, device=device) * 2 - 1) * max_translate
    
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    matrix = torch.zeros(N, 2, 3, device=device)
    matrix[:, 0, 0] = cos_a
    matrix[:, 0, 1] = -sin_a
    matrix[:, 0, 2] = tx
    matrix[:, 1, 0] = sin_a
    matrix[:, 1, 1] = cos_a
    matrix[:, 1, 2] = ty
    
    grid = F.affine_grid(matrix, x.size(), align_corners=False)
    
    # Assuming normalized image where white is 1.0. 
    # Shift so white is 0.0, apply grid_sample (pads with 0.0), then shift back to 1.0
    x_shifted = x - 1.0
    x_transformed = F.grid_sample(x_shifted, grid, padding_mode='zeros', align_corners=False)
    return x_transformed + 1.0


def create_lejepa_iterator(manifest_path: Path, image_dir: Path, batch_size: int, v_views: int = 4):
    from dataset.imslp import load_imslp
    
    meta_gen = shuffle(load_imslp(manifest_path, image_dir))
    
    while True:
        batch_meta = list(itertools.islice(meta_gen, batch_size))
        if not batch_meta:
            break
            
        batch_views = []
        for meta in batch_meta:
            # Using L for grayscale
            pil_img = load_image(meta, image_dir, mode="L").image 
            
            # Simple resize for uniform batching (ViT default is usually 224x224)
            pil_img = pil_img.resize((224, 224), PILImage.Resampling.BILINEAR)
            
            # Convert to tensor manually without torchvision
            tensor_img = torch.from_numpy(np.array(pil_img)).float().unsqueeze(0) / 255.0
            # Normalize: mean=0.5, std=0.5
            tensor_img = (tensor_img - 0.5) / 0.5
            
            # Create identical copies to be uniquely augmented later on the GPU
            views = [tensor_img for _ in range(v_views)]
            batch_views.append(torch.stack(views))
            
        # Yields shape: (batch_size, v_views, C, H, W)
        yield torch.stack(batch_views)
