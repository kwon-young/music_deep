from typing import Literal, Iterator, Generator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage
from dataset.imslp import Metadata
import random
import itertools


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


def random_affine(img: PILImage.Image) -> PILImage.Image:
    angle = random.uniform(-3.0, 3.0)
    tx = random.uniform(-5.0, 5.0)
    ty = random.uniform(-5.0, 5.0)
    
    # Simple affine transform using Pillow's rotate + translate
    # We use a white background assuming sheet music
    return img.rotate(
        angle, 
        translate=(tx, ty), 
        resample=PILImage.Resampling.BILINEAR, 
        fillcolor=255
    )


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
            
            views = []
            for _ in range(v_views):
                aug_img = random_affine(pil_img)
                # Convert to tensor manually without torchvision
                tensor_img = torch.from_numpy(np.array(aug_img)).float().unsqueeze(0) / 255.0
                # Normalize: mean=0.5, std=0.5
                tensor_img = (tensor_img - 0.5) / 0.5
                views.append(tensor_img)
                
            batch_views.append(torch.stack(views))
            
        # Yields shape: (batch_size, v_views, C, H, W)
        yield torch.stack(batch_views)
