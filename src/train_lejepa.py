import itertools
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from PIL import Image as Image

from model.vit import vit_small
from model.lejepa import LeJEPAEncoder, SIGReg
from transform import random_affine, shuffle
from dataset.imslp import load_imslp, load_image, BatchedData


def create_lejepa_iterator(
    manifest_path: Path, image_dir: Path, batch_size: int, v_views: int = 4
):

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
            pil_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)

            # Convert to tensor manually without torchvision
            tensor_img = (
                torch.from_numpy(np.array(pil_img)).float().unsqueeze(0) / 255.0
            )
            # Normalize: mean=0.5, std=0.5
            tensor_img = (tensor_img - 0.5) / 0.5

            # Create identical copies to be uniquely augmented later on the GPU
            views = [tensor_img for _ in range(v_views)]
            batch_views.append(torch.stack(views))

        # Yields shape: (batch_size, v_views, C, H, W)
        yield torch.stack(batch_views)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 128
    v_views = 4
    lamb = 0.05
    epochs = 100
    lr = 5e-4

    manifest_path = Path("data/imslp/imslp.jsonl")
    image_dir = Path("data/imslp/images")

    # Model Setup
    # Note: image channels=1 since `create_lejepa_iterator` uses "L" (Grayscale)
    backbone = vit_small(
        image_size=224, num_classes=0, channels=1, num_keep_patches=128
    )
    encoder = LeJEPAEncoder(backbone, embed_dim=384, proj_dim=16).to(device)
    sigreg = SIGReg().to(device)

    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.05)

    for epoch in range(epochs):
        encoder.train()
        iterator = create_lejepa_iterator(
            manifest_path, image_dir, batch_size, v_views
        )

        for step, batch in enumerate(iterator):
            batch = batch.to(device)
            N, V, C, H, W = batch.shape

            # Flatten batch and views for the model, then apply GPU augmentations
            x_flat = batch.view(N * V, C, H, W)
            x_aug = random_affine(BatchedData([], x_flat)).image

            # Forward pass
            emb, proj = encoder(x_aug, random_drop=True)

            # Reshape projector output to (V, N, D) for the invariance and SIGReg loss
            proj = proj.view(N, V, -1).transpose(0, 1)

            # Compute losses
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)

            loss = sigreg_loss * lamb + inv_loss * (1 - lamb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step}] "
                    f"Loss: {loss.item():.4f} "
                    f"(SIGReg: {sigreg_loss.item():.4f}, Inv: {inv_loss.item():.4f})"
                )


if __name__ == "__main__":
    train()
