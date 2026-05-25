from typing import Generator
import torch
import torch.optim as optim
from pathlib import Path
from functools import partial
from dataclasses import dataclass

from model.vit import vit_small
from model.lejepa import LeJEPAEncoder, SIGReg
from transform import (
    random_affine,
    shuffle,
    random_crop,
    to_numpy,
    to_tensor,
    to,
    make_views,
    to_float1,
    collate,
)
from dataset.imslp import (
    load_imslp,
    load_image,
    BatchedData,
    Metadata,
    Data,
)


@dataclass
class TrainParams:
    manifest_path: Path
    image_dir: Path
    crop_size: int
    n_views: int
    max_angle_deg: float
    max_translate: float
    image_size: int
    num_classes: int
    channels: int
    num_keep_patches: int
    embed_dim: int
    proj_dim: int
    batch_size: int
    lamb: float
    epochs: int
    lr: float
    weight_decay: float
    log_interval: int
    device: torch.device


def transform_image(
    metadata: Metadata,
    params: TrainParams,
) -> Data:
    data_pil = load_image(metadata, image_dir=params.image_dir)
    data_np = to_numpy(data_pil)
    data_t = to_tensor(data_np)
    data_t = random_crop(data_t, crop_size=params.crop_size)
    data_t = to_float1(data_t)
    data_t = make_views(data_t, n=params.n_views)
    data_t = random_affine(data_t, params.max_angle_deg, params.max_translate)
    return data_t


def create_lejepa_iterator(
    params: TrainParams,
) -> Generator[BatchedData]:

    gen = shuffle(load_imslp(params.manifest_path))
    data = map(
        partial(
            transform_image,
            params=params,
        ),
        gen,
    )
    batched_data = collate(data, batch_size=params.batch_size)
    batched_data = map(partial(to, device=params.device), batched_data)
    yield from batched_data


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = TrainParams(
        manifest_path=Path("data/imslp/imslp.jsonl"),
        image_dir=Path("data/imslp/images"),
        crop_size=224,
        n_views=4,
        max_angle_deg=3.0,
        max_translate=0.05,
        image_size=224,
        num_classes=0,
        channels=1,
        num_keep_patches=128,
        embed_dim=384,
        proj_dim=16,
        batch_size=128,
        lamb=0.05,
        epochs=100,
        lr=5e-4,
        weight_decay=0.05,
        log_interval=10,
        device=device,
    )

    # Model Setup
    # Note: image channels=1 since `create_lejepa_iterator` uses "L" (Grayscale)
    backbone = vit_small(
        image_size=params.image_size,
        num_classes=params.num_classes,
        channels=params.channels,
        num_keep_patches=params.num_keep_patches
    )
    encoder = LeJEPAEncoder(backbone, embed_dim=params.embed_dim, proj_dim=params.proj_dim).to(params.device)
    sigreg = SIGReg().to(params.device)

    optimizer = optim.AdamW(encoder.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    for epoch in range(params.epochs):
        encoder.train()
        iterator = create_lejepa_iterator(params)

        for step, batch in enumerate(iterator):
            image = batch.image
            N, V, C, H, W = image.shape

            # Flatten batch and views for the model
            x_aug = image.view(N * V, C, H, W)

            # Forward pass
            emb, proj = encoder(x_aug, random_drop=True)

            # Reshape projector output to (V, N, D) for the invariance and SIGReg loss
            proj = proj.view(N, V, -1).transpose(0, 1)

            # Compute losses
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)

            loss = sigreg_loss * params.lamb + inv_loss * (1 - params.lamb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % params.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{params.epochs}] Step [{step}] "
                    f"Loss: {loss.item():.4f} "
                    f"(SIGReg: {sigreg_loss.item():.4f}, Inv: {inv_loss.item():.4f})"
                )


if __name__ == "__main__":
    train()
