import argparse
from typing import Generator
import torch
import torch.optim as optim
from pathlib import Path
from functools import partial
from dataclasses import dataclass

from model.vit import ViT
from model.lejepa import LeJEPAEncoder, SIGReg
from threaded_generator import ThreadedGenerator, Monitor, partial_generator
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
    n_views: int
    max_angle_deg: float
    max_translate: float
    image_size: int
    num_classes: int
    channels: int
    num_keep_patches: int
    patch_size: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    embed_dim: int
    proj_dim: int
    batch_size: int
    lamb: float
    epochs: int
    lr: float
    weight_decay: float
    log_interval: int
    checkpoint_window_size: int
    device: torch.device
    checkpoint_dir: Path


def transform_image(
    metadata: Metadata,
    params: TrainParams,
) -> Data:
    data_pil = load_image(metadata, image_dir=params.image_dir)
    data_np = to_numpy(data_pil)
    data_t = to_tensor(data_np)
    data_t = random_crop(data_t, crop_size=params.image_size)
    data_t = to_float1(data_t)
    data_t = make_views(data_t, n=params.n_views)
    data_t = random_affine(data_t, params.max_angle_deg, params.max_translate)
    return data_t


@partial_generator
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


def train(params: TrainParams):
    # Model Setup
    # Note: image channels=3 since `create_lejepa_iterator` uses "RGB"
    backbone = ViT(
        image_size=params.image_size,
        patch_size=params.patch_size,
        num_classes=params.num_classes,
        dim=params.dim,
        depth=params.depth,
        heads=params.heads,
        mlp_dim=params.mlp_dim,
        channels=params.channels,
        num_keep_patches=params.num_keep_patches,
    )
    encoder = LeJEPAEncoder(
        backbone, embed_dim=params.embed_dim, proj_dim=params.proj_dim
    ).to(params.device)
    sigreg = SIGReg().to(params.device)

    optimizer = optim.AdamW(
        encoder.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    params.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    running_loss = None
    window_samples = 0

    for epoch in range(params.epochs):
        encoder.train()
        monitor = Monitor()
        iterator = ThreadedGenerator(
            create_lejepa_iterator(params),
            maxsize=2,
            monitor=monitor,
        )

        with monitor:
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

                if running_loss is None:
                    running_loss = loss.item()
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss.item()

                window_samples += N

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % params.log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{params.epochs}] Step [{step}] "
                        f"Loss: {loss.item():.4f} (Running: {running_loss:.4f}) "
                        f"(SIGReg: {sigreg_loss.item():.4f}, Inv: {inv_loss.item():.4f})"
                    )

                if window_samples >= params.checkpoint_window_size:
                    print(
                        f"Sample Window Reached [{params.checkpoint_window_size}]. "
                        f"Running Average Loss: {running_loss:.4f}"
                    )

                    if running_loss < best_loss:
                        best_loss = running_loss
                        torch.save(
                            encoder.state_dict(),
                            params.checkpoint_dir / "best_model.pt",
                        )
                        print(f"Saved new best model with loss {best_loss:.4f}")

                    window_samples = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeJEPA ViT")
    parser.add_argument(
        "--manifest_path", type=Path, default=Path("data/imslp/imslp.jsonl")
    )
    parser.add_argument(
        "--image_dir", type=Path, default=Path("data/imslp/images")
    )
    parser.add_argument("--n_views", type=int, default=4)
    parser.add_argument("--max_angle_deg", type=float, default=3.0)
    parser.add_argument("--max_translate", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_keep_patches", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp_dim", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lamb", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_window_size", type=int, default=10000)
    parser.add_argument(
        "--checkpoint_dir", type=Path, default=Path("data/train_lejepa/")
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = TrainParams(
        manifest_path=args.manifest_path,
        image_dir=args.image_dir,
        n_views=args.n_views,
        max_angle_deg=args.max_angle_deg,
        max_translate=args.max_translate,
        image_size=args.image_size,
        num_classes=args.num_classes,
        channels=args.channels,
        num_keep_patches=args.num_keep_patches,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        batch_size=args.batch_size,
        lamb=args.lamb,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        checkpoint_window_size=args.checkpoint_window_size,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    train(params)
