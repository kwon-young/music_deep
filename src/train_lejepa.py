import argparse
import time
from typing import Generator
import torch
import torch.optim as optim
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from itertools import chain

from model.vit import ViewViT
from model.lejepa import ProjectorMLP, SIGReg
from threaded_generator import (
    ThreadedGenerator,
    ParallelGenerator,
    Monitor,
    partial_generator,
)
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
    extract_patches,
    random_patch_drop,
    unflatten_views,
)
from dataset.imslp import (
    load_imslp,
    load_image,
    Metadata,
)
from music_types import (
    Data,
    TensorImage,
    VCHW,
    RGB,
    Float1,
    BatchedPatchData,
    FlatViewEmbeddings,
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
    drop_rate: float
    patch_size: int
    dim: int
    depth: int
    heads: int
    dim_head: int
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
) -> Data[Metadata, TensorImage[VCHW, RGB, Float1]]:
    data_pil = load_image(metadata, image_dir=params.image_dir)
    data_np = to_numpy(data_pil)
    data_t = to_tensor(data_np)
    data_t = to(data_t, device=params.device)
    data_t = random_crop(data_t, crop_size=params.image_size)
    data_tf = to_float1(data_t)
    data_tfv = make_views(data_tf, n=params.n_views)
    data_tfv = random_affine(
        data_tfv, params.max_angle_deg, params.max_translate
    )
    return data_tfv


@partial_generator
def create_lejepa_iterator(
    params: TrainParams,
    monitor: Monitor,
) -> Generator[BatchedPatchData[Metadata, FlatViewEmbeddings]]:

    gen = partial_generator(shuffle)(
        partial_generator(load_imslp)(params.manifest_path)
    )
    data = partial_generator(map)(
        partial(
            transform_image,
            params=params,
        ),
        gen,
    )
    data_gen = ParallelGenerator(
        data,
        num_workers=2,
        maxsize=params.batch_size,
        # monitor=monitor,
        name="transform",
    )
    batched_data = collate(data_gen, batch_size=params.batch_size)

    for batch in batched_data:
        image = batch.image.data
        N, V, C, H, W = image.shape
        x_aug = image.view(N * V, C, H, W)

        patches = extract_patches(
            x_aug,
            patch_size=(params.patch_size, params.patch_size),
        )
        patches = random_patch_drop(patches, drop_rate=params.drop_rate)

        flat_view_patches = FlatViewEmbeddings(
            data=patches.data,
            indices=patches.indices,
            image_shape=patches.image_shape,
            patch_size=patches.patch_size,
            num_views=params.n_views,
        )

        yield BatchedPatchData(
            metadata=batch.metadata, patches=flat_view_patches
        )


def train(params: TrainParams):
    backbone = ViewViT(
        patch_size=params.patch_size,
        dim=params.dim,
        depth=params.depth,
        heads=params.heads,
        dim_head=params.dim_head,
        mlp_dim=params.mlp_dim,
        channels=params.channels,
    ).to(params.device)

    projector = ProjectorMLP(
        in_features=params.embed_dim,
        hidden_features=2048,
        out_features=params.proj_dim,
    ).to(params.device)

    sigreg = SIGReg().to(params.device)

    optimizer = optim.AdamW(
        chain(backbone.parameters(), projector.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    params.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    running_loss = None
    samples = 0
    checkpoint_number = 0
    start_time = time.time()

    for epoch in range(params.epochs):
        backbone.train()
        projector.train()
        monitor = Monitor()
        iterator = ThreadedGenerator(
            create_lejepa_iterator(params, monitor=monitor),
            maxsize=2,
            # monitor=monitor,
        )

        for step, batch in enumerate(iterator):
            N = len(batch.metadata)

            emb = backbone(batch.patches)
            proj_emb = projector(emb)

            proj_view = unflatten_views(proj_emb)
            proj = proj_view.data.flatten(start_dim=2).transpose(0, 1)

            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)

            loss = sigreg_loss * params.lamb + inv_loss * (1 - params.lamb)

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = 0.99 * running_loss + 0.01 * loss.item()

            samples += N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % params.log_interval == 0:
                elapsed = time.time() - start_time
                speed = samples / elapsed if elapsed > 0 else 0.0
                print(
                    f"Epoch [{epoch}/{params.epochs}] Samples [{samples}] "
                    f"Loss: {loss.item():.4f} (Running: {running_loss:.4f}) "
                    f"(SIGReg: {sigreg_loss.item():.4f}, Inv: {inv_loss.item():.4f}) "
                    f"Speed: {speed:.1f} sample/s"
                )

            if samples > checkpoint_number * params.checkpoint_window_size:
                print(
                    f"Sample Window Reached [{checkpoint_number}]. "
                    f"Running Average Loss: {running_loss:.4f}"
                )
                checkpoint_number += 1

                if running_loss < best_loss:
                    best_loss = running_loss

                    checkpoint = {
                        "backbone": backbone.state_dict(),
                        "projector": projector.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_loss": best_loss,
                    }

                    torch.save(
                        checkpoint,
                        params.checkpoint_dir / "best_model.pt",
                    )
                    print(f"Saved new best model with loss {best_loss:.4f}")


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
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim_head", type=int, default=64)
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
        drop_rate=args.drop_rate,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
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
