import argparse
import time
from typing import Generator
import torch
import torch.optim as optim
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from itertools import chain, batched

from model.vit import ViewViT
from model.lejepa import ProjectorMLP, SIGReg
from threaded_generator import (
    ThreadedGenerator,
    ParallelGenerator,
    Monitor,
    partial_generator,
)
from transform.core import shuffle
import transform.ssl as ssl_tf
from dataset.coco import parse_coco, load_coco_ssl_sample, CocoMetadata, CocoDataset
from logger import ExperimentLogger, BaseMetrics
from music_types import (
    CHW,
    Batch,
    BatchedData,
    View,
    BatchView,
    BVCHW,
    Data,
    FlatViewTensorImage,
    RGB,
    Float1,
    PatchDim,
    NumPatches,
    FlatViewEmbeddings,
    FlatViewPatches,
    SSLSample,
    TensorImage,
    MaskedPair,
)


@dataclass
class LeJEPAMetrics(BaseMetrics):
    epoch: int
    loss_total: float
    loss_sigreg: float
    loss_inv: float
    speed: float


@dataclass
class TrainParams:
    anno_path: Path
    cache_dir: Path | None
    img_dir: Path
    dataset: CocoDataset
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
    prep_device: torch.device
    train_device: torch.device
    exp_dir: Path
    stage_name: str


def transform_image(
    index: int,
    dataset: CocoDataset,
    img_dir: Path,
    params: TrainParams,
) -> Data[CocoMetadata, SSLSample[TensorImage[CHW, RGB, Float1]]]:
    item = load_coco_ssl_sample(dataset, img_dir, index)

    if params.prep_device.type == "cuda":
        try:
            item_decoded = ssl_tf.decode_nvimgcodec(item, device=params.prep_device)
        except Exception:
            item_decoded = ssl_tf.decode_pyvips(item, device=params.prep_device)
        item_cropped = ssl_tf.random_crop(item_decoded, crop_size=params.image_size)
    else:
        item_cropped = ssl_tf.decode_and_crop_pyvips(
            item, crop_size=params.image_size, device=params.prep_device
        )

    item_tf = ssl_tf.to_float1(item_cropped)

    item_padded = ssl_tf.pad_to_patch_size(
        item_tf, patch_size=(params.patch_size, params.patch_size)
    )

    return item_padded


def create_lejepa_iterator(
    params: TrainParams,
    monitor: Monitor | None = None,
) -> Generator[
    BatchedData[CocoMetadata, SSLSample[MaskedPair[Batch, NumPatches, PatchDim]]],
    None,
    None,
]:
    import random
    num_images = len(params.dataset.images)
    indices = list(range(num_images))

    while True:
        random.shuffle(indices)

        transformed_gen = (
            transform_image(idx, params.dataset, params.img_dir, params)
            for idx in indices
        )

        for batch_items in batched(transformed_gen, params.batch_size):
            batched_item = ssl_tf.collate_images(batch_items)
            
            patched_item = ssl_tf.extract_patches(
                batched_item, patch_size=(params.patch_size, params.patch_size)
            )
            
            masked_item = ssl_tf.random_spatial_mask(
                patched_item, drop_ratio=params.drop_rate
            )
            
            final_item = ssl_tf.to_masked_patches(
                masked_item, device=params.train_device
            )
            
            yield final_item


def train(params: TrainParams):
    logger = ExperimentLogger(params.exp_dir, params.stage_name)

    backbone = ViewViT(
        patch_size=params.patch_size,
        dim=params.dim,
        depth=params.depth,
        heads=params.heads,
        dim_head=params.dim_head,
        mlp_dim=params.mlp_dim,
        channels=params.channels,
    ).to(params.train_device)

    projector = ProjectorMLP(
        in_features=params.embed_dim,
        hidden_features=2048,
        out_features=params.proj_dim,
    ).to(params.train_device)

    sigreg = SIGReg().to(params.train_device)

    optimizer = optim.AdamW(
        chain(backbone.parameters(), projector.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    best_loss = float("inf")

    running_loss = None
    samples = 0
    checkpoint_number = 0
    start_time = time.time()
    global_step = 0

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
            global_step += 1
            N = len(batch.metadata)

            emb = backbone(batch.sample.image)
            proj_emb = projector(emb)

            proj_view = ssl_tf.unflatten_views(proj_emb)
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

                metrics = LeJEPAMetrics(
                    step=global_step,
                    epoch=epoch,
                    loss_total=loss.item(),
                    loss_sigreg=sigreg_loss.item(),
                    loss_inv=inv_loss.item(),
                    speed=speed,
                )
                logger.log_metrics(metrics)

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
                        logger.get_checkpoint_dir() / "best_model.pt",
                    )
                    print(f"Saved new best model with loss {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dense LeJEPA ViT on Trompa-COCO")
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--mlp_dim", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lamb", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_window_size", type=int, default=10000)
    parser.add_argument(
        "--prep_device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--train_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--exp_dir", type=Path, default=Path("experiments/default_exp")
    )
    parser.add_argument("--stage_name", type=str, default="pretrain_lejepa")

    args = parser.parse_args()

    prep_device = torch.device(args.prep_device)
    train_device = torch.device(args.train_device)

    dataset = parse_coco(args.anno_path, cache_dir=args.cache_dir)

    params = TrainParams(
        anno_path=args.anno_path,
        cache_dir=args.cache_dir,
        img_dir=args.img_dir,
        dataset=dataset,
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
        prep_device=prep_device,
        train_device=train_device,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
    )

    train(params)
