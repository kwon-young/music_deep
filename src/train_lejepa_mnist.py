import argparse
import time
from typing import Generator
import torch
import torch.nn as nn
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
from transform.core import shuffle
import transform.ssl as ssl_tf
from dataset.mnist import load_mnist, load_image, MNISTMetadata
from dataset.imslp import Data
from dataset.imslp import TensorImage, VCHW, RGB, Float1
from music_types import FlatViewEmbeddings, BatchedData, SSLSample


@dataclass
class TrainParams:
    mnist_dir: Path
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
    device: torch.device


def transform_image(
    metadata: MNISTMetadata,
    params: TrainParams,
) -> Data[MNISTMetadata, SSLSample[TensorImage[VCHW, RGB, Float1]]]:
    data_pil = load_image(metadata)
    data_np = ssl_tf.to_numpy(data_pil)
    data_t = ssl_tf.to_tensor(data_np)
    data_t = ssl_tf.to(data_t, device=params.device)
    data_t = ssl_tf.to_float1(data_t)
    data_tfv = ssl_tf.make_views(data_t, n=params.n_views)
    data_tfv = ssl_tf.random_flatview_affine(
        data_tfv, params.max_angle_deg, params.max_translate
    )
    return data_tfv


@partial_generator
def create_lejepa_mnist_iterator(
    params: TrainParams,
    monitor: Monitor,
) -> Generator[
    BatchedData[MNISTMetadata, SSLSample[FlatViewEmbeddings]],
    None,
    None,
]:

    gen = partial_generator(shuffle)(
        partial_generator(load_mnist)(params.mnist_dir, split="train")
    )
    data = partial_generator(map)(
        partial(transform_image, params=params),
        gen,
    )
    data_gen = ParallelGenerator(
        data,
        num_workers=2,
        maxsize=params.batch_size,
        name="transform",
    )
    batched_data = ssl_tf.collate(data_gen, batch_size=params.batch_size)

    for batch in batched_data:
        patch_seq = ssl_tf.extract_flatviewpatches(
            batch,
            patch_size=(params.patch_size, params.patch_size),
        )
        patch_seq = ssl_tf.random_flatview_patch_drop(
            patch_seq, drop_rate=params.drop_rate
        )

        yield patch_seq


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

    probe = nn.Linear(params.dim, params.num_classes).to(params.device)
    criterion_probe = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        chain(backbone.parameters(), projector.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    optimizer_probe = optim.AdamW(
        probe.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    running_loss_ssl: float | None = None
    running_loss_probe: float | None = None
    running_acc: float | None = None
    samples = 0
    start_time = time.time()

    for epoch in range(params.epochs):
        backbone.train()
        projector.train()
        probe.train()
        monitor = Monitor()
        iterator = ThreadedGenerator(
            create_lejepa_mnist_iterator(params, monitor=monitor),
            maxsize=2,
        )

        for step, batch in enumerate(iterator):
            N = len(batch.metadata)
            V = params.n_views

            labels = torch.tensor(
                [m.label for m in batch.metadata], device=params.device
            )
            labels_v = labels.repeat_interleave(V)

            emb = backbone(batch.sample.image)
            proj_emb = projector(emb)

            global_emb = emb.data.mean(dim=1)

            logits = probe(global_emb.detach())
            probe_loss = criterion_probe(logits, labels_v)

            preds = logits.argmax(dim=-1)
            acc = (preds == labels_v).float().mean().item()

            proj_view = ssl_tf.unflatten_views(proj_emb)
            proj_v = proj_view.data.flatten(start_dim=2).transpose(0, 1)

            inv_loss = (proj_v.mean(0) - proj_v).square().mean()
            sigreg_loss = sigreg(proj_v)

            ssl_loss = sigreg_loss * params.lamb + inv_loss * (1 - params.lamb)

            if (
                running_loss_ssl is None
                or running_loss_probe is None
                or running_acc is None
            ):
                running_loss_ssl = ssl_loss.item()
                running_loss_probe = probe_loss.item()
                running_acc = acc
            else:
                running_loss_ssl = (
                    0.99 * running_loss_ssl + 0.01 * ssl_loss.item()
                )
                running_loss_probe = (
                    0.99 * running_loss_probe + 0.01 * probe_loss.item()
                )
                running_acc = 0.99 * running_acc + 0.01 * acc

            samples += N

            optimizer.zero_grad()
            ssl_loss.backward()
            optimizer.step()

            optimizer_probe.zero_grad()
            probe_loss.backward()
            optimizer_probe.step()

            if step % params.log_interval == 0:
                elapsed = time.time() - start_time
                speed = samples / elapsed if elapsed > 0 else 0.0
                print(
                    f"Epoch [{epoch}/{params.epochs}] Samples [{samples}] "
                    f"SSL Loss: {ssl_loss.item():.4f} (Run: {running_loss_ssl:.4f}) | "
                    f"Probe Loss: {probe_loss.item():.4f} (Run: {running_loss_probe:.4f}) | "
                    f"Probe Acc: {acc * 100:.1f}% (Run: {running_acc * 100:.1f}%) | "
                    f"Speed: {speed:.1f} sample/s"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LeJEPA ViT on MNIST with online probe"
    )
    parser.add_argument(
        "--mnist_dir", type=Path, default=Path("data/mnist-png")
    )
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument("--max_angle_deg", type=float, default=15.0)
    parser.add_argument("--max_translate", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim_head", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lamb", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = TrainParams(
        mnist_dir=args.mnist_dir,
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
        device=device,
    )

    train(params)
