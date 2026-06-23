import argparse
import time
from typing import Generator
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass
from itertools import chain, batched

from model.vit import vit_nano, vit_small, vit_base
from model.lejepa import Predictor, SIGReg
from threaded_generator import (
    ThreadedGenerator,
    Monitor,
)
import transform.ssl as ssl_tf
from dataset.coco import (
    parse_coco,
    load_coco_ssl_sample,
    CocoMetadata,
    CocoDataset,
)
from logger import ExperimentLogger, BaseMetrics
from music_types import (
    CHW,
    Batch,
    BatchedData,
    Data,
    RGB,
    Float1,
    PatchDim,
    NumPatches,
    SSLSample,
    TensorImage,
    MaskedPair,
)


@dataclass
class LeJEPAMetrics(BaseMetrics):
    epoch: float
    loss_total: float
    loss_sigreg: float
    loss_l2: float
    speed: float


@dataclass
class TrainParams:
    anno_path: Path
    cache_dir: Path | None
    img_dir: Path
    dataset: CocoDataset
    crop_size: int | None
    channels: int
    var_threshold: float
    mask_ratio: float
    patch_size: int
    backbone_size: str
    pred_depth: int
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
            item_decoded = ssl_tf.decode_nvimgcodec(
                item, device=params.prep_device
            )
        except Exception:
            item_decoded = ssl_tf.decode_pyvips(item, device=params.prep_device)
        
        if params.crop_size is not None:
            item_cropped = ssl_tf.random_crop(
                item_decoded, crop_size=params.crop_size
            )
        else:
            item_cropped = item_decoded
    else:
        if params.crop_size is not None:
            item_cropped = ssl_tf.decode_and_crop_pyvips(
                item, crop_size=params.crop_size, device=params.prep_device
            )
        else:
            item_cropped = ssl_tf.decode_pyvips(item, device=params.prep_device)

    item_tf = ssl_tf.to_float1(item_cropped)

    item_padded = ssl_tf.pad_to_patch_size(
        item_tf, patch_size=(params.patch_size, params.patch_size)
    )

    return item_padded


def create_lejepa_iterator(
    params: TrainParams,
    monitor: Monitor | None = None,
) -> Generator[
    BatchedData[
        CocoMetadata, SSLSample[MaskedPair[Batch, NumPatches, PatchDim]]
    ],
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

            dropped_item = ssl_tf.variance_patch_drop(
                patched_item, var_threshold=params.var_threshold
            )

            masked_item = ssl_tf.random_spatial_mask(
                dropped_item, drop_ratio=params.mask_ratio
            )

            final_item = ssl_tf.to_masked_patches(
                masked_item, device=params.train_device
            )

            yield final_item


def train(params: TrainParams):
    logger = ExperimentLogger(params.exp_dir, params.stage_name)

    if params.backbone_size == "nano":
        vit_fn = vit_nano
        embed_dim = 192
        heads = 3
        dim_head = 64
        mlp_dim = 768
    elif params.backbone_size == "small":
        vit_fn = vit_small
        embed_dim = 384
        heads = 6
        dim_head = 64
        mlp_dim = 1536
    else:
        vit_fn = vit_base
        embed_dim = 768
        heads = 12
        dim_head = 64
        mlp_dim = 3072

    backbone = vit_fn(
        patch_size=params.patch_size,
        channels=params.channels,
    ).to(params.train_device)

    predictor = Predictor(
        embed_dim=embed_dim,
        depth=params.pred_depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
    ).to(params.train_device)

    sigreg = SIGReg().to(params.train_device)

    optimizer = optim.AdamW(
        chain(backbone.parameters(), predictor.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    running_loss = None
    samples = 0
    start_time = time.time()
    global_step = 0
    dataset_size = len(params.dataset.images)

    backbone.train()
    predictor.train()
    monitor = Monitor()
    iterator = ThreadedGenerator(
        create_lejepa_iterator(params, monitor=monitor),
        maxsize=2,
    )

    for step, batch in enumerate(iterator):
        global_step += 1
        N = len(batch.metadata)
        current_epoch = samples / dataset_size
        if current_epoch > params.epochs:
            break

        target_patches = batch.sample.image.target
        context_patches = batch.sample.image.context

        # 1. Target Encoder (Full Context)
        target_emb = backbone(target_patches)
        global_target_emb = target_emb.data.mean(dim=1)

        # 2. Context Encoder (Masked Input)
        context_emb = backbone(context_patches)

        # 3. Predictor (Grammar Teacher)
        pred_emb = predictor(context_emb, target_emb)

        # 4. Gather target embeddings for the masked patches
        B = target_emb.batch_size
        max_idx = target_emb.indices.max().item() + 1
        pos_map = torch.zeros(
            (B, max_idx), dtype=torch.long, device=params.train_device
        )
        pos_map.scatter_(
            1,
            target_emb.indices,
            torch.arange(
                target_emb.indices.size(1), device=params.train_device
            )
            .unsqueeze(0)
            .expand(B, -1),
        )

        gather_pos = torch.gather(pos_map, 1, pred_emb.indices)
        target_mask_emb = torch.gather(
            target_emb.data,
            1,
            gather_pos.unsqueeze(-1).expand(
                -1, -1, target_emb.data.size(-1)
            ),
        )

        # 5. Losses
        l2_loss = F.mse_loss(pred_emb.data, target_mask_emb)
        sigreg_loss = sigreg(global_target_emb)

        loss = sigreg_loss * params.lamb + l2_loss * (1 - params.lamb)

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
                epoch=current_epoch,
                loss_total=loss.item(),
                loss_sigreg=sigreg_loss.item(),
                loss_l2=l2_loss.item(),
                speed=speed,
            )
            logger.log_metrics(metrics)

            print(
                f"Epoch [{current_epoch:.2f}/{params.epochs}] Samples [{samples}] "
                f"Loss: {loss.item():.4f} (Running: {running_loss:.4f}) "
                f"(SIGReg: {sigreg_loss.item():.4f}, L2: {l2_loss.item():.4f}) "
                f"Speed: {speed:.1f} sample/s"
            )

            checkpoint = {
                "backbone": backbone.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": running_loss,
            }

            torch.save(
                checkpoint,
                logger.get_checkpoint_dir() / "best_model.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Dense LeJEPA ViT on Trompa-COCO"
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path(
            "data/trompa-coco/annotations/instances_trainval2017.json"
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Square crop size. If not provided, the full image is used.",
    )
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument(
        "--backbone_size",
        type=str,
        choices=["nano", "small", "base"],
        default="nano",
    )
    parser.add_argument("--pred_depth", type=int, default=4)
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
        crop_size=args.crop_size,
        channels=args.channels,
        var_threshold=args.var_threshold,
        mask_ratio=args.mask_ratio,
        patch_size=args.patch_size,
        backbone_size=args.backbone_size,
        pred_depth=args.pred_depth,
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
