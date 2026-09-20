import argparse
import math
import os
import time
from typing import Generator, Iterable, NotRequired, TypedDict, cast
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from dataclasses import dataclass
from itertools import chain, batched

from model.vit import ViT, vit_nano, vit_small, vit_base
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
from dataset.imslp import load_imslp, load_image, Metadata as ImslpMetadata
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


type SSLBatch = (
    BatchedData[CocoMetadata, SSLSample[MaskedPair[Batch, NumPatches, PatchDim]]]
    | BatchedData[
        ImslpMetadata, SSLSample[MaskedPair[Batch, NumPatches, PatchDim]]
    ]
)


@dataclass
class LeJEPAMetrics(BaseMetrics):
    epoch: float
    lr: float
    loss_total: float
    loss_sigreg: float
    loss_l2: float
    speed: float


@dataclass
class LeJEPACheckpoint(TypedDict):
    backbone: dict[str, torch.Tensor]
    predictor: dict[str, torch.Tensor]
    optimizer: dict[str, object]
    loss: float
    samples: NotRequired[int]
    step: NotRequired[int]
    epoch: NotRequired[float]


@dataclass
class TrainParams:
    anno_path: Path
    cache_dir: Path | None
    img_dir: Path
    imslp_manifest: Path | None
    dataset: CocoDataset | None
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
    warmup_epochs: float
    min_lr_ratio: float
    weight_decay: float
    log_interval: int
    checkpoint_window_size: int
    prep_device: torch.device
    train_device: torch.device
    use_sdpa: bool
    compile: bool
    detector_checkpoint: Path | None
    backbone_checkpoint: Path | None
    resume: bool
    exp_dir: Path
    stage_name: str


def strip_backbone_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str = "backbone."
) -> dict[str, torch.Tensor]:
    """Strips the ``backbone.`` prefix from a full detector state dict."""
    return {
        k[len(prefix) :]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }


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


def load_imslp_metas(manifest: Path) -> list[ImslpMetadata]:
    return list(load_imslp(manifest))


def transform_image_imslp(
    index: int,
    metadata: list[ImslpMetadata],
    img_dir: Path,
    params: TrainParams,
) -> Data[ImslpMetadata, SSLSample[TensorImage[CHW, RGB, Float1]]]:
    """Decodes an IMSLP page via PIL (CPU) and runs the shared SSL front-end."""
    item = load_image(metadata[index], img_dir)

    item_np = ssl_tf.to_numpy(item)
    item_tensor = ssl_tf.to_tensor(item_np)
    item_device = ssl_tf.to(item_tensor, params.prep_device)

    if params.crop_size is not None:
        item_device = ssl_tf.random_crop(
            item_device, crop_size=params.crop_size
        )

    item_tf = ssl_tf.to_float1(item_device)

    return ssl_tf.pad_to_patch_size(
        item_tf, patch_size=(params.patch_size, params.patch_size)
    )


def collate_mask_and_move[Meta](
    batch_items: tuple[
        Data[Meta, SSLSample[TensorImage[CHW, RGB, Float1]]], ...
    ],
    params: TrainParams,
) -> BatchedData[
    Meta, SSLSample[MaskedPair[Batch, NumPatches, PatchDim]]
]:
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

    return ssl_tf.to_masked_patches(masked_item, device=params.train_device)


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

    dataset = params.dataset
    assert dataset is not None

    num_images = len(dataset.images)
    indices = list(range(num_images))

    while True:
        random.shuffle(indices)

        transformed_gen = (
            transform_image(idx, dataset, params.img_dir, params)
            for idx in indices
        )

        for batch_items in batched(transformed_gen, params.batch_size):
            yield collate_mask_and_move(batch_items, params)


def create_imslp_lejepa_iterator(
    params: TrainParams,
    monitor: Monitor | None = None,
) -> Generator[
    BatchedData[
        ImslpMetadata, SSLSample[MaskedPair[Batch, NumPatches, PatchDim]]
    ],
    None,
    None,
]:
    assert params.imslp_manifest is not None

    import random

    metas = load_imslp_metas(params.imslp_manifest)
    indices = list(range(len(metas)))

    while True:
        random.shuffle(indices)

        transformed_gen = (
            transform_image_imslp(idx, metas, params.img_dir, params)
            for idx in indices
        )

        for batch_items in batched(transformed_gen, params.batch_size):
            yield collate_mask_and_move(batch_items, params)


def train(params: TrainParams):
    # --- DDP Setup & Device Override ---
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        params.prep_device = device
        params.train_device = device
    else:
        global_rank = 0
        local_rank = 0

    is_main_process = global_rank == 0
    # ----------------------------------

    logger = (
        ExperimentLogger(params.exp_dir, params.stage_name)
        if is_main_process
        else None
    )

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

    backbone: ViT | DDP = vit_fn(
        patch_size=params.patch_size,
        channels=params.channels,
        use_sdpa=params.use_sdpa,
    ).to(params.train_device)

    predictor: Predictor | DDP = Predictor(
        embed_dim=embed_dim,
        depth=params.pred_depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        use_sdpa=params.use_sdpa,
    ).to(params.train_device)

    # --- Backbone initialization (precedence: detector checkpoint first) ---
    ssl_ckpt: LeJEPACheckpoint | None = None
    if params.detector_checkpoint is not None:
        assert params.detector_checkpoint.exists(), (
            f"Detector checkpoint not found: {params.detector_checkpoint}"
        )
        if is_main_process:
            print(
                f"Initializing backbone from detector checkpoint: "
                f"{params.detector_checkpoint}"
            )
        detector_ckpt = torch.load(
            params.detector_checkpoint,
            map_location=params.train_device,
            weights_only=True,
        )
        backbone.load_state_dict(
            strip_backbone_prefix(detector_ckpt["model"]), strict=True
        )
    elif params.backbone_checkpoint is not None:
        assert params.backbone_checkpoint.exists(), (
            f"LeJEPA checkpoint not found: {params.backbone_checkpoint}"
        )
        if is_main_process:
            print(
                f"Initializing backbone from LeJEPA checkpoint: "
                f"{params.backbone_checkpoint}"
            )
        ssl_ckpt = torch.load(
            params.backbone_checkpoint,
            map_location=params.train_device,
            weights_only=True,
        )
        backbone.load_state_dict(ssl_ckpt["backbone"], strict=True)
    elif is_main_process:
        print("Training backbone from scratch.")
    # ----------------------------------

    if is_distributed:
        backbone = DDP(
            backbone, device_ids=[local_rank], output_device=local_rank
        )
        predictor = DDP(
            predictor, device_ids=[local_rank], output_device=local_rank
        )

    if params.compile:
        if is_main_process:
            print("Compiling backbone and predictor with torch.compile(dynamic=True)...")
        backbone = cast(ViT | DDP, torch.compile(backbone, dynamic=True))
        predictor = cast(
            Predictor | DDP, torch.compile(predictor, dynamic=True)
        )

    sigreg = SIGReg().to(params.train_device)

    optimizer = optim.AdamW(
        chain(backbone.parameters(), predictor.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    running_loss: float | None = None
    samples = 0
    step_offset = 0
    start_time = time.time()
    if params.dataset is not None:
        dataset_size = len(params.dataset.images)
    else:
        assert params.imslp_manifest is not None
        dataset_size = len(load_imslp_metas(params.imslp_manifest))

    if params.resume:
        assert ssl_ckpt is not None, "--resume requires --backbone_checkpoint"
        assert params.backbone_checkpoint is not None
        if is_main_process:
            print(
                f"Resuming predictor/optimizer/scheduler state from "
                f"{params.backbone_checkpoint}"
            )
        predictor.load_state_dict(ssl_ckpt["predictor"], strict=True)
        optimizer.load_state_dict(ssl_ckpt["optimizer"])
        running_loss = ssl_ckpt["loss"]
        samples = ssl_ckpt["samples"]
        step_offset = ssl_ckpt["step"]

    global_step = step_offset

    backbone.train()
    predictor.train()
    monitor = Monitor() if is_main_process else None
    ssl_iterator: Iterable[SSLBatch]
    if params.dataset is not None:
        ssl_iterator = create_lejepa_iterator(params, monitor=monitor)
    else:
        ssl_iterator = create_imslp_lejepa_iterator(params, monitor=monitor)
    iterator = ThreadedGenerator[SSLBatch](ssl_iterator, maxsize=2)

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
            torch.arange(target_emb.indices.size(1), device=params.train_device)
            .unsqueeze(0)
            .expand(B, -1),
        )

        gather_pos = torch.gather(pos_map, 1, pred_emb.indices)
        target_mask_emb = torch.gather(
            target_emb.data,
            1,
            gather_pos.unsqueeze(-1).expand(-1, -1, target_emb.data.size(-1)),
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

        # --- Sample Budget LR Scheduler (warmup + cosine, mirroring detection) ---
        total_budget = dataset_size * params.epochs
        warmup_samples = params.warmup_epochs * dataset_size
        if samples < warmup_samples:
            current_lr = params.lr * max(
                params.min_lr_ratio, samples / warmup_samples
            )
        else:
            cosine_progress = (samples - warmup_samples) / (
                max(total_budget - warmup_samples, 1e-9)
            )
            current_lr = params.lr * 0.5 * (
                1 + math.cos(math.pi * cosine_progress)
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # ----------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_main_process and step % params.log_interval == 0:
            assert logger is not None

            elapsed = time.time() - start_time
            speed = samples / elapsed if elapsed > 0 else 0.0

            metrics = LeJEPAMetrics(
                step=global_step,
                epoch=current_epoch,
                lr=current_lr,
                loss_total=loss.item(),
                loss_sigreg=sigreg_loss.item(),
                loss_l2=l2_loss.item(),
                speed=speed,
            )
            logger.log_metrics(metrics)

            print(
                f"Epoch [{current_epoch:.2f}/{params.epochs}] Samples [{samples}] "
                f"LR: {current_lr:.2e} | "
                f"Loss: {loss.item():.4f} (Running: {running_loss:.4f}) "
                f"(SIGReg: {sigreg_loss.item():.4f}, L2: {l2_loss.item():.4f}) "
                f"Speed: {speed:.1f} sample/s"
            )

            checkpoint: LeJEPACheckpoint = {
                "backbone": backbone.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": running_loss,
                "samples": samples,
                "step": global_step,
                "epoch": current_epoch,
            }

            torch.save(
                checkpoint,
                logger.get_checkpoint_dir() / "latest_model.pt",
            )

    if is_distributed:
        dist.destroy_process_group()


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
        "--imslp_manifest",
        type=Path,
        default=None,
        help="Path to the IMSLP manifest JSONL. If provided, trains on IMSLP "
        "(set --img_dir data/imslp/images); otherwise on Trompa-COCO.",
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
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=1.0,
        help="Number of epochs to linearly warmup the learning rate",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=1e-4,
        help="Minimum LR multiplier at start of warmup (and cosine floor)",
    )
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
        "--use_sdpa",
        action="store_true",
        help="Enable scaled_dot_product_attention",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the backbone and predictor",
    )
    parser.add_argument(
        "--detector_checkpoint",
        type=Path,
        default=None,
        help="Full detector checkpoint; initializes the backbone from its "
        "trained backbone weights (overrides --backbone_checkpoint).",
    )
    parser.add_argument(
        "--backbone_checkpoint",
        type=Path,
        default=None,
        help="LeJEPA checkpoint; initializes the backbone from its backbone "
        "weights, and (with --resume) restores the full SSL run state.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --backbone_checkpoint, also restoring "
        "the predictor, optimizer, scheduler step/sample counters, and "
        "running loss.",
    )
    parser.add_argument(
        "--exp_dir", type=Path, default=Path("experiments/default_exp")
    )
    parser.add_argument("--stage_name", type=str, default="pretrain_lejepa")

    args = parser.parse_args()

    prep_device = torch.device(args.prep_device)
    train_device = torch.device(args.train_device)

    dataset = (
        parse_coco(args.anno_path, cache_dir=args.cache_dir)
        if args.imslp_manifest is None
        else None
    )

    params = TrainParams(
        anno_path=args.anno_path,
        cache_dir=args.cache_dir,
        img_dir=args.img_dir,
        imslp_manifest=args.imslp_manifest,
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
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        checkpoint_window_size=args.checkpoint_window_size,
        prep_device=prep_device,
        train_device=train_device,
        use_sdpa=args.use_sdpa,
        compile=args.compile,
        detector_checkpoint=args.detector_checkpoint,
        backbone_checkpoint=args.backbone_checkpoint,
        resume=args.resume,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
    )

    train(params)
