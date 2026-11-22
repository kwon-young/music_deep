import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Iterable

import torch
import torch.optim as optim

import transform.det as det_tf
from dataset.coco import CocoDataset, CocoMetadata, load_coco_sample, parse_coco
from logger import BaseMetrics, ExperimentLogger
from model.criterion import LineRefinerCriterion
from model.detector import OMRDetector, create_detector
from model.line_refiner import (
    LineRefiner,
    LineRefinerConfig,
    decode_endpoints,
    gather_line_queries,
)
from model.matcher import HungarianMatcher
from threaded_generator import ThreadedGenerator
from music_types import (
    CHW,
    HW,
    Batch,
    BatchedData,
    BoundingBoxes,
    ClassLabels,
    DetectionSample,
    DetectionTarget,
    EmbedDim,
    Embeddings,
    KeypointDim,
    CoordDim,
    Keypoints,
    LineQueries,
    NumLineClasses,
    NumPatches,
    NumQueries,
    PatchDim,
    Patches,
    PatchUnit,
    TopLeft,
    X1Y1X2Y2,
)


@dataclass
class RefinerParams:
    detector_checkpoint: Path
    exp_dir: Path
    stage_name: str
    feature_cache: Path
    anno_path: Path
    img_dir: Path
    cache_dir: Path | None
    num_images: int | None
    backbone_size: str
    patch_size: int
    channels: int
    num_shapes: int
    base_anchor_size: float
    var_threshold: float
    use_sdpa: bool
    force_pyvips: bool
    device: torch.device
    cost_class: float
    cost_bbox: float
    radius_patches: float
    top_k: int
    matcher_type: str
    refiner: LineRefinerConfig
    epochs: int
    lr: float
    weight_decay: float
    warmup_epochs: float
    min_lr_ratio: float
    pages_per_step: int
    clip_grad_norm: float
    loss_line_l1: float
    loss_line_fgl: float
    log_interval: int
    seed: int


@dataclass
class PageCache:
    """Frozen backbone tokens of one page and its stage-0 line matches."""

    image_id: int
    tokens: torch.Tensor
    indices: torch.Tensor
    image_shape: CHW
    patch_size: HW
    query_indices: torch.Tensor
    target_keypoints: torch.Tensor
    target_labels: torch.Tensor


@dataclass
class PreparedPage:
    epoch: float
    features: Embeddings[Batch, NumPatches, EmbedDim]
    queries: LineQueries[Batch, NumQueries, EmbedDim, KeypointDim, CoordDim]
    target_keypoints: Keypoints[
        tuple[Batch, NumQueries, KeypointDim], X1Y1X2Y2, PatchUnit, TopLeft
    ]
    target_labels: ClassLabels[tuple[Batch, NumQueries], NumLineClasses]


@dataclass
class RefinerMetrics(BaseMetrics):
    epoch: float
    lr: float
    loss_total: float
    loss_line_l1: list[float]
    loss_line_fgl: list[float]
    endpoint_error_px: list[float]
    lines: int
    speed: float


type PatchedPage = BatchedData[
    CocoMetadata,
    DetectionSample[
        Patches[Batch, NumPatches, PatchDim],
        list[BoundingBoxes],
        list[ClassLabels],
        list[Keypoints],
        list[ClassLabels],
    ],
]


def page_cache_path(feature_cache: Path, image_id: int) -> Path:
    return feature_cache / f"{image_id}.pt"


def save_page_cache(page: PageCache, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "image_id": page.image_id,
            "tokens": page.tokens,
            "indices": page.indices,
            "image_shape": list(page.image_shape),
            "patch_size": list(page.patch_size),
            "query_indices": page.query_indices,
            "target_keypoints": page.target_keypoints,
            "target_labels": page.target_labels,
        },
        tmp,
    )
    tmp.replace(path)


def load_page_cache(path: Path) -> PageCache:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a dict, got {type(raw)}")

    def tensor(key: str) -> torch.Tensor:
        value = raw[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path}: '{key}' is not a tensor")
        return value

    def ints(key: str) -> list[int]:
        value = raw[key]
        if not (isinstance(value, list) and all(isinstance(v, int) for v in value)):
            raise ValueError(f"{path}: '{key}' is not a list of ints")
        return value

    image_id = raw["image_id"]
    if not isinstance(image_id, int):
        raise ValueError(f"{path}: 'image_id' is not an int")
    c, h, w = ints("image_shape")
    ph, pw = ints("patch_size")
    return PageCache(
        image_id=image_id,
        tokens=tensor("tokens"),
        indices=tensor("indices"),
        image_shape=(c, h, w),
        patch_size=(ph, pw),
        query_indices=tensor("query_indices"),
        target_keypoints=tensor("target_keypoints"),
        target_labels=tensor("target_labels"),
    )


def preprocess_page(
    index: int, dataset: CocoDataset, params: RefinerParams
) -> PatchedPage:
    """Same preprocessing as inference_coco: no augmentation, fixed patch size."""
    item = load_coco_sample(dataset, params.img_dir, index, params.device)
    if params.force_pyvips or params.device.type != "cuda":
        decoded = det_tf.decode_pyvips(item, device=params.device)
    else:
        decoded = det_tf.decode_nvimgcodec(item, device=params.device)
    patch_size = (params.patch_size, params.patch_size)
    padded = det_tf.pad_to_patch_size(det_tf.to_float1(decoded), patch_size=patch_size)
    patched = det_tf.extract_patches(det_tf.collate((padded,)), patch_size=patch_size)
    normalized = det_tf.normalize_batched_targets(patched)
    return det_tf.variance_patch_drop(normalized, var_threshold=params.var_threshold)


def build_feature_cache(
    params: RefinerParams,
    dataset: CocoDataset,
    detector: OMRDetector,
    matcher: HungarianMatcher,
) -> None:
    params.feature_cache.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(dataset.images)))
    if params.num_images is not None:
        indices = indices[: params.num_images]
    todo = [
        i
        for i in indices
        if not page_cache_path(
            params.feature_cache, dataset.images[i].img_id
        ).exists()
    ]
    print(f"Feature cache: {len(indices) - len(todo)} pages cached, {len(todo)} to build")
    if not todo:
        return

    pages = ThreadedGenerator(
        ((i, preprocess_page(i, dataset, params)) for i in todo),
        maxsize=2,
        name="cache_decode",
    )
    start = time.time()
    with torch.no_grad():
        for done, (index, item) in enumerate(pages, start=1):
            features = detector.backbone(item.sample.image)
            lines = detector.predict_lines(features)
            target = DetectionTarget(
                boxes=item.sample.boxes[0],
                box_labels=item.sample.box_labels[0],
                keypoints=item.sample.keypoints[0],
                keypoint_labels=item.sample.keypoint_labels[0],
            )
            match = matcher.match_lines(lines, [target])[0]
            target_indices = match.target_indices.to(params.device)
            image_id = dataset.images[index].img_id
            save_page_cache(
                PageCache(
                    image_id=image_id,
                    tokens=features.data[0].cpu(),
                    indices=features.indices[0].cpu(),
                    image_shape=features.image_shape,
                    patch_size=features.patch_size,
                    query_indices=match.pred_indices.cpu(),
                    target_keypoints=target.keypoints.data[target_indices].cpu(),
                    target_labels=target.keypoint_labels.data[target_indices].cpu(),
                ),
                page_cache_path(params.feature_cache, image_id),
            )
            elapsed = time.time() - start
            print(
                f"[cache {done}/{len(todo)}] image {image_id}: "
                f"{features.data.shape[1]} tokens, {len(match.pred_indices)} matched lines "
                f"({elapsed / done:.2f}s/page)"
            )


def prepare_pages(
    pages: list[PageCache],
    detector: OMRDetector,
    params: RefinerParams,
) -> Generator[PreparedPage, None, None]:
    """Moves cached pages to the device and recomputes the frozen stage-0 lines."""
    order = list(range(len(pages)))
    with torch.no_grad():
        for epoch in range(params.epochs):
            random.shuffle(order)
            for position, page_index in enumerate(order):
                page = pages[page_index]
                features: Embeddings[Batch, NumPatches, EmbedDim] = Embeddings(
                    data=page.tokens.to(params.device).unsqueeze(0),
                    indices=page.indices.to(params.device).unsqueeze(0),
                    image_shape=page.image_shape,
                    patch_size=page.patch_size,
                )
                lines = detector.predict_lines(features)
                queries = gather_line_queries(
                    features,
                    lines,
                    page.query_indices.to(params.device).unsqueeze(0),
                    params.num_shapes,
                )
                yield PreparedPage(
                    epoch=epoch + (position + 1) / len(order),
                    features=features,
                    queries=queries,
                    target_keypoints=Keypoints(
                        page.target_keypoints.to(params.device).unsqueeze(0)
                    ),
                    target_labels=ClassLabels(
                        page.target_labels.to(params.device).unsqueeze(0)
                    ),
                )


def learning_rate(step: int, total_steps: int, warmup_steps: int, params: RefinerParams) -> float:
    if step < warmup_steps:
        return params.lr * max(params.min_lr_ratio, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    floor = params.lr * params.min_lr_ratio
    return floor + (params.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def endpoint_error_px(
    keypoints: torch.Tensor, targets: torch.Tensor, patch_size: int
) -> torch.Tensor:
    diff = keypoints.reshape(-1, 2, 2) - targets.reshape(-1, 2, 2)
    return diff.norm(dim=-1).sum() * patch_size


def train(
    params: RefinerParams,
    pages: list[PageCache],
    detector: OMRDetector,
    line_weights: list[float],
) -> None:
    logger = ExperimentLogger(params.exp_dir, params.stage_name)
    refiner = LineRefiner(params.refiner).to(params.device)
    criterion = LineRefinerCriterion(
        line_weights=line_weights,
        loss_line_l1=params.loss_line_l1,
        loss_line_fgl=params.loss_line_fgl,
        base_anchor_size=params.base_anchor_size,
        reg_max=params.refiner.reg_max,
    ).to(params.device)
    optimizer = optim.AdamW(
        refiner.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    num_params = sum(p.numel() for p in refiner.parameters())
    print(f"LineRefiner: {num_params / 1e6:.2f}M parameters, {params.refiner}")

    steps_per_epoch = math.ceil(len(pages) / params.pages_per_step)
    total_steps = steps_per_epoch * params.epochs
    warmup_steps = int(params.warmup_epochs * steps_per_epoch)
    num_layers = params.refiner.num_layers

    stream = ThreadedGenerator(
        prepare_pages(pages, detector, params), maxsize=4, name="prepare_pages"
    )

    step = 0
    pages_in_step = 0
    lr = learning_rate(0, total_steps, warmup_steps, params)
    window_pages = 0
    window_lines = 0
    window_total = 0.0
    window_l1 = [0.0] * num_layers
    window_fgl = [0.0] * num_layers
    window_err = [0.0] * (num_layers + 1)
    window_start = time.time()
    epoch = 0.0

    def save_checkpoint() -> None:
        torch.save(
            {
                "config": asdict(params.refiner),
                "refiner": refiner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "detector_checkpoint": str(params.detector_checkpoint),
            },
            logger.get_checkpoint_dir() / "latest_model.pt",
        )

    refiner.train()
    optimizer.zero_grad()
    for page in stream:
        epoch = page.epoch
        num_lines = page.target_labels.data.shape[1]
        if num_lines == 0:
            continue

        refined = refiner(page.features, page.queries)
        losses = criterion(
            refined, page.queries, page.target_keypoints, page.target_labels
        )
        total = torch.stack(losses.loss_line_l1 + losses.loss_line_fgl).sum()
        (total / params.pages_per_step).backward()
        pages_in_step += 1

        with torch.no_grad():
            targets = page.target_keypoints.data[0]
            stage0 = decode_endpoints(
                page.queries.absolute_centers.data[0],
                page.queries.raw_directions.data[0],
                page.queries.endpoint_logits.data[0],
                refiner.weighting_fn.w,
                params.base_anchor_size,
            )
            window_err[0] += endpoint_error_px(stage0, targets, params.patch_size).item()
            for i, keypoints in enumerate(refined.keypoints):
                window_err[i + 1] += endpoint_error_px(
                    keypoints.data[0], targets, params.patch_size
                ).item()
            for i in range(num_layers):
                window_l1[i] += losses.loss_line_l1[i].item()
                window_fgl[i] += losses.loss_line_fgl[i].item()
        window_total += total.item()
        window_pages += 1
        window_lines += num_lines

        if pages_in_step < params.pages_per_step:
            continue

        torch.nn.utils.clip_grad_norm_(refiner.parameters(), params.clip_grad_norm)
        lr = learning_rate(step, total_steps, warmup_steps, params)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad()
        pages_in_step = 0
        step += 1

        if step % params.log_interval == 0:
            elapsed = time.time() - window_start
            metrics = RefinerMetrics(
                step=step,
                epoch=epoch,
                lr=lr,
                loss_total=window_total / window_pages,
                loss_line_l1=[v / window_pages for v in window_l1],
                loss_line_fgl=[v / window_pages for v in window_fgl],
                endpoint_error_px=[v / (2 * window_lines) for v in window_err],
                lines=window_lines,
                speed=window_pages / elapsed,
            )
            logger.log_metrics(metrics)
            errors = " -> ".join(f"{e:.2f}" for e in metrics.endpoint_error_px)
            print(
                f"Epoch [{epoch:.2f}/{params.epochs}] Step [{step}/{total_steps}] "
                f"LR: {lr:.2e} | Loss: {metrics.loss_total:.4f} | "
                f"Endpoint err px (stage0 -> layers): {errors} | "
                f"{metrics.speed:.1f} pages/s"
            )
            window_pages = window_lines = 0
            window_total = 0.0
            window_l1 = [0.0] * num_layers
            window_fgl = [0.0] * num_layers
            window_err = [0.0] * (num_layers + 1)
            window_start = time.time()

        if step % steps_per_epoch == 0:
            save_checkpoint()

    save_checkpoint()
    print(f"Saved {logger.get_checkpoint_dir() / 'latest_model.pt'}")


def main(params: RefinerParams) -> None:
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    dataset = parse_coco(params.anno_path, cache_dir=params.cache_dir)
    detector = create_detector(
        backbone_size=params.backbone_size,
        patch_size=params.patch_size,
        channels=params.channels,
        use_sdpa=params.use_sdpa,
        num_symbol_classes=dataset.num_symbol_classes,
        num_line_classes=dataset.num_line_classes,
        num_shapes=params.num_shapes,
        base_anchor_size=params.base_anchor_size,
    ).to(params.device)
    checkpoint = torch.load(
        params.detector_checkpoint, map_location=params.device, weights_only=True
    )
    detector.load_state_dict(checkpoint["model"])
    detector.eval()
    detector.requires_grad_(False)

    matcher = HungarianMatcher(
        cost_class=params.cost_class,
        cost_bbox=params.cost_bbox,
        radius_patches=params.radius_patches,
        top_k=params.top_k,
        matcher_type=params.matcher_type,
    )
    build_feature_cache(params, dataset, detector, matcher)

    ids = [img.img_id for img in dataset.images]
    if params.num_images is not None:
        ids = ids[: params.num_images]
    pages = [load_page_cache(page_cache_path(params.feature_cache, i)) for i in ids]
    print(
        f"Loaded {len(pages)} cached pages, "
        f"{sum(len(p.target_labels) for p in pages)} matched lines"
    )
    train(params, pages, detector, dataset.line_weights)


def parse_args() -> RefinerParams:
    parser = argparse.ArgumentParser(
        description="Train the iterative D-FINE line refiner on a frozen detector"
    )
    parser.add_argument("--detector_checkpoint", type=Path, required=True)
    parser.add_argument("--exp_dir", type=Path, required=True)
    parser.add_argument("--stage_name", type=str, default="train_line_refiner")
    parser.add_argument(
        "--feature_cache",
        type=Path,
        default=None,
        help="Directory of cached backbone tokens + matches. Defaults to <exp_dir>/feature_cache.",
    )
    parser.add_argument(
        "--anno_path",
        type=Path,
        default=Path("data/trompa-coco/annotations/instances_trainval2017.json"),
    )
    parser.add_argument(
        "--img_dir", type=Path, default=Path("data/trompa-coco/trainval2017")
    )
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument(
        "--backbone_size", type=str, choices=["nano", "small", "base"], default="nano"
    )
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_shapes", type=int, default=5)
    parser.add_argument("--base_anchor_size", type=float, default=1.0)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--use_sdpa", action="store_true")
    parser.add_argument("--force_pyvips", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cost_class", type=float, default=2.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--radius_patches", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--matcher_type", type=str, choices=["scipy", "greedy"], default="scipy"
    )
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--offset_scale", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=float, default=1.0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--pages_per_step", type=int, default=4)
    parser.add_argument("--clip_grad_norm", type=float, default=0.1)
    parser.add_argument("--loss_line_l1", type=float, default=5.0)
    parser.add_argument("--loss_fgl", type=float, default=0.05)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dim = {"nano": 192, "small": 384, "base": 768}[args.backbone_size]
    return RefinerParams(
        detector_checkpoint=args.detector_checkpoint,
        exp_dir=args.exp_dir,
        stage_name=args.stage_name,
        feature_cache=args.feature_cache or args.exp_dir / "feature_cache",
        anno_path=args.anno_path,
        img_dir=args.img_dir,
        cache_dir=args.cache_dir,
        num_images=args.num_images,
        backbone_size=args.backbone_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_shapes=args.num_shapes,
        base_anchor_size=args.base_anchor_size,
        var_threshold=args.var_threshold,
        use_sdpa=args.use_sdpa,
        force_pyvips=args.force_pyvips,
        device=torch.device(args.device),
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        radius_patches=args.radius_patches,
        top_k=args.top_k,
        matcher_type=args.matcher_type,
        refiner=LineRefinerConfig(
            dim=dim,
            num_layers=args.num_layers,
            heads=args.heads,
            points=args.points,
            mlp_ratio=args.mlp_ratio,
            offset_scale=args.offset_scale,
            num_shapes=args.num_shapes,
            base_anchor_size=args.base_anchor_size,
        ),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        pages_per_step=args.pages_per_step,
        clip_grad_norm=args.clip_grad_norm,
        loss_line_l1=args.loss_line_l1,
        loss_line_fgl=args.loss_fgl,
        log_interval=args.log_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main(parse_args())
