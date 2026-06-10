import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from nvidia.nvimgcodec import Decoder


def prepare_images(src_path: Path, jpg_path: Path, tiff_path: Path) -> None:
    """Converts the source PNG to JPEG and TIFF, preserving the color mode."""
    img = Image.open(src_path)

    # JPEG does not support alpha channels (RGBA) or palettes (P)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if not jpg_path.exists():
        print(f"Converting to JPEG: {jpg_path.name}")
        img.save(jpg_path, format="JPEG", quality=95)

    if not tiff_path.exists():
        print(f"Converting to TIFF: {tiff_path.name}")
        # tiff_adobe_deflate is an excellent lossless compression for both Grayscale and RGB
        img.save(tiff_path, format="TIFF", compression="tiff_adobe_deflate")


def current_pipeline(
    img_path: Path, x: int, y: int, crop_size: int, device: torch.device
) -> torch.Tensor:
    """Current: PIL -> Numpy -> Torch -> Crop -> GPU -> Expand/Permute -> Float1"""
    img = Image.open(img_path)
    arr = np.array(img)
    t = torch.as_tensor(arr)

    if t.ndim == 2:  # Grayscale (H, W)
        t_crop = t[y : y + crop_size, x : x + crop_size]
        t_gpu = t_crop.to(device)
        t_rgb = t_gpu.unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_crop = t[y : y + crop_size, x : x + crop_size, :]
        t_gpu = t_crop.to(device)
        t_rgb = t_gpu.permute(2, 0, 1)

    t_float = t_rgb.float() / 255.0
    return t_float


def nvimagecodec_pipeline(
    img_path: Path, x: int, y: int, crop_size: int, decoder: Decoder
) -> torch.Tensor:
    """New: nvimagecodec -> DLPack -> Crop -> Expand/Permute -> Float1"""
    nv_img = decoder.read(str(img_path))
    t_gpu = torch.from_dlpack(nv_img)

    # nvimagecodec might return (H, W, 1) for grayscale. Squeeze to (H, W)
    if t_gpu.ndim == 3 and t_gpu.shape[-1] == 1:
        t_gpu = t_gpu.squeeze(-1)

    if t_gpu.ndim == 2:  # Grayscale (H, W)
        t_crop = t_gpu[y : y + crop_size, x : x + crop_size]
        t_rgb = t_crop.unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_crop = t_gpu[y : y + crop_size, x : x + crop_size, :]
        t_rgb = t_crop.permute(2, 0, 1)

    t_float = t_rgb.float() / 255.0
    return t_float


def benchmark(
    data_dir: Path, crop_size: int, iterations: int, image_name: str | None
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available. Benchmark requires a GPU.")
        return

    if image_name:
        src_img_path = data_dir / image_name
        if not src_img_path.exists():
            print(f"Image not found: {src_img_path}")
            return
    else:
        png_images = list(data_dir.glob("*.png"))
        if not png_images:
            print(f"No PNG images found in {data_dir}")
            return
        src_img_path = png_images[0]

    jpg_img_path = src_img_path.with_suffix(".jpg")
    tiff_img_path = src_img_path.with_suffix(".tiff")

    prepare_images(src_img_path, jpg_img_path, tiff_img_path)

    with Image.open(src_img_path) as img:
        w, h = img.size

    max_x = max(0, w - crop_size)
    max_y = max(0, h - crop_size)

    if max_x < 0 or max_y < 0:
        print(
            f"Error: Crop size ({crop_size}) is larger than image dimensions ({w}x{h})."
        )
        return

    coords = [
        (
            int(torch.randint(0, max_x + 1, (1,)).item()),
            int(torch.randint(0, max_y + 1, (1,)).item()),
        )
        for _ in range(iterations)
    ]

    decoder = Decoder()

    print(
        f"\nBenchmarking {iterations} iterations of {crop_size}x{crop_size} crops on {src_img_path.name}..."
    )

    # --- 1. Current Pipeline (PNG) ---
    current_pipeline(
        src_img_path, coords[0][0], coords[0][1], crop_size, device
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for x, y in coords:
        _ = current_pipeline(src_img_path, x, y, crop_size, device)
    torch.cuda.synchronize()
    current_time = time.perf_counter() - start_time
    print(
        f"Current Pipeline (PIL PNG): {current_time:.4f}s ({iterations / current_time:.2f} it/s)"
    )

    # --- 2. nvimagecodec Pipeline (JPEG) ---
    nvimagecodec_pipeline(
        jpg_img_path, coords[0][0], coords[0][1], crop_size, decoder
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for x, y in coords:
        _ = nvimagecodec_pipeline(jpg_img_path, x, y, crop_size, decoder)
    torch.cuda.synchronize()
    nv_jpg_time = time.perf_counter() - start_time
    print(
        f"nvimagecodec   (GPU JPEG): {nv_jpg_time:.4f}s ({iterations / nv_jpg_time:.2f} it/s) -> {current_time / nv_jpg_time:.2f}x speedup"
    )

    # --- 3. nvimagecodec Pipeline (TIFF) ---
    try:
        nvimagecodec_pipeline(
            tiff_img_path, coords[0][0], coords[0][1], crop_size, decoder
        )
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for x, y in coords:
            _ = nvimagecodec_pipeline(tiff_img_path, x, y, crop_size, decoder)
        torch.cuda.synchronize()
        nv_tiff_time = time.perf_counter() - start_time
        print(
            f"nvimagecodec   (GPU TIFF): {nv_tiff_time:.4f}s ({iterations / nv_tiff_time:.2f} it/s) -> {current_time / nv_tiff_time:.2f}x speedup"
        )
    except Exception as e:
        print(
            f"nvimagecodec   (GPU TIFF): FAILED (Expected on Maxwell/Quadro M1200). Error: {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark nvimagecodec vs PIL"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/trompa-coco/trainval2017"),
        help="Directory containing images",
    )
    parser.add_argument(
        "--crop_size", type=int, default=3584, help="Size of the square crop"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iterations to run"
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Specific image file to use (e.g., image.png)",
    )

    args = parser.parse_args()
    benchmark(args.data_dir, args.crop_size, args.iterations, args.image_name)
