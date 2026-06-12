import argparse
import time
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from nvidia.nvimgcodec import Decoder
import cucim
import pyvips


def prepare_images(src_path: Path, png_path: Path, jpg_path: Path) -> None:
    """Converts the source TIFF to PNG and JPEG for benchmarking."""
    img = Image.open(src_path)

    if not png_path.exists():
        print(f"Converting to PNG: {png_path.name}")
        img.save(png_path, format="PNG")

    # JPEG does not support alpha channels (RGBA) or palettes (P)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if not jpg_path.exists():
        print(f"Converting to JPEG: {jpg_path.name}")
        img.save(jpg_path, format="JPEG", quality=95)


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


def cucim_pipeline(
    img_path: Path, x: int, y: int, crop_size: int, device: torch.device
) -> torch.Tensor:
    """cucim -> read_region -> Numpy -> Torch -> GPU -> Expand/Permute -> Float1"""
    img = cucim.CuImage(str(img_path))
    region = img.read_region(location=(x, y), size=(crop_size, crop_size))
    arr = np.asarray(region)  # Guarantees zero-copy if the buffer is compatible
    t = torch.from_numpy(arr)

    if t.ndim == 2:  # Grayscale (H, W)
        t_gpu = t.to(device)
        t_rgb = t_gpu.unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_gpu = t.to(device)
        t_rgb = t_gpu.permute(2, 0, 1)

    t_float = t_rgb.float() / 255.0
    return t_float


def pyvips_pipeline(
    img_path: Path, x: int, y: int, crop_size: int, device: torch.device
) -> torch.Tensor:
    """pyvips -> crop -> Numpy -> Torch -> GPU -> Expand/Permute -> Float1"""
    # Open lazily
    img = pyvips.Image.new_from_file(str(img_path))

    # Crop lazily
    crop = img.crop(x, y, crop_size, crop_size)

    # Decode to memory and wrap in numpy (zero-copy from the buffer)
    arr = np.ndarray(
        buffer=crop.write_to_memory(),
        dtype=np.uint8,
        shape=(crop.height, crop.width, crop.bands),
    )
    t = torch.from_numpy(arr)

    if t.shape[-1] == 1:  # Grayscale (H, W, 1)
        t_gpu = t.squeeze(-1).to(device)
        t_rgb = t_gpu.unsqueeze(0).expand(3, -1, -1)
    else:  # RGB (H, W, C)
        t_gpu = t.to(device)
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
        tiff_images = list(data_dir.glob("*.tiff"))
        if not tiff_images:
            print(f"No TIFF images found in {data_dir}")
            return
        src_img_path = tiff_images[0]

    png_img_path = src_img_path.with_suffix(".png")
    jpg_img_path = src_img_path.with_suffix(".jpg")

    prepare_images(src_img_path, png_img_path, jpg_img_path)

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

    print(
        f"\nBenchmarking {iterations} iterations of {crop_size}x{crop_size} crops on {src_img_path.name}..."
    )

    # --- Warmups ---
    decoder = Decoder()
    current_pipeline(
        png_img_path, coords[0][0], coords[0][1], crop_size, device
    )
    # cucim_pipeline(
    #     src_img_path, coords[0][0], coords[0][1], crop_size, device
    # )
    pyvips_pipeline(src_img_path, coords[0][0], coords[0][1], crop_size, device)
    nvimagecodec_pipeline(
        jpg_img_path, coords[0][0], coords[0][1], crop_size, decoder
    )
    try:
        nvimagecodec_pipeline(
            src_img_path, coords[0][0], coords[0][1], crop_size, decoder
        )
    except Exception:
        pass
    torch.cuda.synchronize()

    # --- 1. Current Pipeline (PNG) ---
    start_time = time.perf_counter()
    for x, y in coords:
        _ = current_pipeline(png_img_path, x, y, crop_size, device)
    torch.cuda.synchronize()
    current_time = time.perf_counter() - start_time
    print(
        f"Current Pipeline (PIL PNG): {current_time:.4f}s ({iterations / current_time:.2f} it/s)"
    )

    # # --- 2. CuCIM Pipeline (TIFF) ---
    # start_time = time.perf_counter()
    # for x, y in coords:
    #     _ = cucim_pipeline(src_img_path, x, y, crop_size, device)
    # torch.cuda.synchronize()
    # cucim_time = time.perf_counter() - start_time
    # print(
    #     f"CuCIM Pipeline (CPU TIFF):  {cucim_time:.4f}s ({iterations / cucim_time:.2f} it/s) -> {current_time / cucim_time:.2f}x speedup"
    # )

    # --- 3. pyvips Pipeline (TIFF) ---
    start_time = time.perf_counter()
    for x, y in coords:
        _ = pyvips_pipeline(src_img_path, x, y, crop_size, device)
    torch.cuda.synchronize()
    pyvips_time = time.perf_counter() - start_time
    print(
        f"pyvips Pipeline (CPU TIFF): {pyvips_time:.4f}s ({iterations / pyvips_time:.2f} it/s) -> {current_time / pyvips_time:.2f}x speedup"
    )

    # --- 4. nvimagecodec Pipeline (JPEG) ---
    start_time = time.perf_counter()
    for x, y in coords:
        _ = nvimagecodec_pipeline(jpg_img_path, x, y, crop_size, decoder)
    torch.cuda.synchronize()
    nv_jpg_time = time.perf_counter() - start_time
    print(
        f"nvimagecodec   (GPU JPEG):  {nv_jpg_time:.4f}s ({iterations / nv_jpg_time:.2f} it/s) -> {current_time / nv_jpg_time:.2f}x speedup"
    )

    # --- 5. nvimagecodec Pipeline (TIFF) ---
    try:
        start_time = time.perf_counter()
        for x, y in coords:
            _ = nvimagecodec_pipeline(src_img_path, x, y, crop_size, decoder)
        torch.cuda.synchronize()
        nv_tiff_time = time.perf_counter() - start_time
        print(
            f"nvimagecodec   (GPU TIFF):  {nv_tiff_time:.4f}s ({iterations / nv_tiff_time:.2f} it/s) -> {current_time / nv_tiff_time:.2f}x speedup"
        )
    except Exception as e:
        print(
            f"nvimagecodec   (GPU TIFF):  FAILED (Expected on Maxwell/Quadro M1200). Error: {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark nvimagecodec vs cucim vs pyvips vs PIL"
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
        help="Specific image file to use (e.g., image.tiff)",
    )

    args = parser.parse_args()
    benchmark(args.data_dir, args.crop_size, args.iterations, args.image_name)
