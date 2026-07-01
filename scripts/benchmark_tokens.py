# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
# +-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  Quadro M1200                   Off |   00000000:01:00.0 Off |                  N/A |
# | N/A   54C    P8            N/A  /  200W |       0MiB /   4096MiB |      0%      Default |
# Model        | Patch | Batch  | Mode     | Max Tokens
# -------------------------------------------------------
# vit_nano     | 16    | 1      | Infer    | 12769
# vit_nano     | 16    | 1      | Train    | 4225
# vit_nano     | 16    | 8      | Infer    | 4356
# vit_nano     | 16    | 8      | Train    | 1225
# vit_nano     | 16    | 32     | Infer    | 2116
# vit_nano     | 16    | 32     | Train    | 484
# vit_nano     | 16    | 128    | Infer    | 900
# vit_nano     | 16    | 128    | Train    | 144
# vit_nano     | 32    | 1      | Infer    | 12544
# vit_nano     | 32    | 1      | Train    | 4225
# vit_nano     | 32    | 8      | Infer    | 4356
# vit_nano     | 32    | 8      | Train    | 1225
# vit_nano     | 32    | 32     | Infer    | 2025
# vit_nano     | 32    | 32     | Train    | 484
# vit_nano     | 32    | 128    | Infer    | 900
# vit_nano     | 32    | 128    | Train    | 144
# vit_small    | 16    | 1      | Infer    | 8836
# vit_small    | 16    | 1      | Train    | 2916
# vit_small    | 16    | 8      | Infer    | 3025
# vit_small    | 16    | 8      | Train    | 784
# vit_small    | 16    | 32     | Infer    | 1444
# vit_small    | 16    | 32     | Train    | 256
# vit_small    | 16    | 128    | Infer    | 676
# vit_small    | 16    | 128    | Train    | 64
# vit_small    | 32    | 1      | Infer    | 8836
# vit_small    | 32    | 1      | Train    | 2916
# vit_small    | 32    | 8      | Infer    | 3025
# vit_small    | 32    | 8      | Train    | 784
# vit_small    | 32    | 32     | Infer    | 1369
# vit_small    | 32    | 32     | Train    | 256
# vit_small    | 32    | 128    | Infer    | 625
# vit_small    | 32    | 128    | Train    | 64
# vit_base     | 16    | 1      | Infer    | 6084
# vit_base     | 16    | 1      | Train    | 1849
# vit_base     | 16    | 8      | Infer    | 2025
# vit_base     | 16    | 8      | Train    | 441
# vit_base     | 16    | 32     | Infer    | 900
# vit_base     | 16    | 32     | Train    | 121
# vit_base     | 16    | 128    | Infer    | 361
# vit_base     | 16    | 128    | Train    | 36
# vit_base     | 32    | 1      | Infer    | 6084
# vit_base     | 32    | 1      | Train    | 1849
# vit_base     | 32    | 8      | Infer    | 2025
# vit_base     | 32    | 8      | Train    | 441
# vit_base     | 32    | 32     | Infer    | 900
# vit_base     | 32    | 32     | Train    | 121
# vit_base     | 32    | 128    | Infer    | 361
# vit_base     | 32    | 128    | Train    | 36

import torch
import gc
from model.vit import vit_nano, vit_small, vit_base
from music_types import Embeddings


def check_memory(model, batch_size, num_tokens, is_train, patch_size):
    try:
        device = next(model.parameters()).device

        # Use empty instead of randn to save time (we don't need real data)
        patch_dim = 3 * patch_size * patch_size
        dummy_data = torch.empty(batch_size, num_tokens, patch_dim, device=device)
        dummy_indices = (
            torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # image_shape and patch_size must be consistent for compute_freqs
        # grid_h * grid_w >= num_tokens
        grid_h = int(num_tokens**0.5) + 1
        grid_w = grid_h
        h = grid_h * patch_size
        w = grid_w * patch_size
        c = 3

        patches = Embeddings(
            data=dummy_data,
            indices=dummy_indices,
            image_shape=(c, h, w),
            patch_size=(patch_size, patch_size),
        )

        if is_train:
            model.train()
            out = model(patches)
            out.data.mean().backward()
        else:
            model.eval()
            with torch.no_grad():
                out = model(patches)

        # Clean up memory if successful
        del patches, out
        torch.cuda.empty_cache()
        return True

    except torch.cuda.OutOfMemoryError:
        # Clean up on OOM
        torch.cuda.empty_cache()
        return False
    except Exception:
        # Catch other potential size-related errors (e.g. tensor too large)
        torch.cuda.empty_cache()
        return False


def find_max_tokens(model_fn, batch_size, is_train, patch_size):
    # Instantiate model ONCE outside the loop
    device = torch.device("cuda")
    model = model_fn(patch_size=patch_size, channels=3).to(device)

    # Binary search over number of tokens
    low_tokens = 100
    high_tokens = 20000  # lowered max test tokens
    best_tokens = 0

    while low_tokens <= high_tokens:
        mid_tokens = (low_tokens + high_tokens) // 2

        if check_memory(model, batch_size, mid_tokens, is_train, patch_size):
            best_tokens = mid_tokens
            low_tokens = mid_tokens + 1
        else:
            high_tokens = mid_tokens - 1

    # Cleanup model after search
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best_tokens


def main():
    if not torch.cuda.is_available():
        print("CUDA is required to benchmark GPU memory limits.")
        return

    models = {
        "vit_nano": vit_nano,
        "vit_small": vit_small,
        "vit_base": vit_base,
    }

    patch_sizes = [64]  # Only test 64 as it's the project standard
    batch_sizes = [1, 8, 32, 128]

    print(
        f"{'Model':<12} | {'Patch':<5} | {'Batch':<6} | {'Mode':<8} | {'Max Tokens':<12}"
    )
    print("-" * 55)

    for name, model_fn in models.items():
        for ps in patch_sizes:
            for bs in batch_sizes:
                for is_train in [False, True]:
                    mode_str = "Train" if is_train else "Infer"
                    max_tokens = find_max_tokens(model_fn, bs, is_train, ps)
                    print(
                        f"{name:<12} | {ps:<5} | {bs:<6} | {mode_str:<8} | {max_tokens:<12}"
                    )


if __name__ == "__main__":
    main()
