import sys
from pathlib import Path

# Add src to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import torch
import gc
from model.vit import vit_nano, vit_small, vit_base


def check_memory(model_fn, batch_size, img_size, is_train, patch_size):
    try:
        device = torch.device("cuda")
        # Instantiate model with the target image_size and patch_size
        model = model_fn(
            image_size=img_size, patch_size=patch_size, num_classes=0, channels=1
        ).to(device)
        x = torch.randn(batch_size, 1, img_size, img_size, device=device)
        
        if is_train:
            model.train()
            out = model(x)
            out.mean().backward()
        else:
            model.eval()
            with torch.no_grad():
                out = model(x)
                
        # Clean up memory if successful
        del model, x, out
        torch.cuda.empty_cache()
        gc.collect()
        return True
    
    except torch.cuda.OutOfMemoryError:
        # Clean up on OOM
        torch.cuda.empty_cache()
        gc.collect()
        return False
    except Exception:
        # Catch other potential size-related errors (e.g. tensor too large)
        torch.cuda.empty_cache()
        gc.collect()
        return False


def find_max_tokens(model_fn, batch_size, is_train, patch_size):
    # Binary search over image size multiples of the patch size
    low_mult = 1
    high_mult = 250  # max test image size: 250 * patch_size
    best_mult = 0
    
    while low_mult <= high_mult:
        mid_mult = (low_mult + high_mult) // 2
        img_size = mid_mult * patch_size
        
        if check_memory(model_fn, batch_size, img_size, is_train, patch_size):
            best_mult = mid_mult
            low_mult = mid_mult + 1
        else:
            high_mult = mid_mult - 1
            
    if best_mult == 0:
        return 0
        
    # Total tokens = (H // patch_size) * (W // patch_size)
    return best_mult * best_mult


def main():
    if not torch.cuda.is_available():
        print("CUDA is required to benchmark GPU memory limits.")
        return

    models = {
        "vit_nano": vit_nano,
        "vit_small": vit_small,
        "vit_base": vit_base,
    }
    
    patch_sizes = [16, 32]
    batch_sizes = [1, 8, 32, 128]
    
    print(f"{'Model':<12} | {'Patch':<5} | {'Batch':<6} | {'Mode':<8} | {'Max Tokens':<12}")
    print("-" * 55)
    
    for name, model_fn in models.items():
        for ps in patch_sizes:
            for bs in batch_sizes:
                for is_train in [False, True]:
                    mode_str = "Train" if is_train else "Infer"
                    max_tokens = find_max_tokens(model_fn, bs, is_train, ps)
                    print(f"{name:<12} | {ps:<5} | {bs:<6} | {mode_str:<8} | {max_tokens:<12}")


if __name__ == "__main__":
    main()
