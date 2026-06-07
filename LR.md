# Learning Rate Scheduling Strategy

## The Problematic
When scaling up object detection models, changing the `crop_size` or `batch_size` drastically alters the amount of ground truth data (symbols) processed per forward pass. 
* A 224x224 crop might contain 10 symbols.
* A 3584x3584 crop contains 256x more area, and therefore roughly 256x more symbols.

Because the detection loss is averaged over the total number of ground truth boxes (`num_boxes`), the gradient contribution of each individual box becomes proportionally smaller as the crop size increases. 

Standard PyTorch schedulers (like `StepLR` or `CosineAnnealingLR`) advance based on the number of **optimizer steps** or **epochs**. If we scale up the crop size, the model sees the same amount of data in far fewer steps, causing step-based schedulers to become completely misaligned with the actual learning progress.

Furthermore, the classic "Linear Scaling Rule" (multiplying the LR by the batch/area ratio) fails when using **AdamW**. AdamW normalizes gradients by their variance, making it naturally scale-invariant. Manually scaling the LR forces the optimizer to take massive, destructive steps.

## Overview of Schedulers
1. **Step-based Decay (`StepLR`)**: Drops LR at fixed steps. Prone to jarring momentum shocks and requires guessing plateau points.
2. **Adaptive Decay (`ReduceLROnPlateau`)**: Drops LR when validation loss stalls. Reactive and easily tricked by noisy batches.
3. **Continuous Decay (`CosineAnnealingLR`)**: Smoothly decays LR following a cosine curve. Excellent for settling into flat minima.
4. **Warmup**: Linearly increases LR from 0 to `base_lr` over the first few steps. **Critical for Vision Transformers** to prevent early chaotic gradients from destroying weights or dynamic shape biases.

## Compute Budget vs. Signal Budget
We use **Variance-Based Patch Dropping**, which drops up to 90% of the empty background patches.
* **Compute Budget (Tokens):** Varies wildly depending on how much background is dropped.
* **Signal Budget (Ground Truth Symbols):** Remains constant for a given crop, regardless of how much background is dropped.

If we tied the LR schedule to the Compute Budget (tokens processed), the model would advance its schedule faster on dense images and slower on sparse images, destabilizing training. The schedule must be tied strictly to the **Signal Budget**.

## The Chosen Solution: The Symbol Budget Scheduler
To make the learning rate schedule perfectly invariant to `crop_size`, `batch_size`, and patch dropping, we define training progress by the **exact number of ground truth symbols processed**.

1. **True Epoch Definition:** 1 Epoch is defined as processing the total number of ground truth symbols in the entire COCO annotation file.
2. **Total Budget:** `Total Symbol Budget = Total Dataset Symbols * Epochs`.
3. **Dynamic Progress:** At any point, `progress = cumulative_symbols_seen / Total Symbol Budget`.
4. **Schedule:** We apply a Linear Warmup for the first 5% of the budget, followed by a Cosine Decay for the remaining 95%.

This guarantees that the optimizer experiences the exact same learning rate curve relative to the semantic data it has seen, regardless of the hardware, batch size, or crop resolution.
