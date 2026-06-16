# Experiment Log

## Experiment 001: Single Image Overfit (Baseline)
* **Experiment Name/ID**: `experiments/001_single_image_overfit`
* **Hypothesis/Goal**: Verify that the `vit_nano` OMRDetector can successfully overfit a single, large image crop from the Trompa-COCO dataset, establishing that the loss functions, matcher, and gradients are working correctly end-to-end.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16)
  * Crop Size: 224x224
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/001_single_image_overfit`
* **Results**: The model successfully overfit the training data in terms of classification loss (`loss_ce` dropped from 4333.4 to 1.28, and `loss_total` dropped from 4341.5 to ~3.86). However, the Mean Average Precision (`mAP@0.5`) remained extremely low, ending at ~0.017. Visualizations showed the model predicting overlapping bounding boxes clustered near the center intersections of the image, failing to localize precisely.
* **Conclusion**: The experiment verified that the architecture, loss functions, and gradients are working end-to-end (as evidenced by the loss converging). The poor localization was identified as a bug in the dynamic anchor initialization (`softplus` activation causing massive default anchors covering ~70% of the image), which squashed the FGL target residuals and prevented the network from learning fine-grained edge offsets. This bug was subsequently fixed in commit `b2943e6`.

## Experiment 002: Single Image Overfit (Fixed Anchors)
* **Experiment Name/ID**: `experiments/002_single_image_overfit_fixed_anchors`
* **Hypothesis/Goal**: Verify that the inverse softplus initialization fix for `base_anchor_size` resolves the localization issues seen in Experiment 001, allowing the model to achieve a high mAP@0.5 and visually tight bounding boxes.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16)
  * Crop Size: 224x224
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/002_single_image_overfit_fixed_anchors --base_anchor_size 0.0125`
* **Results**: The model successfully learned to classify objects (`loss_ce` dropped from ~4694 to ~1.28), but bounding box regression completely stalled. `loss_bbox` and `loss_giou` remained stuck around 2.7-2.9, and `mAP@0.5` stayed flat at 0.0000.
* **Conclusion**: Fixing the initialization revealed a deeper numerical instability. Because the network was predicting in Image Units (IU), it struggled to output the microscopic values needed to adjust the tiny 1.25% anchors. Furthermore, the FGL target residuals became massive and were heavily clamped to the extreme edge bins, providing poor learning signals. The network needs to operate in a normalized reference frame (Patch Units) to maintain healthy gradients.

## Experiment 003: Single Image Overfit (Patch Units)
* **Experiment Name/ID**: `experiments/003_single_image_overfit_patch_units`
* **Hypothesis/Goal**: Verify that predicting bounding box coordinates and shapes in Patch Units (PU) instead of Image Units (IU) resolves the numerical instability, allowing the model to successfully regress bounding boxes and achieve a high mAP@0.5.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16)
  * Crop Size: 224x224
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/003_single_image_overfit_patch_units --base_anchor_size 1.0`
* **Results**: The model successfully learned to classify objects (`loss_ce` dropped to ~1.28) and the regression losses improved significantly compared to Experiment 002 (`loss_bbox` dropped to ~0.39, `loss_giou` to ~1.5). However, `mAP@0.5` remained very low (peaking around 0.16 and ending at ~0.017). Visualizations revealed that the predicted boxes perfectly hugged the short dimension (thickness) of the staff lines but failed to extend along the long dimension.
* **Conclusion**: The Patch Units conversion successfully stabilized the gradients, allowing the network to learn local offsets. However, a mathematical bottleneck was discovered: the FGL bins are limited to `[-0.5, 0.5]` of the anchor size. With a base anchor of 1.0 patch, the maximum predicted box size is 2.0 patches. Since staff lines span the entire 14-patch width of the image, the network hit a hard mathematical wall and could not stretch the boxes enough.

## Experiment 004: Single Image Overfit (Dynamic Shapes)
* **Experiment Name/ID**: `experiments/004_single_image_overfit_dynamic_shapes`
* **Hypothesis/Goal**: Verify that dynamically predicting the base shapes (width and height) per patch using the MLP, rather than relying on static global anchors, will allow the network to bypass the FGL expansion limits and successfully regress highly elongated bounding boxes (like staff lines), leading to a high mAP@0.5.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16) with dynamic shape prediction in `DFINEDenseHead`.
  * Crop Size: 224x224
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/004_single_image_overfit_dynamic_shapes --base_anchor_size 1.0`
* **Results**: The model successfully learned to classify and localize the objects. `loss_total` dropped to ~3.0, with `loss_bbox` reaching ~0.025 and `loss_giou` reaching ~0.22. Crucially, `mAP@0.5` climbed to ~0.9222 and the newly introduced `mIoU` metric reached ~0.94, indicating that the predicted boxes smoothly and accurately expanded to cover the ground truth objects, including the long staff lines.
* **Conclusion**: Dynamically predicting the base shapes (width and height) per patch completely resolved the FGL expansion bottleneck. The network is no longer constrained by the `[-0.5, 0.5]` limit relative to a static 1-patch anchor, allowing it to successfully regress highly elongated bounding boxes that span across the entire image. The architecture is now mathematically capable of handling the extreme aspect ratios present in Optical Music Recognition.

## Experiment 005: Single Image Overfit (Scale Up - 448x448)
* **Experiment Name/ID**: `experiments/005_single_image_overfit_scale_448`
* **Hypothesis/Goal**: Verify that the `vit_nano` OMRDetector can still successfully overfit a single image when the crop size is doubled from 224 to 448. This tests the scalability of the dynamic shape prediction and checks for any memory or gradient instability issues at larger resolutions.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16)
  * Crop Size: 448x448
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/005_single_image_overfit_scale_448 --crop_size 448 --base_anchor_size 1.0`
* **Results**: The model successfully overfit the 448x448 image. The training dynamics were interesting: it initially fit symbols and lines at their centers, but quickly shared boxes for the same symbol label (e.g., cut staff lines shared boxes, keeping them too short). It then plateaued on box fitting to focus on lowering the classification loss. Finally, after some time, it resumed fine-tuning the boxes to match the ground truth exactly.
* **Conclusion**: The architecture scales well to 448x448 without memory or gradient issues. The two-phase learning dynamic (center/class grouping first, fine-grained boundary regression later) suggests the matcher and loss functions are prioritizing classification and rough localization before committing to exact shape refinement.

## Experiment 006: Single Image Overfit (Scale Up - 896x896)
* **Experiment Name/ID**: `experiments/006_single_image_overfit_scale_896`
* **Hypothesis/Goal**: Verify that the `vit_nano` OMRDetector can scale to an 896x896 crop size and still successfully overfit a single image. This will further test the memory limits and ensure the two-phase learning dynamics observed in Experiment 005 still converge at higher resolutions.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16)
  * Crop Size: 896x896
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/006_single_image_overfit_scale_896 --crop_size 896 --base_anchor_size 1.0`
* **Results**: The model successfully overfit the 896x896 image, though it struggled slightly with ties, the middle of a barline, and a cut bargroup. VRAM usage was extremely low (~650MB) due to variance-based patch dropping, which dropped ~85-90% of patches, reducing self-attention memory by ~97.7%. Convergence was noticeably slower than previous experiments.
* **Conclusion**: The architecture scales exceptionally well in terms of memory thanks to patch dropping. The slower convergence is due to the effective batch size increasing (more ground truth boxes per crop), which dilutes the gradient per box. This necessitates scaling the learning rate linearly with the crop area.

## Experiment 007: Single Image Overfit (Scale Up - 1792x1792)
* **Experiment Name/ID**: `experiments/007_single_image_overfit_scale_1792`
* **Hypothesis/Goal**: Verify that the model can scale to 1792x1792. Test the new `base_lr` scaling rule to see if it restores the convergence speed observed at smaller crop sizes by dynamically adjusting the learning rate based on the crop area.
* **Setup**: 
  * Model: `vit_nano` (patch_size=16)
  * Crop Size: 1792x1792
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/007_single_image_overfit_scale_1792 --crop_size 1792 --base_anchor_size 1.0 --base_lr 1e-4`
* **Results**: The model successfully overfit the 1792x1792 image. The dynamic learning rate scaling (`base_lr` = 1e-4, effective LR = 6.4e-3) resulted in extremely fast convergence, with the total loss dropping from ~14147 to ~4.2 in just 130 steps. VRAM consumption was only 2.2 GB, which is remarkably low for such a massive resolution.
* **Conclusion**: The linear scaling rule for the learning rate based on crop area successfully restored (and even accelerated) convergence speed. The variance-based patch dropping continues to prove highly effective, keeping VRAM at 2.2 GB for a 1792x1792 image. This concludes the single-image overfit scaling track, as the architecture, loss functions, and scaling rules are now fully validated end-to-end.

## Experiment 008: Single Image Overfit (Scale Up - 3584x3584, Patch 64)
* **Experiment Name/ID**: `experiments/008_single_image_overfit_scale_3584_patch_64`
* **Hypothesis/Goal**: Verify that doubling the crop size to 3584x3584 while increasing the patch size to 64x64 keeps the VRAM consumption stable and avoids OOM errors. This tests if we can process massive resolutions on a 4GB GPU by trading off spatial granularity (larger patches) to reduce the token sequence length and intermediate activation sizes.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64)
  * Crop Size: 3584x3584
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/008_single_image_overfit_scale_3584_patch_64 --crop_size 3584 --patch_size 64 --base_anchor_size 1.0 --base_lr 1e-4`
* **Results**: The model failed to converge properly. While the total loss dropped initially, it plateaued around 3.4, and the `mAP@0.5` remained at exactly 0.0000 throughout the run. The effective learning rate was scaled to 2.56e-02 due to the linear scaling rule.
* **Conclusion**: The linear scaling rule (multiplying LR by area ratio) is designed for SGD, not AdamW. AdamW normalizes gradients by their variance, making it naturally scale-invariant. Manually scaling the LR by 256x caused the optimizer to take massive, destructive steps, preventing the network from learning the fine-grained offsets required for the coarse 64x64 patch grid.

## Experiment 009: Single Image Overfit (Scale Up - 3584x3584, Patch 64, Symbol Budget LR)
* **Experiment Name/ID**: `experiments/009_single_image_overfit_scale_3584_patch_64_symbol_budget`
* **Hypothesis/Goal**: Verify that the new "Symbol Budget" LR scheduler (Linear Warmup + Cosine Decay based on the exact number of ground truth symbols processed) allows the model to successfully overfit the 3584x3584 crop with a 64x64 patch size using AdamW.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64)
  * Crop Size: 3584x3584
  * Data: A single image batch repeated infinitely (`repeat(batch)`).
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/009_single_image_overfit_scale_3584_patch_64_symbol_budget --crop_size 3584 --patch_size 64 --base_anchor_size 1.0 --lr 1e-4 --epochs 1000`
* **Results**: The Symbol Budget LR scheduler worked perfectly, smoothly warming up to 1e-4 and decaying to ~2e-8 over the 1000 epochs (365,000 symbols). Total loss dropped from ~2603 to ~12.7, and CE loss dropped significantly, showing the model learned to classify objects. However, localization metrics plateaued: mIoU reached ~0.28 and mAP@0.5 ended at ~0.0093. Visually, the bounding boxes were reasonable, but the strict 0.5 IoU threshold is highly unforgiving for thin objects (4px staff lines, 5px stems) when predicted from a coarse 64x64 patch grid.
* **Conclusion**: The engineering components (patch dropping, dynamic shapes, symbol budget scheduler) are fully validated and working as intended. The poor mAP at patch size 64 is a spatial resolution limitation—predicting pixel-perfect boundaries for thin objects from massive 64x64 patches requires longer training and careful LR tuning. Since the primary goal of the overfitting track (sanity checking the architecture and scaling mechanisms) has been achieved, we will conclude this track here rather than over-optimizing hyperparameters for a single image.

## Experiment 010: Full Dataset Training Baseline
* **Experiment Name/ID**: `experiments/010_full_dataset_baseline`
* **Hypothesis/Goal**: Transition from single-image overfitting to training on the entire Trompa-COCO dataset. Establish a baseline for full-dataset training, verifying that the data pipeline, symbol budget scheduler, and model scale correctly to diverse images and generalize across the dataset.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64)
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset, iterating over all images with a shuffle buffer.
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/010_full_dataset_baseline --patch_size 64 --epochs 10 --use_sdpa --use_amp --prep_device cuda:0 --train_device cuda:1 --match_device cuda:1`
* **Results**: The pipeline successfully processed the full dataset without OOM errors, validating the lazy-loading index strategy and the memory efficiency of AMP and SDPA. The Symbol Budget LR scheduler worked perfectly. The model learned effectively, with total loss dropping from ~1988 to ~4.28 (driven mostly by CE loss dropping to ~1.81). Localization improved, with mIoU climbing to ~0.4889. However, `mAP@0.5` remained low at ~0.0181. Processing speed improved to ~1.0 samples/s.
* **Conclusion**: The full-dataset pipeline, dynamic shapes, patch dropping, and custom LR scheduler work seamlessly at scale. Using full images with AMP and SDPA improved throughput. However, `patch_size=64` is still too coarse for the strict 0.5 IoU threshold required for thin music symbols (staff lines, stems). To achieve high mAP, a smaller patch size (16 or 32) is required, which will necessitate a smaller crop size (896 or 1792) to maintain memory/speed efficiency. This concludes the detection scaling and baseline track.

## Experiment 011: Full Dataset Checkpoint (Fixes & DDP)
* **Experiment Name/ID**: `experiments/011_full_dataset_fixes_and_ddp`
* **Hypothesis/Goal**: This is a checkpoint experiment to validate a series of critical bug fixes and infrastructure improvements made since Experiment 010. Specifically, we want to verify that:
  1. **Tie Label Fix**: Merging the 308 buggy `tie` sub-categories into a single class restores the overall mAP calculation.
  2. **L1 Loss Fix**: Computing the L1 bounding box loss in `CXCYWH` format (instead of `XYXY`) provides better gradient signals for center localization.
  3. **Focal Loss Initialization**: Setting the initial classification bias based on a prior probability prevents massive early loss spikes and stabilizes early training.
  4. **DDP Scaling**: Training on 2 GPUs to smooth out extreme symbol count variances and speed up training.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64)
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset (with cleaned `tie` annotations).
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/011_full_dataset_fixes_and_ddp \
        --patch_size 64 \
        --epochs 10 \
        --anno_path ../input/datasets/kwonyoungchoi/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir ../input/datasets/kwonyoungchoi/trompa-coco/trainval2017 \
        --headless \
        --cache_dir /kaggle/temp/cache/ \
        --use_sdpa \
        --compile \
        --log_epoch_interval 0.5
    ```
    *(Note: `--use_amp` disabled, see note below).*

### Interim Note: AMP (FP16) Numerical Instability
An initial attempt at this experiment using `--use_amp` (FP16) combined with gradient accumulation yielded extremely poor results. This is highly likely due to FP16's limited precision (11 bits of mantissa). In OMR, a 4-pixel thick staff line on a 3584x3584 image has a normalized dimension of `4 / 3584 ≈ 0.0011`. When the network regresses fine-grained edge offsets or computes IoU for these microscopic values, FP16 suffers from catastrophic cancellation and underflow. Dividing the loss by `accumulation_steps` exacerbates this by pushing gradients even closer to the underflow limit. The official run for Exp 011 will proceed in pure FP32.

* **Results**: The experiment successfully completed all 10 epochs. While the in-training batch-level `mAP@0.5` peaked at ~0.2356, the official full-dataset `pycocotools` evaluation yielded a global `mAP@0.5` of **0.047**. This discrepancy is due to `pycocotools` macro-averaging across all ~70 categories: over 40 rare or tiny classes scored 0.0000, heavily penalizing the global average. However, performance on common, distinct symbols was excellent: `noteheadBlack` (**0.9296**), `gClef` (**0.8549**), `stem` (**0.4428**), `accidentalFlat` (**0.4003**), and `fClef` (**0.3645**).
* **Conclusion**: The bug fixes and infrastructure improvements were highly successful. Merging the buggy `tie` categories and fixing the L1 loss format allowed the model to learn meaningful localizations, as evidenced by the >0.85 AP on noteheads and clefs. The low global mAP is primarily a reflection of the dataset's long-tail distribution (rare classes) and the difficulty of localizing extremely thin/tiny symbols (like dots and ties) at the current 64x64 resolution. DDP provided excellent throughput.

## Experiment 012: Log-Space Shape Prediction
* **Experiment Name/ID**: `experiments/012_log_space_shapes`
* **Hypothesis/Goal**: Verify that predicting bounding box width and height in log-space (using `exp` instead of `softplus`) improves the network's ability to regress extreme aspect ratios (like long staff lines) and tiny objects (like dots) by providing better relative precision and scaling.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with log-space shape prediction in `DFINEDenseHead`.
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/012_log_space_shapes \
        --patch_size 64 \
        --epochs 10 \
        --anno_path ../input/datasets/kwonyoungchoi/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir ../input/datasets/kwonyoungchoi/trompa-coco/trainval2017 \
        --headless \
        --cache_dir /kaggle/temp/cache/ \
        --use_sdpa \
        --compile \
        --log_epoch_interval 0.5
    ```
* **Results**: The training metrics showed noticeable improvement in localization: `loss_bbox` dropped to 0.092 (from 0.126 in Exp 011) and `loss_fgl` dropped to 1.246 (from 1.410). The in-training batch `mAP@0.5` peaked higher at 0.262. The official `pycocotools` global `mAP@0.5` increased from 0.047 to **0.0577**. We saw massive jumps in specific classes: `fClef` (0.3645 -> 0.9479), `noteheadBlack` (0.9296 -> 0.9437), `accidentalSharp` (0.1691 -> 0.2437), `flag8thUp` (0.0455 -> 0.1577), and `ledgerLines` (0.0817 -> 0.1577). However, extremely thin objects like `staff` lines dropped to 0.0000.
* **Conclusion**: Log-space shape prediction successfully improved overall localization and global mAP. It relieved the mathematical bottleneck on bounding box regression, allowing the network to scale predictions much more naturally. The trade-off observed on extremely thin objects (staff lines) suggests that while log-space is the correct mathematical approach for general shapes, predicting pixel-perfect boundaries for 4-pixel thick lines from a coarse 64x64 patch grid remains fundamentally difficult. This reinforces the need for specialized representations (like keypoints) for lines, or higher spatial resolution.

## Experiment 013: Dual-Head Architecture for Symbols and Lines
* **Experiment Name/ID**: `experiments/013_dual_head_lines_and_symbols`
* **Hypothesis/Goal**: Verify that the new dual-head architecture (separating symbols as bounding boxes and lines as keypoints) resolves the mathematical bottleneck for extremely thin/long objects (like staff lines and stems). By using the Signed Cartesian + Log Scale formulation for lines, the network should be able to regress extreme aspect ratios without destroying the GIoU metric, leading to a higher mAP for both modalities.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/013_dual_head_lines_and_symbols \
        --patch_size 64 \
        --epochs 10 \
        --anno_path ../input/datasets/kwonyoungchoi/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir ../input/datasets/kwonyoungchoi/trompa-coco/trainval2017 \
        --headless \
        --cache_dir /kaggle/temp/cache/ \
        --use_sdpa \
        --compile \
        --log_epoch_interval 0.5
    ```
* **Results**: [TBD - Run the experiment and evaluate using `scripts/evaluate_coco.py`]
* **Conclusion**: [TBD]
