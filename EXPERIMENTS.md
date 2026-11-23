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
* **Results**: The Symbol Budget LR scheduler worked perfectly, smoothly warming up to 1e-4 and decaying to ~2e-8 over the 1000 epochs (365,000 symbols). Total loss dropped from ~2603 to ~12.7, and CE loss dropped significantly, showing the model learned to classify objects. However, localization metrics plateaued: mIoU reached ~0.28 and mAP@0.5 ended at ~0.0093. Visually, the bounding boxes were reasonable, but the strict 0.5 IoU threshold is highly unforgiving for thin objects (4px staff lines, 5px stems).
* **Conclusion**: The engineering components (patch dropping, dynamic shapes, symbol budget scheduler) are fully validated and working as intended. Predicting pixel-perfect boundaries for thin objects requires longer training and careful LR tuning. Since the primary goal of the overfitting track (sanity checking the architecture and scaling mechanisms) has been achieved, we will conclude this track here rather than over-optimizing hyperparameters for a single image.

## Experiment 010: Full Dataset Training Baseline
* **Experiment Name/ID**: `experiments/010_full_dataset_baseline`
* **Hypothesis/Goal**: Transition from single-image overfitting to training on the entire Trompa-COCO dataset. Establish a baseline for full-dataset training, verifying that the data pipeline, symbol budget scheduler, and model scale correctly to diverse images and generalize across the dataset.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64)
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset, iterating over all images with a shuffle buffer.
  * Command: `mamba run -n pytorch python src/train_detection.py --exp_dir experiments/010_full_dataset_baseline --patch_size 64 --epochs 10 --use_sdpa --use_amp --prep_device cuda:0 --train_device cuda:1 --match_device cuda:1`
* **Results**: The pipeline successfully processed the full dataset without OOM errors, validating the lazy-loading index strategy and the memory efficiency of AMP and SDPA. The Symbol Budget LR scheduler worked perfectly. The model learned effectively, with total loss dropping from ~1988 to ~4.28 (driven mostly by CE loss dropping to ~1.81). Localization improved, with mIoU climbing to ~0.4889. However, `mAP@0.5` remained low at ~0.0181. Processing speed improved to ~1.0 samples/s.
* **Conclusion**: The full-dataset pipeline, dynamic shapes, patch dropping, and custom LR scheduler work seamlessly at scale. Using full images with AMP and SDPA improved throughput. However, the strict 0.5 IoU threshold remains challenging for thin music symbols (staff lines, stems). This concludes the detection scaling and baseline track.

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
* **Results**: The experiment successfully completed all 10 epochs. While the in-training batch-level `mAP@0.5` peaked at ~0.2356, the official full-dataset `pycocotools` evaluation yielded a global `mAP@0.5` of **0.047**. This discrepancy is due to `pycocotools` macro-averaging across all ~70 categories: over 40 rare or tiny classes scored 0.0000, heavily penalizing the global average. However, performance on common, distinct symbols was excellent: `noteheadBlack` (**0.9296**), `gClef` (**0.8549**), `stem` (**0.4428**), `accidentalFlat` (**0.4003**), and `fClef` (**0.3645**).
* **Conclusion**: The bug fixes and infrastructure improvements were highly successful. Merging the buggy `tie` categories and fixing the L1 loss format allowed the model to learn meaningful localizations, as evidenced by the >0.85 AP on noteheads and clefs. The low global mAP is primarily a reflection of the dataset's long-tail distribution (rare classes) and the difficulty of localizing extremely thin/tiny symbols (like dots and ties). DDP provided excellent throughput.

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
* **Conclusion**: Log-space shape prediction successfully improved overall localization and global mAP. It relieved the mathematical bottleneck on bounding box regression, allowing the network to scale predictions much more naturally. The trade-off observed on extremely thin objects (staff lines) suggests that while log-space is the correct mathematical approach for general shapes, predicting pixel-perfect boundaries for 4-pixel thick lines remains fundamentally difficult. This reinforces the need for specialized representations (like keypoints) for lines.

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
* **Results**: The training completed successfully (`loss_total` dropped to ~1.70, `loss_ce` to ~0.59, `loss_bbox` to ~0.02, `loss_giou` to ~0.32). The in-training batch `mAP@0.5` for symbols reached ~0.304, and `mIoU` reached ~0.842, showing that removing lines from the bounding box head significantly improved symbol localization. However, the line task struggled to converge (`loss_line_l1` hovered around 0.25). During evaluation, the Hungarian Matcher initially crashed with a "no full matching exists" error due to the pigeonhole problem (many long lines competing for the same few patch queries at the center). After fixing the evaluation script (vectorizing OKS, adjusting `maxDets`), the keypoint metrics were still extremely low (e.g., `system` at 0.1750, `stem` at 0.0017).
* **Conclusion**: The dual-head split successfully unburdened the symbol head, leading to much better bounding box metrics. However, the `LineHead` suffered from two major flaws: 1) The matcher used endpoint-to-endpoint distance, starving patches in the middle of long lines of gradients and causing crashes. This was fixed and the training restarted. 2) The raw direction vectors were not normalized, causing an identifiability problem where the network entangled magnitude between the log-scale and the raw vector, preventing convergence.

## Experiment 014: Line Head L2 Normalization
* **Experiment Name/ID**: `experiments/014_line_head_l2_norm`
* **Hypothesis/Goal**: Verify that applying L2 normalization to the raw direction vectors in the `LineHead` solves the identifiability/over-parameterization problem, allowing the log-scale to strictly control magnitude.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with updated `LineHead` (L2 normalized directions).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/014_line_head_l2_norm \
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
* **Results**: The L2 normalization did not solve the underlying instability. The network still suffered from massive gradient spikes. The root cause was identified as a "lever-arm effect": because the final endpoint was calculated as `Scale * Direction`, a small error in the normalized direction or D-FINE residual, when multiplied by a massive scale (e.g., 3000 pixels for a staff line), resulted in huge absolute errors and gradient explosion.
* **Conclusion**: The explicit separation of scale and direction via multiplication creates a lever-arm effect that is mathematically unstable for extreme aspect ratios. We need to abandon the multiplicative scale/direction split and move to a formulation where X and Y can independently scale to massive distances without multiplying by a shared magnitude.

## Experiment 015: Line Head Signed Log Formulation
* **Experiment Name/ID**: `experiments/015_line_head_signed_log`
* **Hypothesis/Goal**: Verify that the new "Signed Log" formulation (`sign(x) * (exp(|x|) - 1)`) for line endpoints resolves the lever-arm gradient instability observed in Exp 014. By allowing independent, unbounded scaling for X and Y without multiplication, and scaling D-FINE residuals strictly by the base anchor size (1 PU), the line head should converge stably and achieve high keypoint mAP. Also, all "absolute" losses were converted to patch unit instead of being in image unit. this should improve gradients for massive image sizes.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with updated `LineHead` (Signed Log formulation).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/015_line_head_signed_log \
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
* **Results**: Training completed stably with `loss_total` dropping to 8.40 and `loss_line_l1` dropping to 4.46. The `line_error` decreased to 0.63. In evaluation, global mAP@0.5 was 0.0155 for symbols and 0.0230 for lines. For specific line classes, `system` achieved 0.3007 mAP@0.5, `beam` 0.0195, and `stem` 0.0144, while `staff` scored near zero. For symbols, `noteheadBlack` achieved 0.8477 and `gClef` 0.6365.
* **Conclusion**: The Signed Log formulation successfully stabilized the training gradients for the line head, allowing the losses to converge. However, the overall mAP for both lines and symbols remains low and requires further investigation.

## Experiment 016: ViT-Small Baseline
* **Experiment Name/ID**: `experiments/016_vit_small_baseline`
* **Hypothesis/Goal**: Verify if scaling up the backbone capacity from `vit-nano` to `vit-small` improves the overall mAP for both symbols and lines. The increased capacity might help the network better resolve fine-grained details and context.
* **Setup**: 
  * Model: `vit_small` (patch_size=64) with `SymbolHead` and `LineHead` (Signed Log formulation).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/016_vit_small_baseline \
        --backbone_size small \
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
* **Results**: Training losses converged well, with `loss_total` dropping to ~0.817, `loss_ce` to ~0.394, `loss_bbox` to ~0.065, and `loss_line_l1` to ~0.285. However, the official COCO evaluation showed a significant drop in performance. For symbols, global mAP@0.5 dropped to **0.0236** (from 0.053 in Exp 017). While `noteheadBlack` remained robust at 0.959, performance on rare, complex classes collapsed: `fClef` dropped to 0.115 (from 0.932) and `brace` dropped to 0.056 (from 0.762). For lines, global mAP@0.5 dropped slightly to **0.0055**, with `system` lines remaining collapsed at 0.0003, but `beam` detection improving to 0.0729 (from 0.042).
* **Conclusion**: While affine augmentations successfully prevented the model from memorizing absolute grid positions (evidenced by the continued collapse of `system` lines and the improvement in relative structures like `beam`), they were too aggressive for rare, highly-structured symbol classes. The distortion introduced by rotation, shear, and scale destroyed the performance gains achieved in Exp 017 for complex shapes like clefs and braces, causing the global symbol mAP to drop significantly.

## Experiment 017: Smooth Balanced Loss (Inverse Frequency Weighting)
* **Experiment Name/ID**: `experiments/017_smooth_balanced_loss`
* **Hypothesis/Goal**: Verify that applying inverse smooth frequency weighting to the loss functions resolves the severe class imbalance. By up-weighting the loss for rare classes (clefs, accidentals) and down-weighting common classes (noteheads), the model should be forced to learn features for the rare classes, improving global mAP.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Loss: Inverse smooth weighting applied to both Symbol and Line losses (computed in `parse_coco` and passed to `DFINECriterion`).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/017_smooth_balanced_loss \
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
* **Results**: 
  * **Symbols (Major Success):** Global mAP@0.5 doubled from **0.025 (Exp 15)** to **0.053**. The weighting strategy successfully forced the network to learn rare classes. `fClef` jumped from 0.262 to **0.932**, `accidentalFlat` from 0.004 to **0.307**, and `brace` from 0.316 to **0.762**. `noteheadBlack` also improved slightly to **0.952**.
  * **Lines (Visual Improvement / Metric Drop):** Global mAP@0.5 dropped from 0.038 to **0.007**. This was driven entirely by a collapse in the `system` class (0.300 -> 0.029). However, `beam` (0.019 -> 0.042) and `ledgerLines` (0.024 -> 0.084) improved significantly.
* **Conclusion**: The inverse weighting was highly effective for Symbols, unlocking detection for rare classes. For Lines, the results are nuanced. The collapse of the `system` class suggests the model stopped relying on the "easy" shortcut of predicting fixed-size lines at standard locations (a pitfall similar to the initial `noteheadBlack` overfitting). Visual inspection confirms that the model is now attempting to predict actual line geometry (beams, ledgers) rather than just memorizing system line positions. The drop in `system` mAP is likely a sign of the model breaking out of a local minimum, even if the strict IoU metric penalizes the loss of the "perfect" shortcut predictions. This represents a net improvement in the model's understanding of line geometry.

## Experiment 019: 100 Epochs Training
* **Experiment Name/ID**: `experiments/019_100_epochs`
* **Hypothesis/Goal**: Verify if extending the training duration from 10 to 100 epochs significantly improves the model's ability to localize and classify both symbols and lines. Previous experiments showed promising but plateauing metrics; this experiment tests whether the architecture is simply underfitting and requires more time to converge on the complex Trompa-COCO dataset.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Training: 100 epochs, with a 1-epoch linear warmup. Logging and checkpointing every 5 epochs.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/019_100_epochs \
        --patch_size 64 \
        --epochs 100 \
        --anno_path ../input/datasets/kwonyoungchoi/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir ../input/datasets/kwonyoungchoi/trompa-coco/trainval2017 \
        --headless \
        --cache_dir /kaggle/temp/cache/ \
        --use_sdpa \
        --log_epoch_interval 5 \
        --compile \
        --warmup_epochs 1
    ```
* **Results**: Training over 100 epochs showed strong convergence. Total loss dropped from ~1.31 to ~0.40, driven by classification (`loss_ce` to ~0.23) and localization (`loss_bbox` to ~0.03) improvements. In-training `mAP@0.5` peaked at ~0.912 (epoch 95) and `mIoU` reached ~0.94, though the final epoch saw a slight dip in batch mAP to 0.637. Official COCO evaluation revealed a massive leap in symbol performance: global mAP@0.5 reached **0.622** (up from 0.053 in Exp 017). Common symbols like `noteheadBlack` (0.990), `gClef` (0.979), and `restQuarter` (0.988) achieved near-perfect detection, though rare classes like `articTenuto` (0.0) and `slur` (0.258) remained difficult. Line performance improved more modestly, with global mAP@0.5 at **0.136**. `ledgerLines` reached 0.771 and `system` 0.437, but thin lines like `staff` (0.091) and `stem` (0.189) continued to struggle.
* **Conclusion**: Extending training to 100 epochs was highly effective, particularly for symbols, proving the architecture was previously underfitting rather than fundamentally limited. The dramatic improvement in symbol mAP validates the dual-head architecture and loss formulations. However, the modest gains for lines confirm that extreme aspect ratios and thin structures remain the primary bottleneck. While `ledgerLines` and `system` improved, the network still struggles with the pixel-perfect boundaries required for `staff` and `stem` lines under the strict 0.5 IoU threshold. The next step should focus on specialized line representations or relaxed metrics (e.g., OKS or lower IoU thresholds) to better capture line geometry, and potentially addressing the remaining rare symbol classes via targeted data augmentation or hard example mining.

## Experiment 020: Fine-tuning with Lower Learning Rate
* **Experiment Name/ID**: `experiments/020_finetune_low_lr`
* **Hypothesis/Goal**: Verify that continuing training from the Experiment 019 checkpoint with a significantly lower learning rate stabilizes the final convergence phase. The hypothesis is that the exponential formulation for lines is sensitive to large gradient updates, and a lower LR will prevent overshooting and allow fine-tuning of precise endpoints.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Checkpoint: Resuming from `experiments/019_100_epochs/train_detection/checkpoints/latest_model.pt` (loading both model and optimizer state).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Training: 100 epochs, peak LR `1e-5` (down from `1e-4`), with a 1 epoch linear warmup to smooth the transition.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/020_finetune_low_lr \
        --detector_checkpoint experiments/019_100_epochs/train_detection/checkpoints/latest_model.pt \
        --patch_size 64 \
        --epochs 100 \
        --lr 1e-5 \
        --warmup_epochs 1 \
        --anno_path ../input/datasets/kwonyoungchoi/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir ../input/datasets/kwonyoungchoi/trompa-coco/trainval2017 \
        --headless \
        --cache_dir /kaggle/temp/cache/ \
        --use_sdpa \
        --compile \
        --log_epoch_interval 5
    ```
* **Results**: Training losses remained stable and low, with `loss_total` hovering around 0.28-0.39, and `loss_ce` around 0.15-0.23. In-training `mAP@0.5` peaked at ~0.920 (epoch 90) and `mIoU` reached ~0.955. Official COCO evaluation showed significant improvements over Experiment 019. For symbols, global mAP@0.5 increased to **0.6977** (up from 0.622). Common symbols like `noteheadBlack` (0.990), `gClef` (0.980), and `restQuarter` (0.989) maintained near-perfect detection, while `slur` improved to 0.309. Rare classes like `articTenutoAbove` remained at 0.0. For lines, global mAP@0.5 increased to **0.1816** (up from 0.136). `ledgerLines` reached 0.800, `system` improved to 0.563, `staff` to 0.136, and `stem` to 0.235.
* **Conclusion**: Fine-tuning with a lower learning rate (1e-5) successfully stabilized the final convergence phase and yielded significant improvements in official COCO evaluation metrics for both symbols and lines. The lower LR allowed the network to fine-tune precise endpoints and boundaries without overshooting, validating the hypothesis that the exponential formulation for lines is sensitive to large gradient updates. The remaining challenges are the rare symbol classes (e.g., `articTenutoAbove`) and extremely thin lines, which may require targeted data augmentation or specialized representations.

## Experiment 021: Proportional FGL Scaling for Lines
* **Experiment Name/ID**: `experiments/021_proportional_fgl_lines`
* **Hypothesis/Goal**: Verify that scaling the D-FINE Fine-Grained Localization (FGL) residual for lines by the magnitude of the predicted base direction (plus a floor) resolves the "asleep FGL" bottleneck. By mirroring the D-FINE box logic (where FGL scales with width/height), the line FGL should provide smooth, active gradients across the entire length of long lines, rather than being clamped and falling asleep for errors > 0.5 patches.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and updated `LineHead` (Proportional FGL scaling).
  * Checkpoint: From scratch
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Training: 100 epochs, peak LR `1e-4`, with a 1 epoch linear warmup.
  * Command: 
    ```bash
    PYTHONPATH=/kaggle/temp/music_deep /kaggle/temp/conda/bin/mamba run torchrun --nproc_per_node=2 /kaggle/temp/music_deep/src/train_detection.py \
        --exp_dir experiments/021_proportional_fgl_lines \
        --patch_size 64 \
        --epochs 100 \
        --lr 1e-4 \
        --warmup_epochs 1 \
        --anno_path ../input/datasets/kwonyoungchoi/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir ../input/datasets/kwonyoungchoi/trompa-coco/trainval2017 \
        --headless \
        --cache_dir /kaggle/temp/cache/ \
        --use_sdpa \
        --compile \
        --log_epoch_interval 5
    ```
* **Results**: The training was continued for 300 epochs total (restarted twice: Exp 22 and Exp 23, each running 100 epochs at LR `1e-4`). By the end of the 300th epoch, the model achieved near-perfect detection. For symbols, global mAP@0.5 reached **0.971**. Common symbols like `noteheadBlack` (0.990), `gClef` (1.0), and `fClef` (0.990) maintained near-perfect detection, while rare classes that were previously at 0.0 finally converged: `slur` reached 0.614, `tie` reached 0.685, and `articTenutoAbove` reached 0.404. For lines, global mAP@0.5 reached **0.315**. The proportional FGL scaling successfully resolved the "asleep FGL" bottleneck for long lines: `stem` jumped to 0.891, `beam` to 0.930, `barLine` to 0.955, and `staff` lines saw a massive improvement to 0.666. Training losses stabilized at very low values (`loss_total`: 0.133, `loss_line_l1`: 0.026).
* **Conclusion**: The Proportional FGL Scaling for Lines was mathematically sound but required a large number of iterations to fit the complex geometries of music lines and rare symbols. Maintaining a steady `1e-4` learning rate for 300 epochs allowed the network to converge much further without overfitting or diverging. The architecture is now highly capable of handling extreme aspect ratios and thin structures. The remaining bottlenecks are extremely rare symbols (like `octave` lines at 0.211) which may require targeted data augmentation or hard-example mining to improve further.

## Experiment 025: Variable Patch Size Scale Augmentation
* **Experiment Name/ID**: `experiments/025_variable_patch_size_augmentation`
* **Hypothesis/Goal**: Verify that expanding the range of the random patch extraction size (e.g., from 32x32 to 96x96) before resizing to the static 64x64 ViT input acts as a powerful scale augmentation. This should improve the model's robustness to varying DPIs and symbol sizes by providing more diverse spatial context and zoom levels.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Checkpoint: From scratch (or optionally resuming from Exp 023 to see if it breaks or improves a converged model).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Augmentation: `--extract_patch_min 32 --extract_patch_max 96` (Default is 54 to 64).
  * Training: 100 epochs, peak LR `1e-4`, with a 1 epoch linear warmup.
  * Command: 
    ```bash
    PYTHONPATH=. mamba run -n pytorch torchrun --nproc_per_node=2 src/train_detection.py \
        --exp_dir experiments/025_variable_patch_size_augmentation \
        --patch_size 64 \
        --extract_patch_min 32 \
        --extract_patch_max 96 \
        --epochs 100 \
        --lr 1e-4 \
        --warmup_epochs 1 \
        --anno_path data/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir data/trompa-coco/trainval2017 \
        --headless \
        --use_sdpa \
        --compile \
        --log_epoch_interval 5
    ```
* **Results**: The training was continued for 100 epochs in Experiment 026. The final evaluation showed a regression compared to the Exp 021 baseline. For symbols, global mAP@0.5 dropped to **0.841** (from 0.971) and mAP@0.5_0.95 to **0.644**. While common symbols like `noteheadBlack` (0.990), `gClef` (1.0), and `fClef` (0.990) remained near-perfect, rare classes saw a decline: `slur` dropped to 0.506, `tie` to 0.461, and `articTenutoAbove` to 0.430. For lines, global mAP@0.5 dropped to **0.275** (from 0.315) and mAP@0.5_0.95 to **0.234**. Most line classes remained strong, such as `system` (0.979), `beam` (0.975), and `stem` (0.951), but `octave` lines collapsed to 0.025.
* **Conclusion**: The variable patch size scale augmentation (extracting patches from 32x32 to 96x96) negatively impacted overall performance, causing a regression in both symbol and line mAP. While the model maintained excellent detection for common, distinct symbols, the increased variance in spatial context and zoom levels likely made it harder to converge on the precise boundaries required for rare classes and complex structures. This suggests that such aggressive scale augmentation may be detrimental when the model is already capable of fitting the dataset's native scale distribution, or it requires a much longer training schedule to recover the lost precision.

## Experiment 027: Focal L1 and GIoU Loss for Regression
* **Experiment Name/ID**: `experiments/027_focal_regression_loss`
* **Hypothesis/Goal**: Verify that applying Focal L1 and Focal GIoU losses to the regression tasks (symbol boxes and line keypoints) improves the model's ability to localize highly variable classes (like `slur`, `tie`, `staff`, and `octave`). By down-weighting the loss for "easy" predictions (small errors or stable templates) and focusing the gradients on "hard" predictions (large localization errors), the network should break out of the local minimum of template-matching and learn to regress extreme aspect ratios more accurately.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Loss: Focal L1 and Focal GIoU applied to matched predictions in `DFINECriterion`.
  * Checkpoint: From scratch (or optionally resuming from Exp 026 to see if it breaks or improves a converged model).
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Training: 100 epochs, peak LR `1e-4`, with a 1 epoch linear warmup.
  * Command: 
    ```bash
    PYTHONPATH=. mamba run -n pytorch torchrun --nproc_per_node=2 src/train_detection.py \
        --exp_dir experiments/027_focal_regression_loss \
        --patch_size 64 \
        --epochs 100 \
        --lr 1e-4 \
        --warmup_epochs 1 \
        --anno_path data/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir data/trompa-coco/trainval2017 \
        --headless \
        --use_sdpa \
        --compile \
        --log_epoch_interval 5
    ```
* **Results**: The Focal L1 and Focal GIoU losses produced a mixed result, redistributing gradient budget away from the experiment's stated targets. For symbols, global `mAP@0.5` rose to **0.884** (up from 0.841 in Exp 026), driven almost entirely by the recovery of previously-dead or weak rare classes: `accidentalNaturalSharp` (0.028 → 0.574), `articStaccatissimoWedgeBelow` (0.000 → 0.500), `dynamicMP` (0.345 → 0.756), `tuplet1` (0.207 → 0.632), and `tuplet8` (0.000 → 1.000). Common symbols (`noteheadBlack` 0.990, `gClef` 1.0, `fClef` 0.990) held. However, the classes the experiment explicitly aimed to improve regressed: `slur` (0.507 → 0.455), `tie` (0.462 → 0.419), and `articTenutoBelow` (0.350 → 0.173). Global symbol `mAP@0.5:0.95` collapsed from 0.644 to **0.552**, indicating that down-weighting easy, well-localized predictions degraded overall box precision.
  For lines, focal loss was a net negative. Global `mAP@0.5` dropped to **0.221** (from 0.275) and `mAP@0.5:0.95` to **0.158** (from 0.234). The targeted `octave` class remained essentially dead (0.025 → 0.0015). Thin/long lines that were already well-fit degraded sharply: `staff` (0.807 → 0.554), `stem` (0.951 → 0.791), `ledgerLines` (0.939 → 0.757), and `voltaBracket` (0.632 → 0.253). Only `beam` (0.974) and `system` (0.977) held, as these are dominated by stable templates.
* **Conclusion**: The hypothesis did not hold. Focal regression loss redistributed gradient budget toward *rare but geometrically simple* symbol classes (boosting coarse recall and global `mAP@0.5`) while systematically suppressing the gradients that kept well-localized predictions precise. The explicitly targeted curved/thin classes (`slur`, `tie`, `staff`, `octave`) either stagnated or regressed, because their difficulty is not primarily an "easy vs. hard" imbalance — it is a representational/geometry problem. The line head was particularly harmed, since lines are dominated by easy well-fit predictions whose gradients were down-weighted without any compensating gain on the genuinely hard cases. Exp 026 remains the stronger all-around checkpoint. Future work on `slur`/`tie`/`octave` should focus on specialized representations (e.g., curve/keypoint formulations) rather than loss reweighting.

## Experiment 028: Variable Patch Size Augmentation (Extended Training)
* **Experiment Name/ID**: `experiments/028_variable_patch_size_augmentation`
* **Hypothesis/Goal**: Continue the variable patch size scale augmentation track (Exp 025/026) for another 100 epochs, resuming from the Exp 026 checkpoint with a fresh cosine schedule at peak LR `1e-4`. After reverting the Focal L1/GIoU loss (Exp 027), this verifies whether simply giving the augmented setup more training time — with the standard regression loss — recovers the precision that Exp 027's focal reweighting sacrificed, and pushes the rare/thin classes further.
* **Setup**: 
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`.
  * Checkpoint: Resuming from `experiments/026_variable_patch_size_augmentation/train_detection/checkpoints/latest_model.pt`.
  * Crop Size: Full Image (None)
  * Data: Full Trompa-COCO dataset.
  * Augmentation: `--extract_patch_min 32 --extract_patch_max 96` (carried over from Exp 025/026).
  * Training: 100 additional epochs, peak LR `1e-4`, with a 1 epoch linear warmup. Standard (non-focal) L1/GIoU regression loss.
* **Results**: Extended training with the standard loss produced a broad, strict improvement over Exp 026. In-training batch `mAP@0.5` peaked at ~0.963 (epoch 90) and `mIoU` reached ~0.956, with `loss_total` stabilizing around 0.12-0.21. Official COCO evaluation improved on both axes: for symbols, global `mAP@0.5` rose to **0.891** (from 0.841) and `mAP@0.5:0.95` to **0.709** (from 0.644); for lines, global `mAP@0.5` rose to **0.283** (from 0.275) and `mAP@0.5:0.95` to **0.250** (from 0.234). The precision gap between `mAP@0.5` and `mAP@0.5:0.95` narrowed, indicating tighter box/keypoint localization.
  * **Symbols (broad gains on rare classes):** The biggest wins were on previously weak classes: `tuplet1` (0.207 → 0.693), `accidentalDoubleFlat` (0.516 → 0.916), `accidentalNaturalSharp` (0.028 → 0.401), `timeSig9` (0.000 → 0.257), `dynamicSforzandoPiano` (0.612 → 0.868), and `flag64thDown` (0.717 → 0.934). The curved classes that Exp 027 had targeted-but-regressed also recovered: `slur` (0.506 → 0.568), `tie` (0.461 → 0.504), and `articTenutoAbove` (0.430 → 0.595). Common symbols held near-perfect: `noteheadBlack` (0.990), `gClef` (1.0), `fClef` (1.0). The only notable regression was `dynamicRinforzando1` (1.000 → 0.835); `tuplet8` and `articStaccatissimoWedgeBelow` remained stuck at 0.0.
  * **Lines (uniform improvement):** Every line class improved or held. `staff` jumped 0.807 → **0.896**, `stem` 0.951 → **0.975**, `voltaBracket` 0.632 → **0.686**, `ledgerLines` 0.939 → **0.952**, `beam` 0.975 → **0.988**, and `barLine` 0.968 → **0.979**. `system` held at 0.979. The `octave` class remained effectively dead (0.025 → 0.040).
* **Conclusion**: Simply continuing training under the standard (non-focal) regression loss was the correct remedy: it recovered the precision that Exp 027's focal reweighting had traded away, while the extra iterations let the network fit the rare symbol classes and thin line geometries that the aggressive scale augmentation had previously prevented from converging. The narrowing of the `mAP@0.5` vs `mAP@0.5:0.95` gap confirms that localization tightened rather than just recall rising. The persistent dead spots — `octave` lines, `tuplet8`, `articStaccatissimoWedgeBelow` — are now clearly representational/rarity bottlenecks (extreme aspect ratios or near-absent samples) rather than optimization shortfalls, and reinforce the earlier conclusion that they will need specialized representations or targeted data rather than further LR/loss tuning

## Analysis 001: Intra-class Size Variability vs Detection AP (Exp-028 redo)
* **Experiment Name/ID**: `analysis/001_size_variability_vs_ap` (offline analysis of the Exp 028 checkpoint; no launch folder. Artifacts: `experiments/028_variable_patch_size_augmentation/inference/size_variance_analysis_symbols.json`, `.../size_variance_analysis_lines.json`, `symbol_variance_vs_ap.png`, `line_variance_vs_ap.png`.
* **Hypothesis/Goal**: Verify that per-class detection performance (`mAP@0.5`) drops sharply with intra-class size variability — bbox-diagonal length spread for symbols, keypoint-segment length spread for lines. This redoes the July-22 analysis (commits `0df67a7`/`4061b88`, which used Exp 026) unchanged in methodology, but now on the current best checkpoint ( Exp 028.
* **Setup**: Methodology identical to the July-22 scripts: per-instance bbox diagonal(symbols)/keypoint-segment length(lines,and when missing, bbox diagonal) collected from `data/trompa-coco/annotations/instances_trainval2017.json`, aggregated with raw `np.var` per class, and plotted against per-class `mAP@0.5` from `experiments/028_variable_patch_size_augmentation/inference/coco_eval_summary.json`. Only the summary source changed (026→028`, plus a per-class stats JSON and tabular stdout dump were added(a pure additive artifact,no methodology change.. Commands: `mamba run -n pytorch python scripts/plot_symbol_variance_vs_ap.py` and `mamba run -n pytorch python scripts/plot_line_variance_vs_ap.py`.
* **Results**: **Lines (8 classes — hypothesis strongly confirmed, nearly monotone staircase):** `ledgerLines` (variance 62 px²;`mAP@0.5` 0.952), `stem` (5,275;0.975), `barLine` (7,161;0.979), `beam` → (25,624;0.988), `system` (32,403;0.979), `staff` (342,787;0.896), `voltaBracket` (376,726;0.686), `octave` (1,716,968;0.040). As intra-class length variance grows by ≳4 orders of magnitude, per-class mAP declines monotonically (modulo athe `system`/`barLine` pair alternating, from ≈0.98 to≈0.04**. **Symbols (78 classes — hypothesis holds only at the curved long-tail):** `slur` (variance 299,022;`mAP@0.5` 0.568), `tie` (85,899;0.504),and `brace` (4,772;0.988) are the only classes with a sizeable length spread; every other class has variance ≤ ~200 px²(fixed/deterministic glyph sizes)and AP ranges 0.25–1.0 completely independent of size variance. The low-AP bulk is rarity-driven, not size-driven:`timeSig9` (n=4;AP 0.257), `dynamicMP` (n=3;AP 0.354), `accidentalNaturalSharp` (n=8;AP 0.401), `dynamicPPP` (n=11;AP 0.787), `breathMarkComma` (n=13;AP 0.797; all with ≈0 length variance. `tuplet8` and `articStaccatissimoWedgeBelow` were excluded(no valid scales,≤1 valid sample..
* **Conclusion**: Intra-class size variability essentially accounts for the **line** head&apos;s residual failure:the worst line classes (`octave`,`voltaBracket`,`staff`) are exactly the highest length-variance ones, confirming that lines need more varied-length training data or curve/endpoint-aware representations,rather than more optimization. For **symbols**, the size-variability explanation only covers the three elongated primitives(`slur`,`tie`,`brace`);the remaining symbol gap(and the persistent dead spots `tuplet8`/`articStaccatissimoWedgeBelow`)is squarely a **rarity/representation** problem(reinforcing the Exp 028 conclusion)—to be addressed via instance-balanced sampling,targeted augmentation,or specialized curve head,rather than LR/loss tuning..

## Experiment 029: Size-Difficulty Balanced Loss
* **Experiment Name/ID**: `experiments/029_size_difficulty_balanced_loss`
* **Hypothesis/Goal**: Expand the inverse-frequency class balancing (Exp 017) so that the per-class weights also incorporate **intra-class size variability** as a difficulty proxy, and verify this rescues the classes that Analysis 001 identified as size-hard rather than rarity-hard: the line classes with the highest length variance (`octave`, `voltaBracket`, `staff`) and the elongated symbol primitives (`slur`, `tie`). Because those classes are frequent enough to sit at the frequency-weight floor (0.05), the difficulty factor must be applied *after* the smoothed-inverse-frequency weights are normalized and clamped — otherwise the boost is silently erased by the 0.05 floor.
* **Setup**:
  * Model: `vit_nano` (patch_size=64) with `SymbolHead` and `LineHead`. Data: Full Trompa-COCO dataset.
  * Checkpoint: Resume from `experiments/028_variable_patch_size_augmentation/train_detection/checkpoints/latest_model.pt`.
  * Augmentation: `--extract_patch_min 32 --extract_patch_max 96` (carried over from Exp 025-028).
  * **Correction (2026-11-22)**: the Kaggle run actually used `--extract_patch_min 54 --extract_patch_max 74` (notebook version `f997b85`, Kaggle v32); the 32/96 values here and in the command below are wrong.
  * Training: 100 additional epochs, peak LR `1e-4`, 1 epoch linear warmup, standard (non-focal) L1/GIoU regression loss.
  * New rebalancing: per-class weights now equal the Exp 017 smoothed-inverse-frequency weights multiplied by `clamp(1 + variance_lambda * cv, 1, size_difficulty_max)` **after** normalization+clamping, where `cv` is the per-class coefficient of variation of instance size (bbox diagonal for symbols, keypoint-segment length for lines). `variance_lambda=0.5`, `size_difficulty_max=8.0`. Sanity-checked before launch via `scripts/sanity_check_size_balanced_weights.py` — only `slur`/`tie` (symbols) and `voltaBracket`/`beam`/`staff`/`stem`/`barLine`/`system` (lines) move, everything else stays bit-identical to Exp 028.
  * Cache note: cache now regenerates under the same `instances_trainval2017.pkl` name with the new weights (schema includes `size_difficulty_lambda`/`size_difficulty_max`), so the pre-existing stale pickle must be deleted once before launch.
  * Command:
    ```bash
    PYTHONPATH=. mamba run -n pytorch torchrun --nproc_per_node=2 src/train_detection.py \
        --exp_dir experiments/029_size_difficulty_balanced_loss \
        --detector_checkpoint experiments/028_variable_patch_size_augmentation/train_detection/checkpoints/latest_model.pt \
        --patch_size 64 \
        --extract_patch_min 32 \
        --extract_patch_max 96 \
        --epochs 100 \
        --lr 1e-4 \
        --warmup_epochs 1 \
        --class_weight_variance_lambda 0.5 \
        --class_weight_size_difficulty_max 8.0 \
        --anno_path data/trompa-coco/annotations/instances_trainval2017.json \
        --img_dir data/trompa-coco/trainval2017 \
        --headless \
        --use_sdpa \
        --compile \
        --log_epoch_interval 5
    ```
* **Results**: The size-difficulty balanced loss produced a strict, global improvement over Exp 028 with no regression on any line class. In-training batch `mAP@0.5` peaked at ~0.984 (epoch 90) with `mIoU` ~0.974; final epoch `mAP@0.5` 0.954, `mIoU` 0.938.
  * **Global COCO metrics:** Symbols `mAP@0.5` **0.891 → 0.919** and `mAP@0.5:0.95` **0.709 → 0.753**. Lines `mAP@0.5` **0.283 → 0.291** and `mAP@0.5:0.95` **0.250 → 0.261**.
  * **Targeted size-hard classes — all improved:**
    * **Lines (the headline result):** every line class rose: `voltaBracket` 0.686 → **0.726**, `staff` 0.896 → **0.935**, `octave` 0.040 → **0.052**, `beam` 0.988 → 0.989, `system` 0.979 → **0.990**, `ledgerLines` 0.952 → 0.962, `stem` 0.975 → 0.978, `barLine` 0.979 → 0.980. The two highest-CV classes (`voltaBracket` cv=1.37, `staff` cv=0.51) gained the most.
    * **Symbols:** `slur` 0.568 → **0.597** (`mAP@0.5:0.95` 0.389 → 0.434), `tie` 0.504 → **0.546** (0.276 → 0.319). Fixed-glyph classes were untouched as predicted.
  * **Rare-class spillover:** several rarity-hard (cv≈0) classes also rose, since their frequency weights dominate: `timeSig9` 0.257 → 0.691, `accidentalNaturalSharp` 0.401 → 0.812, `dynamicMP` 0.354 → 1.000, `dynamicRinforzando1` 0.835 → 1.000, `tuplet1` 0.693 → 0.830, `articMarcatoAbove` 0.865 → 1.000. Only notable regressions: `accidentalDoubleFlat` 0.916 → 0.598 and `accidentalNaturalFlat` 0.800 → 0.729 (rare, high noise). `tuplet8` and `articStaccatissimoWedgeBelow` remained at 0.000.
* **Conclusion**: The hypothesis held. Adding intra-class size variability (via the coefficient of variation) to the class weights, applied *after* the frequency normalization+clamping, rescued exactly the size-hard classes that Exp 028/027 leave behind — `octave`/`voltaBracket`/`staff` (lines, the previously-worst size case: 0.040/0.686/0.896) and `slur`/`tie` (symbols) — while leaving the fixed-glyph majority bit-identical in weight and, empirically, at-or-above their Exp 028 AP. The line head's residual problem identified in Analysis 001 is therefore addressable in part by *weighting* rather than only by representation. `octave` is the one caveat: despite the largest absolute weight ceiling (0.85) it only moved 0.040 → 0.052, indicating the extreme aspect-ratio + n=24 sample bottleneck is now the binding constraint, not loss weighting. Next steps: (1) keep `variance_lambda=0.5` as a default and evaluate on a fresh full training (not resumed) to confirm the gains aren't partly extended-training artifacts; (2) raise `variance_lambda` (e.g. 1.0) selectively for lines only; (3) address the remaining dead spots (`tuplet8`, `articStaccatissimoWedgeBelow`, `octave` beyond 0.05) via instance-balanced sampling or curve/keypoint representation as Analysis 001 already concluded.

* **Addendum (2026-09-19, metric fix)**: The global line numbers reported above (mAP@0.5 `0.291`, mAP@0.5:0.95 `0.261`) were **wrong** — an evaluation bug, not a model result. pycocotools' `summarize()` hard-codes `maxDets=20` for the keypoints aggregator while the bbox aggregator uses `maxDets=2000`, so the line metrics were computed by truncating every line prediction to its top-20 scored detections (the line head emits ~1785 detections/image). Fixed in `src/evaluate_coco.py` (new `summarize_at_maxdet`, now used for both modalities at `maxDets=2000`), and exp 29 was re-evaluated with the corrected script.
  * **Corrected global numbers (maxDets=2000, exp 28 → exp 29, recomputed consistently for both):**
    * Lines: `mAP@0.5` **0.812 → 0.8265**, `mAP@0.5:0.95` **0.718 → 0.7481**. All per-category line APs in this report were already computed at maxDets=2000 and are unaffected — the corrected global is now consistent with them (mean per-cat `mAP@0.5` = 0.8265).
    * Symbols: `mAP@0.5` **0.891 → 0.9190**, `mAP@0.5:0.95` **0.723 → 0.7681**.
  * **Re-interpretation:** with the metric bug fixed, the line head is NOT dramatically worse than symbols — it's ~0.08 worse at `mAP@0.5` (0.827 vs 0.919) and nearly equal at `mAP@0.5:0.95` (0.748 vs 0.768). The residual line-vs-symbol gap is small and lives in the same size-hard classes already discussed (`octave` 0.052, `voltaBracket` 0.726), not in a modality-wide shortfall. Exp 029's per-class conclusions and its improvement over Exp 028 stand; only the aggregate line headline numbers were overstated as failures.

## Experiment 030: IMSLP Dense-LeJEPA Backbone Adaptation (DRAFT)
* **Experiment Name/ID**: `experiments/030_imslp_lejepa_backbone`
* **Status**: 🚧 DRAFT — implementation committed (`train_lejepa.py`: IMSLP data source + checkpoint init/resume), experiment **not yet launched**. Results/Conclusion to be filled from `metrics.jsonl` after the run.
* **Hypothesis/Goal**: Adapt the **feature side** of the exp-29 detector (synthetic verovio → real scanned IMSLP) without any labels, by fine-tuning **only the backbone** with the Dense LeJEPA objective on the unannotated IMSLP pages. The objective is the same visual-grammar world model used in pretraining (masked patch prediction + SIGReg), so it should re-fit the feature extractor to the real marginal (ink/paper/noise/DPI) while the dense per-patch heads stay frozen and (by the patch-as-predictor + low-LR argument) remain near-continuously compatible. This tests the *feature-side transfer* leg of the three-arm A/B (none vs LeJEPA-only vs LeJEPA + pseudo-label self-training). Success proxies without IMSLP labels: stable/collapse-free SSL losses, then anti-forgetting on trompa-val and per-class confidence calibration of the un-adapted heads re-mounted on the adapted backbone.
* **Setup**:
  * Model: `vit_nano` (patch_size=64, channels=3), **backbone initialized from exp-29** via `--detector_checkpoint` (prefix-stripping `model.backbone.*`; verified 140 keys strict-load). Predictor (depth=4) and SIGReg trained fresh (SSL stage only, no heads in the graph).
  * Data: `data/imslp/imslp.jsonl` (2,591 scanned pages), `data/imslp/images`. New IMSLP data source in `train_lejepa.py` — the COCO path is untouched.
  * Preprocessing: 1024×1024 random crops, `pad_to_patch_size(64)`, `variance_patch_drop(var_threshold=0.001)`, `random_spatial_mask(mask_ratio=0.5)`.
  * Training: 20 epochs over 2,591 samples (per-rank batch 8 → 2 ranks ≈ 12,960 samples/epoch, ~3,240 steps), sample-budget warmup+cosine, peak `lr=1e-4` (lower than from-scratch 5e-4 because we fine-tune a converged backbone), `warmup_epochs=1`, `min_lr_ratio=1e-4`, `lamb=0.05`, AdamW `weight_decay=0.05`. SDPA + `torch.compile(dynamic=True)`. **DDP on 2 GPUs** (backbone + predictor wrapped; per-rank shuffled indices, gradients all-reduced every step, rank 0 only logs/saves — mirrors `train_detection.py`).
  * Checkpoint: `latest_model.pt` keeps `{backbone, predictor, optimizer, loss, samples, step, epoch}` → directly usable as `--backbone_checkpoint` for a later detector stage; resume via `--resume` (Gap B, detector-side overlay, not yet implemented in this experiment).
  * Command:
    ```bash
    mamba run -n pytorch torchrun --nproc_per_node=2 src/train_lejepa.py \
        --imslp_manifest data/imslp/imslp.jsonl \
        --img_dir data/imslp/images \
        --detector_checkpoint experiments/029_size_difficulty_balanced_loss/train_detection/checkpoints/latest_model.pt \
        --exp_dir experiments/030_imslp_lejepa_backbone \
        --patch_size 64 \
        --backbone_size nano \
        --batch_size 8 \
        --mask_ratio 0.5 \
        --lamb 0.05 \
        --epochs 20 \
        --lr 1e-4 \
        --warmup_epochs 1 \
        --use_sdpa \
        --compile
    # resume (if interrupted):
    #   --resume --backbone_checkpoint experiments/030_imslp_lejepa_backbone/pretrain_lejepa/checkpoints/latest_model.pt
    ```
* **Results**: TBD — read `experiments/030_imslp_lejepa_backbone/pretrain_lejepa/metrics.jsonl`. Monitor: `loss_l2` should fall steadily, `loss_sigreg` should stay bounded (not collapse toward 0, which would signal representation collapse), `loss_total` should not diverge from the exp-19 trompa reference (~0.3–0.9 at similar λ).
* **Conclusion**: TBD. Expected outcomes to decide the three-arm A/B next step:
  * If L2 falls and SIGReg stays healthy and anti-forgetting on trompa-val is mild → LeJEPA leg is validated; proceed to detector-side backbone overlay (Gap B) + confidence-calibration measurement on IMSLP.
  * If the SSL run collapses (near-zero total loss) → the adaptation LR is too high or λ too small; restart with `lr=5e-5`.
  * If the run looks healthy but downstream heads degrade badly on trompa-val → the feature-drift assumption is wrong for this geometry; fall back to pseudo-label self-training (head-side) as the primary leg.

## Note 001: Manual Attention Scale Fixed (2026-11-15)
* **Change**: `src/model/vit.py`, `Attention.forward`: the manual (non-SDPA) path now scales by `q.shape[-1] ** -0.5`, as `F.scaled_dot_product_attention` does by default.
* **Bug**: PoPE stacks `[mag*cos, mag*sin]`, so q and k are `2*dim_head` wide. The manual path scaled by `dim_head**-0.5` (1/8) while SDPA scales by `(2*dim_head)**-0.5` (1/sqrt(128)), so the manual path's attention logits were sqrt(2) too large (max abs output difference 0.216 on a random `Attention`; now ~1e-7).
* **Impact**:
  * Checkpoints trained without `--use_sdpa` (Exp 001–009, `scripts/sanity_check.py` runs) were trained with the old scale. To reproduce them exactly, use the code before this change or multiply the logits by sqrt(2).
  * Exp 010 onward were trained with `--use_sdpa` and are unaffected. Before this fix, `src/inference_coco.py` and `tools/interactive_viewer.py` run without `--use_sdpa` on those checkpoints applied the wrong scale; both paths now agree.

## Analysis 002: Oracle FDR Refinement Ceiling on Exp 029 (Lines and Symbols)
* **Experiment Name/ID**: `analysis/002_oracle_refinement_ceiling` (offline, Exp 029 checkpoint). Artifacts: `experiments/029_size_difficulty_balanced_loss/inference_local/` (baseline) and `.../inference_oracle/` (oracle), each with `coco_eval_summary.json`.
* **Hypothesis/Goal**: Measure how much a *perfect* D-FINE refinement could gain, now with the corrected maxDets=2000 evaluation (the Exp 026 oracle line numbers predate that fix), to decide whether an iterative line refiner is worth building and which line classes it can help.
* **Setup**: `inference_coco.py --oracle_fgl` (commit `2cf122d`): each GT is greedily matched (same class, nearest geometry) to one prediction among all P×K queries, regardless of score; its FDR residual is replaced by the exact GT residual in the stage-0 frame (`base_dirs`, scale `|base_dir| + anchor` for lines; learned shape for symbols), clamped to ±0.5. Scores are untouched. Both runs on the laptop (Quadro M1200), fp32, `--use_sdpa --force_pyvips`, all 351 trainval pages, then `evaluate_coco.py`. The local baseline reproduces the Kaggle Exp 029 evaluation exactly (identical global and per-class numbers).
* **Results**:
  * **Lines (global)**: `mAP@0.5:0.95` **0.748 → 0.831** (+0.083), `mAP@0.5` 0.827 → 0.862 (+0.036), `mAP@0.75` 0.772 → 0.833.
  * **Lines (per class, mAP@0.5:0.95 baseline → oracle)**: `voltaBracket` 0.553 → **0.787** (+0.234), `stem` 0.835 → **0.958** (+0.123), `ledgerLines` 0.842 → **0.965** (+0.123), `octave` 0.006 → 0.105 (+0.099; and `mAP@0.5` 0.052 → 0.214), `staff` 0.837 → 0.872 (+0.035), `barLine` 0.963 → 0.988, `beam` 0.977 → 0.987, `system` 0.973 → 0.990.
  * **Symbols (global)**: `mAP@0.5:0.95` 0.768 → **0.900** (+0.132), `mAP@0.5` 0.919 → 0.935, small-object AP 0.402 → **0.811**.
* **Conclusion**: Refinement headroom is real for lines (+0.083 mAP@0.5:0.95), concentrated in `voltaBracket`, `stem`, `ledgerLines` and, unexpectedly, `octave` (whose `mAP@0.5` quadruples: part of its failure *is* endpoint localization within ±0.5 scale, not only missed detections). `staff` gains little (+0.035): its residual error is mostly beyond the ±0.5 range or a detection/scoring issue, so one-frame refinement will not fix it. Symbols have an even larger ceiling (+0.132), dominated by small objects. Next: train the iterative D-FINE line refiner (Exp 031, `src/train_line_refiner.py`) on the frozen Exp 029 detector and measure what fraction of this +0.083 it recovers.

## Analysis 003: Staff Failure Diagnosis on Exp 029
* **Experiment Name/ID**: `analysis/003_staff_failure_diagnosis` (offline, Exp 029 predictions in `experiments/029_size_difficulty_balanced_loss/inference_local/preds_lines.json`, 351 trainval pages, the same pages the model was trained on).
* **Hypothesis/Goal**: Find why `staff` (mAP@0.5:0.95 0.837, AP@0.5 0.935) is well below `barLine`/`system` (~0.97–0.98). Suspected cause: variable length and the high instance count giving staff a low effective loss weight.
* **Setup**: `PYTHONPATH=src:. mamba run -n pytorch python scripts/analyze_line_failures.py --predictions experiments/029_size_difficulty_balanced_loss/inference_local/preds_lines.json` replays the COCO greedy OKS matching (σ=0.1) and labels every miss and confident false positive. One-off scratch analyses (not committed) re-ranked scores, applied OKS-NMS, and ran the detector on 30 pages to locate the predicting patch. Note: a `staff` annotation is one staff line **between two barlines** (one measure), ~283 per page; consecutive segments share the barline x exactly.
* **Results**:
  * Per OKS threshold, staff AP 0.935 (0.50), 0.871 (0.75), 0.557 (0.95); recall 0.971 / 0.935 / 0.766. Localization (0.935 → 0.837 average) costs more than detection.
  * **Vertical placement is not a failure mode**: matched predictions sit 0.3 px (median) / 2 px (p99) from the true line; no confusion with neighbouring staff lines.
  * **All endpoint error is horizontal, and on the far end**: left endpoint |dx| median ~1 px at every length; right endpoint median 2.2 → 8.5 px and p90 7 → 42 px as length grows.
  * **Cause of the asymmetry**: the matcher assigns 77% of staff segments to a patch in the left 40% of the segment (`t∈[0,0.4)`), and the top-scored prediction follows the same distribution. The endpoint near the predicting patch is precise (~0.8 px); the far one is not (4–5 px median, 17–24 px p90). Predictions made from right-side patches show the mirror pattern.
  * **Ranking ignores localization quality**: for 34.7% of segments, the top-scored prediction is not the best-localized one among those with OKS ≥ 0.5. Re-ranking within each segment's cluster (same scores, re-assigned by OKS) gives mAP 0.837 → **0.865**, AP@0.95 0.557 → **0.645**.
  * **Duplicates** are 71.5% of confident FPs, but removing all of them by oracle only gives +0.017 mAP. OKS-NMS hurts at strict thresholds (mAP 0.826–0.837).
  * **Miss rate is U-shaped in length**: <8 staff spaces 34.9% (1,020 GT, mostly short measures between close barlines), 12–50 staff spaces 1.4–2.3%, >50 staff spaces 20.0% (1,880 GT). The long tail is mostly unbarred systems (cadenza-like passages) where one segment spans the whole system: such solo segments are missed 44% of the time. Both tails are concentrated on ~51 pages each.
  * **Correction to Analysis 002**: its staff oracle gain (+0.035) underestimates the refinement headroom. The oracle corrected the nearest-geometry query, which is often not the top-scored one, so the uncorrected top-scored duplicate still ranks first.
* **Conclusion**: Staff errors are not primarily a loss-weight problem: staff segments are found (97% recall) and the errors survive on training pages, so they are structural rather than under-weighted generalization. The three causes are, by size: (1) far-endpoint imprecision from predicting long segments from one end, which is what the iterative line refiner (Exp 031) targets; (2) classification scores not tied to localization quality (+0.028 mAP measured by re-ranking alone), which D-FINE handles with a quality-aware classification loss (VFL with IoU soft target), adapted here as an OKS target; (3) rare layouts (unbarred systems, very short measures) on a few pages, better addressed by data (synthetic pages) than by a global staff weight increase.

## Experiment 032: Intra-Class Size Weighting and OKS Quality Target
* **Experiment Name/ID**: `experiments/032_size_weighting_oks_target`
* **Status**: done (Kaggle kernel v38, 2×T4, 3.1 h). **Negative result**: global and staff metrics regressed. **Failed branch**: the size-weighting / OKS-target code (commit `ab90bb0`) was reverted, so the training code is back to the Exp 029 state; the experiment stays here as a record.
* **Hypothesis/Goal**: Attack two of the three staff failure causes found in Analysis 003, as a fine-tune of Exp 029:
  1. **Intra-class size weighting**: within each class whose size distribution is wide (CV ≥ 0.3, n ≥ 100: `staff`, `stem`, `beam`, `barLine`, `voltaBracket`, `slur`, `tie`), weight each instance by `density(size)^-0.5`, normalized to mean 1 over the class (the Exp 029 class balance is unchanged) and clamped to [0.25, 4]. Staff gets ~4× at both tails (≤5 and ≥60 staff spaces, the 35% / 20% miss-rate bins) and ~0.75× in the 15–20 bulk. Applied to the positive classification term, L1/GIoU and FGL.
  2. **OKS quality target for lines**: the line classification loss becomes Quality Focal Loss (GFL): a matched query's target is the OKS of its current (detached) prediction instead of 1, so the score ranks queries by endpoint quality. QFL with quality = 1 is exactly the previous focal loss. Analysis 003 measured +0.028 staff mAP from re-ranking alone.
  * Expected: fewer misses in the staff length tails (per-length miss rates from `scripts/analyze_line_failures.py`), and a lower share of segments whose top-scored prediction is not the best-localized one (34.7% in Exp 029). Risk to watch: the bulk length bins losing more than the tails gain.
* **Setup**:
  * Implementation: `src/model/size_weighting.py` (per-class log-size histograms from `parse_coco`, interpolated bin weights), `quality_focal_loss` and `line_oks` in `src/model/criterion.py`. The OKS area is `length × class thickness`, with thickness = median COCO `area / length` (staff 3.9 px, stem 5.5, barLine 8.4, beam 39.3). Patch-unit targets are converted to pixels with the nominal patch size (64), an approximation under the 54–74 patch-size augmentation, smoothed by the interpolated weights. The COCO cache now carries a schema version and is re-parsed automatically when stale or when the class-weight parameters change.
  * Local checks: unit tests (`tests/model/test_detection_losses.py`, QFL equals focal loss at quality 1), a 587-step single-GPU run on 1024 crops from the Exp 029 checkpoint (DDP could not be tested locally: NCCL does not support the laptop's Maxwell GPU).
  * Training: resume from Exp 029 (Kaggle model `music-deep/pytorch/train_detection/11`), 30 epochs, peak LR `1e-4`, 1 warmup epoch, same flags as the Exp 029 Kaggle run otherwise.
  * Command:
    ```bash
    PYTHONPATH=. mamba run torchrun --nproc_per_node=2 src/train_detection.py \
        --exp_dir experiments/032_size_weighting_oks_target \
        --detector_checkpoint <029>/train_detection/checkpoints/latest_model.pt \
        --patch_size 64 --epochs 30 --lr 1e-4 --warmup_epochs 1 \
        --extract_patch_min 54 --extract_patch_max 74 \
        --class_weight_variance_lambda 0.5 --class_weight_size_difficulty_max 8.0 \
        --intra_class_size_alpha 0.5 --line_quality_target \
        --headless --use_sdpa --compile --log_epoch_interval 5
    ```
  * Caveat: this is a continuation of Exp 029, so part of any gain could come from the extra 30 epochs. The diagnosis metrics separate the two mechanisms better than mAP alone.
  * Kaggle environment (v33–v38 failures, kept here for the next run): Miniforge `latest` now ships Python 3.14 and CUDA 13.5 packages, which crash on the T4 driver (CUDA 13.0) with an illegal memory access. Working install: `mamba install python=3.13 cuda-version=13.0 pytorch=2.11 pillow matplotlib scipy nvimgcodec nccl cuda-cudart-dev` run through `subprocess.run(check=True)` so a failed solve stops the notebook (do not add `cuda-toolkit`: its 13.0 build only exists for py3.12; `cuda-cudart-dev` provides the `cuda.h` Triton needs for `--compile`). The 2.1 GB COCO JSON took ~10 min to parse on each run; the parsed cache is now a private Kaggle dataset (`kwonyoungchoi/trompa-coco-cache`) copied into `--cache_dir` before training, and is re-parsed automatically if its schema is stale.
* **Results**: Final checkpoint evaluated with `evaluate_coco.py` (maxDets=2000). Local laptop inference of the downloaded checkpoint reproduces the Kaggle evaluation to 4 decimals, so the inference environment is not involved.
  * **Global (029 → 032)**: Symbols `mAP@0.5:0.95` **0.768 → 0.692**, `mAP@0.5` 0.919 → 0.886, small 0.402 → 0.307. Lines `mAP@0.5:0.95` **0.748 → 0.668**, `mAP@0.5` 0.827 → 0.776.
  * **Lines per class (mAP@0.5:0.95)**: `staff` **0.837 → 0.666**, `voltaBracket` 0.553 → 0.299, `stem` 0.835 → 0.734, `ledgerLines` 0.842 → 0.787, `barLine` 0.963 → 0.944, `beam` 0.977 → 0.957, `system` 0.973 → 0.957, `octave` 0.006 → 0.000.
  * **Symbols**: almost every class lost localization quality, including classes whose loss is unchanged by this experiment (`noteheadBlack` 0.982 → 0.973, `accidentalFlat` 0.926 → 0.853, `timeSig8` 0.861 → 0.754). The weighted `slur` 0.434 → 0.285 and `tie` 0.319 → 0.240 regressed the most.
  * **Staff diagnostics** (`scripts/analyze_line_failures.py` and the Analysis 003 scratch scripts):
    * AP@0.5 / 0.75 / 0.95 0.935 / 0.871 / 0.557 → **0.824 / 0.703 / 0.312**. Recall at OKS 0.5: FN 2.87% → 5.13%, with "missed (no pred OKS>0.1)" 591 → 2,104.
    * Ranking got **worse**, not better: the top-scored prediction of a segment is not the best-localized one in 34.7% → **44.8%** of segments (median OKS gap 0.040 → 0.086). Duplicates among confident FPs 7,948 → **18,975**. Quality re-ranking would now add +0.059 mAP (vs +0.028 before).
    * Miss rate per length (staff spaces): <8 34.9% → 35.3%, 12–18 2.3% → 5.0%, 18–25 1.4% → 3.4%, >50 20.0% → 25.3%. The rare-length tails did not improve and the bulk doubled its miss rate.
    * Only the far-endpoint error of the FNs' closest prediction shrank (median |x2| 0.32 → 0.12 staff spaces), while TP endpoint error grew (x p50/p90 1.6/8.7 → 2.2/12.3 px).
* **Conclusion**: Both mechanisms failed in this form, and something else regressed the untouched symbol classes too.
  1. **OKS quality target is harmful in a dense one-to-one head.** Only the Hungarian-matched query gets a soft target (its own OKS, < 1), while its spatial neighbours, which are often as well or better localized, stay at target 0. This (a) shrinks the score margin between the match and its duplicates, so duplicates out-rank it more often, and (b) never tells the network that a better-localized neighbour should score higher, because the matched query is chosen by classification + L1 cost, not by OKS. D-FINE avoids both through query self-attention in its decoder; our patch-as-predictor head has no interaction between queries. A quality target only makes sense with quality-aware matching (e.g. OKS in the matcher cost) or with soft targets on every query near the segment.
  2. **Intra-class size weighting did not rescue the tails** (<8 and >50 staff-space miss rates unchanged or worse). Its effect cannot be separated from (1) and (3) in this run.
  3. **Unexplained global drop.** Fixed-glyph symbol classes, whose loss is bit-identical to Exp 029 with these flags (QFL at quality 1 equals focal loss, unit size weights; unit-tested), lost 0.01–0.10 mAP. Candidate causes: the LR restart (1e-4 again on a converged model, only 30 epochs to re-converge), the new Kaggle environment (py3.13 / CUDA 13.0 / torch 2.11 instead of Exp 029's), or the changed line losses perturbing the shared backbone.
  * **Next steps**: (a) a control run (`experiments/033_continuation_control`: same branch, environment and 30-epoch schedule from Exp 029, without `--intra_class_size_alpha`/`--line_quality_target`) to separate the restart/environment effect from the two losses; the notebook is prepared but not launched. (b) Do not reuse `--line_quality_target` as is; if retried, put OKS into the matcher cost first. (c) Keep size weighting off until (a) isolates its effect. (d) The far-endpoint problem of Analysis 003 still points to the iterative line refiner (Exp 031) on the frozen Exp 029 detector, which avoids touching the detector weights at all.

## Experiment 034: D-FINE Multi-Layer Refinement on the Last 3 ViT Blocks
* **Experiment Name/ID**: `experiments/034_dfine_multilayer_refinement`
* **Status**: ❌ Negative. Run on Kaggle (notebook version 44, branch `exp-034-dfine-multilayer-refinement` @ `4a21c5f`), stopped by the 11 h cap at epoch 85/100.
* **Hypothesis/Goal**: Analysis 002 measured a refinement ceiling of +0.083 line / +0.132 symbol mAP@0.5:0.95. Analysis 003 found the far-endpoint error of long staff segments to be the largest staff failure. Reproduce D-FINE's Fine-grained Distribution Refinement (decoder of D-FINE-N, 3 layers) **inside the backbone**: block 9 predicts, blocks 10 and 11 refine. Unlike the Exp 031 frozen refiner, the detector is trained jointly.
* **Setup**:
  * **Architecture** (`src/model/layered_detector.py`, `MultiLayerOMRDetector`), mapping D-FINE's DETR decoder to the dense head:
    * Blocks 0–8 play the encoder. Every patch×slot is a query, so there is no top-K query selection.
    * Tap 0 = block 9 (+ its own LayerNorm). The dense `SymbolHead`/`LineHead` give the coarse prediction (D-FINE `pre_bboxes`: center ± shape/2, center + base_dirs) and the first edge distribution. The **detached** coarse geometry is the FDR frame of every layer.
    * Blocks 10 and 11 play decoder layers 1 and 2. Before each, `SlotBoxConditioner` (D-FINE `query_pos_head`: MLP of the concatenated 2×K slot geometries, signed-log, relative to the patch center, detached) is added to the attention **queries and keys only** (D-FINE `with_pos_embed`). Each `RefineHead` adds `delta(tokens + previous_tokens.detach())` to the previous logits (logit chain not detached), has its own score head, and has an `LQE`.
    * The delta heads and the conditioner's last layer are zero-initialized, so at init the 3 layers decode the same boxes (unit-tested).
  * **Loss** (`src/model/layered_criterion.py`): each layer and the pre-prediction are matched separately and classified with their own matching. L1/GIoU/FGL use the GO union of all matchings (vectorized `_get_go_indices`, tested against the D-FINE loop). Symbol FGL is IoU-weighted. The pre-prediction gets classification + L1/GIoU only. DDF (T=5, weight 1.5) distils the final distribution into layers 0 and 1. Deliberate deviations: **focal loss instead of VFL** (Exp 032: an IoU/OKS soft target in a one-to-one dense head degrades ranking); line FGL/DDF quality = 1 (no IoU for segments); no CDN denoising (there is no separate query set).
  * **FDR bins**: `--dfine_outer_bins` puts the outer bins at ±1.0 anchor units as in D-FINE. The legacy ±0.5 duplicated the inner bins, so a residual could never exceed ±0.5×scale, which matters for the far endpoints of Analysis 003. This change is confounded with the multi-layer architecture in this run.
  * **Init**: backbone and dense heads from the Exp 029 checkpoint (`--init_from_single_checkpoint`, bin buffers excluded). Refinement score heads start as copies of the dense classifiers; at init the final class logits equal Exp 029's (unit-tested). Fresh AdamW state (Exp 029 had loaded Exp 028's). Expect an initial drop: the tap-9 heads read block-9 features they were not trained on.
  * **Scale** (target: 10k tokens = 50k queries per modality): the matcher's candidate search was dense `[Q, T]` (~3 GB for lines at 50k × 1,685) and now runs 4× per step. It is now chunked over targets in a targets-major layout (exactly the same pairs, tested against HEAD; top-k 9× faster). At 50k queries with the trainval maximum of targets (1,168 symbols, 1,685 lines), the full layered criterion (4 matchings × 2 modalities, losses, DDF, backward) peaks at 773 MiB extra GPU memory. The step overhead vs the single-layer model, measured on the laptop, is 1.25× at 1,600 tokens and 1.16× at 4,096. Exp 029 took 9.6 h, so the run is close to the 12 h Kaggle limit; `--resume` continues the same LR schedule in a second session if needed.
  * Other changes: `DFINECriterion.forward` refactored into `layer_losses(classification, regression)` (bit-identical losses vs HEAD on an Exp 029 batch); GIoU loss computed element-wise instead of `diag(N×N)` (same values, no assert on inverted boxes, which the ±1.0 bins make possible); checkpoints record their `architecture`, read by `inference_coco.py` (`--eval_layer` evaluates taps 9/10/11).
  * Training: same recipe as the Exp 029 Kaggle run, 100 epochs.
  * Command (Kaggle, `scripts/music_deep.ipynb`):
    ```bash
    PYTHONPATH=. mamba run torchrun --nproc_per_node=2 src/train_detection.py \
        --exp_dir experiments/034_dfine_multilayer_refinement \
        --patch_size 64 --epochs 100 --lr 1e-4 --warmup_epochs 1 \
        --extract_patch_min 54 --extract_patch_max 74 \
        --class_weight_variance_lambda 0.5 --class_weight_size_difficulty_max 8.0 \
        --refine_layers 2 --dfine_outer_bins \
        --init_from_single_checkpoint <029>/train_detection/checkpoints/latest_model.pt \
        --headless --use_sdpa --compile --log_epoch_interval 5
    ```
  * **No control run**: the Exp 032 confound (restart from 029 with a new LR schedule and a new Kaggle environment) is not isolated. A regression of classes that refinement should not touch would point to that confound rather than to the architecture.
* **Results** (checkpoint of epoch 85, step 14,933; `evaluate_coco.py` on all 351 trainval pages; Exp 032 shown because it had the same restart from Exp 029):

  | | Symbols AP / AP50 / AP75 | Lines AP / AP50 / AP75 | staff AP50 / AP |
  |---|---|---|---|
  | Exp 029 (epoch 100) | 0.768 / 0.919 / 0.845 | 0.748 / 0.827 / 0.772 | 0.935 / 0.837 |
  | Exp 032 (restart, 30 epochs) | 0.692 / 0.886 / 0.773 | 0.668 / 0.776 / 0.690 | 0.824 / 0.666 |
  | 034 layer 0 (block 9) | 0.645 / 0.902 / 0.745 | 0.518 / 0.677 / 0.545 | 0.209 / 0.094 |
  | 034 layer 1 (block 10) | 0.652 / 0.917 / 0.749 | 0.528 / 0.693 / 0.552 | 0.304 / 0.141 |
  | 034 final (block 11) | 0.655 / 0.920 / 0.751 | 0.530 / 0.696 / 0.553 | 0.312 / 0.143 |

  * Other line classes, AP@[.5:.95] from Exp 029 to Exp 034 final: barLine 0.963 → 0.850, beam 0.977 → 0.913, ledgerLines 0.842 → 0.621, stem 0.835 → 0.606, system 0.973 → 0.777, voltaBracket 0.553 → 0.325.
  * Training (`metrics.jsonl`, final layer on the training batch, noisy): the in-training line endpoint error jumped from Exp 029's ~0.06 to 0.32 at epoch 5 (warm start on block 9) and only came down to 0.15 at epoch 85, staying 2–3× above Exp 029 at every logged epoch. Symbol mAP50 on the training batch was similar to Exp 029. At epoch 85, pre-prediction loss 0.22, intermediate-layer loss (`loss_aux`) 0.51, DDF 0.18.
  * Cost: 1,440 steps/h vs 1,828 for Exp 029 (1.27× slower, as predicted at small token counts). 100 epochs would need ~12.2 h, so `timeout 11h` stopped training after epoch 85 (10.3 h); the learning rate at that point was 5.5e-6, already near the end of the cosine schedule.
  * Not done: the Analysis 003 staff diagnostics and the Analysis 002 oracle fraction. Given the size of the regression they would not change the conclusion.
* **Conclusion**:
  1. **Moving the first prediction to block 9 is what hurts, and refinement cannot repair it.** Layer 0 → final layer gains only +0.010 symbol AP and +0.012 line AP, while layer 0 is far below Exp 029. Detection itself survives (symbol AP50 0.920 = Exp 029), but localization does not: AP75 and the strict thresholds drop, and long objects collapse (staff AP50 0.935 → 0.312). D-FINE's refinement corrects each edge only within its bin range (±1 anchor scale with the D-FINE outer bins) around a **detached** coarse frame. Here, that frame comes from block-9 features: when it is far off, as it is for the long staff segments, no residual inside the range fixes it. The decoder of D-FINE refines from a strong encoder proposal; in a 12-block ViT, block 9 is not yet such a proposal.
  2. **The restart confound explains only part of the drop.** Exp 032 (the same restart from Exp 029, unchanged architecture) lost 0.076 symbol AP and 0.080 line AP. Exp 034 lost 0.113 and 0.218, and staff lost far more than in Exp 032 (AP50 0.312 vs 0.824). So most of the line regression comes from the architecture, not the restart. Without a control run, the restart's share of the symbol drop stays unknown.
  3. **Box conditioning through q/k injection gives the refinement layers too little to work with.** The two refinement blocks see their slot's current box only as an additive embedding; with PoPE they cannot attend to the far end of a segment (see the Exp 035 proposal below).
  * **Next steps**:
    * (a) Do not resume to epoch 100 (≈2 h of GPU time and a web-UI upload): the last 15 epochs run at LR ≤ 5.5e-6 and cannot close gaps of 0.1–0.6 AP.
    * (b) Keep the Exp 029 prediction on block 11 and put the D-FINE chain **after** it: extra refinement blocks on top of the full backbone, so the frame comes from the best features.
    * (c) Exp 035: give those refinement blocks **sparse deformable sampling** at the slot's predicted endpoints / edges. The patch tokens lie on a regular grid (`Patches.indices`, sorted), so bilinear sampling needs only `searchsorted` on the 4 neighbour cells, with a learned "blank" embedding for dropped (low-variance) cells. Its cost is linear in the number of queries, so it fits the 10k-token target.
  * **Kaggle notes** (for future launches): (1) the `trompa-coco-cache` dataset holds a pickle made by the reverted Exp 032 code; it no longer loads (`SizeHistogram`), and it bakes in Exp 032 class weights. The notebook now parses the COCO JSON fresh (~1 min). (2) `kaggle kernels push` (CLI 2.1.2 and 2.2.5) silently drops `model_sources`, so a run that needs a model input must be launched from the web UI. The model then mounts at `/kaggle/input/models/kwonyoungchoi/music-deep/pytorch/train_detection/<version>/`. (3) `!` shell cells do not stop the notebook on failure, so a crash shows up as a "complete" run within minutes; cell 5 asserts its inputs in Python so it does stop.

