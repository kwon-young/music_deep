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
* **Results**: TBD
* **Conclusion**: TBD
