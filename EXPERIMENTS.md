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
* **Results**: TBD
* **Conclusion**: TBD
