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
