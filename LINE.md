# Upgrading the OMR Detector: Specialized Heads for Symbols and Lines

## 1. The Problem: The "Elephant in the Room"
In Optical Music Recognition (OMR), treating all objects as bounding boxes creates a fundamental geometric flaw. While bounding boxes work well for compact symbols (noteheads, clefs), they are terrible for lines (staff lines, stems, barlines). 
* A bounding box for a diagonal beam or a page-spanning staff line contains 99% empty space.
* This ruins the Generalized IoU (GIoU) metric, confusing the Hungarian Matcher and providing terrible gradient signals during training.
* Furthermore, a single unified detection head struggles to balance the conflicting capacity requirements of tiny, dense symbols and massive, sparse lines.

## 2. The Solution: Dual-Head Architecture
To solve this, we will split the `DFINEDenseHead` into two specialized branches that share the same ViT backbone. We categorize the dataset based on the `keypoints` field in the COCO categories:
* **Symbols** (Categories with `["origin"]` keypoints, including ties and slurs).
* **Lines** (Categories with `["start", "end"]` keypoints).

### A. The Symbol Head
* **Representation:** Bounding Boxes (with optional origin keypoint).
* **Outputs:** `[Classes, cx, cy, log_w, log_h, 4x D-FINE Edge Bins]`
* **Matching & Loss:** GIoU + L1 (on box centers/shapes) + FGL (D-FINE edge distributions).

### B. The Line Head
* **Representation:** Start and End Keypoints (Directed Vectors anchored to the patch center).
* **Outputs:** `[Classes, log_scale1, log_scale2, raw_dx1, raw_dy1, raw_dx2, raw_dy2, 4x D-FINE Keypoint Bins]`
* **Matching & Loss:** **No GIoU.** Matched and penalized using L1 distance on the absolute endpoints, plus FGL for sub-pixel refinement.

## 3. The "Signed Cartesian + Log Scale" Keypoint Formulation
Lines in music scores can be extremely long (staff lines) or very short (barlines), and they can point in any direction. Furthermore, the midpoint of a line is often empty white space, meaning the ViT patch at that location has no visual evidence to predict the line. Therefore, predictions must be anchored strictly to the patch center where the visual evidence exists.

To handle unbounded scaling while allowing lines to point in any direction, we separate the **Scale** from the **Direction** using a Signed Cartesian + Log Scale approach:

1. **Predict Log-Scale Magnitudes (The Reference Scales):**
   For each endpoint, the network predicts a log-space scale. This allows the network to easily reach endpoints that span the entire page without gradient explosion.
   ```python
   S1 = base_anchor_size * torch.exp(log_scale1)
   S2 = base_anchor_size * torch.exp(log_scale2)
   ```

2. **Predict Raw Cartesian Directions:**
   The network predicts raw linear offsets (`raw_dx1, raw_dy1`, etc.). Because these are linear, they can be positive or negative, allowing the vector to point in any direction without the instability of polar coordinates (angles).

3. **Apply D-FINE Residuals:**
   The D-FINE bins output a residual in the range `[-a, a]`. We add this residual to the raw direction, and scale the entire vector by the log-magnitude. Finally, we anchor it to the patch center.
   ```python
   # Final Absolute Endpoints
   x1 = patch_cx + (raw_dx1 + res_x1) * S1
   y1 = patch_cy + (raw_dy1 + res_y1) * S1
   
   x2 = patch_cx + (raw_dx2 + res_x2) * S2
   y2 = patch_cy + (raw_dy2 + res_y2) * S2
   ```

This formulation is mathematically robust: it provides a perfect reference scale (`S1`, `S2`) for the D-FINE sub-pixel refinement, handles extreme aspect ratios via `exp()`, and maintains stable gradients by avoiding angles or arbitrary midpoints.

## 4. Operational Implications
Splitting the head is practically "free" and actually improves system efficiency.

* **Backbone Compute/Memory:** Unchanged (ViT dominates the cost).
* **Head Compute/Memory:** Negligible increase (adding a second tiny MLP).
* **Hungarian Matcher (The Big Win):** Bipartite matching is $O(N^3)$. Splitting the cost matrix into `(Symbols vs Symbols)` and `(Lines vs Lines)` mathematically reduces the complexity and lowers peak VRAM spikes. Dropping GIoU for lines further speeds up the cost calculation.
* **Query Capacity:** We recommend doubling the queries (e.g., 5 for Symbols, 5 for Lines per patch). Because we use Variance-Based Patch Dropping (~90% dropped), doubling the queries on the remaining 10% of patches adds massive detection capacity for dense chords without causing OOM errors.

## 5. Evaluation Strategy (COCO)
Because lines no longer use bounding boxes, standard COCO `mAP` (which relies on box IoU) is mathematically incompatible. We must evaluate them as keypoints.

* **Ground Truth:** Remains a single, unified JSON file. Both symbols and lines keep their respective `bbox` and `keypoints` fields.
* **Predictions:** The inference script will output two separate JSON files: `preds_symbols.json` and `preds_lines.json`.
* **PyCOCOTools:** We will run two separate evaluations, explicitly filtering by Category IDs to prevent cross-contamination:
  1. **Symbols:** `COCOeval(iouType='bbox')` filtered by `catIds` of symbols.
  2. **Lines:** `COCOeval(iouType='keypoints')` filtered by `catIds` of lines. This uses OKS (Object Keypoint Similarity) instead of IoU.

## 6. Summary of Required Code Changes
1. **Dataset:** Ensure the dataset loader extracts and provides the `keypoints` field for line categories.
2. **Detector (`detector.py`):** Split `DFINEDenseHead` into `symbol_mlp` and `line_mlp`. Implement the Signed Cartesian + Log Scale decoding math for the line branch.
3. **Criterion (`criterion.py`):** Create a `loss_line_l1` for endpoints. Drop GIoU for lines. Apply the FGL loss to the endpoint bins using the new `S1/S2` reference scales. Add a weighting factor to balance line loss vs. symbol loss.
4. **Matcher (`matcher.py`):** Split the matching logic. Use GIoU cost for symbols, and Endpoint L1 cost for lines.
5. **Inference & Eval:** Split prediction outputs and implement dual `COCOeval` runs.

## 7. Detailed Implementation Plan

### Step 1: Type Definitions (`src/music_types.py`)
We need to distinguish between Symbol outputs and Line outputs, as well as their respective ground truths.
1. **Add Keypoint Types:** Create `Keypoints` (same shape as `BoundingBoxes` but semantically different).
2. **Update `DetectionTarget` & `DetectionSample`:** Separate the modalities completely to avoid NaN padding.
   * `DetectionSample` will hold `boxes`, `box_labels`, `keypoints`, and `keypoint_labels`.
   * `DetectionTarget` will similarly hold `boxes`, `box_labels`, `keypoints`, and `keypoint_labels`.
3. **Update `DetectionOutput`:** Split the output into two distinct dataclasses: `SymbolOutput` and `LineOutput`. `DetectionOutput` will contain both.
   * `SymbolOutput`: `pred_logits`, `pred_boxes`, `pred_edge_logits`, `absolute_centers`, `learnable_shapes`.
   * `LineOutput`: `pred_logits`, `pred_keypoints`, `pred_endpoint_logits`, `absolute_centers`, `log_scales`, `raw_directions`.

### Step 2: Dataset Parsing (`src/dataset/coco.py`)
We must extract the keypoints and categorize the classes, keeping symbols and lines in separate lists.
1. **Identify Line Categories:** During `parse_coco`, inspect the `keypoints` field of each category. If it contains `["start", "end"]`, mark its ID as a line category. Store a `set` of line category IDs in `CocoDataset`.
2. **Extract Keypoints:** In `load_coco_sample`, when iterating over annotations:
   * If the category is a line (in the line category set), extract the `[x1, y1, v1, x2, y2, v2]` keypoints, convert to `[x1, y1, x2, y2]`, and append to `line_keypoints` and `line_labels`.
   * If the category is a symbol, extract the `bbox`, convert to `[x1, y1, x2, y2]`, and append to `symbol_boxes` and `symbol_labels`.
3. **Return Dual Modalities:** `DetectionSample` now returns the separated tensors for boxes and keypoints.

### Step 3: Transforms (`src/transform/det.py`)
The geometric transformations must apply to both boxes and keypoints.
1. **Crop & Shift:** Update `crop_boxes_xyxy` to also shift the keypoints by `(x, y)`. Unlike boxes, if a keypoint falls outside the crop, we might still want to keep it if the other endpoint is inside (or we can strictly clip them).
2. **Normalization:** Update `normalize_boxes_img` to divide keypoint coordinates by `(width, height)` so they are in `[0, 1]` Float1 space.
3. **Collation:** Update `collate` to stack the new keypoint tensors alongside the box tensors.

### Step 4: Model Architecture (`src/model/detector.py`)
This is where the core mathematical changes happen.
1. **Split the Head:** Rename `DFINEDenseHead` to `SymbolHead`. Create a new `LineHead`.
2. **Implement `LineHead`:**
   * **MLP Output:** `[Classes, log_scale1, log_scale2, raw_dx1, raw_dy1, raw_dx2, raw_dy2, 4x D-FINE Bins]`.
   * **Decoding Math:**
     ```python
     # 1. Log Scales
     S1 = base_anchor_size * torch.exp(log_scale1)
     S2 = base_anchor_size * torch.exp(log_scale2)
     
     # 2. D-FINE Residuals (using the existing DFINEWeightingFunction)
     res = self.weighting_fn(edge_probs) # Shape: (..., 4) -> [res_x1, res_y1, res_x2, res_y2]
     
     # 3. Final Absolute Endpoints
     x1 = patch_cx + (raw_dx1 + res[..., 0]) * S1
     y1 = patch_cy + (raw_dy1 + res[..., 1]) * S1
     x2 = patch_cx + (raw_dx2 + res[..., 2]) * S2
     y2 = patch_cy + (raw_dy2 + res[..., 3]) * S2
     ```
3. **Update `OMRDetector`:** Forward the `patch_tokens` through both `SymbolHead` and `LineHead`. Return the combined `DetectionOutput`.

### Step 5: Bipartite Matching (`src/model/matcher.py`)
The matcher must independently match symbols to symbols, and lines to lines.
1. **Split Targets:** Because the targets are already split in `DetectionTarget` (`boxes` vs `keypoints`), we don't need to filter them by category ID here.
2. **Symbol Matching:** Match `SymbolOutput` against `box` targets using the existing GIoU + L1 cost matrix.
3. **Line Matching:** Match `LineOutput` against `keypoint` targets using **only L1 distance** on the endpoints (no GIoU).
4. **Return:** Return two sets of `MatchIndices` (one for symbols, one for lines).

### Step 6: Loss Computation (`src/model/criterion.py`)
The criterion applies the specific losses to the matched pairs.
1. **Symbol Losses:** Unchanged (Focal Loss, Box L1, Box GIoU, Box FGL).
2. **Line Losses:**
   * **Focal Loss:** Applied to all line queries.
   * **Endpoint L1:** `F.l1_loss(pred_keypoints, gt_keypoints)`.
   * **Endpoint FGL:** Reverse the decoding math to find the target residuals:
     `target_res_x1 = (gt_x1 - patch_cx) / S1 - raw_dx1`
     Pass these target residuals into the D-FINE soft cross-entropy logic.
3. **Total Loss:** Combine all losses. You may need a new weight in `DetectionLossWeights` (e.g., `loss_line_l1`).

### Step 7: Inference (`scripts/inference_coco.py`)
The inference script must output two separate JSON files to prevent evaluation conflicts.
1. **Filter by Confidence:** Apply the confidence threshold to both `SymbolOutput` and `LineOutput`.
2. **Format Symbols:** Convert to `[x, y, w, h]` and append to `preds_symbols.json`.
3. **Format Lines:** Convert to COCO keypoint format `[x1, y1, 1, x2, y2, 1]` (where 1 means visible) and append to `preds_lines.json`.

### Step 8: Evaluation (`scripts/evaluate_coco.py`)
Standard COCO mAP will fail for lines, so we must run two separate evaluations.
1. **Symbol Evaluation:** 
   * Load `preds_symbols.json`.
   * Run `COCOeval(iouType='bbox')`.
   * **Crucial:** Filter the evaluation to only consider symbol `catIds` to prevent it from penalizing the model for "missing" the lines.
2. **Line Evaluation:**
   * Load `preds_lines.json`.
   * Run `COCOeval(iouType='keypoints')`.
   * Filter by line `catIds`.
   * *Note:* Keypoint evaluation requires defining `sigmas` for the Object Keypoint Similarity (OKS). We will need to define a custom sigma array for the start/end points (e.g., `[0.1, 0.1]`).
