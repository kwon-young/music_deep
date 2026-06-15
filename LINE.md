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
