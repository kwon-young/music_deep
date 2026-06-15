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
* **Representation:** Start and End Keypoints (Directed Vectors).
* **Outputs:** `[Classes, dx1, dy1, dx2, dy2, 4x D-FINE Keypoint Bins]`
* **Matching & Loss:** **No GIoU.** Matched and penalized using L1 distance on the absolute endpoints, plus FGL for sub-pixel refinement.

## 3. Applying D-FINE to Line Keypoints
We discovered that the D-FINE logic (Fine-grained Distribution Refinement) can be applied to keypoints with almost zero architectural changes. 

The Line Head directly predicts coarse offsets for the start (`dx1, dy1`) and end (`dx2, dy2`) points relative to the fixed patch center (`patch_cx, patch_cy`). 

We pass the 4 sets of keypoint logits through the exact same `DFINEWeightingFunction` to get relative residuals `[-a, a]`. We scale these residuals by a reference length and add them to the coarse endpoints:

```python
# Absolute endpoints = Fixed Patch Center + Coarse Offset + D-FINE Residual
x1 = patch_cx + dx1 + (x1_res * ref_scale)
y1 = patch_cy + dy1 + (y1_res * ref_scale)
x2 = patch_cx + dx2 + (x2_res * ref_scale)
y2 = patch_cy + dy2 + (y2_res * ref_scale)
```

This perfectly aligns with the ground truth (which only has start/end points) and keeps the math clean and anchored to the patch grid.

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
1. **Dataset:** Compute `cx, cy` for GT lines on-the-fly `(x1+x2)/2, (y1+y2)/2` so the matcher can find the closest patches.
2. **Detector (`detector.py`):** Split `DFINEDenseHead` into `symbol_mlp` and `line_mlp`. Route outputs to their respective decoding math.
3. **Criterion (`criterion.py`):** Create a `loss_line_l1` for endpoints. Drop GIoU for lines. Apply the FGL loss to the endpoint bins. Add a weighting factor to balance line loss vs. symbol loss.
4. **Matcher (`matcher.py`):** Split the matching logic. Use GIoU cost for symbols, and Endpoint L1 cost for lines.
5. **Inference & Eval:** Split prediction outputs and implement dual `COCOeval` runs.
