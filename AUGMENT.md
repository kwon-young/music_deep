# Data Augmentation Strategy for Optical Music Recognition (OMR)

## 1. The Problem: Shortcut Learning and the "Black Blob" Local Minimum
When training a Vision Transformer (ViT) on highly imbalanced and sparse OMR datasets, the network is prone to **Shortcut Learning**. Because black noteheads are extremely frequent and roughly the size of a single patch (e.g., 64x64), the network can achieve a massive initial drop in loss by simply acting as a low-pass density filter: *if the ratio of black-to-white pixels in a patch is high, predict a notehead.*

This creates a deep local minimum. The network relies entirely on absolute pixel density and grid-aligned positions, failing to learn the complex, high-frequency geometric features required to detect rare or thin symbols (clefs, accidentals, staff lines). It also leads to false positives in dense ink areas like bargroups or thick chords.

To break the network out of this local minimum, we must apply targeted data augmentations that destroy these false invariants.

## 2. Recommended Geometric Augmentations

### 2.1. Scale Jittering (Random Resizing)
* **Why:** Currently, the synthetic dataset has a highly stable interline height (~64px). The network memorizes the exact pixel dimensions of symbols.
* **Effect:** Randomly resizing the image before cropping forces the network to learn scale-invariant features and properly utilize the dynamic shape prediction (`w_k, h_k`) rather than relying on static anchor priors.

### 2.2. Random Translation
* **Why:** Symbols in synthetic scores are often perfectly aligned relative to the patch grid.
* **Effect:** Translating the image breaks grid-alignment shortcuts, forcing the network to actually learn and predict the local center offsets (`cx, cy`) rather than assuming objects are always centered in a patch.

## 3. Recommended Pixel-Level Augmentations

### 3.1. Probabilistic Edge Detection
* **Why:** This directly attacks the "black blob" density shortcut. 
* **Effect:** By converting the image to edges (e.g., via a Sobel or Canny filter), the network can no longer rely on counting black pixels. It is forced to learn the actual geometric contours and topology of the symbols.
* **Crucial Detail:** Edge detection perfectly preserves the topological distinction between filled and void symbols (e.g., a filled quarter note has one outer contour, while a hollow half note has two concentric contours).
* **Application:** Because edge detection destroys thickness semantics (making a thick beam look similar to a thin staff line), it should be applied **probabilistically (e.g., 15-20% of the time)**. This forces the network to learn contours as a backup feature without permanently destroying thickness cues.

### 3.2. Morphological Operations (Erosion / Dilation)
* **Why:** Simulates real-world historical score degradation.
* **Effect:** Randomly dilating (thickening) or eroding (thinning) the black ink simulates ink bleed from a heavy printing press or faded ink. This prevents the network from memorizing exact pixel counts and forces it to focus on the structural topology of the symbol.

### 3.3. Elastic Transformations (Slight Warping)
* **Why:** Synthetic scores are perfectly straight.
* **Effect:** Slightly warping the image simulates the curvature of a scanned book page, forcing the network to rely on local relative structures rather than absolute global grids.

## 4. Augmentations to STRICTLY AVOID

### 4.1. Horizontal and Vertical Flips
* **Why:** Music notation is highly chiral (directional) and semantic.
* **Effect:** Flipping a score horizontally turns a valid treble clef into nonsense. Furthermore, the position of a stem relative to a notehead carries semantic meaning (up-stems go on the right, down-stems go on the left). Flipping destroys the visual grammar of music.

### 4.2. Large Rotations and Shears
* **Why:** Music relies on strict horizontal and vertical invariants.
* **Effect:** While a tiny rotation (+/- 2 degrees) is acceptable to simulate scanned page skew, large rotations will destroy the fundamental assumption that staff lines are horizontal and stems are vertical.

## 5. Handling Class Imbalance: Focal Loss Alpha Tuning
While augmentations help with feature extraction, the extreme class imbalance (noteheads vs. rare accidentals) still pulls the gradients toward the majority classes.
* **Avoid Naive Loss Multipliers:** Applying raw inverse-frequency weights to the loss can cause massive gradient explosions for rare classes, destroying the network weights.
* **The Solution:** We already use **Focal Loss**, which naturally down-weights "easy" examples (like the easily detected noteheads). To further balance the classes safely, we should tune the **class-specific `alpha` array** in the Focal Loss. Setting `alpha_c` proportional to the inverse frequency of class `c` provides a bounded, mathematically stable way to balance gradients without explosion.
