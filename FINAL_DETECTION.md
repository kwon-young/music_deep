# Final Architecture: Dense Patch-as-Predictor with Probabilistic FDR for OMR

## 1. Introduction and Context
Optical Music Recognition (OMR) requires detecting thousands of musical symbols on a single high-resolution page. This task presents extreme challenges:
* **High Density & Polyphony:** Symbols are densely packed, often overlapping (e.g., chords), and lack a strict 1D reading order.
* **Extreme Scale Variance:** Targets range from minute (staccato dots) to massive (page-long barlines, slurs).
* **Boundary Ambiguity:** Ink bleed, stylization, and overlapping elements make rigid bounding box coordinates highly error-prone.

The goal is to design a detection architecture on top of a plain Vision Transformer (ViT) pre-trained via LeJEPA. The architecture must be computationally efficient, avoid the quadratic memory explosion of standard attention mechanisms, handle extreme scale variance, and eliminate heuristic post-processing like Non-Maximum Suppression (NMS).

## 2. Rejected Paradigms and Rationale
To arrive at the optimal architecture, several standard paradigms were evaluated and discarded:
* **Standard DETR / D-FINE Decoders:** Relying on global object queries attending to all image patches results in an $O(N^2)$ computational explosion. For OMR, requiring thousands of queries to attend to thousands of patches is computationally intractable.
* **ViTDet / Convolutional Necks:** Reconstructing 2D grids to apply Feature Pyramid Networks (FPNs) and convolutions introduces locality bias and breaks the pure-transformer sequence paradigm.
* **Autoregressive Sequence-to-Sequence:** While elegant for text, generating thousands of symbols sequentially is too slow for inference and struggles with the 2D polyphonic nature of music scores.
* **Top-K Attention Routing:** Filtering predictions before calculating loss leads to "gradient starvation," where untrained but correct predictions never receive the gradients needed to improve.

## 3. The Proposed Architecture
The final architecture is a **Single-Stage, Query-Free, Dense Patch Predictor** that integrates Learnable Shapes and D-FINE's Fine-grained Distribution Refinement (FDR). 

### 3.1. The ViT Backbone
The image is processed by a plain ViT (with patch dropping disabled during fine-tuning). The backbone outputs a sequence of $P$ patch embeddings (e.g., 4,096 patches for a high-resolution grid). Instead of feeding these into a complex decoder, **every single patch token acts as a local predictor**.

### 3.2. Shared Learnable Shapes (Dynamic Anchors)
To handle extreme scale variance without forcing the network to regress massive absolute distances (which is mathematically unstable), the model utilizes $K$ **Shared Learnable Shapes**.
* We define $K$ learnable parameters (e.g., $K=5$) representing default width and height priors ($w_k, h_k$).
* These $K$ shapes are shared globally across all $P$ patches, ensuring **translation invariance**.
* During training, the network optimizes these shapes to match the dataset's most common geometries (e.g., one shape naturally evolves into a tall, thin vertical rectangle for stems, while another becomes a small square for noteheads).

### 3.3. The Detection Head (MLP) and FDR
A Multi-Layer Perceptron (MLP) is applied independently to each of the $P$ patch tokens. Each patch outputs $K$ distinct predictions (one for each learnable shape). 
For each of the $K$ slots, the MLP predicts:
1. **Confidence Score:** The probability that an object exists.
2. **Class Probabilities:** The classification of the musical symbol.
3. **Center Offsets ($\Delta x_c, \Delta y_c$):** Small spatial shifts relative to the physical center of the patch. This allows a patch to claim an object that is slightly off-center.
4. **FDR Edge Distributions (D-FINE / GFocal):** Instead of predicting rigid distances to the bounding box edges, the network outputs logits across $N$ discrete bins for each of the four edges (Top, Bottom, Left, Right). A Softmax is applied to create a probability distribution. The final edge location is the expected value of this distribution, scaled by the Learnable Shape ($w_k, h_k$). This probabilistic approach elegantly models boundary uncertainty.

Total output size: $P \times K$ predictions (e.g., $4096 \times 5 = 20,480$ candidate objects).

## 4. Training and Inference Dynamics

### 4.1. Bipartite (Hungarian) Matching
Despite outputting $P \times K$ predictions, the architecture does not use traditional one-to-many label assignment. Instead, the 20,480 predictions are flattened and fed into a **Hungarian Matcher**.
* The matcher finds the optimal 1-to-1 pairing between the predictions and the ground truth symbols.
* Matched predictions receive localization (FGL Loss) and classification gradients.
* Unmatched predictions are penalized as "background."
* **Solving the Pigeonhole Principle:** Because each patch has $K$ slots, a single patch can successfully predict up to $K$ distinct objects. If a dense chord contains 4 noteheads within a single patch area, the Hungarian matcher will simply assign each notehead to a different slot within that same patch.

### 4.2. NMS-Free Inference
Because the network is trained via 1-to-1 Hungarian matching, it is explicitly penalized for making duplicate predictions of the same object. Consequently, the model learns to suppress its own duplicates natively. 
During inference, the system simply filters the $P \times K$ outputs by the confidence threshold. **Non-Maximum Suppression (NMS) is completely eliminated**, preserving dense, overlapping musical symbols that heuristic IoU thresholds would otherwise destroy.

## 5. Conclusion
This architecture represents an optimal synthesis for OMR. It avoids the $O(N^2)$ memory bottleneck of DETR by utilizing dense patch prediction, solves the scale variance problem of dense predictors via Shared Learnable Shapes, achieves state-of-the-art boundary precision using D-FINE's probabilistic FDR, and eliminates NMS through Hungarian matching. The result is a fast, pure-transformer detector perfectly tailored for the extreme density of musical scores.
