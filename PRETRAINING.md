# Pretraining Strategy: Dense LeJEPA for OMR

## 1. The Challenge of Pretraining for Dense Detection
The downstream architecture for this Optical Music Recognition (OMR) project is a **Dense Patch-as-Predictor** (as detailed in `FINAL_DETECTION.md`). In this setup, an MLP is applied independently to every single patch token to predict bounding boxes and keypoints. This requires strict **spatial locality**: a patch's embedding must strictly represent the visual features present at that specific physical location.

The standard LeJEPA pretraining objective relies on global average pooling across patches to compute the loss between different augmented views. If we apply asymmetric masking (dropping patches in one view) and use a global loss, the network is forced to "smash" or hallucinate the missing information into the remaining tokens so that the global average matches the unmasked view. 

While excellent for global image classification, this destroys spatial locality. A patch containing empty white space might learn to encode "there is a treble clef 5 inches to my left." When the downstream dense detector looks at that white-space patch, it will be confused and predict false positives.

## 2. Learning Music Visual Grammar via Masking
Music notation is highly structured. Staff lines are continuous, stems connect to noteheads, and barlines align vertically. To teach a Vision Transformer this "visual grammar" during self-supervised pretraining, the most effective method is **Masked Prediction** (as seen in I-JEPA and V-JEPA). 

By masking out a section of a staff line and forcing the model to predict it from the surrounding context, the model inherently learns the rules of musical continuity and structure. However, we want to achieve this without the brittle heuristics (EMA teachers, stop-gradients) that standard JEPA models require to prevent representation collapse.

## 3. The Solution: "Dense LeJEPA"
To preserve spatial locality, learn visual grammar, and maintain a heuristic-free training pipeline, we propose **Dense LeJEPA**. This approach combines the patch-level prediction of I-JEPA with the provable collapse-prevention of LeJEPA (SIGReg).

### 3.1. Architecture Components
1. **Target Encoder (Full Context):**
   The full, unmasked image crop is passed through the ViT Encoder. This produces a dense grid of target patch embeddings. 
   *Crucial difference from I-JEPA:* We do **not** use an Exponential Moving Average (EMA) teacher network. We use the active, gradient-receiving Encoder weights.

2. **Context Encoder (Masked Input):**
   A large block of patches is dropped from the image (e.g., masking out a 4x4 grid in the middle of a measure). The *remaining* patches are passed through the exact same ViT Encoder (sharing weights with the Target Encoder).

3. **The Predictor (The Grammar Teacher):**
   A lightweight Transformer network (e.g., 4-6 layers). It takes the encoded context patches, inserts learnable `[MASK]` tokens into the positions of the dropped patches, and adds 2D Positional Encodings. The Predictor processes this sequence and outputs predicted embeddings specifically for the masked locations.

### 3.2. The Objective Function
The total loss is a combination of a dense prediction loss and a regularization loss:

1. **Dense Prediction Loss (L2):**
   We compute the L2 distance between the Predictor's output for the masked patches and the Target Encoder's output for those exact same patches. Because the loss is computed strictly patch-to-patch, spatial locality is perfectly preserved. No token is forced to absorb distant information.

2. **Collapse Prevention (SIGReg):**
   Because we removed the EMA teacher, the network could easily collapse (e.g., outputting a constant vector for all patches). To prevent this, we apply the **Sketched Isotropic Gaussian Regularization (SIGReg)** directly to the **patch-level embeddings** produced by the Target Encoder. 
   By flattening the batch and sequence dimensions into one large pool of patches, SIGReg forces the patch representation space to follow an isotropic Gaussian distribution. This mathematically guarantees that the embeddings cannot collapse to a constant or a low-dimensional subspace.

## 4. Summary of Benefits for OMR
* **Learns Visual Grammar:** The Predictor is forced to understand musical context to fill in missing patches (e.g., inferring a missing notehead based on a visible stem and ledger lines).
* **Preserves Spatial Locality:** Patch-to-patch L2 loss ensures that patch embeddings only represent their specific local receptive field, perfectly aligning with the downstream Dense Patch-as-Predictor head.
* **Heuristic-Free Stability:** SIGReg eliminates the need for asymmetric learning rates, stop-gradients, and EMA teacher networks, making pretraining significantly more stable and easier to tune.
