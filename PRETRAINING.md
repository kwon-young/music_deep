# Pretraining Strategy: Dense LeJEPA for OMR

## 1. The Challenge of Pretraining for Dense Detection
The downstream architecture for this Optical Music Recognition (OMR) project is a **Dense Patch-as-Predictor** (as detailed in `FINAL_DETECTION.md`). In this setup, an MLP is applied independently to every single patch token to predict bounding boxes and keypoints. This requires strict **spatial locality**: a patch's embedding must strictly represent the visual features present at that specific physical location.

The standard LeJEPA pretraining objective relies on global average pooling across patches to compute the loss between different augmented views. If we apply asymmetric masking (dropping patches in one view) and use a global loss, the network is forced to "smash" or hallucinate the missing information into the remaining tokens so that the global average matches the unmasked view. 

While excellent for global image classification, this destroys spatial locality. A patch containing empty white space might learn to encode "there is a treble clef 5 inches to my left." When the downstream dense detector looks at that white-space patch, it will be confused and predict false positives.

## 2. Learning Music Visual Grammar via Masking
Music notation is highly structured. Staff lines are continuous, stems connect to noteheads, and barlines align vertically. To teach a Vision Transformer this "visual grammar" during self-supervised pretraining, the most effective method is **Masked Prediction** (as seen in I-JEPA and V-JEPA). 

By masking out a section of a staff line and forcing the model to predict it from the surrounding context, the model inherently learns the rules of musical continuity and structure. However, we want to achieve this without the brittle heuristics (EMA teachers) that standard JEPA models require to prevent representation collapse.

## 3. The Solution: Global SIGReg + Local L2 (Dense LeJEPA)
To preserve spatial locality, learn visual grammar, and maintain a stable training pipeline, we propose a hybrid approach. 

Initially, one might consider applying LeJEPA's Sketched Isotropic Gaussian Regularization (SIGReg) directly to the patch embeddings. However, this introduces the **White-Space Paradox**: 80-90% of sheet music patches are identical blank white space. SIGReg forces embeddings into a smooth Gaussian distribution. If applied at the patch level, the network would be forced to map identical white-space patches to completely different, random vectors to satisfy the Gaussian constraint, injecting massive artificial noise.

Instead, we use **Global SIGReg + Local L2**:

### 3.1. Architecture Components
1. **Target Encoder (Full Context):**
   The full, unmasked image crop is passed through the ViT Encoder. This produces a dense grid of target patch embeddings. We apply a `stop_gradient` to these patch embeddings to provide stable targets for the prediction task. We also compute a **Global Average Pool** of these embeddings.

2. **Context Encoder (Masked Input):**
   A large block of patches is dropped from the image (e.g., masking out a 4x4 grid in the middle of a measure). The *remaining* patches are passed through the exact same ViT Encoder (sharing weights with the Target Encoder).

3. **The Predictor (The Grammar Teacher):**
   A lightweight Transformer network (e.g., 4-6 layers). It takes the encoded context patches, inserts learnable `[MASK]` tokens into the positions of the dropped patches, and adds 2D Positional Encodings. The Predictor processes this sequence and outputs predicted embeddings specifically for the masked locations.

### 3.2. The Objective Function
The total loss is a combination of a dense prediction loss and a global regularization loss:

1. **Dense Prediction Loss (Local L2):**
   We compute the L2 distance between the Predictor's output for the masked patches and the Target Encoder's output for those exact same patches (which are detached via `stop_gradient`). Because the loss is computed strictly patch-to-patch, spatial locality is perfectly preserved. No token is forced to absorb distant information.

2. **Collapse Prevention (Global SIGReg):**
   To prevent the network from collapsing to a constant vector, we apply **SIGReg** to the **Globally Pooled** embeddings from the Target Encoder. By comparing different pages or large crops of music (which are statistically diverse), we satisfy LeJEPA's theoretical requirement for independent, identically distributed samples. This mathematically guarantees the global representation space follows an isotropic Gaussian distribution, preventing dimensional collapse without destroying the network's ability to confidently identify identical white-space patches.

## 4. Summary of Benefits for OMR
* **Solves the White-Space Paradox:** Global SIGReg allows identical white patches to naturally map to the same embedding vector, while ensuring the overall image representations remain diverse and uncollapsed.
* **Learns Visual Grammar:** The Predictor is forced to understand musical context to fill in missing patches (e.g., inferring a missing notehead based on a visible stem and ledger lines).
* **Preserves Spatial Locality:** Patch-to-patch L2 loss ensures that patch embeddings only represent their specific local receptive field, perfectly aligning with the downstream Dense Patch-as-Predictor head.
* **Theoretical Alignment:** Applying SIGReg globally aligns perfectly with the mathematical proofs of the LeJEPA paper, ensuring optimal downstream risk minimization.
