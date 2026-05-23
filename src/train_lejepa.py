import torch
import torch.optim as optim
from pathlib import Path

from model.vit import vit_small
from model.lejepa import LeJEPAEncoder, SIGReg
from transform import create_lejepa_iterator, gpu_random_affine

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    batch_size = 128
    v_views = 4
    lamb = 0.05
    epochs = 100
    lr = 5e-4
    
    manifest_path = Path("data/imslp/imslp.jsonl")
    image_dir = Path("data/imslp/images")
    
    # Model Setup
    # Note: image channels=1 since `create_lejepa_iterator` uses "L" (Grayscale)
    backbone = vit_small(
        image_size=224, 
        num_classes=0, 
        channels=1, 
        num_keep_patches=128
    )
    encoder = LeJEPAEncoder(backbone, embed_dim=384, proj_dim=16).to(device)
    sigreg = SIGReg().to(device)
    
    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=0.05)
    
    for epoch in range(epochs):
        encoder.train()
        iterator = create_lejepa_iterator(manifest_path, image_dir, batch_size, v_views)
        
        for step, batch in enumerate(iterator):
            batch = batch.to(device)
            N, V, C, H, W = batch.shape
            
            # Flatten batch and views for the model, then apply GPU augmentations
            x_flat = batch.view(N * V, C, H, W)
            x_aug = gpu_random_affine(x_flat)
            
            # Forward pass
            emb, proj = encoder(x_aug, random_drop=True)
            
            # Reshape projector output to (V, N, D) for the invariance and SIGReg loss
            proj = proj.view(N, V, -1).transpose(0, 1)
            
            # Compute losses
            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            
            loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{step}] "
                      f"Loss: {loss.item():.4f} "
                      f"(SIGReg: {sigreg_loss.item():.4f}, Inv: {inv_loss.item():.4f})")

if __name__ == "__main__":
    train()
