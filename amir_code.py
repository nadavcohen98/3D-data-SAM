import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import cv2
import os
import nibabel as nib
from glob import glob
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

# Set CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load SAM2 Model ===
sam_args = {
    'sam_checkpoint': “../../nadavnungi/sam2/sam2_vit_h.pth",
    'model_type': "vit_h",
    'gpu_id': 0,
}
sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
sam.to(device)
transform = ResizeLongestSide(sam.image_encoder.img_size)

# === 3D-UNTER Model Definition ===
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNTER3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=256, base_filters=32):
        super(UNTER3D, self).__init__()
        
        # Encoder
        self.enc1 = Conv3DBlock(in_channels, base_filters)
        self.enc2 = Conv3DBlock(base_filters, base_filters*2)
        self.enc3 = Conv3DBlock(base_filters*2, base_filters*4)
        self.enc4 = Conv3DBlock(base_filters*4, base_filters*8)
        self.pool = nn.MaxPool3d(2)
        
        # 3D Transformer
        transformer_dim = base_filters*8
        self.pos_embedding = nn.Parameter(torch.randn(1, transformer_dim, 
                                                     8, 8, 8) * 0.02)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=8, 
            dim_feedforward=transformer_dim*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose3d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec1 = Conv3DBlock(base_filters*8, base_filters*4)
        self.upconv2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = Conv3DBlock(base_filters*4, base_filters*2)
        self.upconv3 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec3 = Conv3DBlock(base_filters*2, base_filters)
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        # Bottleneck
        x = self.enc4(x)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Apply transformer - reshape for transformer
        b, c, d, h, w = x.shape
        x_flat = x.reshape(b, c, d*h*w).permute(0, 2, 1)  # [B, DHW, C]
        x_transformed = self.transformer(x_flat)
        x = x_transformed.permute(0, 2, 1).reshape(b, c, d, h, w)
        
        # Decoder with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec1(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec3(x)
        
        x = self.final_conv(x)
        return x

# 2D slice extractor for compatibility with SAM
class SliceExtractor(nn.Module):
    def __init__(self, unter_3d, out_channels=256):
        super(SliceExtractor, self).__init__()
        self.unter_3d = unter_3d
        self.adapter = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x_3d, slice_dim=2, slice_indices=None):
        """
        Args:
            x_3d: 5D tensor [B, C, D, H, W]
            slice_dim: dimension to slice (0=batch, 1=channel, 2=depth)
            slice_indices: which indices to extract, if None, use middle slice
        """
        # Process the full 3D volume
        features_3d = self.unter_3d(x_3d)
        
        # Get middle slice if indices not specified
        if slice_indices is None:
            slice_indices = [features_3d.size(slice_dim) // 2] * features_3d.size(0)
        
        # Extract slices
        batch_size = features_3d.size(0)
        slices = []
        
        for b in range(batch_size):
            # Extract slice for current batch
            if slice_dim == 2:  # Slicing depth dimension
                slice_idx = slice_indices[b] if isinstance(slice_indices, list) else slice_indices
                slice_2d = features_3d[b, :, slice_idx, :, :]  # [C, H, W]
            slices.append(slice_2d)
        
        # Stack slices into batch
        slices = torch.stack(slices, dim=0)  # [B, C, H, W]
        
        # Apply 2D adapter for SAM compatibility
        return self.adapter(slices)

# === Enhanced BRATS Dataset Loader for 3D ===
class BRATSDataset3D(data.Dataset):
    def __init__(self, data_dir, transform=None, image_size=(128, 128, 128), 
                 return_3d=True, slice_idx=None, return_original=False):
        self.image_paths = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        self.transform = transform
        self.image_size = image_size
        self.return_3d = return_3d  # Whether to return 3D volume or 2D slice
        self.slice_idx = slice_idx  # Which slice to return if return_3d=False
        self.return_original = return_original  # Whether to return unprocessed original

    def __len__(self):
        return len(self.image_paths)

    def preprocess_volume(self, volume):
        """Preprocess the full 3D volume"""
        # Normalize volume
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
        
        # Resize if needed
        if volume.shape[:3] != self.image_size:
            # Create resized volume
            resized = np.zeros(self.image_size + (volume.shape[3],), dtype=np.float32)
            
            # Resize each channel separately
            for c in range(volume.shape[3]):
                # This is a simple resize - could be improved with interpolation
                vol_c = volume[:,:,:,c]
                zoom_factors = [self.image_size[i]/vol_c.shape[i] for i in range(3)]
                from scipy.ndimage import zoom
                resized[:,:,:,c] = zoom(vol_c, zoom_factors, order=1)  # order=1: linear interpolation
            
            volume = resized
        
        return volume

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load NIfTI images
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()  # Shape: (D, H, W, 4) - 4 modalities
        mask = nib.load(mask_path).get_fdata()  # Shape: (D, H, W)
        
        # Store original for reference if needed
        original = (image.copy(), mask.copy()) if self.return_original else None
        
        # Preprocess volume
        image_processed = self.preprocess_volume(image)
        
        if self.return_3d:
            # Convert to torch tensors
            image_tensor = torch.tensor(image_processed, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, D, H, W)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
            
            if self.return_original:
                return image_tensor, mask_tensor, original
            return image_tensor, mask_tensor
        else:
            # Extract 2D slice for SAM compatibility
            slice_idx = self.slice_idx if self.slice_idx is not None else image.shape[2] // 2
            
            # Extract slice and keep 3 modalities for SAM (exclude FLAIR or use composite channels)
            image_slice = image_processed[:, :, slice_idx, :3]  # (H, W, 3)
            mask_slice = mask[:, :, slice_idx]  # (H, W)
            
            # Convert to torch tensors
            image_tensor = torch.tensor(image_slice, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
            mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            
            if self.return_original:
                return image_tensor, mask_tensor, original
            return image_tensor, mask_tensor

# === Load BRATS Dataset ===
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=1, 
                        num_workers=2, mode="3d", image_size=(128, 128, 128)):
    """
    Create DataLoader for BRATS dataset
    
    Args:
        data_dir: Directory containing BRATS data
        batch_size: Batch size
        num_workers: Number of worker processes
        mode: "3d" or "2d" for full volume or slice-based processing
        image_size: Target image size for resizing
    """
    if mode == "3d":
        dataset = BRATSDataset3D(data_dir, transform=transform, image_size=image_size, return_3d=True)
    else:  # 2d mode
        dataset = BRATSDataset3D(data_dir, transform=transform, image_size=image_size, return_3d=False)
    
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# === Hybrid Architecture: AutoSAM2 + 3D-UNET ===
class HybridSegmenter(nn.Module):
    def __init__(self, sam_model, volume_size=(128, 128, 128), output_classes=1):
        super(HybridSegmenter, self).__init__()
        
        # Components
        self.sam = sam_model
        self.unter_3d = UNTER3D(in_channels=4, out_channels=256)
        self.slice_extractor = SliceExtractor(self.unter_3d)
        
        # Output processing
        self.segmentation_head = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, output_classes, kernel_size=1)
        )
        
    def forward(self, x_3d, guidance_slices=None):
        """
        Args:
            x_3d: 5D tensor [B, C, D, H, W] - The full 3D volume
            guidance_slices: 4D tensor [B, 3, H, W] - 2D slices for SAM guidance (optional)
        """
        batch_size = x_3d.size(0)
        
        # 1. Process full 3D volume with 3D-UNTER
        features_3d = self.unter_3d(x_3d)
        
        # 2. Extract middle slices if guidance not provided
        if guidance_slices is None:
            # Get middle slice indices
            middle_indices = [x_3d.size(2) // 2] * batch_size
            # Extract features for middle slices
            slices_2d = self.slice_extractor(x_3d, slice_indices=middle_indices)
        else:
            # Use pre-selected guidance slices
            with torch.no_grad():
                # Get SAM image embeddings for the guidance slices
                image_embeddings = self.sam.image_encoder(guidance_slices)
            
            # Extract matching feature slices
            slices_2d = self.slice_extractor(x_3d)
        
        # 3. Process selected 2D slices with SAM
        with torch.no_grad():
            # Use extracted 2D features as prompt embeddings
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None, 
                boxes=None, 
                masks=slices_2d  # Use our 2D features as mask prompts
            )
            
            # Process with SAM's mask decoder
            mask_logits, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
        
        # 4. Enhance 3D features using SAM's mask prediction
        # Expand 2D mask to guide 3D segmentation
        mid_depth = features_3d.size(2) // 2
        enhanced_features = features_3d.clone()
        
        # Inject SAM's mask information into 3D features
        for b in range(batch_size):
            # Extract the 2D mask prediction for this batch
            mask_2d = torch.sigmoid(mask_logits[b])  # [1, H, W]
            
            # Create 3D attention weights
            depth = features_3d.size(2)
            attention_3d = torch.zeros(1, 1, depth, mask_2d.size(1), mask_2d.size(2), device=mask_2d.device)
            
            # Apply Gaussian falloff from middle slice
            for d in range(depth):
                # Distance from middle slice
                dist = abs(d - mid_depth) / (depth / 2)
                # Gaussian falloff: exp(-dist²/sigma²)
                falloff = torch.exp(-dist**2 / 0.5**2)
                attention_3d[0, 0, d] = mask_2d * falloff
            
            # Apply attention to features of this batch
            enhanced_features[b] = enhanced_features[b] * (1 + attention_3d)
        
        # 5. Final 3D segmentation using enhanced features
        final_seg = self.segmentation_head(enhanced_features)
        
        return final_seg, mask_logits

# === Define Loss Functions ===
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
        
    def forward(self, pred_3d, target_3d, pred_2d=None, target_2d=None):
        # 3D Dice Loss
        dice = self.dice_loss(pred_3d, target_3d)
        
        # 3D BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred_3d, target_3d)
        
        # Combined loss
        loss = self.dice_weight * dice + self.bce_weight * bce
        
        # Add 2D loss if available
        if pred_2d is not None and target_2d is not None:
            dice_2d = self.dice_loss(pred_2d, target_2d)
            bce_2d = F.binary_cross_entropy_with_logits(pred_2d, target_2d)
            loss_2d = self.dice_weight * dice_2d + self.bce_weight * bce_2d
            loss = loss + 0.5 * loss_2d
            
        return loss

# === Training Loop ===
def train(model, train_loader, optimizer, scaler, criterion, num_epochs=20, 
          validate=True, val_loader=None, save_path='./models/'):
    """Complete training loop with validation"""
    os.makedirs(save_path, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (images_3d, masks_3d) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images_3d, masks_3d = images_3d.to(device), masks_3d.to(device)
            
            # Extract middle slices for SAM guidance
            mid_idx = images_3d.size(2) // 2
            guidance_slices = images_3d[:, :3, mid_idx].clone()  # Use first 3 channels
            
            with autocast(device_type="cuda", dtype=torch.float16):
                # Forward pass
                pred_3d, pred_2d = model(images_3d, guidance_slices)
                
                # Get corresponding 2D masks for the middle slices
                masks_2d = masks_3d[:, :, mid_idx].clone()
                
                # Compute loss
                loss = criterion(pred_3d, masks_3d, pred_2d, masks_2d)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Log batch results
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if validate and val_loader is not None:
            val_loss = validate_model(model, val_loader, criterion)
            print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, f"{save_path}/best_model.pth")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, f"{save_path}/checkpoint_epoch{epoch+1}.pth")
    
    return model

def validate_model(model, val_loader, criterion):
    """Validation loop"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images_3d, masks_3d in tqdm(val_loader, desc="Validation"):
            images_3d, masks_3d = images_3d.to(device), masks_3d.to(device)
            
            # Extract middle slices for SAM guidance
            mid_idx = images_3d.size(2) // 2
            guidance_slices = images_3d[:, :3, mid_idx].clone()
            
            # Forward pass
            pred_3d, pred_2d = model(images_3d, guidance_slices)
            
            # Get corresponding 2D masks for the middle slices
            masks_2d = masks_3d[:, :, mid_idx].clone()
            
            # Compute loss
            loss = criterion(pred_3d, masks_3d, pred_2d, masks_2d)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

# === Main Execution ===
def main():
    # Data loaders
    train_loader = get_brats_dataloader(batch_size=1, mode="3d")
    val_loader = get_brats_dataloader(batch_size=1, mode="3d", data_dir="/home/erezhuberman/data/Task01_BrainTumour_val")
    
    # Model, optimizer, and loss
    model = HybridSegmenter(sam, output_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    criterion = CombinedLoss(dice_weight=1.0, bce_weight=0.5)
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        num_epochs=20,
        validate=True,
        val_loader=val_loader,
        save_path='./models/hybrid_sam_unter/'
    )

if __name__ == "__main__":
    main()
