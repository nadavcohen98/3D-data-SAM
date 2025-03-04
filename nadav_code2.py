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
import skimage.transform as sk_transform

# Set CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === BRATS Dataset Loader ===
class BRATSDataset(data.Dataset):
    def __init__(self, data_dir, target_size=(1024, 1024), transform=None):
        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Find image and mask paths
        self.image_paths = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

        # Validate paths
        if not self.image_paths or not self.mask_paths:
            raise ValueError(f"No NIfTI files found in {data_dir}. Check directory structure.")

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch in number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)})")

        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def _robust_resize(self, image, target_size):
        """
        Robust resizing method using skimage for better handling of different input types
        """
        # Ensure image is float32 and normalized
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalize if not already normalized
        if image.max() > 1 or image.min() < 0:
            image = (image - image.min()) / (image.max() - image.min())
        
        # Resize using skimage with different interpolation methods
        try:
            # Try linear interpolation first
            resized = sk_transform.resize(
                image, 
                target_size, 
                order=1,  # Bilinear interpolation
                preserve_range=True,
                anti_aliasing=True
            )
            return resized
        except Exception as e:
            print(f"Resize failed with linear interpolation: {e}")
            try:
                # Fallback to nearest neighbor
                resized = sk_transform.resize(
                    image, 
                    target_size, 
                    order=0,  # Nearest neighbor
                    preserve_range=True
                )
                return resized
            except Exception as e:
                print(f"Resize failed completely: {e}")
                raise ValueError(f"Could not resize image to {target_size}")

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            # Load NIfTI images
            image_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)

            # Get image data
            image = image_nii.get_fdata()
            mask = mask_nii.get_fdata()

            # Debugging print
            print(f"Original image shape: {image.shape}")
            print(f"Original mask shape: {mask.shape}")

            # Handle 4D images (multiple modalities)
            if image.ndim == 4:
                # Select the first 3 modalities (channels)
                image = image[:, :, :3]
            elif image.ndim != 3:
                raise ValueError(f"Unexpected image dimensions: {image.ndim}")

            # Normalize each channel separately
            normalized_image = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                min_val, max_val = np.min(channel), np.max(channel)
                normalized_image[:, :, i] = (channel - min_val) / (max_val - min_val + 1e-8)

            # Resize image and mask
            resized_image = np.stack([
                self._robust_resize(normalized_image[:, :, i], self.target_size) 
                for i in range(normalized_image.shape[2])
            ], axis=-1)

            # Resize mask (use nearest neighbor for segmentation)
            resized_mask = self._robust_resize(mask, self.target_size)

            # Convert to tensors
            image_tensor = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1)
            mask_tensor = torch.tensor((resized_mask > 0).astype(np.float32)).unsqueeze(0)  # Binary mask

            return image_tensor, mask_tensor

        except Exception as e:
            print(f"Error processing image {image_path}:")
            print(f"Full error: {str(e)}")
            raise

# === Load BRATS Dataset ===
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=1, num_workers=2):
    trainset = BRATSDataset(data_dir)
    return data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


train_loader = get_brats_dataloader()

# === UNTER Model Definition ===
class UNTER(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(UNTER, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Simplified transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,  # Match the channel dimension
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Ensure final output matches SAM's expected embedding size
        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x):
        # Convolutional layers
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(self.pool(x1)))
        x3 = torch.relu(self.conv3(self.pool(x2)))

        # Prepare for transformer
        b, c, h, w = x3.shape
        
        # Reshape to (batch, sequence_length, embedding_dim)
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
        
        # Apply transformer
        x3_transformed = self.transformer(x3_flat)
        
        # Reshape back to (batch, channels, height, width)
        x3_out = x3_transformed.permute(0, 2, 1).view(b, c, h, w)

        # Final convolution to match SAM's embedding size
        x = self.final_conv(x3_out)

        return x

# === Prompt Encoder ===
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=3, out_channels=1024)
        
    def forward(self, x):
        features = self.encoder(x)
        return features

# Initialize models
prompt_encoder = PromptEncoder().to(device)
optimizer = optim.Adam(prompt_encoder.parameters(), lr=0.0003, weight_decay=1e-4)
scaler = GradScaler()

# === Training Loop ===
def train_one_epoch():
    torch.cuda.empty_cache()
    prompt_encoder.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            # Generate prompt embeddings
            prompt_embeddings = prompt_encoder(images)

            # Get image embeddings
            with torch.no_grad():
                image_embeddings = sam.image_encoder(images)

            # Create a dummy sparse embedding tensor
            batch_size = images.size(0)
            sparse_embeddings = torch.zeros(
                (batch_size, 0, 256), 
                device=device
            )

            # Predict masks
            masks_pred, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=prompt_embeddings,
                multimask_output=False,
            )

            # Ensure masks_pred and masks are the same shape
            masks_pred = masks_pred.squeeze(1)  # Remove channel dimension
            masks = masks.squeeze(1)  # Remove channel dimension

            # Compute loss
            loss = loss_fn(masks_pred, masks)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_loss}")

# === Run Training ===
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch()
