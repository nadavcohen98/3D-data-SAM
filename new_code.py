import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import cv2
from glob import glob
from segment_anything import sam_model_registry, SamPredictor

torch.cuda.empty_cache()  # Clear CUDA cache explicitly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset directory
DATASET_PATH = "/home/erezhuberman/data/Task01_BrainTumour"
SAM_CHECKPOINT = "/home/nadavnungi/sam2/sam_vit_h_4b8939.pth"

# Hyperparameters
BATCH_SIZE = 1  # Reduce batch size to prevent OOM errors
LEARNING_RATE = 1e-3
EPOCHS = 10
WEIGHT_DECAY = 1e-5

# Load Frozen SAM Model
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
sam.eval()  # Freeze the entire SAM model

# ==============================
# Loss Functions and Metrics
# ==============================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# ==============================
# Dataset Loader
# ==============================
class BRATSDataset(Dataset):
    def __init__(self, data_dir, image_size=(1024, 1024)):
        self.image_paths = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load NIfTI images
        image = nib.load(image_path).get_fdata()  # Shape: (240, 240, D, 4)
        mask = nib.load(mask_path).get_fdata()  # Shape: (240, 240, D)

        # Select the middle slice
        slice_idx = image.shape[2] // 2  # Middle slice
        image = image[:, :, slice_idx, :3]  # Keep first 3 channels

        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Resize mask and convert to single channel
        mask = cv2.resize(mask[:, :, slice_idx], self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
        
# ==============================
# U-Net Encoder as Prompt Encoder
# ==============================
class UNetPromptEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=256):
        super(UNetPromptEncoder, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.final_conv = nn.Conv2d(1024, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        x = self.final_conv(bottleneck)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x

# Initialize the prompt encoder
prompt_encoder = UNetPromptEncoder().to(device)

# ==============================
# Training Loop
# ==============================
def train_model():
    train_loader = DataLoader(BRATSDataset(DATASET_PATH), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(sam.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        sam.train()
        total_loss, total_dice, total_iou = 0, 0, 0
        num_batches = len(train_loader)

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Resize images for SAM
            images_resized = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)

            with torch.no_grad():
                image_embeddings = sam.image_encoder(images_resized)
                image_pe = sam.prompt_encoder.get_dense_pe()
                sparse_prompt_embeddings = torch.zeros((BATCH_SIZE, 1, sam.prompt_encoder.embed_dim), device=device)

            sam_output, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=None,
                multimask_output=False
            )

            sam_output_resized = F.interpolate(sam_output, size=masks.shape[2:], mode="bilinear", align_corners=False)

            loss = dice_loss(sam_output_resized, masks)
            dice_score = dice_coefficient(sam_output_resized, masks).item()
            iou_score = iou(sam_output_resized, masks).item()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score
            total_iou += iou_score

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    print("\nTraining complete!")

if __name__ == "__main__":
    train_model()
