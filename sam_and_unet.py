import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Ensure use of GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset directory
DATASET_PATH = "/home/erezhuberman/data/Task01_BrainTumour/imagesTr"
SAM_CHECKPOINT = "/home/nadavnungi/sam2/sam_vit_h_4b8939.pth"

# Hyperparameters
BATCH_SIZE = 1  # Reduce batch size to prevent OOM errors on Mac
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
# Dataset Loader with Multi-Channel Input (FLAIR, T1, T1ce, T2)
# ==============================
class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sample_paths = self._find_samples()

        if len(self.sample_paths) == 0:
            raise ValueError(f" No valid samples found in {root_dir}")

    def _find_samples(self):
        sample_paths = []
        for patient_dir in sorted(os.listdir(self.root_dir)):
            full_path = os.path.join(self.root_dir, patient_dir)
            if os.path.isdir(full_path):
                paths = {
                    "flair": os.path.join(full_path, f"{patient_dir}_flair.nii"),
                    "t1": os.path.join(full_path, f"{patient_dir}_t1.nii"),
                    "t1ce": os.path.join(full_path, f"{patient_dir}_t1ce.nii"),
                    "t2": os.path.join(full_path, f"{patient_dir}_t2.nii"),
                    "seg": os.path.join(full_path, f"{patient_dir}_seg.nii"),
                }
                if all(os.path.exists(p) for p in paths.values()):
                    sample_paths.append(paths)
        return sample_paths

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        paths = self.sample_paths[idx]

        # Load all four MRI scans
        scans = {key: nib.load(paths[key]).get_fdata() for key in ["flair", "t1", "t1ce", "t2"]}
        seg_image = nib.load(paths["seg"]).get_fdata()

        # Normalize each scan
        for key in scans:
            scans[key] = (scans[key] - np.min(scans[key])) / (np.max(scans[key]) - np.min(scans[key]) + 1e-8)

        # Convert to PyTorch tensors
        scan_tensors = [torch.tensor(scans[key], dtype=torch.float32) for key in ["flair", "t1", "t1ce", "t2"]]
        seg_image = torch.tensor(seg_image, dtype=torch.float32)

        # Convert segmentation mask to binary (0 = background, 1 = tumor)
        seg_image = (seg_image > 0).float()

        # Extract middle slice
        middle_idx = scan_tensors[0].shape[2] // 2
        scan_tensors = [scan[:, :, middle_idx].unsqueeze(0) for scan in scan_tensors]
        seg_image = seg_image[:, :, middle_idx].unsqueeze(0)

        # Stack all 4 modalities into a single 4-channel tensor
        input_tensor = torch.cat(scan_tensors, dim=0)

        return input_tensor, seg_image

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
    train_loader = DataLoader(BraTSDataset(DATASET_PATH), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(prompt_encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        prompt_encoder.train()
        total_loss, total_dice, total_iou = 0, 0, 0
        num_batches = len(train_loader)

        # Print shape information once per epoch
        first_batch = True  

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            batch_size = images.size(0)

            # Generate prompt embeddings using U-Net
            dense_prompt_embeddings = prompt_encoder(images)

            # Resize MRI images to 1024x1024 for SAM processing
            images_resized = F.interpolate(images[:, :3, :, :], size=(1024, 1024), mode='bilinear', align_corners=False)

            with torch.no_grad():
                # SAM Image Encoder
                image_embeddings = sam.image_encoder(images_resized)

                # SAM Positional Encoding
                image_pe = sam.prompt_encoder.get_dense_pe()

                # Sparse Prompt Embeddings (Zero tensor)
                sparse_prompt_embeddings = torch.zeros((batch_size, 1, sam.prompt_encoder.embed_dim), device=device)

        
            # SAM Mask Decoder
            sam_output, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False
            )

            # Resize SAM output to match ground truth mask size
            sam_output_resized = F.interpolate(sam_output, size=masks.shape[2:], mode="bilinear", align_corners=False)

            # Compute loss & metrics
            loss = dice_loss(sam_output_resized, masks)
            dice_score = dice_coefficient(sam_output_resized, masks).item()
            iou_score = iou(sam_output_resized, masks).item()

            loss.backward()
            optimizer.step()

            # Update epoch metrics
            total_loss += loss.item()
            total_dice += dice_score
            total_iou += iou_score

            # Clear MPS memory cache to prevent OOM
            torch.mps.empty_cache()

        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches

        #  Print loss, Dice coefficient, and IoU for each epoch
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    print("\nTraining complete!")

if __name__ == "__main__":
    train_model()
