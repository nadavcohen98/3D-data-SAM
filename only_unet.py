import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

torch.mps.empty_cache() # Clear MPS cache explicitly

# Define dataset directory
DATASET_PATH = "/Users/nadavcohen/Desktop/Universuty/deep_learning/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 15  
WEIGHT_DECAY = 1e-5  
VALIDATION_SPLIT = 0.2  

# ==============================
# Split Dataset into Training and Validation
# ==============================
def split_dataset(dataset, val_split=VALIDATION_SPLIT):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])

# ==============================
# Dataset Loader with Multi-Channel Input (FLAIR, T1, T1ce, T2)
# ==============================
class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sample_paths = self._find_samples()

        if len(self.sample_paths) == 0:
            raise ValueError(f"No valid samples found in {root_dir}")

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

        # Select first modality if 4D
        for key in scans:
            if len(scans[key].shape) == 4:
                scans[key] = scans[key][0]

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
# Our promot Network (U-Net)
# ==============================
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):  # 4-channel input
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)
        return self.final_conv(dec1)

# ==============================
# DataLoader Function
# ==============================
def get_dataloaders(root_dir, batch_size=BATCH_SIZE):
    dataset = BraTSDataset(root_dir)
    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

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
# Training Loop with Model Saving
# ==============================
def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(DATASET_PATH, batch_size=BATCH_SIZE)
    model = UNet(in_channels=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_dice = 0
    best_model_path = "best_unet_model.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_dice, total_iou = 0, 0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks).item()
            total_iou += iou(outputs, masks).item()

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)
        
        model.eval()
        val_loss, val_dice, val_iou = 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks).item()
                val_iou += iou(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f} | Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete! Best Validation Dice Score: {best_dice}")
    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train_model()
