import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Define dataset directory
DATASET_PATH = "/home/erezhuberman/data/Task01_BrainTumour/imagesTr"

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 10

# ==============================
# ✅ Dataset Loader (Extract Middle Slice of FLAIR)
# ==============================
class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([
            f for f in os.listdir(root_dir) if f.endswith('.nii.gz') and not f.startswith("._")
        ])

        if len(self.file_list) == 0:
            raise ValueError(f"❌ No valid NIfTI files found in {root_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])

        try:
            image = nib.load(file_path).get_fdata()
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            return torch.zeros((1, 240, 240))

        if len(image.shape) == 4:
            image = image[0]  # Select FLAIR

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        image = torch.tensor(image, dtype=torch.float32)
        middle_idx = image.shape[2] // 2
        image = image[:, :, middle_idx]
        image = image.unsqueeze(0)

        return image

# ==============================
# ✅ DataLoader Function
# ==============================
def get_dataloader(root_dir, batch_size=1, num_workers=2):
    dataset = BraTSDataset(root_dir)
    print(f"✅ Found {len(dataset)} samples in {root_dir}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ==============================
# ✅ U-Net Architecture with Batch Normalization
# ==============================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
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
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def crop_tensor(self, target, reference):
        _, _, h, w = reference.shape
        return target[:, :, :h, :w]
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((self.crop_tensor(enc4, dec4), dec4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((self.crop_tensor(enc3, dec3), dec3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((self.crop_tensor(enc2, dec2), dec2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((self.crop_tensor(enc1, dec1), dec1), dim=1)
        dec1 = self.dec1(dec1)
        return self.final_conv(dec1)

# ==============================
# ✅ Loss Functions and Metrics
# ==============================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

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
# ✅ Training Loop
# ==============================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader(DATASET_PATH, batch_size=BATCH_SIZE)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_dice = 0
        total_iou = 0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.randint(0, 2, outputs.shape, dtype=torch.float32).to(device)
            loss = dice_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, labels).item()
            total_iou += iou(outputs, labels).item()
        print(f" Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}, Dice coefficient: {total_dice / len(train_loader):.4f}, IoU: {total_iou / len(train_loader):.4f}")
    print("✅ Training complete!")

if __name__ == "__main__":
    train_model()
