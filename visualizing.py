import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ----------------------- Configuration -----------------------
DATA_PATH = "/home/erezhuberman/data/Task01_BrainTumour"
BATCH_SIZE = 1
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Dataset Loader ----------------------
class BraTSDataset(Dataset):
    def __init__(self, data_path=DATA_PATH):
        self.data_dir = os.path.join(data_path, "imagesTr")
        self.label_dir = os.path.join(data_path, "labelsTr")
        self.image_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('_0000.nii.gz', '.nii.gz'))

        image_data = np.transpose(nib.load(image_path).get_fdata(), (3, 0, 1, 2))
        mask_data = nib.load(label_path).get_fdata()
        mask_data = np.expand_dims((mask_data > 0).astype(np.float32), axis=0)

        image = torch.tensor(image_data, dtype=torch.float32)
        mask = torch.tensor(mask_data, dtype=torch.float32)
        return image, mask

def get_dataloader(batch_size=BATCH_SIZE):
    dataset = BraTSDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ---------------------- Model Definition ----------------------
class AutoSAM2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base_channels = 16
        self.enc1 = nn.Sequential(
            nn.Conv3d(4, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*8),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*8),
            nn.LeakyReLU(inplace=True)
        )
        self.prob_map = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*4, kernel_size=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*4, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout3d(0.3)

    def forward(self, x):
        x1 = self.enc1(x)
        x1 = self.dropout(x1)
        x = self.pool1(x1)
        x2 = self.enc2(x)
        x2 = self.dropout(x2)
        x = self.pool2(x2)
        x3 = self.enc3(x)
        x3 = self.dropout(x3)
        x = self.bottleneck(x3)
        x = self.dropout(x)
        prob_maps = self.prob_map(x)
        return F.interpolate(prob_maps, size=x1.shape[2:], mode='trilinear', align_corners=False)

# ---------------------- Training ----------------------
def visualize_batch(images, masks, outputs, epoch):
    os.makedirs("results", exist_ok=True)
    b = 0
    middle_idx = images.shape[2] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(images[b, 0, middle_idx].cpu().numpy(), cmap='gray')
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(masks[b, 0, middle_idx].cpu().numpy(), cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(outputs[b, 0, middle_idx].cpu().detach().numpy(), cmap='gray')
    plt.title("Predicted Mask")
    plt.savefig(f"results/epoch_{epoch}.png")
    plt.close()

def train_model():
    model = AutoSAM2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    train_loader = get_dataloader()

    for epoch in range(EPOCHS):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        visualize_batch(images, masks, outputs, epoch)

if __name__ == "__main__":
    train_model()
