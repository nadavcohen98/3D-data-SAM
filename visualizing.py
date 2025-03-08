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
DATASET_PATH = "/home/nadavnungi/3D-data-SAM/3D-data-SAM"

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 10


# ==============================
# ✅ Step 1: Dataset Loader (Extract Middle Slice)
# ==============================
class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.nii.gz')])
        
        if len(self.file_list) == 0:
            raise ValueError(f"❌ No NIfTI files found in {root_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        image = nib.load(file_path).get_fdata()  # Shape: [H, W, D]
        
        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32)  # [H, W, D]
        
        # Extract **middle slice** from depth axis
        middle_idx = image.shape[2] // 2
        image = image[:, :, middle_idx]  # Shape: [H, W]
        
        # Add channel dimension for CNNs: [1, H, W]
        image = image.unsqueeze(0)
        
        return image


# ==============================
# ✅ Step 2: DataLoader
# ==============================
def get_dataloader(root_dir, batch_size=1, num_workers=2):
    dataset = BraTSDataset(root_dir)
    print(f"✅ Found {len(dataset)} samples in {root_dir}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ==============================
# ✅ Step 3: Simple CNN Model
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 240 * 240, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (tumor vs. no tumor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==============================
# ✅ Step 4: Training Loop
# ==============================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_loader = get_dataloader(DATASET_PATH, batch_size=BATCH_SIZE)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images in train_loader:
            images = images.to(device)  # Move images to GPU/CPU

            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.randint(0, 2, (images.shape[0],)).to(device)  # Random labels (for testing)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📊 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

    print("✅ Training complete!")


# ==============================
# ✅ Run Training
# ==============================
if __name__ == "__main__":
    train_model()
