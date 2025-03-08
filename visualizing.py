import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ========== DATASET LOADING ========== #
class BraTSDataset(Dataset):
    """
    Custom Dataset for loading BRaTS MRI images.
    - Loads only the **middle slice** of each 3D volume.
    - Normalizes the data (Z-score normalization).
    """
    def __init__(self, root_dir, normalize=True):
        self.root_dir = root_dir
        self.normalize = normalize
        self.file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.file_list)

    def load_nifti(self, file_path):
        """Load a NIfTI file and return a numpy array."""
        data = nib.load(file_path).get_fdata()
        return data.astype(np.float32)

    def preprocess(self, img):
        """Normalize using Z-score normalization (only on non-zero voxels)."""
        nonzero = img > 0
        if np.any(nonzero):
            mean, std = np.mean(img[nonzero]), np.std(img[nonzero])
            img[nonzero] = (img[nonzero] - mean) / (std + 1e-8)
        return img

    def __getitem__(self, idx):
        # Load the MRI scan
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        image = self.load_nifti(file_path)

        # Select the middle slice from the depth dimension
        middle_slice_idx = image.shape[2] // 2  # Get middle slice index
        image_slice = image[:, :, middle_slice_idx]

        # Normalize
        if self.normalize:
            image_slice = self.preprocess(image_slice)

        # Convert to tensor and add channel dimension
        image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)

        return image_tensor


def get_dataloader(root_dir, batch_size=1, num_workers=4):
    """Returns a DataLoader for the BRaTS dataset."""
    dataset = BraTSDataset(root_dir)
    if len(dataset) == 0:
        raise ValueError("No data found! Check dataset path and file structure.")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ========== SIMPLE CNN MODEL ========== #
class SimpleCNN(nn.Module):
    """
    A simple CNN model for MRI slice classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 60 * 60, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (tumor/no tumor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (60, 60))  # Ensure consistent size
        x = x.view(x.shape[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ========== TRAINING FUNCTION ========== #
def train_model():
    """
    Main training function.
    """
    # Set dataset path
    dataset_path = "/home/erezhuberman/data/Task01_BrainTumour/imagesTr"

    # Get DataLoader
    train_loader = get_dataloader(dataset_path, batch_size=4)

    # Initialize model, loss, optimizer
    model = SimpleCNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch.cuda()

            # Fake labels (random for now, real labels should be used)
            labels = torch.randint(0, 2, (images.size(0),)).cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


# ========== VISUALIZATION FUNCTION ========== #
def visualize_batch():
    """
    Visualizes MRI slices from the dataset.
    """
    dataset_path = "/home/erezhuberman/data/Task01_BrainTumour"
    dataloader = get_dataloader(dataset_path, batch_size=4)

    batch = next(iter(dataloader))  # Get first batch
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    for i in range(4):
        img = batch[i].squeeze(0).cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')

    plt.show()


# ========== MAIN EXECUTION ========== #
if __name__ == "__main__":
    train_model()
    visualize_batch()
