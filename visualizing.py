import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
LR = 3e-4
BATCH_SIZE = 1
EPOCHS = 20
DATA_PATH = "/home/erezhuberman/data/Task01_BrainTumour"

# -------------------- Dataset --------------------
class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith("_0000.nii.gz")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        mask_path = img_path.replace("_0000.nii.gz", ".nii.gz")

        # Load image
        image = nib.load(img_path).get_fdata()
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dim

        # Load mask
        mask = nib.load(mask_path).get_fdata()
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dim

        return image, mask

def get_dataloader():
    dataset = BraTSDataset(DATA_PATH)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# -------------------- Model --------------------
class AutoSAM2(nn.Module):
    def __init__(self, num_classes=4):
        super(AutoSAM2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1)  # 4 classes (background + tumors)
        )

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))

# -------------------- Dice Loss --------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        
        intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
        denominator = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
        
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_score.mean()

# -------------------- Training --------------------
def train_model():
    model = AutoSAM2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_dice = DiceLoss()
    criterion_bce = nn.BCELoss()

    train_loader = get_dataloader()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Convert single-channel mask to multi-class one-hot format
            masks = F.one_hot(masks.long(), num_classes=4).squeeze(1).permute(0, 4, 1, 2, 3).float()

            optimizer.zero_grad()
            outputs = model(images)

            # Compute Dice Loss + BCE Loss
            loss_dice = criterion_dice(outputs, masks)
            loss_bce = criterion_bce(outputs, masks)
            loss = 0.5 * loss_dice + 0.5 * loss_bce  # Weighted combination

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
        visualize_batch(images, masks, outputs, epoch)

# -------------------- Visualization --------------------
def visualize_batch(images, masks, outputs, epoch, prefix=""):
    os.makedirs("results", exist_ok=True)
    
    b = 0  # First batch item
    depth = images.shape[2]
    middle_idx = depth // 2

    image_slice = images[b, :, middle_idx].cpu().detach().numpy()
    mask_slice = masks[b, :, middle_idx].cpu().detach().numpy()
    output_slice = F.softmax(outputs[b, :, middle_idx], dim=0).cpu().detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Input Image
    axes[0].imshow(image_slice[0], cmap='gray')
    axes[0].set_title('Input Image (FLAIR)')
    axes[0].axis('off')

    # Ground Truth
    rgb_mask = np.zeros((mask_slice.shape[1], mask_slice.shape[2], 3))
    rgb_mask[mask_slice[1] > 0.5, :] = [1, 1, 0]  # Edema: Yellow
    rgb_mask[mask_slice[2] > 0.5, :] = [0, 1, 0]  # Non-enhancing: Green
    rgb_mask[mask_slice[3] > 0.5, :] = [1, 0, 0]  # Enhancing: Red

    axes[1].imshow(image_slice[0], cmap='gray')
    axes[1].imshow(rgb_mask, alpha=0.5)
    axes[1].set_title('Ground Truth Segmentation')
    axes[1].axis('off')

    # Predicted Output
    rgb_pred = np.zeros((output_slice.shape[1], output_slice.shape[2], 3))
    rgb_pred[output_slice[1] > 0.5, :] = [1, 1, 0]  # Edema: Yellow
    rgb_pred[output_slice[2] > 0.5, :] = [0, 1, 0]  # Non-enhancing: Green
    rgb_pred[output_slice[3] > 0.5, :] = [1, 0, 0]  # Enhancing: Red

    axes[2].imshow(image_slice[0], cmap='gray')
    axes[2].imshow(rgb_pred, alpha=0.5)
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/{prefix}_epoch{epoch}.png")
    plt.close()

# -------------------- Run Training --------------------
if __name__ == "__main__":
    train_model()
