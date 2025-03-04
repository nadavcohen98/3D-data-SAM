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

# Set CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load SAM2 Model ===
sam_args = {
    'sam_checkpoint': "../sam2/sam2_vit_h.pth",
    'model_type': "vit_h",
    'gpu_id': 0,
}
sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
sam.to(device)
transform = ResizeLongestSide(sam.image_encoder.img_size)

# === UNTER Model Definition ===
class UNTER(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(UNTER, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True),
            num_layers=4
        )

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(self.pool(x1)))
        x3 = torch.relu(self.conv3(self.pool(x2)))

        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, h * w).permute(2, 0, 1)
        x3_transformed = self.transformer(x3_flat)
        x3_out = x3_transformed.permute(1, 2, 0).view(b, c, h, w)

        x = torch.relu(self.upconv1(x3_out))
        x = torch.relu(self.upconv2(x))
        x = self.final_conv(x)

        return x

# === BRATS Dataset Loader ===
class BRATSDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, image_size=(1024, 1024)):
        self.image_paths = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load NIfTI images
        image = nib.load(image_path).get_fdata()  # Shape: (240, 240, 3, 4)
        mask = nib.load(mask_path).get_fdata()  # Shape: (240, 240, D)

        # Select the middle slice
        slice_idx = image.shape[2] // 2  # Select the middle slice
        image = image[:, :, slice_idx, :]  # Shape: (H, W, C)
        
        # Keep only the first 3 channels for SAM
        image = image[:, :, :3]  # Keep first 3 channels

        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize between 0-1
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)  # Resize to 1024x1024
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

        # Preprocess mask
        mask = mask[:, :, slice_idx]  # Select middle slice
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)  # Binary mask
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Convert (H, W) to (1, H, W)

        return image, mask
        
# === Load BRATS Dataset ===
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=1, num_workers=2):
    trainset = BRATSDataset(data_dir, transform=transform)
    return data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

train_loader = get_brats_dataloader()

# === Prompt Encoder ===
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=3, out_channels=256)
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return self.conv1x1(features)

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
