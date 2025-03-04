import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import nibabel as nib
from glob import glob

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SAM2 Model
sam_args = {
    'sam_checkpoint': "../sam2/sam2_vit_h.pth",  # Updated path
    'model_type': "vit_h",
    'gpu_id': 0,
}

sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
sam.to(device=device)
transform = ResizeLongestSide(sam.image_encoder.img_size)

# === UNTER Model Definition ===
class UNTER(nn.Module):
    def __init__(self, in_channels=4, out_channels=256):
        super(UNTER, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Transformer Block
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1),
            num_layers=4
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding Path
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(self.pool(x1)))
        x3 = torch.relu(self.conv3(self.pool(x2)))

        # Transformer Processing
        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, h * w).permute(2, 0, 1)  # Reshape for transformer: (seq_len, batch, channels)
        x3_transformed = self.transformer(x3_flat)
        x3_out = x3_transformed.permute(1, 2, 0).view(b, c, h, w)  # Reshape back

        # Decoding Path
        x = torch.relu(self.upconv1(x3_out))
        x = torch.relu(self.upconv2(x))
        x = self.final_conv(x)

        return x

# === BRATS Dataset Loader ===
class BRATSDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, image_size=(240, 240)):
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
    image = nib.load(image_path).get_fdata()  # Shape: (H, W, D, C)
    mask = nib.load(mask_path).get_fdata()  # Shape: (H, W, D)

    # Select a middle slice (2D)
    slice_idx = image.shape[2] // 2
    image = image[:, :, slice_idx, :]  # Shape: (H, W, C)

    print(f"Image shape after slicing: {image.shape}")  # Debugging line

    # Fix: Ensure image has at most 3 channels (RGB) or 1 channel (Grayscale)
    if image.shape[-1] > 3:
        image = image[:, :, :3]  # Keep only the first 3 channels

    mask = mask[:, :, slice_idx]  # Shape: (H, W)

    # Normalize image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize 0-1
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Channels first
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dim

    if self.transform:
        image = self.transform.apply_image(image.numpy())
        mask = self.transform.apply_image(mask.numpy())
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

    return image, mask

# === Load BRATS Dataset ===
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=4, num_workers=4):
    trainset = BRATSDataset(data_dir, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

train_loader = get_brats_dataloader()

# === Define the UNTER Prompt Encoder ===
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=4, out_channels=256)  # BRATS has 4 MRI channels
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return self.conv1x1(features)

# Initialize the model and optimizer
prompt_encoder = PromptEncoder().to(device)
optimizer = optim.Adam(prompt_encoder.parameters(), lr=0.0003, weight_decay=1e-4)

# === Training Loop ===
def train_one_epoch():
    prompt_encoder.train()
    loss_fn = nn.BCELoss()
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)
        prompt_embeddings = prompt_encoder(images)

        # Forward SAM2 with the generated prompt
        with torch.no_grad():
            image_embeddings = sam.image_encoder(images)

        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None, boxes=None, masks=prompt_embeddings
        )

        masks_pred, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Compute Loss
        loss = loss_fn(masks_pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Training loss: {loss.item()}")

# === Run Training ===
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch()
