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

# === BRATS Dataset Loader ===
class BRATSDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, image_size=(1024, 1024)):
        self.image_paths = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        self.transform = transform
        
        # Ensure image_size is a tuple
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 2:
            self.image_size = image_size
        else:
            raise ValueError("image_size must be an integer or a tuple of two integers")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load NIfTI images
        image = nib.load(image_path).get_fdata()  # Shape: (128, 128, 4)
        mask = nib.load(mask_path).get_fdata()  # Shape: (128, 128)

        # Select the first 3 modalities (channels)
        image = image[:, :, :3]  # Shape: (128, 128, 3)

        # Normalize each channel separately
        normalized_image = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            channel = image[:, :, i]
            min_val, max_val = np.min(channel), np.max(channel)
            normalized_image[:, :, i] = (channel - min_val) / (max_val - min_val + 1e-8)

        # Ensure normalized_image is a valid numpy array
        if normalized_image is None or normalized_image.size == 0:
            raise ValueError(f"Failed to process image: {image_path}")

        # Resize image to match SAM's expected input size
        try:
            resized_image = cv2.resize(normalized_image, self.image_size, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"Resize error for image {image_path}")
            print(f"Image shape: {normalized_image.shape}")
            print(f"Desired size: {self.image_size}")
            raise e
        
        # Convert to tensor and reshape to (C, H, W)
        image_tensor = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1)

        # Preprocess mask
        resized_mask = cv2.resize(mask.astype(np.float32), self.image_size, interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.tensor((resized_mask > 0).astype(np.float32)).unsqueeze(0)  # Binary mask

        return image_tensor, mask_tensor

# === Load BRATS Dataset ===
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=1, num_workers=2):
    trainset = BRATSDataset(data_dir, transform=transform)
    return data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


train_loader = get_brats_dataloader()

# === UNTER Model Definition ===
class UNTER(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super(UNTER, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Simplified transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,  # Match the channel dimension
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Ensure final output matches SAM's expected embedding size
        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x):
        # Convolutional layers
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(self.pool(x1)))
        x3 = torch.relu(self.conv3(self.pool(x2)))

        # Prepare for transformer
        b, c, h, w = x3.shape
        
        # Reshape to (batch, sequence_length, embedding_dim)
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
        
        # Apply transformer
        x3_transformed = self.transformer(x3_flat)
        
        # Reshape back to (batch, channels, height, width)
        x3_out = x3_transformed.permute(0, 2, 1).view(b, c, h, w)

        # Final convolution to match SAM's embedding size
        x = self.final_conv(x3_out)

        return x

# === Prompt Encoder ===
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=3, out_channels=1024)
        
    def forward(self, x):
        features = self.encoder(x)
        return features

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
