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
from torch.amp import autocast, GradScaler  # ✅ Import correct module
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
    def __init__(self, in_channels=3, out_channels=256):  # ✅ Ensuring 3-channel input
        super(UNTER, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1),
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
    def __init__(self, data_dir, transform=None, image_size=(1024, 1024)):  # ✅ Match SAM input size
        self.image_paths = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # ✅ Ensure file exists
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing file: {image_path} or {mask_path}")

        # Load NIfTI images
        image = nib.load(image_path).get_fdata()  # Shape: (128, 128, 4)
        mask = nib.load(mask_path).get_fdata()  # Shape: (128, 128)

        # ✅ Ensure image is not empty
        if image is None or image.size == 0:
            raise ValueError(f"Empty image found at {image_path}")

        # Select a middle slice
        if image.shape[-1] < 3:
            raise ValueError(f"Unexpected number of channels in {image_path}: {image.shape[-1]}")

        image = image[:, :, :3]  # 🔥 Keep only first 3 channels for SAM compatibility

        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize between 0-1

        # ✅ Ensure valid shape before resizing
        if len(image.shape) != 3:
            raise ValueError(f"Unexpected image shape {image.shape} in {image_path}")

        # Resize to SAM expected input size (1024x1024)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)  # Resize to 1024x1024
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

        # Ensure image is (3, 1024, 1024)
        if image.shape[0] != 3:
            print(f"⚠️ Warning: Image shape mismatch {image.shape}. Fixing channels.")
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)  # Convert grayscale to RGB

        # Resize mask
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Convert (H, W) to (1, H, W)

        print(f"✅ Final image shape before return: {image.shape}")  # Should be (3, 1024, 1024)
        return image, mask

# === Load BRATS Dataset ===
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=1, num_workers=2):
    trainset = BRATSDataset(data_dir, transform=transform)
    return data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

train_loader = get_brats_dataloader()

# === Prompt Encoder (Ensuring 3-channel Input) ===
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=3, out_channels=256)  # ✅ Fixed in_channels=3
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return self.conv1x1(features)

# === Define Prompt Encoder (Ensuring 3-channel Input) ===
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=3, out_channels=256)  # ✅ Ensure 3-channel input
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return self.conv1x1(features)

# ✅ Directly initialize the model (no need for `del prompt_encoder`)
prompt_encoder = PromptEncoder().to(device)
optimizer = optim.Adam(prompt_encoder.parameters(), lr=0.0003, weight_decay=1e-4)
scaler = GradScaler("cuda")

# === Training Loop ===
def train_one_epoch():
    torch.cuda.empty_cache()
    prompt_encoder.train()
    loss_fn = nn.BCELoss()

    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        print(f"🔍 Input image shape: {images.shape}")  # Debugging check

        with autocast("cuda", dtype=torch.float16):
            prompt_embeddings = prompt_encoder(images)

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

            loss = loss_fn(masks_pred, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Training loss: {loss.item()}")

# === Run Training ===
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch()
