import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
from unter import UNTER  # Assuming UNTER model is available
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import nibabel as nib
from glob import glob

# Load SAM2
sam_args = {
    'sam_checkpoint': "cp/sam2_vit_h.pth",
    'model_type': "vit_h",
    'gpu_id': 0,
}

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
sam.to(device=device)
transform = ResizeLongestSide(sam.image_encoder.img_size)

# BRATS Dataset Loader
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
        
        # Select a middle slice (2D) for training
        slice_idx = image.shape[2] // 2  # Take the middle slice
        image = image[:, :, slice_idx, :]  # Shape: (H, W, C)
        mask = mask[:, :, slice_idx]  # Shape: (H, W)
        
        # Normalize image and convert to tensor
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize 0-1
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Channels first
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        if self.transform:
            image = self.transform.apply_image(image.numpy())
            mask = self.transform.apply_image(mask.numpy())
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask

# Load BRATS Dataset
def get_brats_dataloader(data_dir="/home/erezhuberman/data/Task01_BrainTumour", batch_size=4, num_workers=4):
    trainset = BRATSDataset(data_dir, transform=transform)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

train_loader = get_brats_dataloader()

# Define the UNTER Prompt Encoder
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.encoder = UNTER(in_channels=4, out_channels=256)  # BRATS has 4 MRI channels
        self.conv1x1 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        return self.conv1x1(features)

prompt_encoder = PromptEncoder().to(device)
optimizer = optim.Adam(prompt_encoder.parameters(), lr=0.0003, weight_decay=1e-4)

# Training Loop
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

# Run Training
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch()
