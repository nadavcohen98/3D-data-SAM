# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import SAM2
try:
    import sam2
    from sam2.modeling import Sam2Model
    HAS_SAM2 = True
except ImportError:
    print("WARNING: sam2 package not available. Using mock implementation.")
    HAS_SAM2 = False

class Standard3DUNetEncoder(nn.Module):
    """
    Standard 3D UNet encoder to replace SAM2's prompt encoder
    """
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        
        # Encoder path - simple 3D UNet structure
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # Projection to SAM2 embedding format
        self.prompt_proj = nn.Conv3d(base_channels*8, 256, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.2)
    
    def forward(self, x):
        """
        Forward pass through the encoder
        Returns both intermediate features and final embeddings
        """
        # Encoder path with dropout
        x1 = self.enc1(x)
        x1 = self.dropout(x1)
        
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x2 = self.dropout(x2)
        
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x3 = self.dropout(x3)
        
        x = self.pool3(x3)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # Project to SAM2 prompt format
        embeddings = self.prompt_proj(x)
        
        return embeddings, [x1, x2, x3]

class AutoSAM2(nn.Module):
    """
    Implementation of AutoSAM2 that adapts the Segment Anything Model 2 for medical images
    by replacing its prompt encoder with a custom 3D UNet encoder
    """
    def __init__(self, use_real_sam2=True, process_3d=False):
        super().__init__()
        
        # Store configuration
        self.process_3d = process_3d
        
        # Custom 3D UNet encoder for generating prompts
        self.prompt_encoder = Standard3DUNetEncoder(in_channels=4, base_channels=16)
        
        # Initialize SAM2 model if available
        self.use_real_sam2 = use_real_sam2 and HAS_SAM2
        
        if self.use_real_sam2:
            try:
                # Create SAM2 model
                self.sam2 = Sam2Model.from_pretrained()
                
                # Freeze SAM2 
                for param in self.sam2.parameters():
                    param.requires_grad = False
                
                print("Initialized real SAM2 model")
            except Exception as e:
                print(f"Error initializing SAM2 model: {e}")
                print("Falling back to simplified implementation")
                self.use_real_sam2 = False
        
        # If real SAM2 isn't available, use a simplified decoder
        if not self.use_real_sam2:
            self.simplified_decoder = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)
            )
            print("Using simplified decoder (no actual SAM2)")
    
    def prepare_slice_for_sam2(self, slice_data):
        """Convert BraTS slice format to SAM2-compatible format"""
        # Assume slice_data has shape [B, 4, H, W] (4 MRI modalities)
        
        # For SAM2, we need RGB images
        if slice_data.shape[1] == 4:
            # Use the first 3 modalities as RGB channels
            rgb_slice = slice_data[:, :3]
            
            # Alternatively, use one modality (FLAIR) and repeat it 3 times
            # rgb_slice = slice_data[:, 0:1].repeat(1, 3, 1, 1)
        else:
            # If already 3 channels or 1 channel
            if slice_data.shape[1] == 1:
                rgb_slice = slice_data.repeat(1, 3, 1, 1)
            else:
                rgb_slice = slice_data
        
        return rgb_slice
    
    def prepare_embeddings_for_sam2(self, embeddings, slice_height, slice_width):
        """Format 3D embeddings for SAM2's streaming memory system"""
        # This is a placeholder - the actual implementation will depend on SAM2's API
        # For now, we'll average over the depth dimension and resize to match slice dimensions
        
        # Average over depth dimension
        if len(embeddings.shape) == 5:  # If 5D tensor [B, C, D, H, W]
            embeddings_2d = torch.mean(embeddings, dim=2)
        else:
            embeddings_2d = embeddings
        
        # Resize to match slice dimensions
        embeddings_2d = F.interpolate(
            embeddings_2d, 
            size=(slice_height, slice_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        return embeddings_2d
        
    def forward(self, x):
        """
        Forward pass through the AutoSAM2 model
        x: 3D medical image with shape [B, 4, D, H, W]
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Generate embeddings from custom prompt encoder
        embeddings, features = self.prompt_encoder(x)
        
        if self.process_3d:
            # Process entire 3D volume
            all_masks = []
            
            # Process in chunks to save memory
            chunk_size = min(8, depth)
            
            for start_idx in range(0, depth, chunk_size):
                end_idx = min(start_idx + chunk_size, depth)
                masks_chunk = []
                
                for slice_idx in range(start_idx, end_idx):
                    # Get current slice
                    current_slice = x[:, :, slice_idx, :, :]
                    
                    # Prepare for SAM2
                    rgb_slice = self.prepare_slice_for_sam2(current_slice)
                    
                    # Get corresponding embedding for this slice
                    slice_embed_idx = min(slice_idx // (depth // embeddings.shape[2]), embeddings.shape[2] - 1)
                    slice_embedding = embeddings[:, :, slice_embed_idx]
                    
                    if self.use_real_sam2:
                        try:
                            # Format embeddings for SAM2
                            prompt_embedding = self.prepare_embeddings_for_sam2(slice_embedding, height, width)
                            
                            # Generate mask using SAM2
                            # Note: This is a placeholder - will need to be updated with the actual SAM2 API
                            mask = self.sam2(
                                images=rgb_slice,
                                prompts=prompt_embedding
                            )["masks"]
                        except Exception as e:
                            print(f"Error in SAM2 processing: {e}")
                            # Fallback to simplified decoder
                            prompt_2d = torch.mean(slice_embedding, dim=2)
                            mask = self.simplified_decoder(prompt_2d)
                    else:
                        # Use simplified decoder
                        prompt_2d = F.interpolate(
                            slice_embedding, 
                            size=(height//4, width//4),  # Adjust size based on your decoder
                            mode='bilinear', 
                            align_corners=False
                        )
                        mask = self.simplified_decoder(prompt_2d)
                        
                        # Resize mask to match input dimensions
                        mask = F.interpolate(mask, size=(height, width), mode='bilinear', align_corners=False)
                    
                    masks_chunk.append(mask)
                
                # Stack masks for this chunk
                masks_chunk = torch.stack(masks_chunk, dim=2)
                all_masks.append(masks_chunk)
            
            # Combine all chunks
            all_masks = torch.cat(all_masks, dim=2)
            return all_masks
        else:
            # Process only the middle slice
            middle_slice_idx = depth // 2
            middle_slice = x[:, :, middle_slice_idx, :, :]
            
            # Prepare for SAM2
            rgb_slice = self.prepare_slice_for_sam2(middle_slice)
            
            # Get embedding for middle slice
            middle_embed_idx = min(middle_slice_idx // (depth // embeddings.shape[2]), embeddings.shape[2] - 1)
            middle_embedding = embeddings[:, :, middle_embed_idx]
            
            if self.use_real_sam2:
                try:
                    # Format embeddings for SAM2
                    prompt_embedding = self.prepare_embeddings_for_sam2(middle_embedding, height, width)
                    
                    # Generate mask using SAM2
                    mask = self.sam2(
                        images=rgb_slice,
                        prompts=prompt_embedding
                    )["masks"]
                except Exception as e:
                    print(f"Error in SAM2 processing: {e}")
                    # Fallback to simplified decoder
                    prompt_2d = F.interpolate(
                        middle_embedding, 
                        size=(height//4, width//4),
                        mode='bilinear', 
                        align_corners=False
                    )
                    mask = self.simplified_decoder(prompt_2d)
            else:
                # Use simplified decoder
                prompt_2d = F.interpolate(
                    middle_embedding, 
                    size=(height//4, width//4),
                    mode='bilinear', 
                    align_corners=False
                )
                mask = self.simplified_decoder(prompt_2d)
                
                # Resize mask to match input dimensions
                mask = F.interpolate(mask, size=(height, width), mode='bilinear', align_corners=False)
            
            return mask

# fix_dataloaders.py
import torch
import random
import os
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import numpy as np
import torch.nn.functional as F

class BraTSDatasetFixed(Dataset):
    """A clean implementation of the BraTS dataset loader"""
    def __init__(self, root_dir, normalize=True, target_shape=(64, 128, 128)):
        self.root_dir = root_dir
        self.normalize = normalize
        self.target_shape = target_shape
        
        # Find all image files in the training directory
        self.data_dir = os.path.join(root_dir, "imagesTr")
        self.label_dir = os.path.join(root_dir, "labelsTr")
        
        # Get all files with .nii.gz extension, excluding hidden files
        self.image_files = sorted([f for f in os.listdir(self.data_dir) 
                                   if f.endswith('.nii.gz') and not f.startswith('._')])
        
        print(f"Found {len(self.image_files)} image files in {self.data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Load image
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            image_data = nib.load(image_path).get_fdata()
            
            # Convert to channels-first format
            if len(image_data.shape) == 4:  # Already has channels
                image_data = np.transpose(image_data, (3, 0, 1, 2))
            else:  # Single channel
                image_data = np.expand_dims(image_data, axis=0)
            
            # Load mask
            label_file = self.image_files[idx].replace('_0000.nii.gz', '.nii.gz')
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                mask_data = nib.load(label_path).get_fdata()
                # Convert to binary mask (any tumor class = 1)
                mask_data = (mask_data > 0).astype(np.float32)
                mask_data = np.expand_dims(mask_data, axis=0)  # Add channel dimension
            else:
                mask_data = np.zeros((1,) + image_data.shape[1:])
            
            # Convert to PyTorch tensors
            image = torch.tensor(image_data, dtype=torch.float32)
            mask = torch.tensor(mask_data, dtype=torch.float32)
            
            # Apply normalization
            if self.normalize:
                image = self._normalize(image)
            
            # Resize to target shape if specified
            if self.target_shape is not None:
                image, mask = self._resize(image, mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            # Return dummy data
            dummy_shape = (4, *self.target_shape) if self.target_shape else (4, 240, 240, 155)
            return torch.zeros(dummy_shape, dtype=torch.float32), torch.zeros((1, *dummy_shape[1:]), dtype=torch.float32)
    
    def _normalize(self, image):
        """Normalize image intensities"""
        for c in range(image.shape[0]):  # For each modality
            # Non-zero normalization (ignore background)
            mask = image[c] > 0
            if mask.sum() > 0:  # Avoid division by zero
                mean = torch.mean(image[c][mask])
                std = torch.std(image[c][mask])
                image[c][mask] = (image[c][mask] - mean) / (std + 1e-8)
            
            # For zeros, just keep as zeros
            image[c][~mask] = 0.0
        
        return image
    
    def _resize(self, image, mask):
        """Resize image and mask to target shape"""
        target_depth, target_height, target_width = self.target_shape
        
        # Resize using interpolation
        resized_image = F.interpolate(
            image.unsqueeze(0),  # Add batch dimension
            size=(target_depth, target_height, target_width),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        resized_mask = F.interpolate(
            mask.unsqueeze(0),  # Add batch dimension
            size=(target_depth, target_height, target_width),
            mode='nearest'
        ).squeeze(0)  # Remove batch dimension
        
        return resized_image, resized_mask

def create_dataloaders(root_dir, batch_size=1, max_samples=None, target_shape=(64, 128, 128)):
    """Create train and validation dataloaders with proper splitting"""
    
    # Create dataset
    dataset = BraTSDatasetFixed(
        root_dir=root_dir,
        normalize=True,
        target_shape=target_shape
    )
    
    # Set reproducible random seed
    random.seed(42)
    torch.manual_seed(42)
    
    # Determine total number of samples to use
    total_available = len(dataset)
    if max_samples is not None and max_samples < total_available:
        total_samples = max_samples
    else:
        total_samples = total_available
    
    # Create indices for the dataset
    indices = list(range(total_available))
    random.shuffle(indices)
    selected_indices = indices[:total_samples]
    
    # Split indices into training and validation sets (80/20 split)
    train_size = int(0.8 * len(selected_indices))
    val_size = len(selected_indices) - train_size
    
    train_indices = selected_indices[:train_size]
    val_indices = selected_indices[train_size:]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Created training dataset with {len(train_dataset)} samples")
    print(f"Created validation dataset with {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"Created training dataloader with {len(train_loader)} batches")
    print(f"Created validation dataloader with {len(val_loader)} batches")
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(
        root_dir="./data/Task01_BrainTumour",
        batch_size=1,
        max_samples=64,
        target_shape=(64, 128, 128)
    )
    
    # Check first batch from each loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print(f"Train batch shapes: {train_batch[0].shape}, {train_batch[1].shape}")
    print(f"Val batch shapes: {val_batch[0].shape}, {val_batch[1].shape}")
    
    print(f"Train mask values: min={train_batch[1].min()}, max={train_batch[1].max()}, mean={train_batch[1].mean()}")
    print(f"Val mask values: min={val_batch[1].min()}, max={val_batch[1].max()}, mean={val_batch[1].mean()}")


# dataset.py

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from tqdm import tqdm

def preprocess_brats_data(data, normalize=True, clip_percentile=True):
    """
    Enhanced preprocessing for BRaTS data with improved normalization
    
    Args:
        data: BRaTS MRI data of shape (4, H, W, D) - 4 modalities
        normalize: Whether to apply z-score normalization
        clip_percentile: Whether to clip outliers based on percentiles
    
    Returns:
        Preprocessed data
    """
    # Make a copy to avoid modifying original data
    processed_data = data.clone()
    
    # Apply z-score normalization per modality and per volume
    if normalize:
        for c in range(processed_data.shape[0]):  # For each modality
            # Non-zero normalization (ignore background)
            mask = processed_data[c] > 0
            if mask.sum() > 0:  # Avoid division by zero
                mean = torch.mean(processed_data[c][mask])
                std = torch.std(processed_data[c][mask])
                processed_data[c][mask] = (processed_data[c][mask] - mean) / (std + 1e-8)
            
            # For zeros, just keep as zeros
            processed_data[c][~mask] = 0.0
            
            # Clip outliers to improve stability
            if clip_percentile:
                if mask.sum() > 0:
                    p1, p99 = torch.quantile(processed_data[c][mask], torch.tensor([0.01, 0.99]))
                    processed_data[c] = torch.clamp(processed_data[c], min=p1, max=p99)
    
    return processed_data

def apply_augmentations(image, mask, probability=0.5):
    """Apply optional data augmentations to the image and mask"""
    # Skip augmentation based on probability
    if random.random() > probability:
        return image, mask
    
    # Choose one random augmentation instead of applying multiple
    aug_type = random.choice(['flip_h', 'flip_v', 'rotate', 'intensity'])
    
    if aug_type == 'flip_h':
        # Random horizontal flip
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
    
    elif aug_type == 'flip_v':
        # Random vertical flip
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])
    
    elif aug_type == 'rotate':
        # Random 90-degree rotation
        k = random.choice([1, 2, 3])  # Number of 90-degree rotations
        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])
    
    elif aug_type == 'intensity':
        # Subtle intensity shifts (only 5-10% change to avoid artifacts)
        for c in range(image.shape[0]):
            shift = random.uniform(-0.05, 0.05)
            scale = random.uniform(0.95, 1.05)
            image[c] = image[c] * scale + shift
    
    return image, mask

class BraTSDataset(Dataset):
    def __init__(self, root_dir, train=True, normalize=True, max_samples=None, 
                 filter_empty=False, use_augmentation=False, target_shape=None, 
                 cache_data=False, verbose=True):
        """
        Enhanced BraTS dataset with efficient data loading
        
        Args:
            root_dir: Path to BraTS dataset
            train: Load training or validation data
            normalize: Whether to apply z-score normalization
            max_samples: Maximum number of samples to load
            filter_empty: Whether to filter out scans with no tumor
            use_augmentation: Whether to apply data augmentation (train only)
            target_shape: Target shape for resizing (depth, height, width)
            cache_data: Whether to cache data in memory (speeds up training but uses more RAM)
            verbose: Whether to print progress information
        """
        self.root_dir = root_dir
        self.train = train
        self.normalize = normalize
        self.use_augmentation = use_augmentation and train
        self.target_shape = target_shape
        self.cache_data = cache_data
        self.verbose = verbose
        
        # Cache for faster data loading
        self.data_cache = {} if cache_data else None
        
        # For MONAI dataset structure
        if "Task01_BrainTumour" in root_dir:
            if train:
                self.data_dir = os.path.join(root_dir, "imagesTr")
                self.label_dir = os.path.join(root_dir, "labelsTr")
            else:
                self.data_dir = os.path.join(root_dir, "imagesTs")
                self.label_dir = os.path.join(root_dir, "labelsTs")
                
                # If validation directory doesn't exist, use training data
                if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
                    if verbose:
                        print("Validation directory empty, using training data for validation")
                    self.data_dir = os.path.join(root_dir, "imagesTr")
                    self.label_dir = os.path.join(root_dir, "labelsTr")
            
            # Get all files with .nii.gz extension, excluding hidden files
            self.image_files = sorted([f for f in os.listdir(self.data_dir) 
                                      if f.endswith('.nii.gz') and not f.startswith('._')])
            
            # Filter samples to only include those with tumors
            if filter_empty:
                if verbose:
                    print("Filtering dataset to include only samples with tumors...")
                filtered_image_files = []
                
                for img_file in tqdm(self.image_files, desc="Filtering empty samples", disable=not verbose):
                    label_file = img_file.replace('_0000.nii.gz', '.nii.gz')
                    label_path = os.path.join(self.label_dir, label_file)
                    
                    if os.path.exists(label_path):
                        # Check if mask has any non-zero values (tumor)
                        try:
                            mask_data = nib.load(label_path).get_fdata()
                            has_tumor = np.sum(mask_data) > 0
                            
                            if has_tumor:
                                filtered_image_files.append(img_file)
                        except Exception as e:
                            if verbose:
                                print(f"Error loading {label_path}: {e}")
                
                if verbose:
                    print(f"Found {len(filtered_image_files)} samples with tumors out of {len(self.image_files)} total")
                self.image_files = filtered_image_files
            
            # Split for train/val
            if max_samples is not None:
                # For reproducibility
                random.seed(42)
                
                if train:
                    # Use first portion for training
                    train_size = min(max_samples, len(self.image_files))
                    self.image_files = self.image_files[:train_size]
                    if verbose:
                        print(f"Using {len(self.image_files)} samples for training")
                else:
                    # For validation, use last portion
                    val_size = min(max_samples, len(self.image_files)//5)  # Use 20% for validation
                    # Use last samples for validation to avoid overlap with training
                    self.image_files = self.image_files[-val_size:]
                    if verbose:
                        print(f"Using {len(self.image_files)} samples for validation")
        else:
            # Regular BraTS structure
            if train:
                self.data_dir = os.path.join(root_dir, "BraTS2020_TrainingData")
            else:
                self.data_dir = os.path.join(root_dir, "BraTS2020_ValidationData")
            
            self.patient_dirs = sorted(os.listdir(self.data_dir))
            if max_samples:
                self.patient_dirs = self.patient_dirs[:max_samples]
    
    def __len__(self):
        if "Task01_BrainTumour" in self.root_dir:
            return len(self.image_files)
        else:
            return len(self.patient_dirs)
    
    def _resize_volume(self, image, mask):
        """Resize volume to target dimensions if specified"""
        if self.target_shape is None:
            return image, mask
            
        target_depth, target_height, target_width = self.target_shape
        
        # Resize using interpolation
        resized_image = F.interpolate(
            image.unsqueeze(0),  # Add batch dimension
            size=(target_depth, target_height, target_width),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        resized_mask = F.interpolate(
            mask.unsqueeze(0),  # Add batch dimension
            size=(target_depth, target_height, target_width),
            mode='nearest'
        ).squeeze(0)  # Remove batch dimension
        
        return resized_image, resized_mask
    
    def _load_monai_data(self, idx):
        """Load data from MONAI's Task01_BrainTumour structure with error handling"""
        # Check if data is in cache
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        try:
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            image_data = nib.load(image_path).get_fdata()
            
            # The Task01_BrainTumour dataset has 4 modalities in the 4th dimension
            image_data = np.transpose(image_data, (3, 0, 1, 2))
            
            # Load mask
            label_file = self.image_files[idx].replace('_0000.nii.gz', '.nii.gz')
            label_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(label_path):
                mask_data = nib.load(label_path).get_fdata()
                # Convert to binary mask (any tumor class = 1)
                mask_data = (mask_data > 0).astype(np.float32)
                mask_data = np.expand_dims(mask_data, axis=0)  # Add channel dimension
            else:
                mask_data = np.zeros((1,) + image_data.shape[1:])
            
            # Convert to PyTorch tensors
            image = torch.tensor(image_data, dtype=torch.float32)
            mask = torch.tensor(mask_data, dtype=torch.float32)
            
            # Store in cache if enabled
            if self.cache_data:
                self.data_cache[idx] = (image, mask)
            
            return image, mask
        except Exception as e:
            if self.verbose:
                print(f"Error loading image {self.image_files[idx]}: {e}")
            # Return dummy data
            dummy_shape = (4, 240, 240, 155) if self.target_shape is None else (4, *self.target_shape)
            return torch.zeros(dummy_shape, dtype=torch.float32), torch.zeros((1, *dummy_shape[1:]), dtype=torch.float32)
    
    def _load_brats_data(self, idx):
        """Load data from standard BraTS structure"""
        # Check if data is in cache
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        patient_path = os.path.join(self.data_dir, self.patient_dirs[idx])
        
        try:
            # Load all modalities (FLAIR, T1w, T1gd, T2w)
            flair_path = os.path.join(patient_path, f"{self.patient_dirs[idx]}_flair.nii")
            t1_path = os.path.join(patient_path, f"{self.patient_dirs[idx]}_t1.nii")
            t1ce_path = os.path.join(patient_path, f"{self.patient_dirs[idx]}_t1ce.nii")
            t2_path = os.path.join(patient_path, f"{self.patient_dirs[idx]}_t2.nii")
            
            # Check for .nii.gz extensions if .nii doesn't exist
            if not os.path.exists(flair_path):
                flair_path = flair_path + ".gz"
            if not os.path.exists(t1_path):
                t1_path = t1_path + ".gz"
            if not os.path.exists(t1ce_path):
                t1ce_path = t1ce_path + ".gz"
            if not os.path.exists(t2_path):
                t2_path = t2_path + ".gz"
            
            flair = nib.load(flair_path).get_fdata()
            t1w = nib.load(t1_path).get_fdata()
            t1gd = nib.load(t1ce_path).get_fdata()
            t2w = nib.load(t2_path).get_fdata()
            
            # Stack all four MRI modalities
            image = np.stack([flair, t1w, t1gd, t2w], axis=0)
            
            # Load segmentation mask
            mask_path = os.path.join(patient_path, f"{self.patient_dirs[idx]}_seg.nii")
            if not os.path.exists(mask_path):
                mask_path = mask_path + ".gz"
                
            if os.path.exists(mask_path):
                mask = nib.load(mask_path).get_fdata()
                # Convert to binary mask (any tumor class = 1)
                mask = (mask > 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)  # Add channel dimension
            else:
                mask = np.zeros((1,) + image.shape[1:])
            
            # Convert to PyTorch tensors
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            
            # Store in cache if enabled
            if self.cache_data:
                self.data_cache[idx] = (image, mask)
            
            return image, mask
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading patient {self.patient_dirs[idx]}: {str(e)}")
            # Return dummy tensors in case of failure
            dummy_shape = (4, 240, 240, 155) if self.target_shape is None else (4, *self.target_shape)
            return torch.zeros(dummy_shape, dtype=torch.float32), torch.zeros((1, *dummy_shape[1:]), dtype=torch.float32)
    
    def __getitem__(self, idx):
        # Determine which loading function to use based on dataset structure
        if "Task01_BrainTumour" in self.root_dir:
            image, mask = self._load_monai_data(idx)
        else:
            image, mask = self._load_brats_data(idx)
        
        # Apply preprocessing with improved normalization
        if self.normalize:
            image = preprocess_brats_data(image, normalize=True)
        
        # Resize to target dimensions if specified
        if self.target_shape is not None:
            image, mask = self._resize_volume(image, mask)
        
        # Apply data augmentation in training mode
        if self.use_augmentation:
            image, mask = apply_augmentations(image, mask)
        
        return image, mask

def get_brats_dataloader(root_dir, batch_size=1, train=True, normalize=True, max_samples=None, 
                         num_workers=4, filter_empty=False, use_augmentation=False, 
                         target_shape=None, cache_data=False, verbose=True):
    """Create a DataLoader for BraTS dataset with a proper train/validation split"""
    
    # Step 1: Load ALL available data from the training directory
    # We ensure no filtering/limiting happens at this stage
    full_dataset = BraTSDataset(
        root_dir=root_dir, 
        train=True,  # Always load from training directory
        normalize=normalize,
        max_samples=None,  # Don't limit samples in dataset initialization
        filter_empty=filter_empty,
        use_augmentation=False,  # We'll add augmentation later if needed
        target_shape=target_shape,
        cache_data=cache_data,
        verbose=False  # Turn off verbose in dataset to avoid double messages
    )
    
    # Force class variable reset for clean split
    if hasattr(full_dataset, 'max_samples'):
        full_dataset.max_samples = None
    
    # Step 2: Determine the total number of samples
    total_samples = len(full_dataset)
    if verbose:
        print(f"Full dataset contains {total_samples} samples")
    
    # Step 3: Create fixed indices for reproducible splits
    import random
    random.seed(42)
    
    # Get shuffled indices for the entire dataset
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    # Step 4: Calculate the split sizes (80% train, 20% validation)
    # Use all data unless max_samples is specified
    if max_samples is not None and max_samples < total_samples:
        effective_total = max_samples
    else:
        effective_total = total_samples
    
    # Calculate split sizes
    train_size = int(0.8 * effective_total)
    val_size = min(effective_total - train_size, total_samples - train_size)
    
    # Step 5: Create the actual indices for train and validation
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    
    # Step 6: Create the appropriate subset based on 'train' parameter
    from torch.utils.data import Subset
    if train:
        dataset = Subset(full_dataset, train_indices)
        # Apply augmentation if requested
        if use_augmentation:
            full_dataset.use_augmentation = True
        if verbose:
            print(f"Created training dataset with {len(dataset)} samples")
    else:
        dataset = Subset(full_dataset, val_indices)
        if verbose:
            print(f"Created validation dataset with {len(dataset)} samples")
    
    # Step 7: Create and return the DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,  # Shuffle only for training
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train  # Drop incomplete batches only during training
    )
    
    if verbose:
        print(f"Created {'training' if train else 'validation'} dataloader with {len(loader)} batches")
    
    return loader




# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc
import time
from datetime import datetime, timedelta
import random

# Import our modules
from model import AutoSAM2
from dataset import get_brats_dataloader

#------------------------------------------------
# Loss Functions
#------------------------------------------------
def dice_loss(y_pred, y_true, smooth=1.0):
    """Calculate Dice loss for binary segmentation"""
    # Apply sigmoid to get probabilities
    y_pred = torch.sigmoid(y_pred)
    
    # Flatten tensors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return 1 - dice

def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    bce = nn.BCEWithLogitsLoss(reduction='none')(y_pred, y_true)
    
    # Apply sigmoid to get probabilities for focal term
    y_pred_sigmoid = torch.sigmoid(y_pred)
    pt = y_true * y_pred_sigmoid + (1 - y_true) * (1 - y_pred_sigmoid)
    focal_term = alpha * (1 - pt) ** gamma
    
    # Apply focal term to BCE
    loss = focal_term * bce
    
    return loss.mean()

def combined_loss(y_pred, y_true, dice_weight=0.7, focal_weight=0.3):
    """Combined loss function: Dice Loss + Focal Loss"""
    dice = dice_loss(y_pred, y_true)
    focal = focal_loss(y_pred, y_true)
    
    return dice_weight * dice + focal_weight * focal

#------------------------------------------------
# Evaluation Metrics
#------------------------------------------------
def calculate_dice_coefficient(y_pred, y_true, threshold=0.5):
    """Calculate Dice coefficient for evaluation"""
    # Check if inputs are numpy arrays and convert accordingly
    if isinstance(y_pred, np.ndarray):
        y_pred_binary = (y_pred > threshold).astype(np.float32)
    else:
        y_pred_binary = (y_pred > threshold).float()
    
    if isinstance(y_true, np.ndarray):
        y_true_binary = (y_true > threshold).astype(np.float32)
    else:
        y_true_binary = (y_true > threshold).float()
    
    # Flatten arrays
    y_pred_binary = y_pred_binary.reshape(-1)
    y_true_binary = y_true_binary.reshape(-1)
    
    intersection = (y_pred_binary * y_true_binary).sum()
    union = y_pred_binary.sum() + y_true_binary.sum()
    
    if union == 0:
        return 1.0  # If both prediction and ground truth are empty, consider it a perfect match
    
    return (2. * intersection) / union

#------------------------------------------------
# Visualization Functions
#------------------------------------------------
def visualize_slice(image_slice, mask_slice, pred_slice=None, title=None):
    """Visualize a slice with its mask and prediction"""
    fig, axes = plt.subplots(1, 3 if pred_slice is not None else 2, figsize=(12, 4))
    
    # Show the image (using first modality)
    axes[0].imshow(image_slice[0], cmap='gray')
    axes[0].set_title('Input Slice (FLAIR)')
    axes[0].axis('off')
    
    # Show the mask
    axes[1].imshow(mask_slice[0], cmap='hot')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Show the prediction if available
    if pred_slice is not None:
        axes[2].imshow(pred_slice[0], cmap='hot')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    if title:
        fig.suptitle(title)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f"results/{title}.png" if title else "results/visualization.png")
    plt.close(fig)  # Explicitly close figure to prevent memory leak

def save_training_history(history, filename):
    """Save training history plot"""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot Dice scores
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Training and Validation Dice Score')
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    plt.savefig(f"results/{filename}")
    plt.close()  # Explicitly close to prevent memory leak

#------------------------------------------------
# Training Functions
#------------------------------------------------
def train_epoch(model, train_loader, optimizer, device, epoch, process_3d=True, use_mixed_precision=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    dice_scores = []
    processed_batches = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    
    # Set up mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            # Clear GPU cache before processing batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Get original dimensions for potential resizing
            batch_size, channels, depth, height, width = images.shape
            mask_shape = masks.shape
            
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    output = model(images)
                    
                    # Ensure output dimensions match mask dimensions
                    if output.shape != masks.shape:
                        output = F.interpolate(
                            output, 
                            size=(mask_shape[2], mask_shape[3], mask_shape[4]), 
                            mode='trilinear', 
                            align_corners=False
                        )
                    
                    # Calculate loss
                    loss = combined_loss(output, masks)
                
                # Backward and optimize with scaled gradients
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training (no mixed precision)
                output = model(images)
                
                # Ensure output dimensions match mask dimensions
                if output.shape != masks.shape:
                    output = F.interpolate(
                        output, 
                        size=(mask_shape[2], mask_shape[3], mask_shape[4]), 
                        mode='trilinear', 
                        align_corners=False
                    )
                
                loss = combined_loss(output, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Calculate Dice score for monitoring
            with torch.no_grad():
                pred = torch.sigmoid(output).detach()
                dice = calculate_dice_coefficient(pred.cpu().numpy(), masks.cpu().numpy())
                dice_scores.append(dice)
            
            # Update statistics
            total_loss += loss.item()
            processed_batches += 1
            pbar.set_postfix({"loss": loss.item(), "dice": dice})
            
            # Visualize first batch of first epoch
            if batch_idx == 0 and epoch == 0:
                # Visualize middle slice of 3D prediction
                middle_idx = depth // 2
                visualize_slice(
                    images[0, :, middle_idx].cpu().detach().numpy(),
                    masks[0, :, middle_idx].cpu().detach().numpy(),
                    torch.sigmoid(output[0, :, middle_idx]).cpu().detach().numpy(),
                    title=f"train_sample_epoch_{epoch}"
                )
                
            # Explicitly free memory
            del images, masks, output, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM in batch {batch_idx}. Skipping batch...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            else:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # Calculate average metrics
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    else:
        avg_loss = float('inf')
        avg_dice = 0.0
    
    return avg_loss, avg_dice

def validate(model, dataloader, device, epoch, process_3d=True, use_mixed_precision=True):
    """Validate the model"""
    model.eval()
    val_loss = 0
    dice_scores = []
    processed_batches = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validation")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                # Clear GPU cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Get original dimensions for potential resizing
                batch_size, channels, depth, height, width = images.shape
                mask_shape = masks.shape
                
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass with mixed precision
                if use_mixed_precision and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        output = model(images)
                        
                        # Ensure output dimensions match mask dimensions
                        if output.shape != masks.shape:
                            output = F.interpolate(
                                output, 
                                size=(mask_shape[2], mask_shape[3], mask_shape[4]), 
                                mode='trilinear', 
                                align_corners=False
                            )
                        
                        loss = combined_loss(output, masks)
                else:
                    output = model(images)
                    
                    # Ensure output dimensions match mask dimensions
                    if output.shape != masks.shape:
                        output = F.interpolate(
                            output, 
                            size=(mask_shape[2], mask_shape[3], mask_shape[4]), 
                            mode='trilinear', 
                            align_corners=False
                        )
                    
                    loss = combined_loss(output, masks)
                
                # Calculate Dice for evaluation
                pred = torch.sigmoid(output).cpu().numpy()
                true = masks.cpu().numpy()
                dice = calculate_dice_coefficient(pred, true)
                
                # Update statistics
                val_loss += loss.item()
                dice_scores.append(dice)
                processed_batches += 1
                pbar.set_postfix({"val_loss": loss.item(), "dice": dice})
                
                # Visualize first batch
                if batch_idx == 0:
                    # Visualize middle slice of 3D prediction
                    middle_idx = depth // 2
                    visualize_slice(
                        images[0, :, middle_idx].cpu().numpy(),
                        masks[0, :, middle_idx].cpu().numpy(),
                        pred[0, :, middle_idx],
                        title=f"val_sample_epoch_{epoch}"
                    )
                
                # Free memory
                del images, masks, output, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM in validation batch {batch_idx}. Skipping batch...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Calculate average metrics
    if processed_batches > 0:
        avg_loss = val_loss / processed_batches
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    else:
        avg_loss = float('inf')
        avg_dice = 0.0
    
    return avg_loss, avg_dice

#------------------------------------------------
# Main Training Function
#------------------------------------------------
def train_model(data_path, batch_size=1, epochs=20, learning_rate=3e-4,
                use_mixed_precision=True, test_run=False, reset=True, 
                target_shape=(64, 128, 128), cache_data=False):
    """
    Train AutoSAM2 for 3D medical image segmentation
    
    Args:
        data_path: Path to the BraTS dataset
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Initial learning rate
        use_mixed_precision: Whether to use mixed precision training
        test_run: Whether to run with fewer samples for testing
        reset: If True, start training from scratch
        target_shape: Target shape for resizing (depth, height, width)
        cache_data: Whether to cache data in memory
    """


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    # Make directories for results and checkpoints
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize AutoSAM2 model
    print("Initializing AutoSAM2 with 3D UNet prompt encoder")
    model = AutoSAM2(
        use_real_sam2=True,
        process_3d=True  # Always process 3D for BraTS
    ).to(device)
    
    # Check if model file exists to resume training
    model_path = f"checkpoints/best_autosam2_model.pth"
    start_epoch = 0
    best_dice = 0.0
    
    if os.path.exists(model_path) and not reset:
        print(f"Found existing model checkpoint at {model_path}, loading...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            print(f"Resuming from epoch {start_epoch} with best dice score: {best_dice:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch...")
            start_epoch = 0
            best_dice = 0.0
    else:
        if reset and os.path.exists(model_path):
            print(f"Reset flag is set. Ignoring existing checkpoint and starting from epoch 0.")
        else:
            print("No existing checkpoint found. Starting from epoch 0.")
    
    # Define optimizer with weight decay - only optimize the prompt encoder
    optimizer = optim.AdamW(
        model.prompt_encoder.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5,
        amsgrad=True
    )
    
    # Load optimizer state if resuming
    if os.path.exists(model_path) and not reset and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        except Exception as e:
            print(f"Error loading optimizer state: {e}. Using fresh optimizer.")
    
    # Define learning rate scheduler with patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize data loaders
    max_samples = 8 if test_run else None
    
    print("Creating data loaders...")
    train_loader = get_brats_dataloader(
        data_path, 
        batch_size=batch_size, 
        train=True,
        normalize=True, 
        max_samples=None,  # Use all available data
        num_workers=4,
        filter_empty=False,
        use_augmentation=False,
        target_shape=target_shape,
        cache_data=cache_data,
        verbose=True
    )

    val_loader = get_brats_dataloader(
        data_path, 
        batch_size=batch_size, 
        train=False,
        normalize=True, 
        max_samples=None,  # Use all available data
        num_workers=4,
        filter_empty=False,
        use_augmentation=False,
        target_shape=target_shape,
        cache_data=cache_data,
        verbose=True
    )
        
    # Add diagnostic code here - after the loaders are created
    print("Checking training data...")
    train_batch = next(iter(train_loader))
    train_images, train_masks = train_batch
    print(f"Training images shape: {train_images.shape}")
    print(f"Training masks shape: {train_masks.shape}")
    print(f"Training masks values: min={train_masks.min()}, max={train_masks.max()}, mean={train_masks.mean()}")

    print("Checking validation data...")
    try:
        val_batch = next(iter(val_loader))
        val_images, val_masks = val_batch
        print(f"Validation images shape: {val_images.shape}")
        print(f"Validation masks shape: {val_masks.shape}")
        print(f"Validation masks values: min={val_masks.min()}, max={val_masks.max()}, mean={val_masks.mean()}")
    except StopIteration:
        print("WARNING: Validation dataloader is empty!")
    
    print(f"Training with {len(train_loader)} batches, validating with {len(val_loader)} batches")
    
    print(f"Training with {len(train_loader)} batches, validating with {len(val_loader)} batches")
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'lr': []
    }
    
    # Early stopping parameters
    patience = 10
    counter = 0
    
    # Training loop
    training_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, device, epoch, 
            process_3d=True, use_mixed_precision=use_mixed_precision
        )
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        
        # Validate
        val_loss, val_dice = validate(
            model, val_loader, device, epoch, 
            process_3d=True, use_mixed_precision=use_mixed_precision
        )
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Update learning rate based on validation dice
        scheduler.step(val_dice)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        estimated_total_time = elapsed_time / (epoch - start_epoch + 1) * (epochs - start_epoch) if epoch > start_epoch else epochs * epoch_time
        remaining_time = max(0, estimated_total_time - elapsed_time)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Dice = {train_dice:.4f}")
        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Dice = {val_dice:.4f}")
        print(f"Epoch Time: {timedelta(seconds=int(epoch_time))}, Remaining: {timedelta(seconds=int(remaining_time))}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_dice': train_dice,
                'val_dice': val_dice,
                'best_dice': best_dice,
            }, model_path)
            print(f"Saved new best model with Dice score: {best_dice:.4f}")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"No improvement in validation Dice score for {counter} epochs")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, f"checkpoints/autosam2_model_epoch_{epoch}.pth")
            
            # Save intermediate training history
            save_training_history(history, f"training_history_epoch_{epoch}.png")
        
        # Check for early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save final training history
    save_training_history(history, "final_training_history.png")
    print(f"Training complete! Best Dice score: {best_dice:.4f}")
    
    # Load best model for further use
    best_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    return model, history, best_dice

#------------------------------------------------
# Main function to run the whole pipeline
#------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train AutoSAM2 for brain tumor segmentation")
    parser.add_argument('--data_path', type=str, default="./data/Task01_BrainTumour",
                    help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learning rate')
    parser.add_argument('--test_run', action='store_true',
                    help='Run with limited samples for testing')
    parser.add_argument('--reset', action='store_true', default=True,
                    help='Reset training from scratch, ignoring checkpoints')
    parser.add_argument('--no_mixed_precision', action='store_true',
                    help='Disable mixed precision training')
    parser.add_argument('--memory_limit', type=float, default=0.9,
                    help='Memory limit for GPU (0.0-1.0)')
    parser.add_argument('--target_shape', type=str, default="64,128,128",
                    help='Target shape for resizing (depth,height,width)')
    parser.add_argument('--max_samples', type=int, default=32,  # הגדלנו מ-None ל-32
                    help='Maximum number of samples to use (default: 32)')
    parser.add_argument('--cache_data', action='store_true',
                    help='Cache data in memory for faster training')

    
    args = parser.parse_args()
    
    # Set memory limit for GPU
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_limit)
    
    # Parse target shape if provided
    target_shape = tuple(map(int, args.target_shape.split(',')))
    print(f"Using target shape: {target_shape}")
    
    # Use max_samples if provided (overrides test_run)
    max_samples = args.max_samples if args.max_samples is not None else (8 if args.test_run else None)
    
    # Train the model
    model, history, best_dice = train_model(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_mixed_precision=not args.no_mixed_precision,
        test_run=(max_samples is not None),
        reset=args.reset,
        target_shape=target_shape,
        cache_data=args.cache_data
    )
    
    print(f"Final best Dice score: {best_dice:.4f}")

if __name__ == "__main__":
    main()


    
#sam2.py

class Sam2Model:
    """
    Placeholder for SAM2 model for development purposes.
    This allows the code to import without having the actual SAM2 model.
    """
    
    @staticmethod
    def from_pretrained(checkpoint_path=None):
        """Placeholder for loading pretrained model"""
        print("Warning: Using placeholder SAM2 model. This is not the actual implementation.")
        return Sam2Model()
    
    def __init__(self):
        # Placeholders for SAM2 components
        self.image_encoder = PlaceholderModule()
        self.mask_decoder = PlaceholderModule()
    
    def __call__(self, images, prompts):
        """Placeholder forward function"""
        # Return a dummy mask with the same batch size and spatial dimensions as the input
        batch_size = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        
        import torch
        return torch.zeros((batch_size, 1, height, width), device=images.device)


class PlaceholderModule:
    """Placeholder for SAM2 submodules"""
    
    def __init__(self):
        self.parameters = lambda: []  # Empty parameters list
    
    def __call__(self, *args, **kwargs):
        return None 