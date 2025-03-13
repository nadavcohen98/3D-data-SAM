#dataset.py
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from tqdm import tqdm
import argparse
import torch

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
            
            # Limit samples if specified
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
    
    def __getitem__(self, idx):
        # Only use MONAI data loader for simplicity
        image, mask = self._load_monai_data(idx)
        
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


#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

# Import SAM2 with proper error handling
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
    print("Successfully imported SAM2")
except ImportError:
    print("ERROR: sam2 package not available. This implementation requires SAM2.")
    HAS_SAM2 = False

class Encoder3D(nn.Module):
    """
    3D encoder based on UNet architecture with reduced downsampling
    to preserve spatial information
    """
    def __init__(self, in_channels=4, base_channels=16):
        super().__init__()
        
        # First encoder block
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Second encoder block
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.3)
    
    def forward(self, x):
        # Encoder pathway with skip connections
        x1 = self.enc1(x)
        x1 = self.dropout(x1)
        
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x2 = self.dropout(x2)
        
        x = self.pool2(x2)
        
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        return [x1, x2, x]  # Return features for skip connections

class MiniDecoder(nn.Module):
    """
    Mini-decoder that produces embeddings for SAM2, 
    similar to AutoSAM's approach - the key component that interfaces with SAM2
    """
    def __init__(self, base_channels=16):
        super().__init__()
        
        # Upsampling blocks
        self.up1 = nn.ConvTranspose3d(
            base_channels*4, base_channels*2, 
            kernel_size=2, stride=2
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True)
        )
        
        # Feature projection to get RGB-like output
        self.final = nn.Conv3d(base_channels*2, 3, kernel_size=1)
        self.tanh = nn.Tanh()  # Similar to AutoSAM - normalizes to [-1, 1] range
    
    def forward(self, features):
        # Unpack features from encoder
        x1, x2, bottleneck = features
        
        # First upsampling with skip connection
        x = self.up1(bottleneck)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        
        # Final projection with tanh (like in AutoSAM)
        x = self.final(x)
        x = self.tanh(x)
        
        return x

class AutoSAM2(nn.Module):
    """
    AutoSAM2 model for 3D medical image segmentation
    Specialized for multi-class tumor segmentation with 4 classes:
    - Class 0: Background
    - Class 1: Edema (whole tumor region)
    - Class 2: Non-enhancing tumor (tumor core)
    - Class 3: Enhancing tumor
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        
        # Create encoder
        self.encoder = Encoder3D(in_channels=4, base_channels=16)
        
        # Create mini-decoder (the key component for AutoSAM)
        self.mini_decoder = MiniDecoder(base_channels=16)
        
        # Segmentation head for auxiliary supervision and fallback
        self.seg_head = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )
        
        # Initialize SAM2
        if HAS_SAM2:
            try:
                self.sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
                print("SAM2 initialized successfully")
                
                # Freeze SAM2 weights
                for param in self.sam2_predictor.model.parameters():
                    param.requires_grad = False
                
                self.has_sam2 = True
            except Exception as e:
                print(f"Error initializing SAM2: {e}")
                self.has_sam2 = False
                self.sam2_predictor = None
        else:
            self.has_sam2 = False
            self.sam2_predictor = None
    
    def _convert_features_to_image(self, features_slice):
        """Convert feature slice to RGB image for SAM2"""
        # Ensure features_slice is numpy array
        if torch.is_tensor(features_slice):
            features_slice = features_slice.detach().cpu().numpy()
        
        # Normalize each channel to [0, 255]
        image = np.zeros_like(features_slice)
        for c in range(features_slice.shape[0]):
            channel = features_slice[c]
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            # Avoid division by zero
            if max_val > min_val:
                image[c] = (channel - min_val) / (max_val - min_val) * 255
        
        # Convert to [H, W, C] format and uint8 type
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        return image
    
    def _process_slice_with_sam2(self, embeddings_slice, prob_slice, device):
        """
        Process a single slice with SAM2 for multi-class segmentation
        
        Args:
            embeddings_slice: Feature embeddings for the slice [3, H, W]
            prob_slice: Probability maps for the slice [C, H, W]
            device: Device to place tensors on
            
        Returns:
            Multi-class segmentation mask [C, H, W]
        """
        # Convert to image for SAM2
        rgb_image = self._convert_features_to_image(embeddings_slice)
        height, width = rgb_image.shape[:2]
        
        # Initialize output mask
        output_mask = torch.zeros((self.num_classes, height, width), device=device)
        
        try:
            # Set image in SAM2
            self.sam2_predictor.set_image(rgb_image)
            
            # Create points for foreground and background
            # Use center point for simplicity and stability
            center_point = np.array([[width // 2, height // 2]])
            center_label = np.array([1])  # 1 = foreground
            
            # Add background points at the corners
            bg_points = np.array([
                [10, 10], 
                [width-10, 10], 
                [10, height-10], 
                [width-10, height-10]
            ])
            bg_labels = np.zeros(len(bg_points))  # 0 = background
            
            # Combine points
            points = np.vstack([center_point, bg_points])
            labels = np.concatenate([center_label, bg_labels])
            
            # Get predictions from SAM2
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            # Process output masks
            if len(masks) > 0 and len(scores) > 0:
                # Get best mask
                best_idx = np.argmax(scores)
                foreground_mask = torch.tensor(masks[best_idx], dtype=torch.float32, device=device)
                
                # Background is inverse of foreground
                output_mask[0] = 1 - foreground_mask
                
                # Convert probability maps to tensor on same device
                prob_device = torch.tensor(prob_slice.detach().cpu().numpy(), device=device)
                
                # Distribute foreground to classes 1-3 based on probability distribution
                # First calculate summed probability for all tumor classes
                tumor_prob_sum = torch.clamp(
                    prob_device[1] + prob_device[2] + prob_device[3],
                    min=1e-6
                )
                
                # Then distribute foreground mask based on class probabilities
                for c in range(1, self.num_classes):
                    class_weight = prob_device[c] / tumor_prob_sum
                    output_mask[c] = foreground_mask * class_weight
                
                # Apply hierarchical constraints:
                # ET ⊆ TC ⊆ WT (class 3 ⊆ class 2 ⊆ class 1)
                output_mask[1] = torch.max(output_mask[1], output_mask[2])  # WT includes TC
                output_mask[1] = torch.max(output_mask[1], output_mask[3])  # WT includes ET
                output_mask[2] = torch.max(output_mask[2], output_mask[3])  # TC includes ET
                
                return output_mask
            else:
                # Return probabilities if SAM2 fails
                return torch.tensor(prob_slice.detach().cpu().numpy(), device=device)
            
        except Exception as e:
            print(f"Error in SAM2 processing: {e}")
            # Return probabilities if SAM2 fails
            return torch.tensor(prob_slice.detach().cpu().numpy(), device=device)
    
    def forward(self, x):
        """
        Forward pass of the model, using SAM2 for both training and validation
        Specialized for multi-class tumor segmentation
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            Segmentation masks [B, num_classes, D, H, W]
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Get features from encoder
        encoder_features = self.encoder(x)
        
        # Get embeddings from mini-decoder (the key AutoSAM component)
        embeddings = self.mini_decoder(encoder_features)
        
        # Generate auxiliary segmentation with segmentation head
        seg_output = self.seg_head(embeddings)
        seg_probs = torch.softmax(seg_output, dim=1)
        
        # Initialize output tensor
        output_masks = torch.zeros_like(seg_probs)
        
        # Determine which slices to process with SAM2
        # Use every 8th slice during training, every 4th during inference
        slice_stride = 8 if self.training else 4
        
        # IMPORTANT: Make sure slice indices are in bounds
        max_idx = depth - 1
        key_slices = [i for i in range(0, depth, slice_stride) if i <= max_idx]
        
        # Only use SAM2 if it's available
        if self.has_sam2 and self.sam2_predictor is not None:
            # Track which slices were processed
            processed_slices = {}
            
            for b in range(batch_size):
                # Process key slices with SAM2
                for slice_idx in key_slices:
                    try:
                        # Safety check (should be redundant now)
                        if slice_idx >= depth:
                            continue
                            
                        # Get slice data
                        emb_slice = embeddings[b, :, slice_idx].detach().cpu()
                        prob_slice = seg_probs[b, :, slice_idx]
                        
                        # Process with SAM2
                        mask = self._process_slice_with_sam2(emb_slice, prob_slice, x.device)
                        
                        if mask is not None:
                            output_masks[b, :, slice_idx] = mask
                            processed_slices[(b, slice_idx)] = True
                        else:
                            # If SAM2 processing failed, use segmentation head output
                            output_masks[b, :, slice_idx] = seg_probs[b, :, slice_idx]
                    except Exception as e:
                        print(f"Error processing slice {slice_idx} for batch {b}: {e}")
                        # Use segmentation head output
                        output_masks[b, :, slice_idx] = seg_probs[b, :, slice_idx]
                
                # Fill in non-key slices with interpolation
                for slice_idx in range(depth):
                    # Skip already processed slices
                    if (b, slice_idx) in processed_slices:
                        continue
                    
                    # Find nearest processed slices
                    prev_slice = None
                    next_slice = None
                    
                    for s in key_slices:
                        if s < slice_idx and (b, s) in processed_slices:
                            prev_slice = s
                        if s > slice_idx and (b, s) in processed_slices:
                            next_slice = s
                            break
                    
                    # Interpolate if we have both prev and next
                    if prev_slice is not None and next_slice is not None:
                        # Linear interpolation
                        weight = (slice_idx - prev_slice) / (next_slice - prev_slice)
                        output_masks[b, :, slice_idx] = (1 - weight) * output_masks[b, :, prev_slice] + weight * output_masks[b, :, next_slice]
                    # Use nearest if we only have one
                    elif prev_slice is not None:
                        output_masks[b, :, slice_idx] = output_masks[b, :, prev_slice]
                    elif next_slice is not None:
                        output_masks[b, :, slice_idx] = output_masks[b, :, next_slice]
                    else:
                        # Use segmentation head output if no processed slices
                        output_masks[b, :, slice_idx] = seg_probs[b, :, slice_idx]
            
            # During training, we need to mix SAM2 results with segmentation head
            # to ensure gradient flow through the network
            if self.training:
                # During training, use a blend of SAM2 output and segmentation head
                # with more weight to SAM2 to guide learning while allowing gradients
                blend_weight = 0.7
                blended_output = blend_weight * output_masks + (1 - blend_weight) * seg_probs
                
                # Return logits for loss computation
                return torch.log(blended_output + 1e-6)
            else:
                # During evaluation, return probabilities directly
                return output_masks
        else:
            # If SAM2 not available, return segmentation head output
            print("Warning: SAM2 not available, using segmentation head output only")
            return seg_output


#train.py - Enhanced version with optimizations
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
import torch.nn.functional as F
import torch.serialization

torch.serialization.add_safe_globals([np.core.multiarray.scalar])


def calculate_dice_score(y_pred, y_true):
    """
    Efficient Dice score calculation with minimal printing
    """
    # For raw logits, apply softmax first
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    
    # Check if input contains logits or probabilities
    if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
        probs = F.softmax(y_pred, dim=1)
    else:
        probs = y_pred
    
    # Apply threshold to get binary predictions
    preds = (probs > 0.5).float()
    
    # Calculate Dice for each class except background (class 0)
    dice_scores = []
    
    # Iterate over all non-background classes
    for c in range(1, y_pred.size(1)):
        # Flatten tensors
        pred_c = preds[:, c].reshape(preds.size(0), -1)
        true_c = y_true[:, c].reshape(y_true.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_c * true_c).sum(1)
        pred_sum = pred_c.sum(1)
        true_sum = true_c.sum(1)
        
        # Skip if no ground truth or prediction
        valid_samples = ((pred_sum > 0) | (true_sum > 0))
        
        if valid_samples.sum() > 0:
            # Calculate Dice: (2 * intersection) / (pred_sum + true_sum)
            dice_c = (2.0 * intersection[valid_samples] / (pred_sum[valid_samples] + true_sum[valid_samples] + 1e-5)).mean().item()
            dice_scores.append(dice_c)
        else:
            # No valid samples, add 0
            dice_scores.append(0.0)
    
    # If no valid classes, return 0
    if len(dice_scores) == 0:
        return {'mean': 0.0}
    
    # Return average
    mean_dice = sum(dice_scores) / len(dice_scores)
    
    result = {'mean': mean_dice}
    
    # Add per-class metrics
    for i, score in enumerate(dice_scores, 1):
        result[f'class_{i}'] = score
    
    return result

# Simple dice loss for multiclass segmentation with improved numeric stability
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        
    def forward(self, y_pred, y_true):
        # Get dimensions
        batch_size, num_classes = y_pred.size(0), y_pred.size(1)
        
        # Check if input is probabilities or logits
        if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        
        # Calculate Dice for each class
        dice_per_class = []
        
        # Start from class 1 if ignoring background
        start_class = 1 if self.ignore_background else 0
        
        for c in range(start_class, num_classes):
            # Get predictions and targets for this class
            pred_c = y_pred[:, c]
            true_c = y_true[:, c]
            
            # Flatten
            pred_c = pred_c.contiguous().view(-1)
            true_c = true_c.contiguous().view(-1)
            
            # Calculate intersection and union
            intersection = (pred_c * true_c).sum()
            pred_sum = pred_c.sum()
            true_sum = true_c.sum()
            
            # Only calculate loss for non-empty masks to avoid division by zero
            if true_sum > 0 or pred_sum > 0:
                # Calculate Dice
                dice = (2.0 * intersection + self.smooth) / (pred_sum + true_sum + self.smooth)
                dice_per_class.append(1.0 - dice)
        
        # Return mean Dice loss
        if len(dice_per_class) == 0:
            # Return zero tensor if no valid classes
            return torch.tensor(0.0, requires_grad=True, device=y_pred.device)
        else:
            return torch.stack(dice_per_class).mean()
        
def calculate_iou(y_pred, y_true, threshold=0.5, eps=1e-6):
    """
    Calculate Intersection over Union (IoU) for each class.

    Args:
        y_pred (Tensor): Model predictions (logits or probabilities) of shape [B, C, D, H, W]
        y_true (Tensor): Ground truth masks of shape [B, C, D, H, W]
        threshold (float): Threshold for binarization of predictions.
        eps (float): Small value to avoid division by zero.

    Returns:
        dict: Mean IoU and per-class IoU.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")

    # Apply threshold to convert probability predictions into binary masks
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()

    iou_scores = []
    
    # Loop over each class (excluding background)
    for c in range(1, y_pred.shape[1]):
        pred_c = y_pred_bin[:, c]
        true_c = y_true[:, c]

        intersection = (pred_c * true_c).sum(dim=(1, 2, 3))
        union = (pred_c + true_c).clamp(0, 1).sum(dim=(1, 2, 3))

        iou_c = (intersection + eps) / (union + eps)  # Compute IoU per sample
        iou_scores.append(iou_c.mean().item())  # Get mean IoU for this class

    # Compute mean IoU across all classes
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    result = {'mean_iou': mean_iou}
    for i, iou in enumerate(iou_scores, 1):
        result[f'class_{i}_iou'] = iou

    return result

# Combined loss for better convergence
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.6, bce_weight=0.2, focal_weight=0.2):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).cuda())  # Higher weight for tumors
        self.focal_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        bce = self.bce_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true.argmax(dim=1))

        return self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal

def preprocess_batch(batch, device=None):
    """
    Preprocess batch for binary masks
    """
    images, masks = batch
    
    # Convert binary masks to multi-class format
    if masks.shape[1] == 1:
        # For binary masks with just tumor/no-tumor
        multi_class_masks = torch.zeros((masks.shape[0], 4, *masks.shape[2:]), dtype=torch.float32)
        
        # Class 0: Background (where mask is 0)
        multi_class_masks[:, 0] = (masks[:, 0] == 0).float()
        
        # Class 1: Primary tumor region (all tumor pixels)
        multi_class_masks[:, 1] = (masks[:, 0] == 1).float()
        
        # For training completeness, create synthetic values for classes 2 and 3
        # This is only for demonstration - you may want to adjust or remove this
        if torch.sum(multi_class_masks[:, 1]) > 0:
            # Use a portion of class 1 for classes 2 and 3
            rnd = torch.rand_like(multi_class_masks[:, 1])
            multi_class_masks[:, 2] = (multi_class_masks[:, 1] * (rnd < 0.2)).float()
            multi_class_masks[:, 3] = (multi_class_masks[:, 1] * (rnd < 0.1) * (rnd > 0.05)).float()
        
        masks = multi_class_masks
    
    # Ensure mask values are within expected range
    masks = torch.clamp(masks, 0, 1)
    
    # Move to device if specified
    if device is not None:
        images = images.to(device)
        masks = masks.to(device)
    
    return images, masks

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler=None):
    """
    Training epoch function with proper logging of Dice Loss, Dice Score, IoU, and BCE Loss.
    """
    model.train()
    total_loss = 0
    all_metrics = []
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Clear GPU cache if needed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Preprocess batch
            images, masks = preprocess_batch(batch, device=device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute BCE Loss and Dice Loss
            bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
            dice_loss = DiceLoss()(outputs, masks)

            # Compute combined loss
            loss = 0.7 * dice_loss + 0.3 * bce_loss
            
            # Compute IoU
            iou_metrics = calculate_iou(outputs, masks)

            # Compute Dice Score
            dice_metrics = calculate_dice_score(outputs.detach(), masks)
            
            # Store metrics
            all_metrics.append({
                'bce_loss': bce_loss.item(),
                'dice_loss': dice_loss.item(),
                'mean_dice': dice_metrics['mean'],
                'mean_iou': iou_metrics['mean_iou']
            })

            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Learning rate scheduling
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{dice_metrics['mean']:.4f}",
                'iou': f"{iou_metrics['mean_iou']:.4f}",
                'bce': f"{bce_loss.item():.4f}"
            })
                
            # Visualize first batch
            if batch_idx == 0 and epoch % 5 == 0:
                visualize_batch(images, masks, outputs, epoch, "train")
            
            # Update total loss
            total_loss += loss.item()
            
            # Free memory
            del images, masks, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nCUDA OOM in batch {batch_idx}. Skipping batch...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            else:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                raise e
    
    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {
        'mean_dice': np.mean([m['mean_dice'] for m in all_metrics]) if all_metrics else 0.0,
        'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]) if all_metrics else 0.0,
        'bce_loss': np.mean([m['bce_loss'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_loss': np.mean([m['dice_loss'] for m in all_metrics]) if all_metrics else 0.0
    }

    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model while logging Dice Loss, Dice Score, IoU, and BCE Loss.
    """
    model.eval()
    total_loss = 0
    all_metrics = []
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # Clear GPU cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Preprocess batch
                images, masks = preprocess_batch(batch, device=device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute BCE Loss and Dice Loss
                bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
                dice_loss = DiceLoss()(outputs, masks)

                # Compute combined loss
                loss = 0.7 * dice_loss + 0.3 * bce_loss
                
                # Compute IoU
                iou_metrics = calculate_iou(outputs, masks)

                # Compute Dice Score
                dice_metrics = calculate_dice_score(outputs, masks)
                
                # Store metrics
                all_metrics.append({
                    'bce_loss': bce_loss.item(),
                    'dice_loss': dice_loss.item(),
                    'mean_dice': dice_metrics['mean'],
                    'mean_iou': iou_metrics['mean_iou']
                })
                                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice_metrics['mean']:.4f}",
                    'iou': f"{iou_metrics['mean_iou']:.4f}",
                    'bce': f"{bce_loss.item():.4f}"
                })
                
                # Visualize first batch
                if batch_idx == 0:
                    visualize_batch(images, masks, outputs, epoch, "val")
                
                # Update total loss
                total_loss += loss.item()
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\nCUDA OOM in validation batch {batch_idx}. Skipping batch...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    print(f"\nError in validation batch {batch_idx}: {str(e)}")
                    raise e
            
        # Calculate average metrics
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {
            'mean_dice': np.mean([m['mean_dice'] for m in all_metrics]) if all_metrics else 0.0,
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]) if all_metrics else 0.0,
            'bce_loss': np.mean([m['bce_loss'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_loss': np.mean([m['dice_loss'] for m in all_metrics]) if all_metrics else 0.0
        }

        return avg_loss, avg_metrics
        
def visualize_batch(images, masks, outputs, epoch, prefix=""):
    """
    Visualize a batch of images, masks, and predictions
    """
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Get middle slice of first batch item
    b = 0
    depth = images.shape[2]
    middle_idx = depth // 2
    
    # Get slice data
    image_slice = images[b, :, middle_idx].cpu().detach().numpy()
    mask_slice = masks[b, :, middle_idx].cpu().detach().numpy()
    
    # Apply softmax if outputs are logits
    if torch.min(outputs) < 0 or torch.max(outputs) > 1:
        output_slice = F.softmax(outputs[b, :, middle_idx], dim=0).cpu().detach().numpy()
    else:
        output_slice = outputs[b, :, middle_idx].cpu().detach().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show FLAIR image
    axes[0].imshow(image_slice[0], cmap='gray')
    axes[0].set_title('FLAIR')
    axes[0].axis('off')
    
    # Create RGB mask for ground truth
    rgb_mask = np.zeros((mask_slice.shape[1], mask_slice.shape[2], 3))
    rgb_mask[mask_slice[1] > 0.5, :] = [1, 1, 0]  # Edema: Yellow
    rgb_mask[mask_slice[2] > 0.5, :] = [0, 1, 0]  # Non-enhancing: Green
    rgb_mask[mask_slice[3] > 0.5, :] = [1, 0, 0]  # Enhancing: Red
    
    # Show ground truth
    axes[1].imshow(image_slice[0], cmap='gray')
    axes[1].imshow(rgb_mask, alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Create RGB mask for prediction
    rgb_pred = np.zeros((output_slice.shape[1], output_slice.shape[2], 3))
    rgb_pred[output_slice[1] > 0.5, :] = [1, 1, 0]  # Edema: Yellow
    rgb_pred[output_slice[2] > 0.5, :] = [0, 1, 0]  # Non-enhancing: Green
    rgb_pred[output_slice[3] > 0.5, :] = [1, 0, 0]  # Enhancing: Red
    
    # Show prediction
    axes[2].imshow(image_slice[0], cmap='gray')
    axes[2].imshow(rgb_pred, alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/{prefix}_epoch{epoch}.png")
    plt.close()

def save_training_history(history, filename):
    """
    Save training history plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot Dice scores
    plt.subplot(2, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Mean Dice Score')
    
    # If we have BraTS metrics, plot them
    if 'train_dice_wt' in history:
        # Plot WT Dice
        plt.subplot(2, 2, 3)
        plt.plot(history['train_dice_wt'], label='Train WT')
        plt.plot(history['val_dice_wt'], label='Val WT')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.title('Whole Tumor (WT) Dice')
        
        # Plot TC and ET Dice
        plt.subplot(2, 2, 4)
        plt.plot(history['train_dice_tc'], label='Train TC')
        plt.plot(history['val_dice_tc'], label='Val TC')
        plt.plot(history['train_dice_et'], label='Train ET')
        plt.plot(history['val_dice_et'], label='Val ET')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.title('Tumor Core (TC) and Enhancing Tumor (ET) Dice')
    elif 'lr' in history:
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(history['lr'])
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

def train_model(data_path, batch_size=1, epochs=20, learning_rate=1e-3,
               use_mixed_precision=False, test_run=False, reset=True, 
               target_shape=(64, 128, 128)):
    """
    Optimized train function with better learning rate schedule
    Including IoU and BCE loss in history.
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
    print("Initializing AutoSAM2 for multi-class segmentation")
    model = AutoSAM2(num_classes=4).to(device)
    
    # Check if model file exists to resume training
    model_path = "checkpoints/best_autosam2_model.pth"
    start_epoch = 0
    best_dice = 0.0
    
    if os.path.exists(model_path) and not reset:
        print(f"Found existing model checkpoint at {model_path}, loading...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            print(f"Resuming from epoch {start_epoch} with best Dice score: {best_dice:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch...")
            start_epoch = 0
            best_dice = 0.0
    else:
        print("No existing checkpoint found. Starting from epoch 0.")
    
    # Define loss criterion
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.encoder.parameters(),
        lr=learning_rate,
        weight_decay=5e-4
    )
    
    # Load optimizer state if resuming
    if os.path.exists(model_path) and not reset and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        except Exception as e:
            print(f"Error loading optimizer state: {e}. Using fresh optimizer.")
    
    # Get data loaders
    max_samples = 64 if test_run else None
    
    train_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=True,
        normalize=True, max_samples=max_samples, num_workers=4,
        filter_empty=False, use_augmentation=True,
        target_shape=target_shape, cache_data=False, verbose=True
    )

    val_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=False,
        normalize=True, max_samples=max_samples, num_workers=4,
        filter_empty=False, use_augmentation=False,
        target_shape=target_shape, cache_data=False, verbose=True
    )
    
    # Set up OneCycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Initialize history with Dice, IoU, and BCE loss
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_bce': [], 'val_bce': [],
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
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_metrics.get('mean_dice', 0.0))
        history['train_iou'].append(train_metrics.get('mean_iou', 0.0))
        history['train_bce'].append(train_metrics.get('bce_loss', 0.0))
        
        # Validate
        try:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
            print(f"Validation metrics: {val_metrics}")  # Debugging

            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_metrics.get('mean_dice', 0.0))
            history['val_iou'].append(val_metrics.get('mean_iou', 0.0))
            history['val_bce'].append(val_metrics.get('bce_loss', 0.0))

        except Exception as e:
            print(f"Error during validation: {e}")
            history['val_loss'].append(float('inf'))
            history['val_dice'].append(0.0)
            history['val_iou'].append(0.0)
            history['val_bce'].append(0.0)
                
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        estimated_total_time = elapsed_time / (epoch - start_epoch + 1) * (epochs - start_epoch) if epoch > start_epoch else epochs * epoch_time
        remaining_time = max(0, estimated_total_time - elapsed_time)
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Dice = {train_metrics.get('mean_dice', 0.0):.4f}, IoU = {train_metrics.get('mean_iou', 0.0):.4f}, BCE = {train_metrics.get('bce_loss', 0.0):.4f}")
        print(f"Epoch {epoch+1}: Val Dice = {val_metrics.get('mean_dice', 0.0):.4f}, IoU = {val_metrics.get('mean_iou', 0.0):.4f}, BCE = {val_metrics.get('bce_loss', 0.0):.4f}")
        print(f"Epoch Time: {timedelta(seconds=int(epoch_time))}, Remaining: {timedelta(seconds=int(remaining_time))}")
        
        # Save best model based on mean Dice score
        if val_metrics.get('mean_dice', 0.0) > best_dice:
            best_dice = val_metrics.get('mean_dice', 0.0)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_dice': best_dice,
            }, model_path)
            print(f"Saved new best model with Dice score: {best_dice:.4f}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, f"checkpoints/autosam2_model_epoch_{epoch}.pth")
            
            # Save training history
            save_training_history(history, f"training_history_epoch_{epoch}.png")
        
        # Early stopping
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

    return model, history, val_metrics

def main():
    parser = argparse.ArgumentParser(description="Train AutoSAM2 for brain tumor segmentation")
    parser.add_argument('--data_path', type=str, default="/home/erezhuberman/data/Task01_BrainTumour",
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,  
                        help='Learning rate')
    parser.add_argument('--test_run', action='store_true',
                        help='Run with limited samples for testing')
    parser.add_argument('--reset', action='store_true', default=False,
                        help='Reset training from scratch, ignoring checkpoints')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--memory_limit', type=float, default=0.9,
                        help='Memory limit for GPU (0.0-1.0)')
    parser.add_argument('--target_shape', type=str, default="64,128,128",
                        help='Target shape for resizing (depth,height,width)')
    
    args = parser.parse_args()
    
    # Set memory limit for GPU if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.memory_limit)
            print(f"Set GPU memory fraction to {args.memory_limit * 100:.1f}%")
        except Exception as e:
            print(f"Warning: Could not set GPU memory fraction: {e}")

    # Parse target shape argument
    try:
        target_shape = tuple(map(int, args.target_shape.split(',')))
        if len(target_shape) != 3:
            raise ValueError("Target shape must have exactly 3 dimensions (depth, height, width).")
    except ValueError as e:
        print(f"Error parsing target shape: {e}")
        return  # Exit early if parsing fails
    
    print(f"Using target shape: {target_shape}")

    # Train the model
    try:
        model, history, best_metrics = train_model(
            data_path=args.data_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            use_mixed_precision=not args.no_mixed_precision,
            test_run=args.test_run,
            reset=args.reset,
            target_shape=target_shape
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return  # Exit if training fails

    # Print final metrics safely
    if best_metrics:
        print("\nFinal best metrics:")
        for key, value in best_metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print("\nNo final metrics were returned. Training may have failed.")

if __name__ == "__main__":
    main()
