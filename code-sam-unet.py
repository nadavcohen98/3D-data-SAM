#dataset.py
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
import matplotlib.pyplot as plt

# Import SAM2 with better error handling
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
    print("Successfully imported SAM2")
except ImportError:
    print("ERROR: sam2 package not available. This implementation requires SAM2.")
    HAS_SAM2 = False
    raise ImportError("SAM2 must be installed for this implementation to work")

class MultiscaleEncoder(nn.Module):
    """
    Encoder that extracts multi-scale features from 3D medical images
    with less aggressive downsampling to preserve spatial information
    """
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder path with reduced downsampling
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(inplace=True)
        )
        # First pooling - reduce dimensions by factor of 2
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*2),
            nn.LeakyReLU(inplace=True)
        )
        # Second pooling - reduce dimensions by factor of 2 (total 4x reduction)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True)
        )
        # We stop at 4x downsampling (no third pooling layer)
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*8),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_channels*8),
            nn.LeakyReLU(inplace=True)
        )
        
        # Probability map generator without further reduction
        self.prob_map = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*4, kernel_size=1),
            nn.InstanceNorm3d(base_channels*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(base_channels*4, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(0.3)
    
    def forward(self, x):
        # Encoder path with dropout
        x1 = self.enc1(x)
        x1 = self.dropout(x1)
        
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x2 = self.dropout(x2)
        
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x3 = self.dropout(x3)
        
        # Bottleneck (no more pooling)
        x = self.bottleneck(x3)
        x = self.dropout(x)
        
        # Generate probability maps for each class
        prob_maps = self.prob_map(x)
        
        return {'features': [x1, x2, x3, x], 'prob_maps': prob_maps}

class TumorBoundingBoxGenerator:
    """
    Utility class to extract bounding boxes from tumor probability maps
    """
    def __init__(self, threshold=0.5, padding=10):
        self.threshold = threshold
        self.padding = padding
    
    def get_bounding_boxes(self, prob_maps, min_region_size=10):
        """
        Extract bounding boxes for each class from probability maps
        
        Args:
            prob_maps: Probability maps [B, C, D, H, W]
            min_region_size: Minimum size of tumor region to consider
            
        Returns:
            Dictionary of bounding boxes per class for each batch item
        """
        batch_size, num_classes, depth, height, width = prob_maps.shape
        all_boxes = []
        
        for b in range(batch_size):
            batch_boxes = []
            
            for c in range(1, num_classes):  # Skip background class (0)
                # Get binary mask for this class
                prob_map = prob_maps[b, c].cpu().detach().numpy()
                binary_mask = prob_map > self.threshold
                
                # Skip if no region found
                if not np.any(binary_mask):
                    batch_boxes.append(None)
                    continue
                
                # Label connected components
                labeled_mask, num_features = ndimage.label(binary_mask)
                
                # Find largest connected component
                if num_features == 0:
                    batch_boxes.append(None)
                    continue
                
                sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features+1))
                largest_component_idx = np.argmax(sizes) + 1
                
                # Skip if largest component is too small
                if sizes[largest_component_idx-1] < min_region_size:
                    batch_boxes.append(None)
                    continue
                
                # Extract component mask
                component_mask = labeled_mask == largest_component_idx
                
                # Find bounding box of tumor region
                z_indices, y_indices, x_indices = np.where(component_mask)
                
                # Add padding to bounding box
                z_min = max(0, np.min(z_indices) - self.padding)
                z_max = min(depth - 1, np.max(z_indices) + self.padding)
                y_min = max(0, np.min(y_indices) - self.padding)
                y_max = min(height - 1, np.max(y_indices) + self.padding)
                x_min = max(0, np.min(x_indices) - self.padding)
                x_max = min(width - 1, np.max(x_indices) + self.padding)
                
                # Store bounding box
                batch_boxes.append({
                    'z_min': z_min, 'z_max': z_max,
                    'y_min': y_min, 'y_max': y_max,
                    'x_min': x_min, 'x_max': x_max,
                    'class': c
                })
            
            all_boxes.append(batch_boxes)
        
        return all_boxes


class EnhancedRichSliceGenerator:
    """
    Enhanced generator for rich 2D representations that uses multi-level features
    from the 3D encoder for better information transfer to SAM2
    """
    def __init__(self, prob_map_weight=0.5):
        self.prob_map_weight = prob_map_weight
    
    def generate_rich_slice(self, volume, slice_idx, prob_maps=None, encoder_features=None):
        """
        Generate rich 2D representation using both volume data and encoder features
        
        Args:
            volume: Input volume [B, C, D, H, W]
            slice_idx: Index of slice to process
            prob_maps: Probability maps from encoder [B, num_classes, D, H, W]
            encoder_features: List of feature maps at different resolutions
                             [enc1, enc2, enc3, bottleneck]
        
        Returns:
            Rich 2D representation for SAM2
        """
        batch_size, modalities, depth, height, width = volume.shape
        
        # Prepare output batch
        rgb_batch = []
        
        for b in range(batch_size):
            # Get slice with safety bounds checking
            slice_idx_safe = min(max(slice_idx, 0), depth-1)
            current_slice = volume[b, :, slice_idx_safe].cpu().detach().numpy()
            
            # Extract individual modalities
            flair = current_slice[0].copy()
            t1 = current_slice[1].copy()
            t1ce = current_slice[2].copy() if modalities > 2 else t1.copy()
            t2 = current_slice[3].copy() if modalities > 3 else t1.copy()
            
            # Enhanced contrast function with adaptive percentile clipping
            def enhance_contrast(img, p_low=1, p_high=99):
                # Handle uniform regions
                if np.max(img) - np.min(img) < 1e-6:
                    return np.zeros_like(img)
                
                low, high = np.percentile(img, [p_low, p_high])
                img_norm = np.clip((img - low) / (high - low + 1e-8), 0, 1)
                return img_norm * 255
            
            # Create base RGB channels highlighting different tumor characteristics
            r_channel = enhance_contrast(flair).astype(np.uint8)  # FLAIR - edema
            g_channel = enhance_contrast(t1ce).astype(np.uint8)   # T1CE - enhancing tumor
            b_channel = enhance_contrast(t2).astype(np.uint8)     # T2 - general tumor region
            
            # ENHANCEMENT 1: Incorporate multi-level features from the encoder
            if encoder_features is not None:
                # Extract feature maps for this slice
                feature_maps = []
                for feat_level in encoder_features:
                    # Skip if feature level is empty
                    if feat_level is None or feat_level.shape[0] <= b:
                        continue
                        
                    # Get the correct slice from this feature level
                    # Need to handle different resolutions
                    feat_depth = feat_level.shape[2]
                    feat_slice_idx = min(int(slice_idx_safe * feat_depth / depth), feat_depth-1)
                    
                    # Extract slice from feature map
                    feat_slice = feat_level[b, :, feat_slice_idx].cpu().detach()
                    
                    # Resize to match the slice dimensions
                    if feat_slice.shape[1] != height or feat_slice.shape[2] != width:
                        feat_slice = F.interpolate(
                            feat_slice.unsqueeze(0),  # Add batch dim
                            size=(height, width),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    # Add to list of feature maps
                    feature_maps.append(feat_slice)
                
                # Enhance channels with feature information if available
                if feature_maps:
                    # Use deepest features for tumor highlighting (bottleneck)
                    if len(feature_maps) >= 4:
                        # Enhance red channel with bottleneck features (tumor focus)
                        deep_feat = feature_maps[3]
                        # Get mean across channels and normalize
                        deep_feat_mean = torch.mean(deep_feat, dim=0).numpy()
                        deep_feat_norm = (deep_feat_mean - deep_feat_mean.min()) / (deep_feat_mean.max() - deep_feat_mean.min() + 1e-8)
                        deep_feat_vis = (deep_feat_norm * 255).astype(np.uint8)
                        
                        # Blend with red channel (tumor highlighting)
                        r_channel = np.maximum(r_channel, deep_feat_vis)
                    
                    # Use middle-level features for edge enhancement (enc3)
                    if len(feature_maps) >= 3:
                        # Enhance green channel with mid-level features (edges)
                        mid_feat = feature_maps[2]
                        # Use variance across channels to detect edges
                        mid_feat_var = torch.var(mid_feat, dim=0).numpy()
                        mid_feat_norm = (mid_feat_var - mid_feat_var.min()) / (mid_feat_var.max() - mid_feat_var.min() + 1e-8)
                        mid_feat_vis = (mid_feat_norm * 255).astype(np.uint8)
                        
                        # Blend with green channel (edge enhancement)
                        g_channel = np.maximum(g_channel, mid_feat_vis)
                    
                    # Use shallow features for texture (enc1)
                    if len(feature_maps) >= 1:
                        # Enhance blue channel with shallow features (texture)
                        shallow_feat = feature_maps[0]
                        # Get max across channels for texture details
                        shallow_feat_max = torch.max(shallow_feat, dim=0)[0].numpy()
                        shallow_feat_norm = (shallow_feat_max - shallow_feat_max.min()) / (shallow_feat_max.max() - shallow_feat_max.min() + 1e-8)
                        shallow_feat_vis = (shallow_feat_norm * 255).astype(np.uint8)
                        
                        # Blend with blue channel (texture details)
                        b_channel = np.maximum(b_channel, shallow_feat_vis)
            
            # Add probability map influence if available
            if prob_maps is not None:
                # Ensure prob_maps has a valid slice index
                prob_depth = prob_maps.shape[2]
                prob_slice_idx = min(slice_idx_safe, prob_depth-1)
                prob_slice = prob_maps[b, 1:, prob_slice_idx].cpu().detach().numpy()
                
                # Resize probability maps to match the slice dimensions if needed
                if prob_slice.shape[1] != height or prob_slice.shape[2] != width:
                    from scipy.ndimage import zoom
                    # Calculate zoom factors
                    h_factor = height / prob_slice.shape[1]
                    w_factor = width / prob_slice.shape[2]
                    # Resize each class probability map
                    resized_prob = np.zeros((prob_slice.shape[0], height, width))
                    for c in range(prob_slice.shape[0]):
                        resized_prob[c] = zoom(prob_slice[c], (h_factor, w_factor), order=1)
                    prob_slice = resized_prob
                
                # ENHANCEMENT 2: Use class-specific color coding for probability maps
                # Class 1 (Edema) - enhance yellow (R+G)
                if prob_slice.shape[0] >= 1:
                    edema_prob = (prob_slice[0] * 255 * self.prob_map_weight).astype(np.uint8)
                    r_channel = np.maximum(r_channel, edema_prob)
                    g_channel = np.maximum(g_channel, edema_prob)
                
                # Class 2 (Non-enhancing) - enhance green
                if prob_slice.shape[0] >= 2:
                    non_enh_prob = (prob_slice[1] * 255 * self.prob_map_weight).astype(np.uint8)
                    g_channel = np.maximum(g_channel, non_enh_prob)
                
                # Class 3 (Enhancing) - enhance red
                if prob_slice.shape[0] >= 3:
                    enh_prob = (prob_slice[2] * 255 * self.prob_map_weight).astype(np.uint8)
                    r_channel = np.maximum(r_channel, enh_prob)
            
            # Stack channels to create RGB image
            rgb_slice = np.stack([r_channel, g_channel, b_channel], axis=2)
            
            # Ensure array is contiguous in memory
            rgb_slice = np.ascontiguousarray(rgb_slice)
            
            rgb_batch.append(rgb_slice)
        
        return np.array(rgb_batch)

class MulticlassPointPromptGenerator:
    """
    Generates intelligent point prompts for SAM2 for multiclass segmentation
    """
    def __init__(self, num_points_per_class=3):
        self.num_points_per_class = num_points_per_class
    
    def generate_prompts(self, prob_maps, slice_idx, height, width, threshold=0.7):
        """
        Generate point prompts for each class based on probability maps
        
        Args:
            prob_maps: Probability maps [B, C, D, H, W]
            slice_idx: Current slice index
            height, width: Dimensions of the slice
            threshold: Probability threshold for points
            
        Returns:
            Point coordinates and labels for each batch item
        """
        batch_size, num_classes, depth, _, _ = prob_maps.shape
        all_points = []
        all_labels = []
        
        # Ensure slice_idx is within bounds of prob_maps depth
        slice_idx_safe = min(max(slice_idx, 0), depth-1)
        
        for b in range(batch_size):
            # Initialize lists for points and labels
            batch_points = []
            batch_labels = []
            
            # Process each tumor class (skip background class 0)
            for c in range(1, num_classes):
                # Get probability map for this class in the current slice
                class_prob = prob_maps[b, c, slice_idx_safe].cpu().detach().numpy()
                
                # Resize probability map to match height and width if needed
                if class_prob.shape[0] != height or class_prob.shape[1] != width:
                    from scipy.ndimage import zoom
                    h_factor = height / class_prob.shape[0]
                    w_factor = width / class_prob.shape[1]
                    class_prob = zoom(class_prob, (h_factor, w_factor), order=1)
                
                # Find regions above threshold
                high_prob_mask = class_prob > threshold
                
                # If we found high probability regions
                if np.any(high_prob_mask):
                    # Label connected components
                    labeled_mask, num_features = ndimage.label(high_prob_mask)
                    
                    # Sort components by size
                    sizes = ndimage.sum(high_prob_mask, labeled_mask, range(1, num_features+1))
                    sorted_indices = np.argsort(-sizes)  # Descending order
                    
                    # Take up to the specified number of largest components
                    num_components = min(self.num_points_per_class, num_features)
                    points_added = 0
                    
                    for i in range(num_components):
                        component_idx = sorted_indices[i] + 1
                        component_mask = labeled_mask == component_idx
                        
                        # Find center of mass of the component
                        cy, cx = ndimage.center_of_mass(component_mask)
                        
                        # Add point to the list
                        batch_points.append([int(cx), int(cy)])
                        batch_labels.append(c)  # Use class as label
                        points_added += 1
                    
                    # If we didn't add enough points, add additional points from the same component
                    if points_added < self.num_points_per_class and num_features > 0:
                        largest_component = labeled_mask == (sorted_indices[0] + 1)
                        
                        # Find additional points within the largest component
                        y_indices, x_indices = np.where(largest_component)
                        
                        # If we have enough points
                        if len(y_indices) > 0:
                            # Randomly select additional points
                            remaining = self.num_points_per_class - points_added
                            indices = np.random.choice(len(y_indices), 
                                                      min(remaining, len(y_indices)), 
                                                      replace=False)
                            
                            for idx in indices:
                                batch_points.append([int(x_indices[idx]), int(y_indices[idx])])
                                batch_labels.append(c)
            
            # Add background points (class 0)
            # Add points near the edges of the image
            edge_points = [
                [width//4, height//4],
                [3*width//4, height//4],
                [width//4, 3*height//4],
                [3*width//4, 3*height//4]
            ]
            
            for point in edge_points:
                batch_points.append(point)
                batch_labels.append(0)  # Background class
            
            # Convert to numpy arrays
            all_points.append(np.array(batch_points))
            all_labels.append(np.array(batch_labels))
        
        return all_points, all_labels

class SAM2TumorSegmenter:
    """
    Uses SAM2 to segment tumors from rich 2D slices with multi-class support
    """
    def __init__(self, sam2_predictor, num_classes=4):
        self.sam2_predictor = sam2_predictor
        self.num_classes = num_classes
        
    def segment_slice(self, rich_slice, point_coords, point_labels, bounding_boxes=None):
        """
        Segment tumor in a rich 2D slice using SAM2
        
        Args:
            rich_slice: Rich 2D representation [H, W, 3]
            point_coords: Point prompts [N, 2]
            point_labels: Point labels [N]
            bounding_boxes: Optional bounding boxes
            
        Returns:
            Multi-class segmentation mask
        """
        height, width = rich_slice.shape[:2]
        
        # Initialize multi-class mask
        multi_class_mask = np.zeros((self.num_classes, height, width), dtype=np.float32)
        
        # Process each class separately
        for c in range(1, self.num_classes):
            # Get points for this class
            class_indices = np.where(point_labels == c)[0]
            
            # Skip if no points for this class
            if len(class_indices) == 0:
                continue
            
            # Extract points for this class
            class_points = point_coords[class_indices]
            class_point_labels = np.ones(len(class_points))  # All foreground
            
            # Also add some background points
            background_indices = np.where(point_labels == 0)[0]
            
            if len(background_indices) > 0:
                bg_points = point_coords[background_indices]
                
                # Combine foreground and background points
                combined_points = np.vstack([class_points, bg_points])
                combined_labels = np.concatenate([class_point_labels, 
                                               np.zeros(len(bg_points))])
            else:
                combined_points = class_points
                combined_labels = class_point_labels
            
            # Set image in SAM2 predictor
            try:
                self.sam2_predictor.set_image(rich_slice)
                
                # Get bounding box for this class if available
                box = None
                if bounding_boxes is not None and bounding_boxes[c-1] is not None:
                    # Extract 2D box from 3D box
                    box_info = bounding_boxes[c-1]
                    box = np.array([
                        box_info['x_min'], box_info['y_min'], 
                        box_info['x_max'], box_info['y_max']
                    ])
                
                # Get SAM2 prediction
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=combined_points,
                    point_labels=combined_labels,
                    box=box,
                    multimask_output=True
                )
                
                # Get best mask
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    class_mask = masks[best_idx].astype(np.float32)
                    
                    # Store in multi-class mask
                    multi_class_mask[c] = class_mask
            
            except Exception as e:
                print(f"Error in SAM2 segmentation for class {c}: {e}")
        
        # Resolve overlaps between classes
        # Priority: Enhancing tumor (3) > Non-enhancing tumor (2) > Edema (1)
        for y in range(height):
            for x in range(width):
                if multi_class_mask[3, y, x] > 0.5:  # Enhancing tumor
                    multi_class_mask[1, y, x] = 0
                    multi_class_mask[2, y, x] = 0
                elif multi_class_mask[2, y, x] > 0.5:  # Non-enhancing tumor
                    multi_class_mask[1, y, x] = 0
        
        return multi_class_mask

class AutoSAM2(nn.Module):
    """
    Implementation of AutoSAM2 for multi-class 3D medical image segmentation
    Enhanced with improved feature transfer from encoder to SAM2
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        
        # Create 3D encoder with reduced downsampling
        self.encoder = MultiscaleEncoder(in_channels=4, base_channels=16, num_classes=num_classes)
        
        # Create enhanced rich slice generator that uses multi-level features
        self.slice_generator = EnhancedRichSliceGenerator(prob_map_weight=0.5)
        
        # Create bounding box generator
        self.bbox_generator = TumorBoundingBoxGenerator()
        
        # Create point prompt generator
        self.prompt_generator = MulticlassPointPromptGenerator()
        
        # Initialize SAM2 model from Hugging Face
        try:
            print(f"Loading SAM2 from Hugging Face...")
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            
            # Freeze weights
            for param in self.sam2_predictor.model.parameters():
                param.requires_grad = False
            
            # Create tumor segmenter
            self.segmenter = SAM2TumorSegmenter(self.sam2_predictor, num_classes=num_classes)
            
            print("Initialized SAM2 model successfully")
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            raise RuntimeError(f"Failed to initialize SAM2: {e}")
    
    def process_slice_with_sam2(self, volume, prob_maps, encoder_features, slice_idx):
        """
        Process a single slice using the complete AutoSAM2 pipeline
        
        Args:
            volume: Input volume [B, C, D, H, W]
            prob_maps: Probability maps [B, num_classes, D, H, W]
            encoder_features: List of feature maps from encoder
            slice_idx: Index of slice to process
            
        Returns:
            Segmentation masks for the slice [B, num_classes, H, W]
        """
        batch_size, _, depth, height, width = volume.shape
        
        # Generate rich 2D representation using multi-level features
        rich_slices = self.slice_generator.generate_rich_slice(
            volume, slice_idx, prob_maps, encoder_features
        )
        
        # Generate bounding boxes from probability maps
        boxes = self.bbox_generator.get_bounding_boxes(prob_maps)
        
        # Generate point prompts
        points, labels = self.prompt_generator.generate_prompts(
            prob_maps, slice_idx, height, width
        )
        
        # Process each slice with SAM2
        result_masks = []
        
        for b in range(batch_size):
            # Process slice
            mask = self.segmenter.segment_slice(
                rich_slices[b], 
                points[b], 
                labels[b], 
                boxes[b] if b < len(boxes) else None
            )
            
            # Convert to tensor
            mask_tensor = torch.tensor(mask, device=volume.device)
            result_masks.append(mask_tensor)
        
        # Stack results
        stacked_masks = torch.stack(result_masks) if result_masks else torch.zeros(
            (batch_size, self.num_classes, height, width), 
            device=volume.device
        )
        
        return stacked_masks
        
    def forward(self, x, visualize=False):
        """
        Forward pass through AutoSAM2 with enhanced feature transfer
        
        Args:
            x: Input volume [B, C, D, H, W]
            visualize: Whether to visualize results
            
        Returns:
            Multi-class segmentation masks [B, num_classes, D, H, W]
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Run encoder to get features and probability maps
        encoder_output = self.encoder(x)
        encoder_features = encoder_output['features']  # [enc1, enc2, enc3, bottleneck]
        prob_maps = encoder_output['prob_maps']
        
        # Resize probability maps to match input dimensions
        resized_prob_maps = F.interpolate(
            prob_maps, 
            size=(depth, height, width),
            mode='trilinear', 
            align_corners=False
        )
        
        # Initialize output tensor
        output_masks = torch.zeros_like(resized_prob_maps)
        
        # Process the middle slice for efficient computation during training
        middle_slice = depth // 2
        
        # Process the slice with SAM2 using multi-level features
        slice_masks = self.process_slice_with_sam2(
            x, resized_prob_maps, encoder_features, middle_slice
        )
        
        # Add the processed slice to the output volume
        output_masks[:, :, middle_slice] = slice_masks
        
        # For other slices, use the encoder predictions
        # This helps maintain computational efficiency while still leveraging the SAM2 segmentation
        # for the most important slice
        
        # Fill in other slices with encoder predictions
        for z in range(depth):
            if z != middle_slice:  # Skip the middle slice that was processed with SAM2
                output_masks[:, :, z] = resized_prob_maps[:, :, z]
        
        # Optional visualization
        if visualize:
            self._visualize_slice(x, resized_prob_maps, slice_masks, middle_slice)
        
        return output_masks
    
    def _visualize_slice(self, volume, prob_maps, sam_masks, slice_idx):
        """
        Visualize the segmentation results for debugging
        
        Args:
            volume: Input volume [B, C, D, H, W]
            prob_maps: Probability maps from encoder [B, num_classes, D, H, W]
            sam_masks: Segmentation masks from SAM2 [B, num_classes, H, W]
            slice_idx: Index of processed slice
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Use first batch item
        b = 0
        
        # Get slice data
        slice_data = volume[b, 0, slice_idx].cpu().detach().numpy()  # FLAIR
        prob_slice = prob_maps[b, 1:, slice_idx].cpu().detach().numpy()  # Skip background
        sam_slice = sam_masks[b, 1:].cpu().detach().numpy()  # Skip background
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Plot original slice
        axes[0, 0].imshow(slice_data, cmap='gray')
        axes[0, 0].set_title('Original FLAIR')
        axes[0, 0].axis('off')
        
        # Create color maps for tumors
        def create_color_overlay(base_img, masks):
            overlay = np.zeros((*base_img.shape, 3))
            
            # Normalize base image
            base_norm = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
            
            # Set as grayscale background
            overlay[..., 0] = base_norm
            overlay[..., 1] = base_norm
            overlay[..., 2] = base_norm
            
            # Add colored overlays for each class
            if masks.shape[0] >= 1:
                # Edema: Yellow
                edema_mask = masks[0] > 0.5
                overlay[edema_mask, 0] = 1.0  # Red
                overlay[edema_mask, 1] = 1.0  # Green
                overlay[edema_mask, 2] = 0.0  # No blue
            
            if masks.shape[0] >= 2:
                # Non-enhancing tumor: Green
                non_enh_mask = masks[1] > 0.5
                overlay[non_enh_mask, 0] = 0.0  # No red
                overlay[non_enh_mask, 1] = 1.0  # Green
                overlay[non_enh_mask, 2] = 0.0  # No blue
            
            if masks.shape[0] >= 3:
                # Enhancing tumor: Red
                enh_mask = masks[2] > 0.5
                overlay[enh_mask, 0] = 1.0  # Red
                overlay[enh_mask, 1] = 0.0  # No green
                overlay[enh_mask, 2] = 0.0  # No blue
            
            return overlay
        
        # Plot encoder predictions
        axes[0, 1].imshow(create_color_overlay(slice_data, prob_slice))
        axes[0, 1].set_title('Encoder Predictions')
        axes[0, 1].axis('off')
        
        # Plot SAM2 segmentation
        axes[1, 0].imshow(create_color_overlay(slice_data, sam_slice))
        axes[1, 0].set_title('SAM2 Segmentation')
        axes[1, 0].axis('off')
        
        # Generate rich slice representation
        rich_slice = self.slice_generator.generate_rich_slice(
            volume[:1], slice_idx, prob_maps[:1], self.encoder(volume[:1])['features']
        )[0]
        
        # Convert RGB to proper format for display
        rich_slice_display = rich_slice.astype(np.float32) / 255.0
        
        # Plot rich slice representation
        axes[1, 1].imshow(rich_slice_display)
        axes[1, 1].set_title('Rich Slice Representation')
        axes[1, 1].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"results/slice_{slice_idx}_visualization.png")
        plt.close()



# train.py - Enhanced version with optimizations
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
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, y_pred, y_true):
        # Apply sigmoid to logits for BCE loss
        bce = self.bce_loss(y_pred, y_true)
        
        # Dice loss expects raw logits
        dice = self.dice_loss(y_pred, y_true)
        
        # Weighted sum
        return self.dice_weight * dice + self.bce_weight * bce

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
    Fixed train_epoch function to ensure Dice scores are properly updated
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
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate metrics for EVERY batch, not just the first one
            with torch.no_grad():
                metrics = calculate_dice_score(outputs.detach(), masks)
                all_metrics.append(metrics)
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Update learning rate within epoch if using OneCycleLR
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Update progress bar with CURRENT batch metrics, not first batch metrics
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['mean']:.4f}"  # Current batch metrics
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
    avg_metrics = {'mean': np.mean([m['mean'] for m in all_metrics]) if all_metrics else 0.0}
    
    # Add per-class metrics if available
    if all_metrics:
        for key in all_metrics[0]:
            if key != 'mean':
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model
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
                
                # Calculate BCE Loss
                bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)

                # Calculate Dice Loss
                dice_loss = DiceLoss()(outputs, masks)

                # Combine Losses
                loss = 0.7 * dice_loss + 0.3 * bce_loss
                
                # Compute metrics (Dice and IoU)
                dice_metrics = calculate_dice_score(outputs, masks)
                iou_metrics = calculate_iou(outputs, masks)
                
                # Store metrics
                all_metrics.append({**dice_metrics, **iou_metrics, 'bce_loss': bce_loss.item()})
                                
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
            'mean_dice': np.mean([m['mean'] for m in all_metrics]) if all_metrics else 0.0,
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]) if all_metrics else 0.0,
            'bce_loss': np.mean([m['bce_loss'] for m in all_metrics]) if all_metrics else 0.0
        }

        # Add per-class metrics if available
        if all_metrics:
            for key in all_metrics[0]:
                if key not in ['mean', 'mean_iou', 'bce_loss']:
                    avg_metrics[key] = np.mean([m[key] for m in all_metrics])

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
    model = AutoSAM2(
        num_classes=4  # Background + 3 tumor classes
    ).to(device)
    
    # Check if model file exists to resume training
    model_path = "checkpoints/best_autosam2_model.pth"
    start_epoch = 0
    best_dice = 0.0
    
    if os.path.exists(model_path) and not reset:
        print(f"Found existing model checkpoint at {model_path}, loading...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            print(f"Resuming from epoch {start_epoch} with best Dice score: {best_dice:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch...")
            start_epoch = 0
            best_dice = 0.0
    else:
        if reset and os.path.exists(model_path):
            print(f"Reset flag is set. Ignoring existing checkpoint and starting from epoch 0.")
        else:
            print("No existing checkpoint found. Starting from epoch 0.")
    
    # Define criterion - combined loss for better convergence
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)
    
    # Define optimizer with higher learning rate
    optimizer = optim.AdamW(
        model.encoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-4  # Increased weight decay for better regularization
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
        data_path, 
        batch_size=batch_size, 
        train=True,
        normalize=True, 
        max_samples=max_samples,  
        num_workers=4,
        filter_empty=False,
        use_augmentation=True,  # Enable augmentation
        target_shape=target_shape,
        cache_data=False,
        verbose=True
    )

    val_loader = get_brats_dataloader(
        data_path, 
        batch_size=batch_size, 
        train=False,
        normalize=True, 
        max_samples=max_samples,  
        num_workers=4,
        filter_empty=False,
        use_augmentation=False,
        target_shape=target_shape,
        cache_data=False,
        verbose=True
    )
    
    # Set up OneCycle learning rate scheduler 
    # This is one of the recommended changes - provides better convergence
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # Initial LR is max_lr/25
        final_div_factor=1000  # Final LR is max_lr/1000
    )
    
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
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scheduler
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_metrics['mean'])
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update history
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_metrics['mean'])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        estimated_total_time = elapsed_time / (epoch - start_epoch + 1) * (epochs - start_epoch) if epoch > start_epoch else epochs * epoch_time
        remaining_time = max(0, estimated_total_time - elapsed_time)
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Dice = {train_metrics['mean']:.4f}")
        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Dice = {val_metrics['mean']:.4f}")
        print(f"Epoch Time: {timedelta(seconds=int(epoch_time))}, Remaining: {timedelta(seconds=int(remaining_time))}")
        
        # Save best model based on mean Dice score
        if val_metrics['mean'] > best_dice:
            best_dice = val_metrics['mean']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_dice': best_dice,
            }, model_path)
            print(f"Saved new best model with Dice score: {best_dice:.4f}")
            print(f"Model saved at: {os.path.abspath(model_path)}")  # PRINT THE SAVED MODEL PATH
            counter = 0  # Reset early stopping counter
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"No improvement in Dice score for {counter} epochs")
        
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
    print(f"Best trained model saved at: {os.path.abspath(model_path)}")
    
    # Load best model for further use
    best_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    best_metrics = best_checkpoint.get('val_metrics', None)
    
    return model, history, best_metrics

def main():
    parser = argparse.ArgumentParser(description="Train AutoSAM2 for brain tumor segmentation")
    parser.add_argument('--data_path', type=str, default="/home/erezhuberman/data/Task01_BrainTumour",
                    help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,  # Increased learning rate
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
    
    args = parser.parse_args()
    
    # Set memory limit for GPU
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.memory_limit)
    
    # Parse target shape if provided
    target_shape = tuple(map(int, args.target_shape.split(',')))
    print(f"Using target shape: {target_shape}")
    
    # Train the model
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
    
    # Print final metrics
    print("Final best metrics:")
    for key, value in best_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
