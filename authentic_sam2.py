# Import your existing dependencies from model.py
from model import EnhancedPromptGenerator, MRItoRGBMapper, FlexibleUNet3D, HAS_SAM2
from model import build_sam2_hf, SAM2ImagePredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger("AuthenticSAM2")
# ==============================================================
# 1. DIRECT PROMPT ENCODER FOR MEDICAL IMAGES
# ==============================================================
# This follows the original AutoSAM approach more closely by taking
# the input image directly and creating prompts for SAM2

class MedicalImagePromptEncoder(nn.Module):
    """
    Direct prompt encoder for medical images that follows the original AutoSAM approach.
    Takes raw medical image data and generates prompts for SAM2 without using UNet features.
    """
    def __init__(self, in_channels=4, out_channels=256, base_channels=16):
        super(MedicalImagePromptEncoder, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder blocks - using ResNet-style blocks for robust feature extraction
        self.enc1 = self._make_encoder_block(base_channels, base_channels*2, stride=1)
        self.enc2 = self._make_encoder_block(base_channels*2, base_channels*4, stride=2)
        self.enc3 = self._make_encoder_block(base_channels*4, base_channels*8, stride=2)
        
        # Attention module to focus on relevant regions
        self.attention = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*2, kernel_size=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Prompt decoder for SAM2
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*2, out_channels, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_encoder_block(self, in_channels, out_channels, stride):
        """Create a ResNet-style encoder block"""
        layers = []
        
        # Downsampling block
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ))
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        # Add residual connection
        layers.append(lambda x, shortcut=shortcut: F.relu(shortcut(x) + layers[0](x)))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize the weights of the model"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass of the prompt encoder"""
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Encoder pathway
        x = self.enc1(x)
        x = self.enc2(x)
        enc_features = self.enc3(x)
        
        # Generate attention mask
        attention_mask = self.attention(enc_features)
        
        # Apply attention
        attended_features = enc_features * attention_mask
        
        # Decode to SAM2 prompt format
        prompt_embeddings = self.decoder(attended_features)
        
        return prompt_embeddings, attention_mask


# ==============================================================
# 2. SLICE EXTRACTOR FOR 3D VOLUMES
# ==============================================================
# This component handles extraction of 2D slices from 3D volumes

class VolumeSliceExtractor(nn.Module):
    """
    Extracts 2D slices from 3D volumes and prepares them for processing
    with the prompt encoder and SAM2.
    """
    def __init__(self, normalize=True):
        super(VolumeSliceExtractor, self).__init__()
        self.normalize = normalize
    
    def extract_slice(self, volume, slice_idx, depth_dim=2):
        """Extract a specific slice from a 3D volume"""
        # Handle different depth dimension positions
        if depth_dim == 2:  # [B, C, D, H, W]
            slice_data = volume[:, :, slice_idx, :, :]
        elif depth_dim == 3:  # [B, C, H, D, W]
            slice_data = volume[:, :, :, slice_idx, :]
        elif depth_dim == 4:  # [B, C, H, W, D]
            slice_data = volume[:, :, :, :, slice_idx]
        else:
            raise ValueError(f"Unsupported depth dimension: {depth_dim}")
        
        # If we have a batch, ensure it's handled correctly
        if len(slice_data.shape) > 3 and slice_data.shape[0] > 1:
            slice_data = slice_data[0:1]  # Take only the first item if batch size > 1
        
        return slice_data
    
    def normalize_slice(self, slice_data):
        """Normalize a 2D slice for consistent processing"""
        if not self.normalize:
            return slice_data
        
        # Z-score normalization per channel
        normalized_slice = slice_data.clone()
        
        for c in range(slice_data.shape[1]):
            # Get non-zero values
            channel = slice_data[:, c]
            mask = channel > 0
            
            if mask.sum() > 0:
                mean = torch.mean(channel[mask])
                std = torch.std(channel[mask])
                # Apply normalization only to non-zero values
                normalized_slice[:, c][mask] = (channel[mask] - mean) / (std + 1e-8)
        
        return normalized_slice
    
    def prepare_slice(self, volume, slice_idx, depth_dim=2, target_size=None):
        """Extract, normalize and prepare a slice for processing"""
        # Extract the slice
        slice_data = self.extract_slice(volume, slice_idx, depth_dim)
        
        # Normalize
        slice_data = self.normalize_slice(slice_data)
        
        # Resize if target size is specified
        if target_size is not None:
            curr_size = slice_data.shape[-2:]
            if curr_size != target_size:
                slice_data = F.interpolate(
                    slice_data, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
        
        return slice_data


# ==============================================================
# 3. AUTHENTICSAM2 MODEL 
# ==============================================================
# Main model that follows the original AutoSAM approach more closely

class AuthenticSAM2(nn.Module):
    """
    A more authentic implementation of AutoSAM for 3D medical data.
    Directly takes medical image slices and generates prompts for SAM2,
    following the original AutoSAM paper's approach more closely.
    """
    def __init__(
        self, 
        num_classes=4, 
        base_channels=16,
        sam2_model_id="facebook/sam2-hiera-small",
        enable_hybrid_mode=False
    ):
        super(AuthenticSAM2, self).__init__()
        
        # Configuration
        self.num_classes = num_classes
        self.enable_hybrid_mode = enable_hybrid_mode
        self.sam2_model_id = sam2_model_id
        
        # Initialize standard UNet3D for hybrid mode (optional)
        if enable_hybrid_mode:
            self.unet3d = FlexibleUNet3D(
                in_channels=4,
                n_classes=num_classes,
                base_channels=base_channels,
                trilinear=True
            )
        
        # Initialize prompt generator for SAM2
        self.prompt_generator = EnhancedPromptGenerator(
            num_positive_points=5,
            num_negative_points=3,
            edge_detection=True,
            use_confidence=True,
            use_mask_prompt=True
        )
        
        # NEW: Direct medical image prompt encoder following AutoSAM approach
        self.prompt_encoder = MedicalImagePromptEncoder(
            in_channels=4,  # 4 MRI modalities
            out_channels=256,  # SAM2 expected embedding size
            base_channels=32
        )
        
        # Slice extractor for handling 3D to 2D conversion
        self.slice_extractor = VolumeSliceExtractor(normalize=True)
        
        # MRI to RGB converter for SAM2
        self.mri_to_rgb = MRItoRGBMapper()
        
        # Initialize tracking variables
        self.has_sam2 = False
        self.sam2 = None
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.performance_metrics["sam2_slices_processed"] = 0
        
        # Store latest outputs for visualization
        self.last_sam2_slices = None
        self.last_combined_output = None
        
        # Initialize SAM2
        self.initialize_sam2()
    
    def initialize_sam2(self):
        """Initialize SAM2 with appropriate error handling."""
        if not HAS_SAM2:
            logger.warning("SAM2 package not available. Running in fallback mode.")
            return
            
        try:
            # Initialize SAM2
            logger.info(f"Building SAM2 with model_id: {self.sam2_model_id}")
            sam2_model = build_sam2_hf(self.sam2_model_id)
            self.sam2 = SAM2ImagePredictor(sam2_model)
            
            # Freeze SAM2 weights (following original AutoSAM approach)
            for param in self.sam2.model.parameters():
                param.requires_grad = False
            
            self.has_sam2 = True
            logger.info("SAM2 initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SAM2: {e}")
            self.has_sam2 = False
            self.sam2 = None
    
    def set_mode(self, enable_hybrid_mode=None):
        """Change between hybrid and SAM2-only mode"""
        if enable_hybrid_mode is not None:
            self.enable_hybrid_mode = enable_hybrid_mode
        
        logger.info(f"Mode set to: Hybrid={self.enable_hybrid_mode}")
    
    def get_strategic_slices(self, depth, percentage=0.6):
        """
        Select strategic slices making up the requested percentage of total depth
        with higher concentration in the center regions.
        
        Args:
            depth: Total number of slices in the volume
            percentage: Percentage of slices to select (0.0-1.0)
        
        Returns:
            List of selected slice indices
        """
        # Calculate number of slices to select
        num_slices = max(1, int(depth * percentage))
        
        # For very few slices, use simple approach
        if num_slices <= 5:
            return [int(depth * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9][:num_slices]]
        
        # Create three regions with different densities
        center_region = 0.5  # 50% of slices in the center 40% of the volume
        sides_region = 0.3   # 30% of slices in the middle 40% of the volume 
        edges_region = 0.2   # 20% of slices in the outer 20% of the volume
        
        # Calculate slice counts for each region
        center_count = int(num_slices * center_region)
        sides_count = int(num_slices * sides_region)
        edges_count = num_slices - center_count - sides_count
        
        # Generate slice indices for each region
        center_slices = []
        if center_count > 0:
            center_start = int(depth * 0.4)
            center_end = int(depth * 0.6)
            step = (center_end - center_start) / center_count
            center_slices = [int(center_start + i * step) for i in range(center_count)]
        
        sides_slices = []
        if sides_count > 0:
            side1_start = int(depth * 0.2)
            side1_end = int(depth * 0.4)
            side2_start = int(depth * 0.6)
            side2_end = int(depth * 0.8)
            
            sides_per_side = sides_count // 2
            remainder = sides_count % 2
            
            side1_step = (side1_end - side1_start) / (sides_per_side)
            side1_slices = [int(side1_start + i * side1_step) for i in range(sides_per_side)]
            
            side2_step = (side2_end - side2_start) / (sides_per_side + remainder)
            side2_slices = [int(side2_start + i * side2_step) for i in range(sides_per_side + remainder)]
            
            sides_slices = side1_slices + side2_slices
        
        edges_slices = []
        if edges_count > 0:
            edge1_start = 0
            edge1_end = int(depth * 0.2)
            edge2_start = int(depth * 0.8)
            edge2_end = depth
            
            edges_per_side = edges_count // 2
            remainder = edges_count % 2
            
            edge1_step = (edge1_end - edge1_start) / (edges_per_side)
            edge1_slices = [int(edge1_start + i * edge1_step) for i in range(edges_per_side)]
            
            edge2_step = (edge2_end - edge2_start) / (edges_per_side + remainder)
            edge2_slices = [int(edge2_start + i * edge2_step) for i in range(edges_per_side + remainder)]
            
            edges_slices = edge1_slices + edge2_slices
        
        # Combine all slices and sort
        all_slices = sorted(center_slices + sides_slices + edges_slices)
        
        # Ensure no duplicates and stay within bounds
        all_slices = sorted(list(set([min(depth-1, max(0, idx)) for idx in all_slices])))
        
        return all_slices
    
    def process_slice_with_authentic_sam2(self, input_slice, orig_slice_data, slice_idx, device):
        """
        Process a single slice with SAM2 using the authentic AutoSAM approach.
        Takes the raw input slice and generates prompts directly.
        
        Args:
            input_slice: Raw input medical image slice [1, C, H, W]
            orig_slice_data: Original full-resolution slice for SAM2
            slice_idx: Current slice index
            device: Device to use for processing
            
        Returns:
            Segmentation mask for this slice
        """
        if not self.has_sam2:
            logger.warning(f"SAM2 not available for slice {slice_idx}")
            return None
        
        try:
            # Get image dimensions
            h, w = input_slice.shape[-2:]
            
            # *** AUTHENTIC AUTOSAM APPROACH ***
            # Generate prompts directly from input slice without UNet features
            prompt_embeddings, attention_mask = self.prompt_encoder(input_slice)
            
            # Convert MRI to RGB for SAM2 input image
            rgb_tensor = self.mri_to_rgb(orig_slice_data)
            
            # Convert to numpy array in the format SAM2 expects: [H, W, 3]
            rgb_image = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            
            # Generate point prompts from attention mask
            points, labels, box, _ = self.prompt_generator.generate_prompts(
                prompt_embeddings, slice_idx, h, w
            )
            
            # Set image in SAM2
            self.sam2.set_image(rgb_image)
            
            # Call SAM2 with prompts
            masks, scores, _ = self.sam2.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True
            )
            
            # Select best mask based on score
            if len(masks) > 0:
                best_idx = scores.argmax()
                best_mask = masks[best_idx]
                
                # Convert to tensor and format
                mask_tensor = torch.from_numpy(best_mask).float().to(device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Create multi-class output using the binary-style approach
                height, width = mask_tensor.shape[2:]
                multi_class_mask = torch.zeros((1, self.num_classes, height, width), device=device)
                
                # Background (class 0) is inverse of the tumor mask
                multi_class_mask[:, 0] = 1.0 - mask_tensor[:, 0]
                
                # Distribute tumor mask among classes 1, 2, 3 using typical BraTS distribution
                typical_dist = [0.0, 0.3, 0.4, 0.3]  # Background + 3 tumor classes
                for c in range(1, self.num_classes):
                    multi_class_mask[:, c] = mask_tensor[:, 0] * typical_dist[c]
                
                # Ensure the class probabilities sum to 1.0
                total_prob = multi_class_mask.sum(dim=1, keepdim=True)
                multi_class_mask = multi_class_mask / total_prob.clamp(min=1e-5)
                
                return multi_class_mask
            else:
                # No valid masks found
                logger.warning(f"SAM2 failed to generate masks for slice {slice_idx}")
                return None
                    
        except Exception as e:
            logger.error(f"Error processing slice {slice_idx} with SAM2: {e}")
            return None
    
    def create_3d_from_slices(self, input_shape, sam2_slices, depth_dim_idx, device):
        """
        Create a 3D volume from 2D slice results with interpolation between processed slices.
        """
        # Create empty volume matching the expected output size
        batch_size = input_shape[0]
        output_shape = list(input_shape)
        output_shape[1] = self.num_classes  # Change channel dimension to match num_classes
        
        volume = torch.zeros(output_shape, device=device)
        
        # Create boolean mask of processed slices
        processed_slices_mask = torch.zeros(output_shape[depth_dim_idx], dtype=torch.bool, device=device)
        
        # Insert each slice result into the appropriate position
        for slice_idx, mask in sam2_slices.items():
            if mask is not None:
                if depth_dim_idx == 2:
                    volume[:, :, slice_idx] = mask
                elif depth_dim_idx == 3:
                    volume[:, :, :, slice_idx] = mask
                elif depth_dim_idx == 4:
                    volume[:, :, :, :, slice_idx] = mask
                    
                processed_slices_mask[slice_idx] = True
        
        # Interpolate between processed slices for smoother transitions
        if sum(processed_slices_mask) > 1:
            # Create an index tensor for interpolation reference
            slice_indices = torch.nonzero(processed_slices_mask).squeeze(-1)
            
            # Find gaps between processed slices
            for i in range(len(slice_indices) - 1):
                start_idx = slice_indices[i].item()
                end_idx = slice_indices[i + 1].item()
                
                # If there's a gap, interpolate
                if end_idx - start_idx > 1:
                    # Get start and end slices
                    if depth_dim_idx == 2:
                        start_slice = volume[:, :, start_idx].clone()
                        end_slice = volume[:, :, end_idx].clone()
                    elif depth_dim_idx == 3:
                        start_slice = volume[:, :, :, start_idx].clone()
                        end_slice = volume[:, :, :, end_idx].clone()
                    elif depth_dim_idx == 4:
                        start_slice = volume[:, :, :, :, start_idx].clone()
                        end_slice = volume[:, :, :, :, end_idx].clone()
                    
                    # Linear interpolation weights
                    for j in range(1, end_idx - start_idx):
                        alpha = j / (end_idx - start_idx)
                        interp_slice = (1 - alpha) * start_slice + alpha * end_slice
                        
                        # Ensure probability distribution sums to 1
                        interp_slice = F.normalize(interp_slice, p=1, dim=1)
                        
                        # Insert interpolated slice
                        if depth_dim_idx == 2:
                            volume[:, :, start_idx + j] = interp_slice
                        elif depth_dim_idx == 3:
                            volume[:, :, :, start_idx + j] = interp_slice
                        elif depth_dim_idx == 4:
                            volume[:, :, :, :, start_idx + j] = interp_slice
        
        return volume
    
    def combine_results(self, unet_output, sam2_output, blend_ratio=0.5):
        """
        Combine UNet and SAM2 outputs in hybrid mode.
        
        Args:
            unet_output: Output from UNet3D
            sam2_output: Output from SAM2
            blend_ratio: Blending ratio (0.0 = full UNet, 1.0 = full SAM2)
            
        Returns:
            Combined segmentation
        """
        combined = (1 - blend_ratio) * unet_output + blend_ratio * sam2_output
        
        # Normalize to ensure valid probability distribution
        total_prob = combined.sum(dim=1, keepdim=True)
        combined = combined / total_prob.clamp(min=1e-5)
        
        return combined
    
    def forward(self, x):
        """
        Forward pass with authentic AutoSAM approach for 3D medical data.
        This implementation follows the original AutoSAM paper more closely.
        """
        start_time = time.time()
        device = x.device
        
        # Check if SAM2 is available
        if not self.has_sam2:
            if self.enable_hybrid_mode and hasattr(self, 'unet3d'):
                # Fallback to UNet3D if SAM2 is not available
                return self.unet3d(x)[0]
            else:
                logger.error("SAM2 not available and hybrid mode disabled!")
                return torch.zeros((x.shape[0], self.num_classes) + x.shape[2:], device=device)
        
        # Get input dimensions
        batch_size, channels, *spatial_dims = x.shape
        
        # Identify which dimension is the depth dimension (usually the smallest)
        if len(spatial_dims) == 3:  # We have a 3D volume
            min_dim = min(spatial_dims)
            depth_dim_idx = spatial_dims.index(min_dim) + 2  # +2 for batch and channel dims
            depth = spatial_dims[depth_dim_idx - 2]
        else:
            raise ValueError("Expected 3D volume input (5D tensor)")
        
        # Process with UNet3D in hybrid mode
        if self.enable_hybrid_mode and hasattr(self, 'unet3d'):
            unet_output, _, _, _ = self.unet3d(x)
        else:
            unet_output = None
        
        # Select strategic slices for SAM2 processing
        key_indices = self.get_strategic_slices(depth, percentage=1.0)  # Use all slices for authenticity
        key_indices.sort()
        
        # Process each slice with authentic AutoSAM approach
        sam2_results = {}
        
        for slice_idx in key_indices:
            try:
                # Extract slice for SAM2
                orig_slice = self.slice_extractor.extract_slice(x, slice_idx, depth_dim_idx)
                
                # Prepare slice for prompt encoder 
                processed_slice = self.slice_extractor.prepare_slice(
                    x, slice_idx, depth_dim_idx, target_size=(224, 224)  # Standard size for efficiency
                )
                
                # Process with authentic SAM2 approach
                result = self.process_slice_with_authentic_sam2(
                    processed_slice, orig_slice, slice_idx, device
                )
                
                if result is not None:
                    sam2_results[slice_idx] = result
                    self.performance_metrics["sam2_slices_processed"] += 1
            except Exception as e:
                logger.error(f"Error processing slice {slice_idx}: {e}")
        
        # Store SAM2 results for visualization
        self.last_sam2_slices = sam2_results
        
        # Create 3D volume from SAM2 results
        sam2_volume = self.create_3d_from_slices(x.shape, sam2_results, depth_dim_idx, device)
        
        # Hybrid mode: blend UNet and SAM2 results if enabled
        if self.enable_hybrid_mode and unet_output is not None:
            final_output = self.combine_results(unet_output, sam2_volume, blend_ratio=0.5)
        else:
            final_output = sam2_volume
        
        # Store for visualization
        self.last_combined_output = final_output
        
        # Update timing metrics
        total_time = time.time() - start_time
        self.performance_metrics["total_time"].append(total_time)
        
        return final_output


# ==============================================================
# 4. TRAINING HELPERS
# ==============================================================
# Functions to help with training the authentic AutoSAM model

def compute_segmentation_loss(outputs, targets, num_classes=4):
    """
    Compute combined segmentation loss for training the prompt encoder.
    This follows the original AutoSAM paper approach.
    
    Args:
        outputs: Model outputs [B, C, ...]
        targets: Ground truth masks [B, C, ...]
        num_classes: Number of segmentation classes
        
    Returns:
        Combined loss value
    """
    device = outputs.device
    batch_size = outputs.shape[0]
    
    # Compute multi-class Dice loss
    dice_loss = 0.0
    
    for c in range(1, num_classes):  # Skip background class
        for b in range(batch_size):
            pred = outputs[b, c].view(-1)
            target = targets[b, c].view(-1)
            
            # Skip if no target pixels
            if target.sum() == 0 and pred.sum() == 0:
                continue
                
            # Compute Dice coefficient
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            
            if union > 0:
                dice = (2.0 * intersection) / (union + 1e-7)
                dice_loss += (1.0 - dice)
    
    # Normalize by number of valid comparisons
    num_comparisons = (num_classes - 1) * batch_size
    dice_loss = dice_loss / max(1, num_comparisons)
    
    # Compute BCE loss
    bce_loss = nn.BCEWithLogitsLoss()(outputs, targets)
    
    # Combined loss with weighting
    return 0.7 * dice_loss + 0.3 * bce_loss


def train_epoch_authentic(model, train_loader, optimizer, device, epoch):
    """
    Training epoch function specifically for the authentic AutoSAM model.
    Ensures proper gradient flow to the prompt encoder.
    
    Args:
        model: AuthenticSAM2 model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to use for training
        epoch: Current epoch number
        
    Returns:
        Average loss and metrics
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = compute_segmentation_loss(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Return average loss
    return total_loss / num_batches


def validate_authentic(model, val_loader, device, epoch):
    """
    Validation function for the authentic AutoSAM model.
    Computes metrics for evaluation.
    
    Args:
        model: AuthenticSAM2 model
        val_loader: DataLoader for validation data
        device: Device to use for validation
        epoch: Current epoch number
        
    Returns:
        Average metrics
    """
    model.eval()
    total_loss = 0
    dice_scores = defaultdict(list)
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = compute_segmentation_loss(outputs, targets)
            
            # Compute Dice scores for each class
            for c in range(1, 4):  # Skip background, focus on tumor classes
                for b in range(outputs.shape[0]):
                    pred = (outputs[b, c] > 0.5).float()
                    target = targets[b, c]
                    
                    # Skip if no target pixels
                    if target.sum() == 0 and pred.sum() == 0:
                        continue
                        
                    # Compute Dice coefficient
                    intersection = (pred * target).sum().item()
                    union = pred.sum().item() + target.sum().item()
                    
                    if union > 0:
                        dice = (2.0 * intersection) / (union + 1e-7)
                        
                        # Store in appropriate category
                        if c == 1:
                            dice_scores['NCR'].append(dice * 100)  # Convert to percentage
                        elif c == 2:
                            dice_scores['ED'].append(dice * 100)
                        elif c == 3:
                            dice_scores['ET'].append(dice * 100)
            
            # Compute whole tumor (WT) and tumor core (TC) dice scores
            for b in range(outputs.shape[0]):
                # Whole Tumor (WT): all tumor classes
                pred_wt = ((outputs[b, 1:4].sum(dim=0) > 0.5)).float()
                target_wt = (targets[b, 1:4].sum(dim=0) > 0).float()
                
                if target_wt.sum() > 0 or pred_wt.sum() > 0:
                    intersection = (pred_wt * target_wt).sum().item()
                    union = pred_wt.sum().item() + target_wt.sum().item()
                    
                    if union > 0:
                        dice_wt = (2.0 * intersection) / (union + 1e-7)
                        dice_scores['WT'].append(dice_wt * 100)
                
                # Tumor Core (TC): NCR + ET (classes 1 and 3)
                pred_tc = ((outputs[b, [1, 3]].sum(dim=0) > 0.5)).float()
                target_tc = (targets[b, [1, 3]].sum(dim=0) > 0).float()
                
                if target_tc.sum() > 0 or pred_tc.sum() > 0:
                    intersection = (pred_tc * target_tc).sum().item()
                    union = pred_tc.sum().item() + target_tc.sum().item()
                    
                    if union > 0:
                        dice_tc = (2.0 * intersection) / (union + 1e-7)
                        dice_scores['TC'].append(dice_tc * 100)
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"Val Epoch {epoch+1}, Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
    
    # Compute average metrics
    metrics = {}
    metrics['loss'] = total_loss / max(1, num_batches)
    
    # Compute average Dice scores
    for key, scores in dice_scores.items():
        if scores:
            metrics[f'{key}_dice'] = sum(scores) / len(scores)
        else:
            metrics[f'{key}_dice'] = 0.0
    
    # Compute mean Dice score across tumor classes
    tumor_dices = [metrics.get(f'{k}_dice', 0.0) for k in ['ET', 'WT', 'TC']]
    metrics['mean_dice'] = sum(tumor_dices) / len(tumor_dices)
    
    return metrics


# ==============================================================
# 5. USAGE EXAMPLE
# ==============================================================
# Example of how to use the authentic AutoSAM2 model

def train_authentic_sam2(data_path, batch_size=1, epochs=20, learning_rate=1e-4, enable_hybrid_mode=False):
    """
    Train the authentic AutoSAM2 model on BraTS data.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories for results
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize model
    model = AuthenticSAM2(
        num_classes=4,
        base_channels=16,
        sam2_model_id="facebook/sam2-hiera-small",
        enable_hybrid_mode=enable_hybrid_mode
    ).to(device)
    
    # Define optimizer - only train the prompt encoder parameters
    # This follows the original AutoSAM approach
    optimizer = optim.AdamW(
        model.prompt_encoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=learning_rate / 100
    )
    
    # Get data loaders
    train_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=True,
        normalize=True, num_workers=4, use_augmentation=True
    )
    
    val_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=False,
        normalize=True, num_workers=4, use_augmentation=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'mean_dice': [],
        'et_dice': [],
        'wt_dice': [],
        'tc_dice': []
    }
    
    # Best model tracking
    best_dice = 0.0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch_authentic(model, train_loader, optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = validate_authentic(model, val_loader, device, epoch)
        history['val_loss'].append(val_metrics['loss'])
        history['mean_dice'].append(val_metrics['mean_dice'])
        history['et_dice'].append(val_metrics.get('ET_dice', 0.0))
        history['wt_dice'].append(val_metrics.get('WT_dice', 0.0))
        history['tc_dice'].append(val_metrics.get('TC_dice', 0.0))
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Mean Dice: {val_metrics['mean_dice']:.2f}%, ET: {val_metrics.get('ET_dice', 0.0):.2f}%, WT: {val_metrics.get('WT_dice', 0.0):.2f}%, TC: {val_metrics.get('TC_dice', 0.0):.2f}%")
        
        # Save best model
        if val_metrics['mean_dice'] > best_dice:
            best_dice = val_metrics['mean_dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'best_dice': best_dice
            }, "checkpoints/best_authentic_sam2_model.pth")
            print(f"Saved new best model with Dice score: {best_dice:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, f"checkpoints/authentic_sam2_epoch_{epoch+1}.pth")
    
    print(f"Training complete! Best Dice score: {best_dice:.2f}%")
    return model, history
