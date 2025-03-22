# Standard library imports
import os
import time
import gc
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

# External libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bidirectional_autosam2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BidirectionalAutoSAM2")

# Import SAM2 with error handling
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2
    from sam2.build_sam import build_sam2_hf
    HAS_SAM2 = True
    logger.info("Successfully imported SAM2")
except ImportError:
    logger.error("ERROR: sam2 package not available.")
    HAS_SAM2 = False

# Import necessary components from model.py
from model import (
    get_strategic_slices,
    MRItoRGBMapper,
    EnhancedPromptGenerator,
    ResidualBlock3D,
    EncoderBlock3D,
    DecoderBlock3D,
    FlexibleUNet3D
)

# ==== Medical Image Prompt Encoder (follows original AutoSAM more closely) ====

class MedicalImagePromptEncoder(nn.Module):
    """
    Direct prompt encoder for medical images that follows the original AutoSAM approach.
    Takes UNet3D features and generates prompts for SAM2.
    """
    def __init__(self, in_channels=32, out_channels=256):
        super(MedicalImagePromptEncoder, self).__init__()
        
        # Channel attention module to emphasize important feature channels
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial Attention module to focus on relevant regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Transformer block for feature transformation
        self.transformer_path = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True)
        )
        
        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, 128, kernel_size=1)
        
        # Final projection to SAM2 expected dimension
        self.final_proj = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Quality prediction head (for monitoring only)
        self.quality_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Tracking variables
        self.last_quality = None
    
    def forward(self, x):
        """
        Forward pass to generate SAM2 prompts
        
        Args:
            x: UNet3D features [B, C, H, W]
            
        Returns:
            prompts: Enhanced feature map for SAM2 [B, 256, H, W]
            quality: Predicted quality score (0-1)
        """
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(x_channel)
        x_attended = x_channel * spatial_weights
        
        # Apply transformer path
        transformed = self.transformer_path(x_attended)
        
        # Add shortcut connection
        shortcut = self.shortcut(x_attended)
        features = transformed + shortcut
        
        # Final projection to SAM2 expected dimension
        prompts = self.final_proj(features)
        
        # Quality prediction (for monitoring)
        quality = self.quality_predictor(prompts)
        self.last_quality = quality
        
        return prompts, quality

# ==== Slice Processor for 3D Medical Data ====

class MedicalSliceProcessor(nn.Module):
    """Process 3D medical volumes by extracting and normalizing slices"""
    def __init__(self, normalize=True):
        super(MedicalSliceProcessor, self).__init__()
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
            slice_data = slice_data[0:1]  # Take only the first batch item
        
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
            
            # Only normalize if we have enough non-zero values
            if mask.sum() > 1:  # Need at least 2 values for std
                mean = torch.mean(channel[mask])
                std = torch.std(channel[mask], unbiased=False)  # Use biased std
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

# ==== Main BidirectionalAutoSAM2 Model ====

class BidirectionalAutoSAM2(nn.Module):
    """
    BidirectionalAutoSAM2 model with improved design that follows the original AutoSAM more closely.
    
    This model:
    1. Uses UNet3D encoder and mini-decoder to process 3D volumes
    2. Uses a dedicated prompt encoder to generate prompts for SAM2
    3. Creates a fully automatic segmentation pipeline for 3D medical volumes
    """
    def __init__(
        self,
        num_classes=4,
        base_channels=16,
        sam2_model_id="facebook/sam2-hiera-small",
        enable_auxiliary_head=True
    ):
        super(BidirectionalAutoSAM2, self).__init__()
        
        # Configuration
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.enable_auxiliary_head = enable_auxiliary_head
        
        # UNet3D encoder and mini-decoder (early stages only)
        self.unet3d = FlexibleUNet3D(
            in_channels=4,
            n_classes=num_classes,
            base_channels=base_channels,
            trilinear=True
        )
        
        # Prompt encoder - follows original AutoSAM approach more closely
        self.prompt_encoder = MedicalImagePromptEncoder(
            in_channels=base_channels * 2,  # Match UNet3D dec2 output
            out_channels=256  # Match SAM2 expected size
        )
        
        # Initialize MRI to RGB converter and prompt generator
        self.mri_to_rgb = MRItoRGBMapper()
        self.prompt_generator = EnhancedPromptGenerator(
            num_positive_points=5,
            num_negative_points=3,
            edge_detection=True,
            use_confidence=True,
            use_mask_prompt=True
        )
        
        # Initialize SAM2
        self.sam2 = None
        self.has_sam2 = False
        self.initialize_sam2(sam2_model_id)
        
        # Slice processor
        self.slice_processor = MedicalSliceProcessor(normalize=True)
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
        # Training parameters
        self.training_slice_percentage = 0.3
        self.eval_slice_percentage = 0.6
        self.current_epoch = 0
    
    def initialize_sam2(self, sam2_model_id):
        """Initialize SAM2"""
        if not HAS_SAM2:
            logger.warning("SAM2 package not available.")
            return
        
        try:
            logger.info(f"Building SAM2 with model_id: {sam2_model_id}")
            sam2_model = build_sam2_hf(sam2_model_id)
            self.sam2 = SAM2ImagePredictor(sam2_model)
            
            # Freeze SAM2 weights
            for param in self.sam2.model.parameters():
                param.requires_grad = False
            
            self.has_sam2 = True
            logger.info("SAM2 initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SAM2: {e}")
            self.has_sam2 = False
    
    def set_epoch(self, epoch):
        """Set current epoch for adaptive slice selection"""
        self.current_epoch = epoch
        
        # Adaptively increase slice percentage during training
        if epoch > 5:
            self.training_slice_percentage = min(0.5, 0.3 + 0.04 * (epoch - 5))
    
    def process_slice_with_sam2(self, input_slice, prompt_embeddings, slice_idx, device):
        """
        Process a single slice with SAM2 using prompts from our prompt encoder
        
        Args:
            input_slice: Input MRI slice [B, C, H, W]
            prompt_embeddings: Embeddings from prompt encoder [B, 256, H', W']
            slice_idx: Current slice index
            device: Device to run computation on
            
        Returns:
            mask_tensor: Generated segmentation mask [B, 1, H, W]
            metrics: Performance metrics dictionary
        """
        if not self.has_sam2:
            logger.warning(f"SAM2 not available for slice {slice_idx}")
            return None, {}
        
        metrics = {}
        
        try:
            batch_size, channels, height, width = input_slice.shape
            
            # Convert MRI to RGB for SAM2
            rgb_tensor = self.mri_to_rgb(input_slice)
            
            # Convert to numpy for SAM2 (from first batch item)
            rgb_image = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            
            # Generate point prompts from embeddings
            points, labels, box, _ = self.prompt_generator.generate_prompts(
                prompt_embeddings, slice_idx, height, width
            )
            
            # Set image in SAM2
            self.sam2.set_image(rgb_image)
            
            # Call SAM2 with the prompts
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
                
                # Convert to tensor
                mask_tensor = torch.from_numpy(best_mask).float().to(device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Update metrics
                metrics['sam2_score'] = scores[best_idx]
                
                return mask_tensor, metrics
            else:
                logger.warning(f"No masks generated for slice {slice_idx}")
                return None, metrics
        
        except Exception as e:
            logger.error(f"Error processing slice {slice_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, metrics
    
    def forward(self, x, targets=None):
        """
        Forward pass with improved design following original AutoSAM
        
        Args:
            x: Input 3D volume [B, C, D, H, W]
            targets: Optional target masks [B, C, D, H, W]
            
        Returns:
            output_volume: Segmented 3D volume [B, num_classes, D, H, W]
            aux_output: Optional auxiliary output
            losses: Dictionary with loss terms
        """
        # Track whether we're in training mode
        training_mode = self.training
        
        # Get device information
        device = x.device
        batch_size, channels, *spatial_dims = x.shape
        
        # Identify depth dimension
        if len(spatial_dims) == 3:
            depth_dim_idx = spatial_dims.index(min(spatial_dims)) + 2  # +2 for batch and channel
            depth = spatial_dims[depth_dim_idx - 2]
        else:
            depth_dim_idx = 2
            depth = x.shape[depth_dim_idx]
        
        # Get UNet3D features (run on full volume)
        _, unet_features, _, metadata = self.unet3d(x, use_full_decoder=False)
        
        # Select slices to process
        slice_percentage = self.eval_slice_percentage if not training_mode else self.training_slice_percentage
        key_indices = get_strategic_slices(depth, percentage=slice_percentage)
        key_indices.sort()
        
        # Process slices
        slice_results = {}
        slice_qualities = []
        
        for slice_idx in key_indices:
            try:
                # Get UNet features for this slice
                ds_idx = min(slice_idx // 4, unet_features.shape[2] - 1)
                if depth_dim_idx == 2:
                    slice_features = unet_features[:, :, ds_idx].clone()
                elif depth_dim_idx == 3:
                    slice_features = unet_features[:, :, :, ds_idx].clone()
                elif depth_dim_idx == 4:
                    slice_features = unet_features[:, :, :, :, ds_idx].clone()
                
                # Extract original slice at full resolution
                original_slice = self.slice_processor.extract_slice(x, slice_idx, depth_dim_idx)
                
                # Generate prompts with the prompt encoder (AutoSAM approach)
                prompt_embeddings, quality = self.prompt_encoder(slice_features)
                slice_qualities.append(quality.item())
                
                # Process with SAM2
                mask_tensor, metrics = self.process_slice_with_sam2(
                    original_slice, prompt_embeddings, slice_idx, device
                )
                
                if mask_tensor is not None:
                    # Convert binary mask to multi-class format
                    h, w = mask_tensor.shape[2:]
                    
                    # For multi-class segmentation, convert binary mask to class probabilities
                    if self.num_classes > 2:
                        multi_class_mask = torch.zeros((batch_size, self.num_classes, h, w), device=device)
                        
                        # Background (class 0) is inverse of tumor mask
                        multi_class_mask[:, 0] = 1.0 - mask_tensor[:, 0]
                        
                        # Distribute tumor probability across tumor classes using typical BraTS distribution
                        typical_dist = [0.0, 0.3, 0.4, 0.3]  # Background + 3 tumor classes
                        for c in range(1, self.num_classes):
                            multi_class_mask[:, c] = mask_tensor[:, 0] * typical_dist[c]
                        
                        # Ensure probabilities sum to 1.0
                        total_prob = multi_class_mask.sum(dim=1, keepdim=True)
                        multi_class_mask = multi_class_mask / total_prob.clamp(min=1e-5)
                        
                        # Store result
                        slice_results[slice_idx] = multi_class_mask
                    else:
                        # Binary segmentation
                        binary_mask = torch.zeros((batch_size, 2, h, w), device=device)
                        binary_mask[:, 0] = 1.0 - mask_tensor[:, 0]  # Background
                        binary_mask[:, 1] = mask_tensor[:, 0]  # Foreground
                        
                        # Store result
                        slice_results[slice_idx] = binary_mask
                
                # Update performance metrics
                for key, value in metrics.items():
                    self.performance_metrics[key].append(value)
            
            except Exception as e:
                logger.error(f"Error processing slice {slice_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Reconstruct 3D volume from processed slices
        output_volume = self.create_3d_from_slices(
            x.shape, slice_results, depth_dim_idx, device
        )
        
        # Compute auxiliary loss if we have quality predictions
        losses = {}
        if training_mode and slice_qualities:
            # Encourage high quality predictions
            avg_quality = sum(slice_qualities) / len(slice_qualities)
            quality_loss = torch.tensor(1.0 - avg_quality, device=device, requires_grad=True)
            losses['quality_loss'] = quality_loss
        
        # No auxiliary output for now
        aux_output = None
        
        return output_volume, aux_output, losses
    
    def create_3d_from_slices(self, input_shape, slice_results, depth_dim_idx, device):
        """
        Create a 3D volume from 2D slice results with interpolation between slices.
        """
        # Create empty volume matching the expected output size
        batch_size = input_shape[0]
        output_shape = list(input_shape)
        output_shape[1] = self.num_classes  # Change channel dimension
        
        volume = torch.zeros(output_shape, device=device)
        
        # Create mask of processed slices
        processed_slices_mask = torch.zeros(output_shape[depth_dim_idx], dtype=torch.bool, device=device)
        
        # Insert processed slices into volume
        for slice_idx, mask in slice_results.items():
            if mask is not None:
                if depth_dim_idx == 2:
                    volume[:, :, slice_idx] = mask
                elif depth_dim_idx == 3:
                    volume[:, :, :, slice_idx] = mask
                elif depth_dim_idx == 4:
                    volume[:, :, :, :, slice_idx] = mask
                
                processed_slices_mask[slice_idx] = True
        
        # Interpolate between processed slices
        if torch.sum(processed_slices_mask) > 1:
            # Get indices of processed slices
            slice_indices = torch.nonzero(processed_slices_mask).squeeze(-1)
            
            # Interpolate between adjacent processed slices
            for i in range(len(slice_indices) - 1):
                start_idx = slice_indices[i].item()
                end_idx = slice_indices[i + 1].item()
                
                # If gap exists, interpolate
                if end_idx - start_idx > 1:
                    # Get slices at endpoints
                    if depth_dim_idx == 2:
                        start_slice = volume[:, :, start_idx].clone()
                        end_slice = volume[:, :, end_idx].clone()
                    elif depth_dim_idx == 3:
                        start_slice = volume[:, :, :, start_idx].clone()
                        end_slice = volume[:, :, :, end_idx].clone()
                    elif depth_dim_idx == 4:
                        start_slice = volume[:, :, :, :, start_idx].clone()
                        end_slice = volume[:, :, :, :, end_idx].clone()
                    
                    # Linear interpolation for each position in the gap
                    for j in range(1, end_idx - start_idx):
                        alpha = j / (end_idx - start_idx)
                        interp_slice = (1 - alpha) * start_slice + alpha * end_slice
                        
                        # Normalize probability distribution
                        total_prob = interp_slice.sum(dim=1, keepdim=True)
                        interp_slice = interp_slice / total_prob.clamp(min=1e-5)
                        
                        # Insert interpolated slice
                        if depth_dim_idx == 2:
                            volume[:, :, start_idx + j] = interp_slice
                        elif depth_dim_idx == 3:
                            volume[:, :, :, start_idx + j] = interp_slice
                        elif depth_dim_idx == 4:
                            volume[:, :, :, :, start_idx + j] = interp_slice
        
        return volume
    
    def get_performance_stats(self):
        """Return performance statistics"""
        stats = {
            "has_sam2": self.has_sam2,
            "training_slice_percentage": self.training_slice_percentage,
            "eval_slice_percentage": self.eval_slice_percentage,
            "current_epoch": self.current_epoch
        }
        
        # Add statistics from metrics
        for key, values in self.performance_metrics.items():
            if values:
                stats[f"avg_{key}"] = sum(values) / len(values)
                stats[f"max_{key}"] = max(values)
                stats[f"min_{key}"] = min(values)
        
        return stats

# ==== Adapter for train.py Compatibility ====

class BidirectionalAutoSAM2Adapter(BidirectionalAutoSAM2):
    """
    Adapter class that makes BidirectionalAutoSAM2 compatible with the existing train.py.
    
    This adapter:
    1. Provides properties expected by train.py (encoder, unet3d)
    2. Implements set_mode to match the behavior expected by train.py
    """
    def __init__(self, num_classes=4, base_channels=16, sam2_model_id="facebook/sam2-hiera-small", enable_auxiliary_head=True):
        super().__init__(num_classes, base_channels, sam2_model_id, enable_auxiliary_head)
        
        # Properties that train.py expects
        self.encoder = self.unet3d  # train.py expects access to encoder
        
        # Additional compatibility properties
        self.enable_unet_decoder = True  # Variable checked in train.py
        self.enable_sam2 = self.has_sam2
        self.has_sam2_enabled = self.has_sam2
        
        # Debug information
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"BidirectionalAutoSAM2: Total trainable parameters: {trainable_count}")
    
    def forward(self, x, targets=None):
        """
        Adapted forward pass for train.py compatibility
        
        Args:
            x: Input 3D volume [B, C, D, H, W]
            targets: Optional target masks [B, C, D, H, W]
            
        Returns:
            outputs: Segmentation volume
        """
        # Ensure prompt encoder parameters require gradients
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        
        # Call parent's forward method
        output_volume, aux_output, losses = super().forward(x, targets)
        
        # Debug once
        if not hasattr(self, '_debug_printed'):
            print(f"Forward pass - Output shape: {output_volume.shape}")
            self._debug_printed = True
        
        return output_volume
    
    def set_mode(self, enable_unet_decoder=None, enable_sam2=None, sam2_percentage=None, bg_blend=None, tumor_blend=None):
        """
        Function to change model mode - required by train.py
        """
        if enable_unet_decoder is not None:
            self.enable_unet_decoder = enable_unet_decoder
        
        if enable_sam2 is not None:
            self.enable_sam2 = enable_sam2
            self.has_sam2_enabled = enable_sam2
            # Actually disable SAM2 if requested
            if not enable_sam2:
                self.has_sam2 = False
        
        # Set slice percentage for processing
        if sam2_percentage is not None:
            self.training_slice_percentage = min(sam2_percentage, 0.5)  # Limit to half during training
            self.eval_slice_percentage = sam2_percentage
        
        # These parameters aren't used in this implementation but we accept them for compatibility
        if bg_blend is not None:
            self.bg_blend = bg_blend
            
        if tumor_blend is not None:
            self.tumor_blend = tumor_blend
            
        print(f"Model mode: UNet={self.enable_unet_decoder}, SAM2={self.enable_sam2}, Slices={self.eval_slice_percentage}")
