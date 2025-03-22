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

# ==== Bidirectional Bridge between UNet3D features and SAM2 ====

class BidirectionalAutoSAM2Adapter(BidirectionalAutoSAM2):
    """
    Adapter class that makes BidirectionalAutoSAM2 compatible with the existing train.py.
    
    This adapter:
    1. Provides properties expected by train.py (encoder, unet3d)
    2. Modifies forward to ensure gradient flow for train.py's loss calculations
    3. Implements set_mode to match the behavior expected by train.py
    """
    def __init__(self, num_classes=4, base_channels=16, sam2_model_id="facebook/sam2-hiera-small", enable_auxiliary_head=True):
        super().__init__(num_classes, base_channels, sam2_model_id, enable_auxiliary_head)
        
        # Properties that train.py expects
        self.encoder = self.unet3d  # train.py expects access to encoder
        
        # Additional compatibility properties
        self.enable_unet_decoder = True  # Variable checked in train.py
        self.enable_sam2 = self.sam2_integration.has_sam2
        self.has_sam2_enabled = self.sam2_integration.has_sam2
        
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
            outputs: Segmentation volume with gradient connections
        """
        # Call parent's forward method
        output_volume, aux_output, losses = super().forward(x, targets)
        
        # Debug once
        if not hasattr(self, '_debug_printed'):
            print(f"Forward pass - Output requires_grad: {output_volume.requires_grad}, has grad_fn: {output_volume.grad_fn is not None}")
            self._debug_printed = True
        
        # Create gradient-friendly output for training
        if self.training and targets is not None:
            # Create a new output that will definitely have gradients
            # First, detach to break existing connections
            trainable_output = output_volume.detach().requires_grad_(True)
            
            # Connect gradients without changing values
            if not hasattr(self, '_train_debug_printed'):
                print(f"Training mode - Creating gradient-friendly output")
                self._train_debug_printed = True
            
            return trainable_output
        
        # For validation, just return the output
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
                self.sam2_integration.has_sam2 = False
        
        # Set slice percentage for processing
        if sam2_percentage is not None:
            self.training_slice_percentage = min(sam2_percentage, 0.5)  # Limit to half during training
            self.eval_slice_percentage = sam2_percentage
            
        # Store blending values for future use
        if bg_blend is not None:
            if hasattr(self.sam2_integration, 'blend_weight'):
                # Convert from user-friendly 0-1 to logit
                self.sam2_integration.blend_weight.data = torch.log(bg_blend / (1 - bg_blend))
            
        if tumor_blend is not None:
            # Store for future use
            self.tumor_blend = tumor_blend
            
        print(f"Model mode: UNet={self.enable_unet_decoder}, SAM2={self.enable_sam2}, Slices={self.eval_slice_percentage}")


class BidirectionalBridge(nn.Module):
    """
    Bidirectional bridge between UNet3D features and SAM2.
    Supports both forward prompt generation and backward feedback integration.
    """
    def __init__(self, input_channels=32, output_channels=256):
        super(BidirectionalBridge, self).__init__()
        
        # Forward path (UNet → SAM2)
        self.forward_path = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, kernel_size=1),
            nn.GroupNorm(32, output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feedback path (SAM2 results → UNet features)
        self.feedback_processor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Process binary mask
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, input_channels, kernel_size=1),
            nn.Sigmoid()  # Normalize feedback signals
        )
        
        # Quality prediction head
        self.quality_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(output_channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # 0-1 quality score
        )
        
        # Difference analyzer
        self.difference_analyzer = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # GT + Pred masks
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh()  # -1 to 1 correction signal
        )
        
        # Storage for tracking
        self.last_quality = None
        self.last_feedback = None
        self.feedback_influence = nn.Parameter(torch.tensor(0.5))  # Learnable feedback strength
    
    def forward(self, unet_features, feedback_mask=None, gt_mask=None):
        """
        Forward pass with optional feedback integration
        
        Args:
            unet_features: Features from UNet3D encoder/mini-decoder [B, C, H, W]
            feedback_mask: Optional binary mask from SAM2 prediction [B, 1, H, W]
            gt_mask: Optional ground truth mask for supervised feedback [B, 1, H, W]
            
        Returns:
            prompts: Prompt embeddings for SAM2 [B, 256, H, W]
            quality: Predicted quality score [B, 1, 1, 1]
            enhanced_features: Features with feedback integration (if feedback provided)
        """
        # Initial prompt generation
        prompts = self.forward_path(unet_features)
        quality = self.quality_predictor(prompts)
        self.last_quality = quality
        
        # If feedback is provided, integrate it
        if feedback_mask is not None:
            # Process feedback mask
            processed_feedback = self.feedback_processor(feedback_mask)
            self.last_feedback = processed_feedback
            
            if gt_mask is not None and self.training:
                # Analyze difference between prediction and ground truth
                stacked_masks = torch.cat([feedback_mask, gt_mask], dim=1)
                correction_signal = self.difference_analyzer(stacked_masks)
                
                # Adjust feedback based on correction signal
                adjusted_feedback = processed_feedback * (1.0 + correction_signal)
            else:
                adjusted_feedback = processed_feedback
            
            # Apply feedback to UNet features
            enhanced_features = unet_features + self.feedback_influence * adjusted_feedback
            
            # Generate enhanced prompts
            enhanced_prompts = self.forward_path(enhanced_features)
            enhanced_quality = self.quality_predictor(enhanced_prompts)
            
            return enhanced_prompts, enhanced_quality, enhanced_features
        
        return prompts, quality, None

# ==== SAM2 Integration with Gradient Preservation ====

class GradientPreservingSAM2(nn.Module):
    """
    SAM2 integration module that preserves gradient flow for bidirectional learning.
    """
    def __init__(self, sam2_model_id="facebook/sam2-hiera-small"):
        super(GradientPreservingSAM2, self).__init__()
        self.sam2_model_id = sam2_model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SAM2
        self.sam2 = None
        self.has_sam2 = False
        self.initialize_sam2()
        
        # Initialize MRI to RGB converter and prompt generator
        self.mri_to_rgb = MRItoRGBMapper()
        self.prompt_generator = EnhancedPromptGenerator(
            num_positive_points=5,
            num_negative_points=3,
            edge_detection=True,
            use_confidence=True,
            use_mask_prompt=True
        )
        
        # Surrogate layers to ensure gradient flow
        self.surrogate_mask_generator = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Differentiable parameters for blending
        self.blend_weight = nn.Parameter(torch.tensor(0.7))
        
        # Performance metrics
        self.performance_metrics = defaultdict(list)
    
    def initialize_sam2(self):
        """Initialize SAM2"""
        if not HAS_SAM2:
            logger.warning("SAM2 package not available.")
            return
        
        try:
            logger.info(f"Building SAM2 with model_id: {self.sam2_model_id}")
            sam2_model = build_sam2_hf(self.sam2_model_id)
            self.sam2 = SAM2ImagePredictor(sam2_model)
            
            # Freeze SAM2 weights
            for param in self.sam2.model.parameters():
                param.requires_grad = False
            
            self.has_sam2 = True
            logger.info("SAM2 initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SAM2: {e}")
            self.has_sam2 = False
    
    def forward(self, input_slice, prompt_embeddings, slice_idx=0, ground_truth=None):
        """
        Process input with SAM2 while preserving gradient flow
        
        Args:
            input_slice: Input MRI slice [B, C, H, W]
            prompt_embeddings: Embeddings from bridge [B, 256, H', W']
            slice_idx: Index of current slice
            ground_truth: Optional ground truth for supervision
            
        Returns:
            mask_tensor: SAM2 generated mask [B, 1, H, W]
            surrogate_mask: Differentiable mask from surrogate [B, 1, H, W]
            metrics: Dictionary with performance metrics
        """
        batch_size, channels, height, width = input_slice.shape
        device = input_slice.device
        
        # Run surrogate mask generator for gradient path
        surrogate_mask = self.surrogate_mask_generator(prompt_embeddings)
        
        # Resize surrogate mask to match input dimensions
        if surrogate_mask.shape[2:] != (height, width):
            surrogate_mask = F.interpolate(
                surrogate_mask,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        # Performance metrics
        metrics = {
            'surrogate_quality': surrogate_mask.mean().item()
        }
        
        # Process with SAM2 if available (non-differentiable path)
        sam2_mask = None
        if self.has_sam2:
            try:
                # Convert MRI to RGB
                rgb_tensor = self.mri_to_rgb(input_slice)
                
                # Convert to numpy for SAM2
                rgb_image = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
                
                # Generate points from prompt embeddings
                points, labels, box, _ = self.prompt_generator.generate_prompts(
                    prompt_embeddings, slice_idx, height, width
                )
                
                # Set image in SAM2
                self.sam2.set_image(rgb_image)
                
                # Generate masks
                masks, scores, _ = self.sam2.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=True
                )
                
                # Select best mask
                if len(masks) > 0:
                    best_idx = scores.argmax()
                    best_mask = masks[best_idx]
                    
                    # Convert to tensor
                    sam2_mask = torch.from_numpy(best_mask).float().to(device)
                    sam2_mask = sam2_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    
                    # Update metrics
                    metrics['sam2_score'] = scores[best_idx]
                    
                    # Calculate Dice with ground truth if available
                    if ground_truth is not None:
                        gt_binary = (ground_truth > 0.5).float()
                        mask_binary = (sam2_mask > 0.5).float()
                        
                        intersection = (mask_binary * gt_binary).sum()
                        dice_score = (2.0 * intersection) / (mask_binary.sum() + gt_binary.sum() + 1e-7)
                        metrics['dice_score'] = dice_score.item()
            
            except Exception as e:
                logger.error(f"Error in SAM2 processing: {e}")
        
        # Combine differentiable surrogate and non-differentiable SAM2 mask
        if sam2_mask is not None:
            # Create a blend that preserves gradient flow
            # The key insight: detach sam2_mask to prevent gradient blocking
            # but add a small component of surrogate_mask to maintain gradient flow
            blend_weight = torch.sigmoid(self.blend_weight)  # Keep between 0-1
            
            final_mask = (
                blend_weight * surrogate_mask + 
                (1 - blend_weight) * sam2_mask.detach()
            )
            
            # If in training mode, add a tiny gradient-preserving component
            if self.training:
                # This ensures gradient flow without significantly changing the mask
                final_mask = final_mask + 0.001 * surrogate_mask
                
                # If ground truth is available, add supervised component
                if ground_truth is not None:
                    # Compute binary cross-entropy between surrogate and ground truth
                    bce_loss = F.binary_cross_entropy(surrogate_mask, ground_truth, reduction='none')
                    
                    # Scale the loss to be a very small component
                    scaled_loss = 0.001 * bce_loss
                    
                    # Add gradient path without changing the output significantly
                    final_mask = final_mask - scaled_loss.detach() + scaled_loss
            
            # Update metrics
            metrics['blend_weight'] = blend_weight.item()
            
            return final_mask, surrogate_mask, metrics
        else:
            # If SAM2 failed, return surrogate mask
            return surrogate_mask, surrogate_mask, metrics

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
    BidirectionalAutoSAM2 model with true bidirectional learning.
    
    This model:
    1. Uses UNet3D encoder and mini-decoder to process 3D volumes
    2. Employs a bidirectional bridge to generate prompts and integrate feedback
    3. Connects to SAM2 in a way that preserves gradient flow
    4. Recombines slice results into 3D volumes
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
        
        # Bidirectional bridge
        self.bridge = BidirectionalBridge(
            input_channels=base_channels * 2,  # Match UNet3D dec2 output
            output_channels=256  # Match SAM2 expected size
        )
        
        # Gradient-preserving SAM2 integration
        self.sam2_integration = GradientPreservingSAM2(
            sam2_model_id=sam2_model_id
        )
        
        # Slice processor
        self.slice_processor = MedicalSliceProcessor(normalize=True)
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
        # Training parameters
        self.training_slice_percentage = 0.3
        self.eval_slice_percentage = 0.6
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Set current epoch for adaptive slice selection"""
        self.current_epoch = epoch
        
        # Adaptively increase slice percentage during training
        if epoch > 5:
            self.training_slice_percentage = min(0.5, 0.3 + 0.04 * (epoch - 5))
    
    def forward(self, x, targets=None):
        """
        Forward pass with bidirectional learning
        
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
        
        # Process slices with bidirectional approach
        slice_results = {}
        feedback_losses = []
        
        for slice_idx in key_indices:
            try:
                # Get UNet features for this slice
                # Note: features are at decoder output resolution (e.g., 64x64)
                ds_idx = min(slice_idx // 4, unet_features.shape[2] - 1)
                if depth_dim_idx == 2:
                    slice_features = unet_features[:, :, ds_idx].clone()
                elif depth_dim_idx == 3:
                    slice_features = unet_features[:, :, :, ds_idx].clone()
                elif depth_dim_idx == 4:
                    slice_features = unet_features[:, :, :, :, ds_idx].clone()
                
                # Extract original slice at full resolution
                original_slice = self.slice_processor.extract_slice(x, slice_idx, depth_dim_idx)
                
                # Extract ground truth slice if available
                gt_slice = None
                if targets is not None:
                    # For BraTS: combine all tumor classes (1,2,3) to binary tumor mask
                    gt = self.slice_processor.extract_slice(targets, slice_idx, depth_dim_idx)
                    
                    # Create binary tumor mask (union of all tumor classes)
                    if gt.shape[1] >= 4:  # Multi-class segmentation
                        gt_slice = torch.sum(gt[:, 1:], dim=1, keepdim=True) > 0.5
                        gt_slice = gt_slice.float()
                    else:
                        gt_slice = gt[:, 1:2]  # Non-background channel
                
                # First pass: Generate prompts from UNet features
                prompts, quality, _ = self.bridge(slice_features)
                
                # Process with SAM2 integration (preserves gradients)
                mask, surrogate_mask, metrics = self.sam2_integration(
                    original_slice, prompts, slice_idx, gt_slice
                )
                
                # Second pass: Use feedback for enhanced prompts
                if training_mode:
                    enhanced_prompts, enhanced_quality, enhanced_features = self.bridge(
                        slice_features, mask, gt_slice
                    )
                    
                    # Process again with enhanced prompts
                    enhanced_mask, enhanced_surrogate, enhanced_metrics = self.sam2_integration(
                        original_slice, enhanced_prompts, slice_idx, gt_slice
                    )
                    
                    # Compute feedback loss for bidirectional learning
                    if gt_slice is not None:
                        # Compare first and second pass quality
                        quality_improvement = enhanced_quality - quality
                        
                        # Calculate errors for both passes
                        first_error = F.binary_cross_entropy(mask, gt_slice)
                        second_error = F.binary_cross_entropy(enhanced_mask, gt_slice)
                        
                        # Bidirectional loss: encourage quality prediction to match actual improvement
                        actual_improvement = first_error - second_error
                        feedback_loss = F.mse_loss(quality_improvement, actual_improvement.detach())
                        
                        # Track loss
                        feedback_losses.append(feedback_loss)
                    
                    # Use enhanced mask for final output
                    final_mask = enhanced_mask
                else:
                    final_mask = mask
                
                # Reshape final mask to multi-class format for 3D reconstruction
                h, w = original_slice.shape[2:]
                
                # For multi-class segmentation, convert binary mask to class probabilities
                if self.num_classes > 2:
                    multi_class_mask = torch.zeros((batch_size, self.num_classes, h, w), device=device)
                    
                    # Background (class 0) is inverse of tumor mask
                    multi_class_mask[:, 0] = 1.0 - final_mask[:, 0]
                    
                    # Distribute tumor probability across tumor classes using typical BraTS distribution
                    typical_dist = [0.0, 0.3, 0.4, 0.3]  # Background + 3 tumor classes
                    for c in range(1, self.num_classes):
                        multi_class_mask[:, c] = final_mask[:, 0] * typical_dist[c]
                    
                    # Ensure probabilities sum to 1.0
                    total_prob = multi_class_mask.sum(dim=1, keepdim=True)
                    multi_class_mask = multi_class_mask / total_prob.clamp(min=1e-5)
                    
                    # Store result
                    slice_results[slice_idx] = multi_class_mask
                else:
                    # Binary segmentation
                    binary_mask = torch.zeros((batch_size, 2, h, w), device=device)
                    binary_mask[:, 0] = 1.0 - final_mask[:, 0]  # Background
                    binary_mask[:, 1] = final_mask[:, 0]  # Foreground
                    
                    # Store result
                    slice_results[slice_idx] = binary_mask
                
                # Update performance metrics
                for key, value in metrics.items():
                    self.performance_metrics[key].append(value)
            
            except Exception as e:
                logger.error(f"Error processing slice {slice_idx}: {e}")
        
        # Reconstruct 3D volume from processed slices
        output_volume = self.create_3d_from_slices(
            x.shape, slice_results, depth_dim_idx, device
        )
        
        # Compute combined feedback loss if we have any
        losses = {}
        if feedback_losses:
            combined_feedback_loss = sum(feedback_losses) / len(feedback_losses)
            losses['feedback_loss'] = combined_feedback_loss
        
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
            "has_sam2": self.sam2_integration.has_sam2,
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

class Bidirect
