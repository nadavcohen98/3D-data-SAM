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

# Import necessary components from your existing code
from model import get_strategic_slices, MRItoRGBMapper, EnhancedPromptGenerator
from visualization import visualize_batch_comprehensive

# ======= Medical Image Prompt Encoder (following original AutoSAM paper) =======

class MedicalPromptEncoder(nn.Module):
    """
    Medical image prompt encoder that follows the original AutoSAM approach.
    Takes a 2D slice from a medical 3D volume and generates prompts for SAM2.
    
    This encoder is specifically designed to be trained with gradients from
    SAM2's mask decoder, following the original AutoSAM paper methodology.
    """
    def __init__(self, in_channels=4, prompt_embed_dim=256, base_channels=32):
        super(MedicalPromptEncoder, self).__init__()
        
        # Downsampling path (encoder)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.GroupNorm(16, base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.GroupNorm(16, base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels*8, base_channels*16, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*16, base_channels*16, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels*16),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling path (decoder)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*16, base_channels*8, kernel_size=3, padding=1),
            nn.GroupNorm(16, base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.GroupNorm(16, base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Final projection to prompt embedding
        self.to_prompt_embed = nn.Sequential(
            nn.Conv2d(base_channels*4, prompt_embed_dim, kernel_size=1),
            nn.GroupNorm(32, prompt_embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Lightweight auxiliary segmentation head for additional supervision
        self.aux_seg_head = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, 1, kernel_size=1)
        )
        
        # Quality prediction head - predicts how good the prompts will be for SAM2
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(prompt_embed_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the medical prompt encoder
        
        Args:
            x: Input medical image slice [B, C, H, W]
            
        Returns:
            prompt_embeddings: Prompt embeddings for SAM2
            aux_seg: Auxiliary segmentation output
            quality_score: Predicted quality of the prompts
        """
        # Encoder path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder path
        x = self.up1(x5)
        x = self.up2(x)
        
        # Generate outputs
        prompt_embeddings = self.to_prompt_embed(x)
        aux_seg = self.aux_seg_head(x)
        quality_score = self.quality_head(prompt_embeddings)
        
        return prompt_embeddings, aux_seg, quality_score

# ======= Slice Processor for 3D Medical Volumes =======

class MedicalSliceProcessor(nn.Module):
    """
    Process 3D medical volumes by extracting strategic slices and preparing
    them for the prompt encoder.
    """
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
            
            if mask.sum() > 0:
                mean = torch.mean(channel[mask])
                std = torch.std(channel[mask])
                # Apply normalization only to non-zero values
                normalized_slice[:, c][mask] = (channel[mask] - mean) / (std + 1e-8)
        
        return normalized_slice
    
    def prepare_slice(self, volume, slice_idx, depth_dim=2, target_size=(224, 224)):
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

# ======= SAM2 Integration Module =======

class SAM2Integration(nn.Module):
    """
    Module for integrating SAM2 into the BidirectionalAutoSAM2 pipeline.
    Handles the interface between medical prompt encoder and SAM2, including
    prompt conversion, mask prediction, and feedback generation.
    """
    def __init__(self, sam2_model_id="facebook/sam2-hiera-small", device="cuda"):
        super(SAM2Integration, self).__init__()
        self.sam2_model_id = sam2_model_id
        self.device = device
        
        # Initialize SAM2 predictor
        self.sam2 = None
        self.has_sam2 = False
        self.initialize_sam2()
        
        # Initialize MRI to RGB converter for SAM2 input
        self.mri_to_rgb = MRItoRGBMapper()
        
        # Initialize prompt generator helper
        self.prompt_generator = EnhancedPromptGenerator(
            num_positive_points=5,
            num_negative_points=3,
            edge_detection=True,
            use_confidence=True,
            use_mask_prompt=True
        )
        
        # Feedback metrics
        self.feedback_history = defaultdict(list)
    
    def initialize_sam2(self):
        """Initialize SAM2 with appropriate error handling"""
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
    
    def process_with_sam2(self, 
                          input_slice, 
                          prompt_embeddings, 
                          quality_score,
                          ground_truth=None,
                          slice_idx=None):
        """
        Process a single slice with SAM2 using the generated prompt embeddings
        
        Args:
            input_slice: Original input medical image slice [1, C, H, W]
            prompt_embeddings: Prompt embeddings from encoder [1, 256, H/4, W/4]
            quality_score: Predicted quality score [1, 1, 1, 1]
            ground_truth: Optional ground truth mask for feedback calculation
            slice_idx: Optional slice index for logging
            
        Returns:
            mask_tensor: Predicted mask [1, 1, H, W]
            feedback: Dictionary with feedback metrics for bidirectional learning
        """
        if not self.has_sam2:
            logger.warning(f"SAM2 not available for slice {slice_idx}")
            return None, None
        
        try:
            # Convert MRI to RGB for SAM2 input image
            rgb_tensor = self.mri_to_rgb(input_slice)
            
            # Get dimensions
            h, w = input_slice.shape[2:]
            
            # Convert to numpy array in the format SAM2 expects: [H, W, 3]
            rgb_image = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            
            # Generate point prompts from prompt embeddings
            points, labels, box, _ = self.prompt_generator.generate_prompts(
                prompt_embeddings, slice_idx or 0, h, w
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
            
            # Initialize feedback dictionary
            feedback = {
                'prompt_quality': quality_score.item(),
                'sam2_score': 0.0,
                'dice_score': 0.0,
                'success': False
            }
            
            # Select best mask based on score
            if len(masks) > 0:
                best_idx = scores.argmax()
                best_mask = masks[best_idx]
                best_score = scores[best_idx]
                
                # Convert to tensor and format
                mask_tensor = torch.from_numpy(best_mask).float().to(self.device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Update feedback with SAM2 score
                feedback['sam2_score'] = best_score
                feedback['success'] = True
                
                # Calculate Dice score if ground truth is available
                if ground_truth is not None:
                    gt_binary = (ground_truth > 0.5).float()
                    pred_binary = (mask_tensor > 0.5).float()
                    
                    # Calculate Dice score
                    intersection = (pred_binary * gt_binary).sum()
                    dice_score = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-7)
                    
                    feedback['dice_score'] = dice_score.item()
                
                # Store feedback in history
                for key, value in feedback.items():
                    self.feedback_history[key].append(value)
                
                return mask_tensor, feedback
            else:
                # No valid masks found
                logger.warning(f"SAM2 failed to generate masks for slice {slice_idx}")
                return None, feedback
                    
        except Exception as e:
            logger.error(f"Error processing slice {slice_idx} with SAM2: {e}")
            return None, {'prompt_quality': quality_score.item(), 'error': str(e), 'success': False}

# ======= Bidirectional Feedback System =======

class BidirectionalFeedback(nn.Module):
    """
    Bidirectional feedback system that converts SAM2 feedback into gradient-friendly
    supervision signals for the medical prompt encoder.
    """
    def __init__(self):
        super(BidirectionalFeedback, self).__init__()
        
        # Prediction loss scalers
        self.quality_to_actual_ratio = nn.Parameter(torch.tensor(1.0))
        self.dice_weight = nn.Parameter(torch.tensor(0.7))
        self.sam2_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, 
               predicted_quality, 
               actual_feedback, 
               prompt_embeddings):
        """
        Generate bidirectional feedback loss
        
        Args:
            predicted_quality: Predicted quality from prompt encoder [B, 1, 1, 1]
            actual_feedback: Dictionary with actual SAM2 feedback metrics
            prompt_embeddings: The generated prompt embeddings
            
        Returns:
            feedback_loss: Loss term that provides feedback
            feedback_metrics: Dictionary with feedback metrics
        """
        device = predicted_quality.device
        batch_size = predicted_quality.shape[0]
        
        # Extract actual quality metrics
        if actual_feedback and actual_feedback.get('success', False):
            # Combine dice_score and sam2_score for the "actual quality"
            dice_score = actual_feedback.get('dice_score', 0.0)
            sam2_score = actual_feedback.get('sam2_score', 0.0)
            
            actual_quality = self.dice_weight * dice_score + self.sam2_weight * sam2_score
            quality_target = torch.tensor([actual_quality], device=device).view_as(predicted_quality)
            
            # Compute quality prediction loss
            quality_prediction_loss = F.mse_loss(predicted_quality, quality_target)
            
            # For successful cases, add a loss that grows quadratically when quality is low
            success_loss = torch.exp(-5 * actual_quality) * (1.0 - predicted_quality)
            
            # Combine losses
            feedback_loss = quality_prediction_loss + 0.2 * success_loss
            
            # Update metrics
            feedback_metrics = {
                'predicted_quality': predicted_quality.item(),
                'actual_quality': actual_quality,
                'quality_prediction_loss': quality_prediction_loss.item(),
                'success_loss': success_loss.item(),
                'feedback_loss': feedback_loss.item()
            }
        else:
            # For failed cases, push quality prediction down
            target_quality = torch.tensor([0.1], device=device).view_as(predicted_quality)
            feedback_loss = 0.5 * F.mse_loss(predicted_quality, target_quality)
            
            feedback_metrics = {
                'predicted_quality': predicted_quality.item(),
                'actual_quality': 0.0,
                'feedback_loss': feedback_loss.item()
            }
        
        return feedback_loss, feedback_metrics

# ======= BidirectionalAutoSAM2 Main Model =======

class BidirectionalAutoSAM2(nn.Module):
    """
    BidirectionalAutoSAM2 model that follows the original AutoSAM approach 
    for 3D medical volumes, with bidirectional learning.
    
    This model:
    1. Takes a 3D medical volume as input
    2. Extracts 2D slices at strategic positions
    3. Processes each slice with a medical prompt encoder
    4. Uses the generated prompts with SAM2 for segmentation
    5. Creates a bidirectional feedback loop for learning optimal prompts
    6. Reconstructs the 3D volume from 2D slice segmentations
    """
    def __init__(
        self, 
        num_classes=4,
        base_channels=32,
        sam2_model_id="facebook/sam2-hiera-small",
        enable_auxiliary_head=True
    ):
        super(BidirectionalAutoSAM2, self).__init__()
        
        # Configuration
        self.num_classes = num_classes
        self.enable_auxiliary_head = enable_auxiliary_head
        
        # Initialize medical prompt encoder
        self.prompt_encoder = MedicalPromptEncoder(
            in_channels=4,  # 4 MRI modalities
            prompt_embed_dim=256,  # SAM2 expected embedding size
            base_channels=base_channels
        )
        
        # Initialize slice processor for handling 3D to 2D conversion
        self.slice_processor = MedicalSliceProcessor(normalize=True)
        
        # Initialize SAM2 integration module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_integration = SAM2Integration(
            sam2_model_id=sam2_model_id,
            device=self.device
        )
        
        # Initialize bidirectional feedback system
        self.feedback_system = BidirectionalFeedback()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.slice_predictions = {}
        
        # Set training parameters
        self.training_slice_percentage = 0.3  # Start with 30% of slices during training
        self.eval_slice_percentage = 0.6  # Use more slices during evaluation
        self.current_epoch = 0
        
        # UNet3D for comparative/hybrid mode if needed
        self.auxiliary_3d_head = None
        if enable_auxiliary_head:
            self.auxiliary_3d_head = AuxiliarySegHead(
                num_classes=num_classes,
                base_channels=base_channels
            )
    
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
            targets: Optional target segmentation masks for training [B, C, D, H, W]
            
        Returns:
            output_volume: Segmented 3D volume [B, num_classes, D, H, W]
            aux_output: Auxiliary segmentation if enabled
            losses: Dictionary with loss terms during training
        """
        # Move to device if needed
        device = x.device
        batch_size, channels, *spatial_dims = x.shape
        
        # Identify which dimension is the depth dimension (usually the smallest)
        depth_dim_idx = None
        if len(spatial_dims) == 3:  # We have a 3D volume
            min_dim = min(spatial_dims)
            depth_dim_idx = spatial_dims.index(min_dim) + 2  # +2 for batch and channel dims
            depth = spatial_dims[depth_dim_idx - 2]
        else:
            # Default to standard [B, C, D, H, W] format
            depth_dim_idx = 2
            depth = x.shape[depth_dim_idx]
        
        # Select strategic slices for processing
        slice_percentage = self.eval_slice_percentage if not self.training else self.training_slice_percentage
        key_indices = get_strategic_slices(depth, percentage=slice_percentage)
        key_indices.sort()
        
        # Process each slice with prompt encoder and SAM2
        sam2_results = {}
        aux_seg_slices = {}
        overall_feedback_loss = 0.0
        feedback_metrics = defaultdict(list)
        
        for slice_idx in key_indices:
            try:
                # Prepare slice for prompt encoder
                input_slice = self.slice_processor.prepare_slice(
                    x, slice_idx, depth_dim_idx, target_size=(224, 224)
                )
                
                # Extract ground truth slice for training if available
                gt_slice = None
                if targets is not None:
                    # For BraTS segmentation, combine all tumor classes (1, 2, 3) to get binary tumor mask
                    gt = self.slice_processor.extract_slice(targets, slice_idx, depth_dim_idx)
                    if gt.shape[1] >= 4:  # Multi-class segmentation
                        # Get binary tumor mask (union of all tumor classes)
                        gt_slice = torch.sum(gt[:, 1:], dim=1, keepdim=True) > 0.5
                        gt_slice = gt_slice.float()
                    else:
                        gt_slice = gt[:, 1:2]  # Non-background channel
                
                # Extract features and generate prompts with the medical prompt encoder
                prompt_embeddings, aux_seg, quality_score = self.prompt_encoder(input_slice)
                
                # Store auxiliary segmentation result
                aux_seg_slices[slice_idx] = aux_seg
                
                # Process with SAM2 integration module
                mask_tensor, feedback = self.sam2_integration.process_with_sam2(
                    input_slice, 
                    prompt_embeddings,
                    quality_score,
                    ground_truth=gt_slice,
                    slice_idx=slice_idx
                )
                
                # Calculate bidirectional feedback loss during training
                if self.training and feedback is not None:
                    feedback_loss, fb_metrics = self.feedback_system(
                        quality_score,
                        feedback,
                        prompt_embeddings
                    )
                    
                    # Accumulate feedback loss
                    overall_feedback_loss += feedback_loss
                    
                    # Store feedback metrics
                    for key, value in fb_metrics.items():
                        feedback_metrics[key].append(value)
                
                # Store mask tensor if valid
                if mask_tensor is not None:
                    # Resize mask to match original input dimensions
                    orig_h, orig_w = self.slice_processor.extract_slice(
                        x, slice_idx, depth_dim_idx
                    ).shape[2:4]
                    
                    if mask_tensor.shape[2:4] != (orig_h, orig_w):
                        mask_tensor = F.interpolate(
                            mask_tensor,
                            size=(orig_h, orig_w),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # For multi-class output, convert binary mask to class probabilities
                    if self.num_classes > 2:
                        # Create multi-class output using the binary-style approach
                        multi_class_mask = torch.zeros(
                            (batch_size, self.num_classes, orig_h, orig_w), 
                            device=device
                        )
                        
                        # Background (class 0) is inverse of the tumor mask
                        multi_class_mask[:, 0] = 1.0 - mask_tensor[:, 0]
                        
                        # Distribute tumor mask among classes 1, 2, 3 using typical BraTS distribution
                        typical_dist = [0.0, 0.3, 0.4, 0.3]  # Background + 3 tumor classes
                        for c in range(1, self.num_classes):
                            multi_class_mask[:, c] = mask_tensor[:, 0] * typical_dist[c]
                        
                        # Ensure the class probabilities sum to 1.0
                        total_prob = multi_class_mask.sum(dim=1, keepdim=True)
                        multi_class_mask = multi_class_mask / total_prob.clamp(min=1e-5)
                        
                        # Store in results
                        sam2_results[slice_idx] = multi_class_mask
                    else:
                        # Binary segmentation
                        binary_mask = torch.zeros(
                            (batch_size, 2, orig_h, orig_w),
                            device=device
                        )
                        binary_mask[:, 0] = 1.0 - mask_tensor[:, 0]  # Background
                        binary_mask[:, 1] = mask_tensor[:, 0]  # Foreground
                        
                        # Store in results
                        sam2_results[slice_idx] = binary_mask
            except Exception as e:
                logger.error(f"Error processing slice {slice_idx}: {e}")
        
        # Create output volume from processed slices
        output_volume = self.create_3d_from_slices(
            x.shape, sam2_results, depth_dim_idx, device
        )
        
        # Process with auxiliary 3D head if enabled
        aux_output = None
        if self.enable_auxiliary_head and self.auxiliary_3d_head is not None:
            aux_output = self.auxiliary_3d_head(x)
        
        # Compute auxiliary segmentation loss during training
        losses = {}
        if self.training:
            # Auxiliary segmentation loss
            if aux_seg_slices and targets is not None:
                aux_seg_loss = self.compute_auxiliary_seg_loss(
                    aux_seg_slices, targets, key_indices, depth_dim_idx
                )
                losses['aux_seg_loss'] = aux_seg_loss
            
            # Bidirectional feedback loss
            if overall_feedback_loss > 0:
                losses['feedback_loss'] = overall_feedback_loss
                
                # Store feedback metrics in losses dictionary
                for key, values in feedback_metrics.items():
                    if values:
                        losses[f'fb_{key}'] = sum(values) / len(values)
        
        return output_volume, aux_output, losses
    
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
        if torch.sum(processed_slices_mask) > 1:
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
    
    def compute_auxiliary_seg_loss(self, aux_seg_slices, targets, key_indices, depth_dim_idx):
        """Compute auxiliary segmentation loss for prompt encoder auxiliary head"""
        device = targets.device
        aux_loss = 0.0
        valid_slices = 0
        
        for slice_idx in key_indices:
            if slice_idx in aux_seg_slices:
                # Get auxiliary segmentation prediction
                aux_seg = aux_seg_slices[slice_idx]
                
                # Get ground truth slice
                gt = self.slice_processor.extract_slice(targets, slice_idx, depth_dim_idx)
                
                # For BraTS segmentation, combine all tumor classes (1, 2, 3) to get binary tumor mask
                if gt.shape[1] >= 4:  # Multi-class segmentation
                    # Get binary tumor mask (union of all tumor classes)
                    gt_slice = torch.sum(gt[:, 1:], dim=1, keepdim=True) > 0.5
                    gt_slice = gt_slice.float()
                else:
                    gt_slice = gt[:, 1:2]  # Non-background channel
                
                # Compute binary cross-entropy loss
                bce_loss = F.binary_cross_entropy_with_logits(aux_seg, gt_slice)
                
                # Compute Dice loss
                pred = torch.sigmoid(aux_seg)
                intersection = (pred * gt_slice).sum()
                dice_loss = 1.0 - (2. * intersection) / (pred.sum() + gt_slice.sum() + 1e-7)
                
                # Combined loss
                combined_loss = 0.5 * bce_loss + 0.5 * dice_loss
                aux_loss += combined_loss
                valid_slices += 1
        
        # Return average loss
        return aux_loss / max(1, valid_slices)
    
    def get_performance_stats(self):
        """Return performance statistics"""
        stats = {
            "has_sam2": self.sam2_integration.has_sam2,
            "training_slice_percentage": self.training_slice_percentage,
            "eval_slice_percentage": self.eval_slice_percentage,
            "current_epoch": self.current_epoch
        }
        
        # Add SAM2 feedback statistics
        if self.sam2_integration.feedback_history:
            for key, values in self.sam2_integration.feedback_history.items():
                if values:
                    stats[f"avg_{key}"] = sum(values) / len(values)
                    stats[f"max_{key}"] = max(values)
                    stats[f"min_{key}"] = min(values)
        
        return stats


# ======= Auxiliary 3D Segmentation Head =======

class AuxiliarySegHead(nn.Module):
    """
    Lightweight auxiliary 3D segmentation head that can be used alongside
    the primary SAM2-based segmentation for additional supervision or
    as a fallback when SAM2 is not available.
    """
    def __init__(self, num_classes=4, base_channels=16):
        super(AuxiliarySegHead, self).__init__()
        
        # Encoder pathway
        self.enc1 = nn.Sequential(
            nn.Conv3d(4, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder pathway
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv3d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        """Forward pass"""
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Decoder
        x = self.dec2(x3)
        x = self.dec1(x)
        
        # Final layer
        logits = self.final(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        return probs

# Add this adapter class at the end of BidirectionalAutoSAM2.py

class BidirectionalAutoSAM2Adapter(BidirectionalAutoSAM2):
    """
    Adapter version of BidirectionalAutoSAM2 that works with the existing train.py
    """
    def __init__(self, num_classes=4, base_channels=32, sam2_model_id="facebook/sam2-hiera-small", enable_auxiliary_head=True):
        super().__init__(num_classes, base_channels, sam2_model_id, enable_auxiliary_head)
        
        # Properties that train.py expects
        self.encoder = self.prompt_encoder  # train.py expects access to encoder
        self.unet3d = self.auxiliary_3d_head if self.enable_auxiliary_head else None
        
        # Additional compatibility properties
        self.enable_unet_decoder = True  # Variable checked in train.py
        self.enable_sam2 = self.sam2_integration.has_sam2
        self.has_sam2_enabled = self.sam2_integration.has_sam2
    
    def forward(self, x, targets=None):
        """
        Adapt the forward interface to be compatible with train.py
        
        Args:
            x: Input 3D volume [B, C, D, H, W]
            targets: Optional target masks [B, C, D, H, W]
            
        Returns:
            outputs: Segmentation volume [B, num_classes, D, H, W]
        """
        # Call the original forward
        output_volume, aux_output, losses = super().forward(x, targets)
        
        # Update performance metrics
        if losses:
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    self.performance_metrics[key].append(value.item())
                else:
                    self.performance_metrics[key].append(value)
        
        # Return output in the format train.py expects
        return output_volume
    
    def set_mode(self, enable_unet_decoder=None, enable_sam2=None, sam2_percentage=None, bg_blend=None, tumor_blend=None):
        """
        Function to change model mode - required by train.py
        """
        if enable_unet_decoder is not None:
            self.enable_unet_decoder = enable_unet_decoder
            
            # Update model based on new mode
            if not enable_unet_decoder:
                self.enable_auxiliary_head = False
            else:
                self.enable_auxiliary_head = True
        
        if enable_sam2 is not None:
            self.enable_sam2 = enable_sam2
            self.has_sam2_enabled = enable_sam2
        
        # Set slice percentage for processing
        if sam2_percentage is not None:
            self.training_slice_percentage = min(sam2_percentage, 0.5)  # Limit to half during training
            self.eval_slice_percentage = sam2_percentage
            
        # Store blending values for future use
        if bg_blend is not None:
            self.bg_blend = bg_blend
            
        if tumor_blend is not None:
            self.tumor_blend = tumor_blend
            
        print(f"Model mode: UNet={self.enable_unet_decoder}, SAM2={self.enable_sam2}, Slices={self.eval_slice_percentage}")

