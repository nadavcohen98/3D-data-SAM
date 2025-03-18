#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
import logging
import os
from collections import defaultdict
from scipy.ndimage import zoom, binary_erosion, binary_dilation, label, distance_transform_edt
import matplotlib.pyplot as plt

plt.ion()  # enable interactive mode

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autosam2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoSAM2")

print("=== LOADING AUTOSAM2 WITH FLEXIBLE ARCHITECTURE ===")

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

# ======= Base model components =======

class ResidualBlock3D(nn.Module):
    """3D convolutional block with residual connections and group normalization."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(ResidualBlock3D, self).__init__()
        
        # Main convolutional path
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        )
        
        # Residual connection with projection if needed
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # Final activation after residual connection
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.conv_block(x)
        return self.activation(x + residual)

class EncoderBlock3D(nn.Module):
    """Encoder block that combines downsampling with residual convolutions."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(EncoderBlock3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock3D(in_channels, out_channels, num_groups)
        )
    
    def forward(self, x):
        return self.encoder(x)

class DecoderBlock3D(nn.Module):
    """Decoder block with upsampling and residual convolutions."""
    def __init__(self, in_channels, out_channels, num_groups=8, trilinear=True):
        super(DecoderBlock3D, self).__init__()
        
        # Upsampling method
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        # Residual convolution block after concatenation
        self.conv = ResidualBlock3D(in_channels, out_channels, num_groups)
    
    def forward(self, x1, x2):
        # Upsample x1
        x1 = self.up(x1)
        
        # Pad x1 if needed to match x2 dimensions
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,  # Left, Right
            diffY // 2, diffY - diffY // 2,  # Top, Bottom
            diffZ // 2, diffZ - diffZ // 2   # Front, Back
        ])
        
        # Concatenate x2 (encoder features) with x1 (decoder features)
        x = torch.cat([x2, x1], dim=1)
        
        # Apply residual convolution block
        return self.conv(x)

# ======= UNet3D flexible architecture with mid-decoder hooks =======

class FlexibleUNet3D(nn.Module):
    """
    UNet3D architecture with hooks for mid-decoder features at 64x64 resolution.
    Allows flexible configuration with enable/disable switches for different paths.
    """
    def __init__(self, in_channels=4, n_classes=4, base_channels=16,trilinear=True):
        super(FlexibleUNet3D, self).__init__()
        
        # Configuration
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        
        # Initial convolution block
        self.initial_conv = ResidualBlock3D(in_channels, base_channels)
        
        # Encoder pathway
        self.enc1 = EncoderBlock3D(base_channels, base_channels * 2)
        self.enc2 = EncoderBlock3D(base_channels * 2, base_channels * 4)
        self.enc3 = EncoderBlock3D(base_channels * 4, base_channels * 8)
        self.enc4 = EncoderBlock3D(base_channels * 8, base_channels * 8)  # Keep channel count at 128
        
        # Decoder pathway with skip connections
        # Early decoder stages (to reach 64x64 resolution)
        self.dec1 = DecoderBlock3D(base_channels * 16, base_channels * 4, trilinear=trilinear)  # 8 + 8 = 16
        self.dec2 = DecoderBlock3D(base_channels * 8, base_channels * 2, trilinear=trilinear)   # 4 + 4 = 8
        
        # Late decoder stages (after 64x64 resolution)
        self.dec3 = DecoderBlock3D(base_channels * 4, base_channels, trilinear=trilinear)       # 2 + 2 = 4
        self.dec4 = DecoderBlock3D(base_channels * 2, base_channels, trilinear=trilinear)       # 1 + 1 = 2
        
        # Final output layer
        self.output_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)
        
        # Projection for SAM2 embeddings (from mid-decoder features)
        self.sam_projection = nn.Conv3d(base_channels * 2, 256, kernel_size=1)
    
    def forward(self, x, use_full_decoder=True):
        """
        Forward pass with flexible options
        """
        # Get batch dimensions
        batch_size, channels, dim1, dim2, dim3 = x.shape
        
        # Identify depth dimension (smallest one)
        dims = [dim1, dim2, dim3]
        depth_idx = dims.index(min(dims))
        depth = dims[depth_idx]
        
        # Ultra-defensive slice selection
        # Select only slices that are guaranteed to be within bounds
        max_slice_idx = depth - 1  # Maximum valid index
        
        # Generate slices based on the actual depth
        key_indices = []
        
        # Add slices at fixed percentages of the depth
        percentages = [
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.35, 0.38, 0.4, 0.42, 0.44, 
        0.46, 0.47, 0.48, 0.485, 0.49, 0.5, 0.51, 0.515, 0.52, 0.53, 0.54, 0.56, 
        0.58, 0.6, 0.62, 0.65, 0.68, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
        ]
        for p in percentages:
            idx = min(int(depth * p), max_slice_idx)
            if idx not in key_indices:
                key_indices.append(idx)
        
        # Add extra slices around the middle if possible
        middle = depth // 2
        extra_indices = []
        for offset in [-5, -3, -2, -1, 1, 2, 3, 5]:
            idx = middle + offset
            if 0 <= idx <= max_slice_idx and idx not in key_indices:
                extra_indices.append(idx)
        
        # Sort all indices
        key_indices.extend(extra_indices)
        key_indices.sort()
        
    
        # Encoder pathway
        x1 = self.initial_conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        # Early decoder stages
        dec_out1 = self.dec1(x5, x4)
        dec_out2 = self.dec2(dec_out1, x3)
        
        # Generate SAM2 embeddings
        sam_embeddings = self.sam_projection(dec_out2)
        
        # Calculate downsampled indices safely
        downsampled_depth = max(1, depth // 4)  # Prevent divide by zero
        ds_key_indices = [min(idx // 4, downsampled_depth-1) for idx in key_indices]
        
        # Store metadata
        metadata = {
            "key_indices": key_indices,
            "ds_key_indices": ds_key_indices,
            "depth_dim_idx": depth_idx,
            "mid_decoder_shape": dec_out2.shape
        }
        
        # If not using full decoder, return mid-decoder features
        if not use_full_decoder:
            return None, dec_out2, sam_embeddings, metadata
        
        # Late decoder stages
        dec_out3 = self.dec3(dec_out2, x2)
        dec_out4 = self.dec4(dec_out3, x1)
        
        # Final convolution
        output = self.output_conv(dec_out4)
        
        # Apply sigmoid
        segmentation = torch.sigmoid(output)
        
        return segmentation, dec_out2, sam_embeddings, metadata


class MultiPointPromptGenerator:
    """Generates strategic point prompts for SAM2 based on probability maps"""
    def __init__(self, num_points=3):
        self.num_points = num_points
        
    def _get_strategic_points(self, region_mask, num_points, prob_map):
        """Get strategic points within a specific region"""

        
        points = []
        
        # Get distance transform (distance from boundary)
        distance = distance_transform_edt(region_mask)
        
        # Weight by probability for finding confident points
        weighted_map = distance * prob_map * region_mask
        
        # Get coordinates of high weighted values
        flat_idx = np.argsort(weighted_map.flatten())[-num_points*2:]  # Get more candidates than needed
        y_coords, x_coords = np.unravel_index(flat_idx, weighted_map.shape)
        
        # Select diverse points
        selected_indices = self._select_diverse_points(x_coords, y_coords, num_points, 5)
        
        # Convert to point list
        for idx in selected_indices:
            points.append([int(x_coords[idx]), int(y_coords[idx])])
        
        return points

    def _select_diverse_points(self, x_coords, y_coords, num_points, min_distance):
        """
        Select a diverse set of points that maintain minimum distance from each other.
        """
        
        if len(x_coords) <= num_points:
            return list(range(len(x_coords)))
        
        # Start with the last point (highest confidence)
        selected_indices = [len(x_coords) - 1]
        
        # Greedily add points that maintain minimum distance
        for _ in range(num_points - 1):
            max_min_distance = -1
            best_idx = -1
            
            # Find point with maximum minimum distance to already selected points
            for i in range(len(x_coords)):
                if i in selected_indices:
                    continue
                
                # Calculate minimum distance to any selected point
                min_dist = float('inf')
                for idx in selected_indices:
                    dist = (x_coords[i] - x_coords[idx])**2 + (y_coords[i] - y_coords[idx])**2
                    min_dist = min(min_dist, dist)
                
                # Update best point if this one is better
                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    best_idx = i
            
            # If we found a valid point and it's far enough away, add it
            if best_idx != -1 and max_min_distance >= min_distance**2:
                selected_indices.append(best_idx)
            elif best_idx != -1:
                # If we can't find a point far enough away, just use the best we have
                selected_indices.append(best_idx)
            else:
                # No more valid points
                break
        
        return selected_indices

    def _get_negative_points(self, binary_mask, prob_map, num_points):
        """Generate strategic negative points"""
        
        # Dilate the mask to find nearby background
        dilated = binary_dilation(binary_mask, iterations=3)
        
        # Convert to boolean explicitly before using bitwise NOT
        dilated_bool = np.bool_(dilated)
        binary_mask_bool = np.bool_(binary_mask)
        
        # Now use bitwise NOT on boolean arrays
        near_boundary = dilated_bool & ~binary_mask_bool
        
        # Find regions with low probability that are near boundary
        background_regions = (prob_map < 0.3) & near_boundary
        
        points = []
        if np.sum(background_regions) > 0:
            # Get coordinates of background regions
            bg_y, bg_x = np.where(background_regions)
            
            # Select points
            num_to_select = min(num_points, len(bg_y))
            if num_to_select > 0:
                # Select diverse background points
                selected_indices = self._select_diverse_points(
                    bg_x, bg_y, 
                    num_to_select, 
                    10  # Larger min distance for background points
                )
                
                # Add selected points
                for idx in selected_indices:
                    points.append([int(bg_x[idx]), int(bg_y[idx])])
        
        # If not enough points, add some far from tumor
        if len(points) < num_points:
            # Find regions far from tumor - use boolean type explicitly
            far_bg_mask = ~dilated_bool
            if np.sum(far_bg_mask) > 0:
                far_y, far_x = np.where(far_bg_mask)
                remaining = num_points - len(points)
                num_to_select = min(remaining, len(far_y))
                if num_to_select > 0:
                    indices = np.random.choice(len(far_y), num_to_select, replace=False)
                    for idx in indices:
                        points.append([int(far_x[idx]), int(far_y[idx])])
        
        return points
    
    def generate_prompts(self, probability_maps, slice_idx, height, width):
        """
        Generate optimized point prompts for SAM2 based on probability maps
        
        Args:
            probability_maps: UNet3D feature maps [B, C, H, W]
            slice_idx: Current slice index (for diagnostic purposes)
            height, width: Target dimensions for points
            
        Returns:
            points: np.array of point coordinates
            labels: np.array of point labels (1=foreground, 0=background)
        """

        
        # Extract tumor probability (combine tumor classes with weighted importance)
        if probability_maps.shape[1] >= 4:  # Check if we have enough channels
            # Use class channels 1, 2, 3 (tumor classes) with weighted combination
            # Weight ET (class 3) higher as it's typically the most important region
            tumor_prob = (
                0.3 * torch.sigmoid(probability_maps[0, 1]) +  # NCR
                0.3 * torch.sigmoid(probability_maps[0, 2]) +  # ED
                0.4 * torch.sigmoid(probability_maps[0, 3])    # ET (enhancing tumor)
            ).cpu().detach().numpy()
        else:
            # Fallback to first non-background channel
            tumor_prob = torch.sigmoid(probability_maps[0, min(1, probability_maps.shape[1]-1)]).cpu().detach().numpy()
        
        # Resize to target dimensions if necessary
        if tumor_prob.shape != (height, width):
            tumor_prob = zoom(tumor_prob, (height / tumor_prob.shape[0], width / tumor_prob.shape[1]), order=1)
        
        # Create binary mask from probability map
        binary_mask = tumor_prob > 0.5
        
        # Initialize points lists
        foreground_points = []
        background_points = []
        
        # Only proceed if we have a non-empty tumor mask
        if np.sum(binary_mask) > 10:
            # ===== REGION-BASED POINT SELECTION =====
            
            # Find tumor regions and their centers
            labeled_mask, num_regions = label(binary_mask)
            
            if num_regions > 1:
                # If multiple regions, process each separately
                region_sizes = [(i+1, np.sum(labeled_mask == (i+1))) for i in range(num_regions)]
                region_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size
                
                # Number of points per region based on relative size
                points_remaining = self.num_positive_points
                
                for region_id, size in region_sizes[:3]:  # Process top 3 regions
                    if size < 20:  # Skip tiny regions
                        continue
                        
                    # Mask for this region
                    region_mask = (labeled_mask == region_id)
                    
                    # Calculate points for this region based on relative size
                    region_points = max(1, min(3, int(points_remaining * size / np.sum(binary_mask))))
                    
                    # Get strategic points for this region
                    region_points = self._get_strategic_points(region_mask, region_points, tumor_prob)
                    foreground_points.extend(region_points)
                    
                    points_remaining -= len(region_points)
                    if points_remaining <= 0:
                        break
            
            # ===== DISTANCE-BASED POINT SELECTION =====
            
            # If we need more points, use distance transform to find points far from boundaries
            if len(foreground_points) < self.num_positive_points:
                distance = distance_transform_edt(binary_mask)
                
                # Weight by probability to find high-confidence internal points
                confidence = distance * tumor_prob * binary_mask
                
                # Get top confidence points
                flat_idx = np.argsort(confidence.flatten())[-10:]  # Get top 10 candidates
                high_conf_y, high_conf_x = np.unravel_index(flat_idx, confidence.shape)
                
                # Select diverse high-confidence points
                selected_indices = self._select_diverse_points(
                    high_conf_x, high_conf_y, 
                    min(self.num_positive_points - len(foreground_points), len(high_conf_y)), 
                    5  # Minimum distance between points
                )
                
                # Add selected high-confidence points
                for idx in selected_indices:
                    foreground_points.append([int(high_conf_x[idx]), int(high_conf_y[idx])])
            
            # ===== BOUNDARY POINT SELECTION =====
            
            # If we still need more points, add boundary points
            if len(foreground_points) < self.num_positive_points:
                # Create boundary mask (dilated - eroded)
                eroded = binary_erosion(binary_mask, iterations=2)
                dilated = binary_dilation(binary_mask, iterations=2)
                
                # Convert to boolean explicitly before bitwise operations
                dilated_bool = np.bool_(dilated)
                eroded_bool = np.bool_(eroded)
                
                # Now use bitwise operations on boolean arrays
                boundary = dilated_bool & ~eroded_bool
                
                # Find boundary points
                boundary_y, boundary_x = np.where(boundary)
                if len(boundary_y) > 0:
                    # Select random points from boundary
                    num_boundary_points = min(self.num_positive_points - len(foreground_points), len(boundary_y))
                    indices = np.random.choice(len(boundary_y), num_boundary_points, replace=False)
                    for idx in indices:
                        foreground_points.append([int(boundary_x[idx]), int(boundary_y[idx])])
            
            # ===== NEGATIVE (BACKGROUND) POINT SELECTION =====
            
            # Add strategic background points near boundary
            background_points = self._get_negative_points(binary_mask, tumor_prob, self.num_negative_points)
        
        # If no foreground points found, use center point
        if not foreground_points:
            foreground_points.append([width//2, height//2])
        
        # If no background points, add some corners
        if not background_points:
            background_points.append([width//4, height//4])
            background_points.append([width*3//4, height*3//4])
        
        # Combine points and labels
        all_points = foreground_points + background_points
        all_labels = [1] * len(foreground_points) + [0] * len(background_points)
        
        return np.array(all_points), np.array(all_labels)



# ======= SAM2 integration components =======

class SliceProcessor(nn.Module):
    """Process 3D volume slices for SAM2 integration."""
    def __init__(self, input_channels=256, output_size=(64, 64)):
        super().__init__()
        
        self.output_size = output_size
        
        # Refinement for slice features
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=input_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pooling to ensure correct output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
    
    def extract_slice(self, volume, idx, depth_dim=2):
        """Extract a specific slice from a 3D volume."""
        if depth_dim == 0:
            return volume[:, :, idx]
        elif depth_dim == 1:
            return volume[:, :, :, idx]
        else:  # default to dim 2
            return volume[:, :, :, :, idx]
    
    def forward(self, features, indices, depth_dim):
        """Process slices for SAM2."""
        processed_slices = {}
        
        for idx in indices:
            # Extract slice
            slice_2d = self.extract_slice(features, idx, depth_dim)
            
            # Apply refinement
            refined_slice = self.refine(slice_2d)
            
            # Ensure correct size with adaptive pooling
            if refined_slice.shape[2:] != self.output_size:
                refined_slice = self.adaptive_pool(refined_slice)
            
            # Store processed slice
            processed_slices[idx] = refined_slice
        
        return processed_slices

class MRItoRGBMapper(nn.Module):
    """Advanced mapping from 4 MRI channels to 3 RGB channels with attention"""
    def __init__(self):
        super().__init__()
        
        # Combined processing - simpler and more robust
        self.initial_features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        
        # Feature extraction and integration
        self.features = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        
        # Simple self-attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final RGB mapping
        self.to_rgb = nn.Conv2d(16, 3, kernel_size=1)
        
    def forward(self, x):
        # Process all modalities together
        features = self.initial_features(x)
        
        # Extract integrated features
        features = self.features(features)
        
        # Apply attention
        attention_map = self.attention(features)
        attended_features = features * attention_map
        
        # Generate RGB
        rgb = self.to_rgb(attended_features)
        
        # Ensure proper range with sigmoid
        enhanced_rgb = torch.sigmoid(rgb)
        
        return enhanced_rgb

class UNet3DtoSAM2Bridge(nn.Module):
    def __init__(self, input_channels=32, output_channels=256):
        super(UNet3DtoSAM2Bridge, self).__init__()
        # Channel Attention module to emphasize important feature channels
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 4, input_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Attention module to focus on relevant regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # Shortcut branch to match dimensions for residual connection
        self.shortcut = nn.Conv2d(input_channels, 128, kernel_size=1)
        # Transformer branch: two convolutional layers with GroupNorm and ReLU
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(16, 128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(16, 128)
        self.relu = nn.ReLU(inplace=True)
        # Final layer to convert features to SAM2 expected dimensions
        self.final_conv = nn.Conv2d(128, output_channels, kernel_size=1)
        self.final_norm = nn.GroupNorm(32, output_channels)
        # Learnable parameters for dynamic scaling of the output
        self.scale_factor = 1.5

    def forward(self, x):
        # Apply channel attention and weight the input accordingly
        att_channels = self.channel_attention(x)
        x_ca = x * att_channels

        # Apply spatial attention on the channel-attended features
        att_spatial = self.spatial_attention(x_ca)
        x_sa = x_ca * att_spatial

        # Transformer branch: process features through two convolution layers
        out = self.conv1(x_sa)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Compute shortcut projection for residual connection
        shortcut = self.shortcut(x_sa)
        out = self.relu(out + shortcut)

        # Final transformation to match SAM2 input dimensions
        out = self.final_conv(out)
        out = self.final_norm(out)

        # Apply learnable scaling (importance) with adaptive scaling factor
        out = out * self.scale_factor

        return out

class EnhancedPromptGenerator:
    def __init__(self, 
                 num_positive_points=10,  # Increased from 5 for better coverage
                 num_negative_points=3,  
                 edge_detection=True,    
                 use_confidence=True,    
                 use_mask_prompt=True):
        self.num_positive_points = num_positive_points
        self.num_negative_points = num_negative_points
        self.edge_detection = edge_detection
        self.use_confidence = use_confidence
        self.use_mask_prompt = use_mask_prompt
        self.prompt_stats = {
            'boxes_generated': 0,
            'points_generated': 0,
            'masks_generated': 0,
            'multi_region_cases': 0
        }

    def generate_optimal_box(self, binary_mask_or_probability_maps, prob_map_or_slice_idx=None, height=None, width=None):
        # Handle different parameter patterns
        if height is not None and width is not None:
            probability_maps = binary_mask_or_probability_maps
            slice_idx = prob_map_or_slice_idx
            
            # Extract tumor probability (weighted combination)
            if probability_maps.shape[1] >= 4:
                tumor_prob = torch.sigmoid(probability_maps[0, 1:4]).sum(dim=0).cpu().detach().numpy()
            else:
                tumor_prob = torch.sigmoid(probability_maps[0, min(1, probability_maps.shape[1]-1)]).cpu().detach().numpy()
            
            # Resize if necessary
            curr_h, curr_w = tumor_prob.shape
            if curr_h != height or curr_w != width:
                tumor_prob = zoom(tumor_prob, (height / curr_h, width / curr_w), order=1)
            
            # Create binary mask
            binary_mask = tumor_prob > 0.4
            prob_map = tumor_prob
        else:
            binary_mask = binary_mask_or_probability_maps
            prob_map = prob_map_or_slice_idx
            height, width = prob_map.shape
        
        # If no tumor detected, return None
        if np.sum(binary_mask) < 10:
            return None
        
        # Apply dilation to create a more inclusive box
        dilated_mask = binary_dilation(binary_mask, iterations=3)
        
        # Find coordinates of tumor region
        y_coords, x_coords = np.where(dilated_mask)
        if len(y_coords) == 0:
            return None
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Reduce padding factor from 15% to 10% for tighter bounding box
        padding_x = max(5, int((max_x - min_x) * 0.10))
        padding_y = max(5, int((max_y - min_y) * 0.10))
        
        x1 = max(0, min_x - padding_x)
        y1 = max(0, min_y - padding_y)
        x2 = min(width - 1, max_x + padding_x)
        y2 = min(height - 1, max_y + padding_y)
        
        self.prompt_stats['boxes_generated'] += 1
        return np.array([x1, y1, x2, y2])
    
    def generate_prompts(self, probability_maps, slice_idx, height, width):
        """
        Generate optimized point prompts for SAM2 based on probability maps.
        Returns:
            - all_points: numpy array of point coordinates.
            - all_labels: numpy array of corresponding labels (1=foreground, 0=background).
            - bounding_box: optimal bounding box computed from the probability map.
            - tumor_prob: the processed tumor probability map.
        """
        # Extract tumor probability with increased intensity multiplier
        if probability_maps.shape[1] >= 4:
            tumor_prob = (
                0.3 * torch.sigmoid(probability_maps[0, 1]) +  
                0.3 * torch.sigmoid(probability_maps[0, 2]) +  
                0.4 * torch.sigmoid(probability_maps[0, 3])
            ).cpu().detach().numpy() * 2  # Increase intensity by factor of 2
        else:
            tumor_prob = torch.sigmoid(probability_maps[0, min(1, probability_maps.shape[1]-1)]).cpu().detach().numpy() * 1.5

        # Resize probability map to target dimensions if necessary
        if tumor_prob.shape != (height, width):
            tumor_prob = zoom(tumor_prob, (height / tumor_prob.shape[0], width / tumor_prob.shape[1]), order=1)
        
        binary_mask = tumor_prob > 0.6
        
        # Initialize lists for foreground and background points
        foreground_points = []
        background_points = []
        
        # Process tumor regions and generate strategic positive points
        if np.sum(binary_mask) > 10:
            labeled_mask, num_regions = label(binary_mask)
            if num_regions > 1:
                region_sizes = [(i+1, np.sum(labeled_mask == (i+1))) for i in range(num_regions)]
                region_sizes.sort(key=lambda x: x[1], reverse=True)
                points_remaining = self.num_positive_points
                for region_id, size in region_sizes[:3]:
                    if size < 20:
                        continue
                    region_mask = (labeled_mask == region_id)
                    region_points = self._get_strategic_points(region_mask, min(3, points_remaining), tumor_prob)
                    foreground_points.extend(region_points)
                    points_remaining -= len(region_points)
                    if points_remaining <= 0:
                        break

            # If additional points needed, use confidence-based selection
            if len(foreground_points) < self.num_positive_points:
                distance = distance_transform_edt(binary_mask)
                confidence = distance * tumor_prob * binary_mask
                flat_idx = np.argsort(confidence.flatten())[-10:]
                high_conf_y, high_conf_x = np.unravel_index(flat_idx, confidence.shape)
                selected_indices = self._select_diverse_points(high_conf_x, high_conf_y, 
                                                                self.num_positive_points - len(foreground_points), 5)
                for idx in selected_indices:
                    foreground_points.append([int(high_conf_x[idx]), int(high_conf_y[idx])])
            
            # Generate background (negative) points
            background_points = self._get_negative_points(binary_mask, tumor_prob, self.num_negative_points)
        
        if not foreground_points:
            foreground_points.append([width // 2, height // 2])
        if not background_points:
            background_points.append([width // 4, height // 4])
            background_points.append([width * 3 // 4, height * 3 // 4])
        
        all_points = foreground_points + background_points
        all_labels = [1] * len(foreground_points) + [0] * len(background_points)
        self.prompt_stats['points_generated'] += len(all_points)
        
        bounding_box = self.generate_optimal_box(binary_mask, tumor_prob)
        return np.array(all_points), np.array(all_labels), bounding_box, tumor_prob

    def _get_strategic_points(self, region_mask, num_points, prob_map):
        """Get strategic points within a specific region using distance transform and probability weighting."""
        points = []
        distance_map = distance_transform_edt(region_mask)
        weighted_map = distance_map * prob_map * region_mask
        flat_idx = np.argsort(weighted_map.flatten())[-num_points * 2:]  # Get more candidates
        y_coords, x_coords = np.unravel_index(flat_idx, weighted_map.shape)
        
        if len(y_coords) > 0:
            selected_indices = [len(y_coords) - 1]  # Start with the highest value point
            for _ in range(min(num_points - 1, len(y_coords) - 1)):
                remaining_indices = [i for i in range(len(y_coords)) if i not in selected_indices]
                if not remaining_indices:
                    break
                max_min_dist = -1
                best_idx = -1
                for i in remaining_indices:
                    min_dist = float('inf')
                    for sel in selected_indices:
                        dist = (x_coords[i] - x_coords[sel]) ** 2 + (y_coords[i] - y_coords[sel]) ** 2
                        min_dist = min(min_dist, dist)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = i
                if best_idx != -1:
                    selected_indices.append(best_idx)
            for idx in selected_indices:
                points.append([int(x_coords[idx]), int(y_coords[idx])])
        return points

    def _select_diverse_points(self, x_coords, y_coords, num_points, min_distance):
        """
        Select a diverse set of points ensuring that each selected point is at least
        min_distance away from the others.
        """
        if len(x_coords) <= num_points:
            return list(range(len(x_coords)))
        
        selected_indices = [len(x_coords) - 1]  # Start with the highest confidence point
        
        while len(selected_indices) < num_points:
            max_min_dist = -1
            best_idx = None
            for i in range(len(x_coords)):
                if i in selected_indices:
                    continue
                # Calculate minimum distance from i to already selected points
                dists = [((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2) for j in selected_indices]
                min_dist = min(dists)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            if best_idx is not None and max_min_dist >= min_distance**2:
                selected_indices.append(best_idx)
            else:
                break
        return selected_indices

    def _get_negative_points(self, binary_mask, prob_map, num_points):
        """Generate strategic negative points near the tumor boundary."""
        dilated = binary_dilation(binary_mask, iterations=3)
        dilated_bool = np.bool_(dilated)
        binary_mask_bool = np.bool_(binary_mask)
        near_boundary = dilated_bool & ~binary_mask_bool
        background_regions = (prob_map < 0.3) & near_boundary
        
        points = []
        if np.sum(background_regions) > 0:
            bg_y, bg_x = np.where(background_regions)
            num_to_select = min(num_points, len(bg_y))
            if num_to_select > 0:
                indices = np.random.choice(len(bg_y), num_to_select, replace=False)
                for idx in indices:
                    points.append([int(bg_x[idx]), int(bg_y[idx])])
        
        if len(points) < num_points:
            far_bg_mask = ~dilated_bool
            if np.sum(far_bg_mask) > 0:
                far_y, far_x = np.where(far_bg_mask)
                remaining = num_points - len(points)
                num_to_select = min(remaining, len(far_y))
                if num_to_select > 0:
                    indices = np.random.choice(len(far_y), num_to_select, replace=False)
                    for idx in indices:
                        points.append([int(far_x[idx]), int(far_y[idx])])
        return points

    



# ======= Main AutoSAM2 model =======

class AutoSAM2(nn.Module):
    """
    AutoSAM2 with flexible architecture configurations.
    Supports multiple modes:
    1. Full UNet3D only (SAM2 disabled)
    2. SAM2 only using mid-decoder features (UNet3D decoder disabled)
    3. Hybrid mode with both paths
    """
    def __init__(
        self, 
        num_classes=4, 
        base_channels=16, 
        trilinear=True,
        enable_unet_decoder=True,
        enable_sam2=True,
        sam2_model_id="facebook/sam2-hiera-small"
    ):
        super().__init__()

        self.prompt_generator = EnhancedPromptGenerator(
            num_positive_points=5,
            num_negative_points=3,
            edge_detection=True,
            use_confidence=True,
            use_mask_prompt=True
        )

        self.unet_sam_bridge = UNet3DtoSAM2Bridge(
            input_channels=32, 
            output_channels=256 
        )
        
        # Configuration
        self.num_classes = num_classes
        self.enable_unet_decoder = enable_unet_decoder
        self.enable_sam2 = enable_sam2
        self.sam2_model_id = sam2_model_id
        
        
        # Create flexible UNet3D
        self.unet3d = FlexibleUNet3D(
            in_channels=4,
            n_classes=num_classes,
            base_channels=base_channels,
            trilinear=trilinear
        )
        
        # Slice processor for SAM2 integration
        self.slice_processor = SliceProcessor(
            input_channels=256,
            output_size=(64, 64)
        )

        self.mri_to_rgb = MRItoRGBMapper()
        
        # For compatibility with train.py
        self.encoder = self.unet3d  # Point to the UNet3D model
        
        # Initialize tracking variables
        self.has_sam2 = False
        self.has_sam2_enabled = False
        self.sam2 = None
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.performance_metrics["sam2_slices_processed"] = 0
        
        # Initialize SAM2 if enabled
        if enable_sam2:
            self.initialize_sam2()
    
    def initialize_sam2(self):
        """Initialize SAM2."""
        if not HAS_SAM2:
            logger.warning("SAM2 package not available. Running in fallback mode.")
            return
            
        try:
            # Initialize SAM2
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
            self.sam2 = None
    
    def set_mode(self, enable_unet_decoder=None, enable_sam2=None):
        """
        Change the processing mode dynamically.
        
        Args:
            enable_unet_decoder: Whether to use the full UNet3D decoder
            enable_sam2: Whether to use SAM2 for selected slices
        """
        if enable_unet_decoder is not None:
            self.enable_unet_decoder = enable_unet_decoder
        
        if enable_sam2 is not None:
            self.enable_sam2 = enable_sam2
            
        logger.info(f"Mode set to: UNet Decoder={self.enable_unet_decoder}, SAM2={self.enable_sam2}")

    
    def extract_tumor_box(self, prob_map, threshold=0.5):
        # Threshold the probability map
        binary_mask = prob_map > threshold
        
        # Find coordinates of non-zero elements
        y_indices, x_indices = np.where(binary_mask)
        
        # If no tumor found, return None
        if len(y_indices) == 0:
            return None
        
        # Find bbox coordinates
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)
        
        # Return as [x1, y1, x2, y2] format
        return np.array([x1, y1, x2, y2])
    
    def create_mask_prompt(self, prob_map, threshold=0.5):
        # Create binary mask
        mask = (prob_map > threshold).astype(np.float32)
        
        # Ensure correct shape for SAM2
        return mask
        
    
    def preprocess_slice_for_sam2(self, img_slice):
        """Preprocess a slice for SAM2."""
        # Convert to numpy and standardize
        img_np = img_slice[0, 0].detach().cpu().numpy()
        
        # Apply contrast enhancement
        try:
            p1, p99 = np.percentile(img_np, (1, 99))
            if p99 > p1:
                img_np = np.clip((img_np - p1) / (p99 - p1), 0, 1)
            else:
                # If percentile fails, use min-max normalization
                min_val, max_val = np.min(img_np), np.max(img_np)
                if max_val > min_val:
                    img_np = (img_np - min_val) / (max_val - min_val)
                else:
                    img_np = np.zeros_like(img_np)
        except Exception as e:
            logger.warning(f"Error in contrast enhancement: {e}. Using basic normalization.")
            min_val, max_val = np.min(img_np), np.max(img_np)
            if max_val > min_val:
                img_np = (img_np - min_val) / (max_val - min_val)
            else:
                img_np = np.zeros_like(img_np)
        
        # Create RGB image for SAM2
        img_rgb = np.stack([img_np, img_np, img_np], axis=2)
        
        return img_rgb
    
    def process_slice_with_sam2(self, input_vol, slice_idx, slice_features, depth_dim_idx, device):
        """Process a single slice with SAM2 using both point and box prompts"""
        if not self.has_sam2:
            logger.warning(f"SAM2 not available for slice {slice_idx}")
            return None
                
        try:
            # Extract original slice
            if depth_dim_idx == 0:
                orig_slice = input_vol[:, :, slice_idx]
            elif depth_dim_idx == 1:
                orig_slice = input_vol[:, :, :, slice_idx]
            else:  # depth_dim_idx == 2
                orig_slice = input_vol[:, :, :, :, slice_idx]
            
            # If we have a batch, just take the first item
            if orig_slice.shape[0] > 1:
                orig_slice = orig_slice[0:1]
            
            # Convert MRI to RGB using our hybrid mapper
            rgb_tensor = self.mri_to_rgb(orig_slice)
            
            # Convert to numpy array in the format SAM2 expects: [H, W, 3]
            rgb_image = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            
            # Get image dimensions
            h, w = rgb_image.shape[:2]
            
            # Process through bridge network to get enhanced features
            enhanced_features = self.unet_sam_bridge(slice_features)
        
            
            # Generate point prompts
            points, labels, box, _ = self.prompt_generator.generate_prompts(
                enhanced_features, slice_idx, h, w
            )
            
           
            # Set image in SAM2
            self.sam2.set_image(rgb_image)
            
            # Call SAM2 with both point and box prompts (if box exists)
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
                
                # Create multi-class output
                height, width = mask_tensor.shape[2:]
                multi_class_mask = torch.zeros((1, self.num_classes, height, width), device=device)
                
                # Fill tumor classes (1,2,3)
                for c in range(1, self.num_classes):
                    multi_class_mask[:, c] = mask_tensor[:, 0]
                
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
        Create a 3D volume from 2D slice results.
        For mode 2 (SAM2 only) where we need to build a volume from individual slices.
        """
        # Create empty volume matching the expected output size
        batch_size = input_shape[0]
        output_shape = list(input_shape)
        output_shape[1] = self.num_classes  # Change channel dimension to match num_classes
        
        volume = torch.zeros(output_shape, device=device)
        
        # Insert each slice result into the appropriate position
        for slice_idx, mask in sam2_slices.items():
            if mask is not None:
                if depth_dim_idx == 0:
                    volume[:, :, slice_idx] = mask
                elif depth_dim_idx == 1:
                    volume[:, :, :, slice_idx] = mask
                else:  # depth_dim_idx == 2
                    volume[:, :, :, :, slice_idx] = mask
        
        return volume

    def process_volume_with_3d_context(self, input_vol, features, metadata, device):
        """Process volume using a sliding window approach for 3D context"""
        depth_dim_idx = metadata["depth_dim_idx"]
        key_indices = sorted(metadata["key_indices"])
        
        # Results dictionary
        sam2_results = {}
        
        # Process slices in overlapping groups of 3
        context_window = 3
        
        # Track previous masks for continuity
        previous_masks = {}  # Store masks by slice index
        
        # Process each slice with context from neighbors
        for i in range(len(key_indices)):
            center_idx = key_indices[i]
            
            # Get context indices (current slice plus neighbors)
            context_indices = []
            for offset in range(-context_window//2, context_window//2 + 1):
                neighbor_pos = i + offset
                if neighbor_pos >= 0 and neighbor_pos < len(key_indices):
                    context_indices.append(key_indices[neighbor_pos])
            
            # Process this slice with context
            result = self._process_slice_with_context(
                input_vol, context_indices, center_idx, features, 
                depth_dim_idx, previous_masks, device
            )
            
            if result is not None:
                # Store the result
                sam2_results[center_idx] = result
                
                # Store mask for future context (just class 1, flattened to 2D)
                mask_np = result.detach().cpu().numpy()[0, 1].squeeze() > 0.5
                previous_masks[center_idx] = mask_np.astype(bool)
                
                # Limit memory usage by keeping only recent slices
                if len(previous_masks) > context_window*2:
                    oldest_key = min(k for k in previous_masks.keys() if k != center_idx)
                    del previous_masks[oldest_key]
        
        return sam2_results

    
    def _process_slice_with_context(self, input_vol, context_indices, center_idx, 
                                  features, depth_dim_idx, previous_masks, device):
        """Process a slice with context from neighboring slices"""
        try:
            # Extract and process the center slice (same as before)
            orig_slice = self._extract_slice(input_vol, center_idx, depth_dim_idx)
            rgb_tensor = self.mri_to_rgb(orig_slice)
            rgb_image = rgb_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            h, w = rgb_image.shape[:2]
            
            slice_features = self.slice_processor.extract_slice(features, center_idx // 4, depth_dim_idx)
            enhanced_features = self.unet_sam_bridge(slice_features)
            points, labels, box, _ = self.prompt_generator.generate_prompts(enhanced_features, center_idx, h, w)
            
            # Set image in SAM2
            self.sam2.set_image(rgb_image)
            
            # Just use points and box without mask_input for now
            masks, scores, _ = self.sam2.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True
            )
            
            # Process results (same as before)
            if len(masks) > 0:
                best_idx = scores.argmax()
                best_mask = masks[best_idx]
                mask_tensor = torch.from_numpy(best_mask).float().to(device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                multi_class_mask = torch.zeros((1, self.num_classes, h, w), device=device)
                for c in range(1, self.num_classes):
                    multi_class_mask[:, c] = mask_tensor[:, 0]
                return multi_class_mask
            
            return None
        except Exception as e:
            logger.error(f"Error processing slice {center_idx} with context: {e}")
            return None
                                   
    
    def _extract_slice(self, volume, idx, depth_dim=2):
        """Extract a specific slice from a 3D volume."""
        if depth_dim == 0:
            slice_data = volume[:, :, idx]
        elif depth_dim == 1:
            slice_data = volume[:, :, :, idx]
        else:  # default to dim 2
            slice_data = volume[:, :, :, :, idx]
        
        # If we have a batch, just take the first item
        if slice_data.shape[0] > 1:
            slice_data = slice_data[0:1]
        
        return slice_data
    
    def combine_results_adaptively(self, unet_output, sam2_slices, depth_dim_idx, blend_weight=0.7):
        """
        Combine UNet output with SAM2 results.
        For each slice, if SAM2 produced a valid mask, blend it with UNet output using the blend_weight.
        Otherwise, leave the UNet output unchanged.
        """
        combined = unet_output.clone()
        for slice_idx, mask in sam2_slices.items():
            if mask is not None:
                if depth_dim_idx == 0:
                    unet_slice = combined[:, :, slice_idx]
                    blended = blend_weight * unet_slice + (1 - blend_weight) * mask
                    combined[:, :, slice_idx] = blended
                elif depth_dim_idx == 1:
                    unet_slice = combined[:, :, :, slice_idx]
                    blended = blend_weight * unet_slice + (1 - blend_weight) * mask
                    combined[:, :, :, slice_idx] = blended
                else:
                    unet_slice = combined[:, :, :, :, slice_idx]
                    blended = blend_weight * unet_slice + (1 - blend_weight) * mask
                    combined[:, :, :, :, slice_idx] = blended
        return combined

    
    def combine_results(self, unet_output, sam2_slices, depth_dim_idx, blend_weight=0.7):
        """
        Combine UNet output with SAM2 results with detailed comparisons.
        """
        # Start with UNet output
        combined = unet_output.clone()
        
        # Initialize tracking if needed
        if not hasattr(self, 'slice_comparison'):
            self.slice_comparison = {
                'batch_count': 0,
                'detailed_slices': []
            }
        
        self.slice_comparison['batch_count'] += 1
        
        # For detailed comparison every 20 batches
        save_detailed = (self.slice_comparison['batch_count'] % 40 == 0)
        if save_detailed:
            print("\n===== Detailed SAM2 vs UNet3D Comparison =====")
        
        # Replace or blend slices with SAM2 results
        for slice_idx, mask in sam2_slices.items():
            if mask is not None:
                # Get UNet slice
                if depth_dim_idx == 0:
                    unet_slice = combined[:, :, slice_idx]
                elif depth_dim_idx == 1:
                    unet_slice = combined[:, :, :, slice_idx]
                else:  # depth_dim_idx == 2
                    unet_slice = combined[:, :, :, :, slice_idx]
                
                # Get binary tumor segmentations 
                sam2_tumor = (mask[:, 1:].sum(dim=1) > 0).float()
                unet_tumor = (unet_slice[:, 1:].sum(dim=1) > 0).float()
                
                # Calculate metrics
                sam2_pixels = torch.sum(sam2_tumor).item()
                unet_pixels = torch.sum(unet_tumor).item()
                models_overlap = torch.sum(sam2_tumor * unet_tumor).item()
                
                # Compute Dice score between models
                models_dice = 2 * models_overlap / (sam2_pixels + unet_pixels) if (sam2_pixels + unet_pixels) > 0 else 0
                
                
                # Apply blending
                if depth_dim_idx == 0:
                    combined[:, :, slice_idx] = (
                        blend_weight * unet_slice + (1 - blend_weight) * mask
                    )
                elif depth_dim_idx == 1:
                    combined[:, :, :, slice_idx] = (
                        blend_weight * unet_slice + (1 - blend_weight) * mask
                    )
                else:  # depth_dim_idx == 2:
                    combined[:, :, :, :, slice_idx] = (
                        blend_weight * unet_slice + (1 - blend_weight) * mask
                    )
        
        # Reset detailed slices list if we've printed them
        if save_detailed:
            self.slice_comparison['detailed_slices'] = []
            print("==========================================\n")
        
        return combined
    
    def get_boundary_pixels(self, mask, threshold=0.5):
        """Get boundary pixels of a mask for comparing segmentation boundaries"""
        mask_binary = (mask[:, 1:] > threshold).float()
        mask_sum = torch.sum(mask_binary, dim=1, keepdim=True)
        
        # Apply erosion to get inner region
        kernel_size = 3
        padding = kernel_size // 2
        pooled = F.max_pool2d(
            -mask_sum, kernel_size=kernel_size, stride=1, padding=padding
        )
        eroded = (-pooled > -0.5).float()
        
        # Boundary is the difference between mask and eroded mask
        boundary = mask_sum - eroded
        
        return boundary
    
    def forward(self, x):
        """
        Forward pass with 3D context-aware SAM2 integration
        """
        start_time = time.time()
        device = x.device
        
        # Flag for SAM2 usage tracking
        self.has_sam2_enabled = self.enable_sam2 and self.has_sam2
        
        # Process with UNet3D to get mid-decoder features
        unet_output, mid_features, sam_embeddings, metadata = self.unet3d(
            x, 
            use_full_decoder=self.enable_unet_decoder
        )
        
        # Mode 1: UNet3D only
        if not self.enable_sam2 or not self.has_sam2:
            self.performance_metrics["total_time"].append(time.time() - start_time)
            return unet_output
        
        # Process entire volume with 3D context for SAM2
        sam2_results = self.process_volume_with_3d_context(
            x, mid_features, metadata, device
        )
        
        # Mode 2: SAM2 only (without UNet decoder)
        if not self.enable_unet_decoder:
            # Create 3D volume from SAM2 slice results
            final_output = self.create_3d_from_slices(
                x.shape, sam2_results, metadata["depth_dim_idx"], device
            )
        
        # Mode 3: Hybrid mode (UNet + SAM2)
        else:
            # Combine UNet output with 3D context-aware SAM2 results
            final_output = self.combine_results_adaptively(
                unet_output, sam2_results, metadata["depth_dim_idx"]
            )
        
        # Update timing metrics
        total_time = time.time() - start_time
        self.performance_metrics["total_time"].append(total_time)
        
        return final_output
    
    def get_performance_stats(self):
        """Return performance statistics."""
        stats = {
            "has_sam2": self.has_sam2,
            "sam2_enabled": self.has_sam2_enabled,
            "unet_enabled": self.enable_unet_decoder,
            "sam2_slices_processed": self.performance_metrics["sam2_slices_processed"],
            "model_mode": self._get_current_mode()
        }
        
        if self.performance_metrics["total_time"]:
            stats["avg_time_per_forward"] = np.mean(self.performance_metrics["total_time"])
            stats["total_forward_passes"] = len(self.performance_metrics["total_time"])
        
        return stats
    
    def _get_current_mode(self):
        """Return string describing current mode."""
        if self.enable_unet_decoder and not self.has_sam2_enabled:
            return "unet3d_only"
        elif not self.enable_unet_decoder and self.has_sam2_enabled:
            return "sam2_only"
        elif self.enable_unet_decoder and self.has_sam2_enabled:
            return "hybrid"
        else:
            return "invalid_config"

    def visualize_slice_comparison(self, input_vol, unet_output, sam2_output, ground_truth, slice_indices):
        """
        For each selected slice, compute the Dice score for UNet and SAM2 outputs separately,
        comparing each to the ground truth. The Dice is computed per channel (channels 1-3)
        and then averaged.
        """
        # Move tensors to CPU and detach them
        input_vol = input_vol.detach().cpu()
        unet_output = unet_output.detach().cpu()
        sam2_output = sam2_output.detach().cpu()
        ground_truth = ground_truth.detach().cpu()
        
        import numpy as np
        b = 0  # Visualize the first sample in the batch
        for idx in slice_indices:
            print(f"Slice {idx}:")
            dice_unet_channels = []
            dice_sam2_channels = []
            for ch in range(1, ground_truth.shape[1]):  # assume channels 1,2,3 represent tumor classes
                gt_ch = ground_truth[b, ch, idx].numpy()
                unet_ch = unet_output[b, ch, idx].numpy()
                sam2_ch = sam2_output[b, ch, idx].numpy()
                
                # Convert to binary masks using threshold 0.5
                gt_bin = (gt_ch > 0.5).astype(np.float32)
                unet_bin = (unet_ch > 0.5).astype(np.float32)
                sam2_bin = (sam2_ch > 0.5).astype(np.float32)
                
                # Compute Dice for UNet channel
                intersection_unet = np.sum(gt_bin * unet_bin)
                dice_unet = (2 * intersection_unet) / (np.sum(gt_bin) + np.sum(unet_bin) + 1e-5)
                dice_unet_channels.append(dice_unet)
                
                # Compute Dice for SAM2 channel
                intersection_sam2 = np.sum(gt_bin * sam2_bin)
                dice_sam2 = (2 * intersection_sam2) / (np.sum(gt_bin) + np.sum(sam2_bin) + 1e-5)
                dice_sam2_channels.append(dice_sam2)
                
                print(f"  Channel {ch} - UNet Dice: {dice_unet:.4f}, SAM2 Dice: {dice_sam2:.4f}")
            
            avg_dice_unet = np.mean(dice_unet_channels) if dice_unet_channels else 0.0
            avg_dice_sam2 = np.mean(dice_sam2_channels) if dice_sam2_channels else 0.0
            
            print(f"  Average UNet Dice: {avg_dice_unet:.4f}")
            print(f"  Average SAM2 Dice: {avg_dice_sam2:.4f}\n")




print("=== AUTOSAM2 WITH FLEXIBLE ARCHITECTURE LOADED SUCCESSFULLY ===")


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
                 filter_empty=False, use_augmentation=False, 
                 cache_data=False, verbose=True):
        """
        Enhanced BraTS dataset with efficient data loading
        """
        self.root_dir = root_dir
        self.train = train
        self.normalize = normalize
        self.use_augmentation = use_augmentation and train
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
            dummy_shape = (4, 240, 240, 155) 
            return torch.zeros(dummy_shape, dtype=torch.float32), torch.zeros((1, *dummy_shape[1:]), dtype=torch.float32)
    
    def __getitem__(self, idx):
        # Only use MONAI data loader for simplicity
        image, mask = self._load_monai_data(idx)
        
        # Apply preprocessing with improved normalization
        if self.normalize:
            image = preprocess_brats_data(image, normalize=True)
        
        
        # Apply data augmentation in training mode
        if self.use_augmentation:
            image, mask = apply_augmentations(image, mask)
        
        return image, mask

def get_brats_dataloader(root_dir, batch_size=1, train=True, normalize=True, max_samples=None, 
                         num_workers=4, filter_empty=False, use_augmentation=False, 
                         cache_data=False, verbose=True):
    """Create a DataLoader for BraTS dataset with a proper train/validation split"""
    
    # Step 1: Load ALL available data from the training directory
    full_dataset = BraTSDataset(
        root_dir=root_dir, 
        train=True,  # Always load from training directory
        normalize=normalize,
        max_samples=None,  # Don't limit samples in dataset initialization
        filter_empty=filter_empty,
        use_augmentation=False,  # We'll add augmentation later if needed
        cache_data=cache_data,
        verbose=False  # Turn off verbose in dataset to avoid double messages
    )
    
    # Step 2: Determine the total number of samples
    total_samples = len(full_dataset)
    if verbose:
        print(f"Full dataset contains {total_samples} samples")
    
    # Step 3: Create fixed indices for reproducible but random splits
    import random
    random.seed(42)
    
    # Get shuffled indices for the entire dataset
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)  # This ensures random selection
    
    # Step 4: Calculate the split sizes (80% train, 20% validation)
    if max_samples is not None and max_samples < total_samples:
        effective_total = max_samples
    else:
        effective_total = total_samples
    
    # Calculate split sizes
    train_size = int(0.8 * effective_total)
    val_size = min(effective_total - train_size, total_samples - train_size)
    
    # Step 5: Create the actual indices for train and validation (randomly)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    
    # Step 6: Create the appropriate subset
    from torch.utils.data import Subset
    if train:
        dataset = Subset(full_dataset, train_indices)
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
    Enhanced Dice score calculation for BraTS data with correct class mapping.
    
    BraTS label convention:
    - 0: Background
    - 1: Necrotic Tumor Core (NCR)
    - 2: Peritumoral Edema (ED)
    - 4: Enhancing Tumor (ET) (represented at index 3 in tensors)
    
    Regions:
    - ET: Enhancing Tumor (class 4)
    - WT: Whole Tumor (classes 1+2+4)
    - TC: Tumor Core (classes 1+4)
    
    Args:
        y_pred (Tensor): Model predictions of shape [B, C, D, H, W]
        y_true (Tensor): Ground truth masks of shape [B, C, D, H, W]
        
    Returns:
        dict: Dictionary with Dice metrics
    """
    # For raw logits, apply sigmoid first
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    
    # Check if input contains logits or probabilities
    if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
        probs = torch.sigmoid(y_pred)
    else:
        probs = y_pred
    
    # Apply threshold to get binary predictions
    preds = (probs > 0.5).float()
    
    # Initialize results dictionary
    result = {}
    
    # Store all per-sample Dice scores for statistics
    all_dice_scores = {
        'class_1': [],  # NCR
        'class_2': [],  # ED
        'class_4': [],  # ET (at index 3)
        'ET': [],       # Enhancing Tumor (class 4, at index 3)
        'WT': [],       # Whole Tumor (classes 1+2+4)
        'TC': []        # Tumor Core (classes 1+4)
    }
    
    batch_size = y_pred.size(0)
    
    # Calculate per-sample Dice scores for each class and region
    for b in range(batch_size):
        # Calculate per-class Dice (assuming 4 channels: 0=background, 1=NCR, 2=ED, 3=ET)
        # Class 1 (NCR) - at index 1
        pred_ncr = preds[b, 1].reshape(-1)
        true_ncr = y_true[b, 1].reshape(-1)
        if true_ncr.sum() > 0 or pred_ncr.sum() > 0:
            intersection_ncr = (pred_ncr * true_ncr).sum()
            dice_ncr = (2.0 * intersection_ncr / (pred_ncr.sum() + true_ncr.sum() + 1e-5)).item() * 100
            all_dice_scores['class_1'].append(dice_ncr)
            
        # Class 2 (ED) - at index 2
        pred_ed = preds[b, 2].reshape(-1)
        true_ed = y_true[b, 2].reshape(-1)
        if true_ed.sum() > 0 or pred_ed.sum() > 0:
            intersection_ed = (pred_ed * true_ed).sum()
            dice_ed = (2.0 * intersection_ed / (pred_ed.sum() + true_ed.sum() + 1e-5)).item() * 100
            all_dice_scores['class_2'].append(dice_ed)
            
        # Class 4 (ET) - at index 3
        pred_et = preds[b, 3].reshape(-1)
        true_et = y_true[b, 3].reshape(-1)
        if true_et.sum() > 0 or pred_et.sum() > 0:
            intersection_et = (pred_et * true_et).sum()
            dice_et = (2.0 * intersection_et / (pred_et.sum() + true_et.sum() + 1e-5)).item() * 100
            all_dice_scores['class_4'].append(dice_et)
            all_dice_scores['ET'].append(dice_et)  # ET is the same as class 4
        
        # Calculate WT (Whole Tumor) - Classes 1+2+4 (indices 1,2,3)
        pred_wt = (preds[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        true_wt = (y_true[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        if true_wt.sum() > 0 or pred_wt.sum() > 0:
            intersection_wt = (pred_wt * true_wt).sum()
            dice_wt = (2.0 * intersection_wt / (pred_wt.sum() + true_wt.sum() + 1e-5)).item() * 100
            all_dice_scores['WT'].append(dice_wt)
        
        # Calculate TC (Tumor Core) - Classes 1+4 (indices 1,3)
        # Create TC masks by combining NCR and ET
        pred_tc = ((preds[b, 1] + preds[b, 3]) > 0).float().reshape(-1)
        true_tc = ((y_true[b, 1] + y_true[b, 3]) > 0).float().reshape(-1)
        if true_tc.sum() > 0 or pred_tc.sum() > 0:
            intersection_tc = (pred_tc * true_tc).sum()
            dice_tc = (2.0 * intersection_tc / (pred_tc.sum() + true_tc.sum() + 1e-5)).item() * 100
            all_dice_scores['TC'].append(dice_tc)
    
    # Calculate statistics for each class and region
    for key, scores in all_dice_scores.items():
        if scores:  # Only calculate if we have scores
            scores_tensor = torch.tensor(scores)
            result[f'{key}_mean'] = scores_tensor.mean().item()
            result[f'{key}_std'] = scores_tensor.std().item() if len(scores_tensor) > 1 else 0.0
            result[f'{key}_median'] = torch.median(scores_tensor).item()
            
            # Calculate IQR (75th percentile - 25th percentile)
            if len(scores_tensor) > 1:
                q1, q3 = torch.tensor(scores_tensor.tolist()).quantile(torch.tensor([0.25, 0.75])).tolist()
                result[f'{key}_iqr'] = q3 - q1
            else:
                result[f'{key}_iqr'] = 0.0
        else:
            # Set default values if no scores available
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0
            result[f'{key}_median'] = 0.0
            result[f'{key}_iqr'] = 0.0
    
    # Calculate overall mean Dice across all regions
    region_means = [result.get(f'{region}_mean', 0.0) for region in ['ET', 'WT', 'TC']]
    result['mean'] = sum(region_means) / len(region_means) if region_means else 0.0
    
    return result

def calculate_iou(y_pred, y_true, threshold=0.5, eps=1e-6):
    """
    Calculate IoU for BraTS tumor regions with correct class mapping.
    
    BraTS label convention:
    - 0: Background
    - 1: Necrotic Tumor Core (NCR)
    - 2: Peritumoral Edema (ED)
    - 4: Enhancing Tumor (ET) (represented at index 3 in tensors)
    
    Args:
        y_pred (Tensor): Model predictions of shape [B, C, D, H, W]
        y_true (Tensor): Ground truth masks of shape [B, C, D, H, W]
        threshold (float): Threshold for binarization of predictions.
        eps (float): Small value to avoid division by zero.

    Returns:
        dict: IoU metrics for individual classes and tumor regions.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Shape mismatch: y_pred and y_true must have the same shape.")

    # Apply threshold to convert probability predictions into binary masks
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()

    batch_size = y_pred.shape[0]
    result = {}
    
    # For storing per-sample IoU values
    all_iou_values = {
        'class_1': [],  # NCR
        'class_2': [],  # ED
        'class_4': [],  # ET (at index 3)
        'ET': [],       # Enhancing Tumor (class 4)
        'WT': [],       # Whole Tumor (classes 1+2+4)
        'TC': []        # Tumor Core (classes 1+4)
    }
    
    for b in range(batch_size):
        # Class 1 (NCR) - at index 1
        pred_ncr = y_pred_bin[b, 1].reshape(-1)
        true_ncr = y_true[b, 1].reshape(-1)
        intersection_ncr = (pred_ncr * true_ncr).sum()
        union_ncr = (pred_ncr + true_ncr).clamp(0, 1).sum()
        if union_ncr > eps:
            iou_ncr = (intersection_ncr / union_ncr).item() * 100
            all_iou_values['class_1'].append(iou_ncr)
        
        # Class 2 (ED) - at index 2
        pred_ed = y_pred_bin[b, 2].reshape(-1)
        true_ed = y_true[b, 2].reshape(-1)
        intersection_ed = (pred_ed * true_ed).sum()
        union_ed = (pred_ed + true_ed).clamp(0, 1).sum()
        if union_ed > eps:
            iou_ed = (intersection_ed / union_ed).item() * 100
            all_iou_values['class_2'].append(iou_ed)
        
        # Class 4 (ET) - at index 3
        pred_et = y_pred_bin[b, 3].reshape(-1)
        true_et = y_true[b, 3].reshape(-1)
        intersection_et = (pred_et * true_et).sum()
        union_et = (pred_et + true_et).clamp(0, 1).sum()
        if union_et > eps:
            iou_et = (intersection_et / union_et).item() * 100
            all_iou_values['class_4'].append(iou_et)
            all_iou_values['ET'].append(iou_et)  # ET is the same as class 4
        
        # Calculate WT (Whole Tumor) - Classes 1+2+4 (indices 1,2,3)
        pred_wt = (y_pred_bin[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        true_wt = (y_true[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        intersection_wt = (pred_wt * true_wt).sum()
        union_wt = (pred_wt + true_wt).clamp(0, 1).sum()
        if union_wt > eps:
            iou_wt = (intersection_wt / union_wt).item() * 100
            all_iou_values['WT'].append(iou_wt)
        
        # Calculate TC (Tumor Core) - Classes 1+4 (indices 1,3)
        pred_tc = ((y_pred_bin[b, 1] + y_pred_bin[b, 3]) > 0).float().reshape(-1)
        true_tc = ((y_true[b, 1] + y_true[b, 3]) > 0).float().reshape(-1)
        intersection_tc = (pred_tc * true_tc).sum()
        union_tc = (pred_tc + true_tc).clamp(0, 1).sum()
        if union_tc > eps:
            iou_tc = (intersection_tc / union_tc).item() * 100
            all_iou_values['TC'].append(iou_tc)
    
    # Calculate statistics for each category
    for key, values in all_iou_values.items():
        if values:  # Only calculate if we have values
            values_tensor = torch.tensor(values)
            result[f'{key}_mean'] = values_tensor.mean().item()
            result[f'{key}_std'] = values_tensor.std().item() if len(values_tensor) > 1 else 0.0
            result[f'{key}_median'] = torch.median(values_tensor).item()
            
            # Calculate IQR
            if len(values_tensor) > 1:
                q1, q3 = torch.tensor(values_tensor.tolist()).quantile(torch.tensor([0.25, 0.75])).tolist()
                result[f'{key}_iqr'] = q3 - q1
            else:
                result[f'{key}_iqr'] = 0.0
        else:
            # Set default values if no scores available
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0
            result[f'{key}_median'] = 0.0
            result[f'{key}_iqr'] = 0.0
    
    # Calculate mean IoU across regions
    region_means = [result.get(f'{region}_mean', 0.0) for region in ['ET', 'WT', 'TC']]
    result['mean_iou'] = sum(region_means) / len(region_means) if region_means else 0.0
    
    return result

class BraTSDiceLoss(nn.Module):
    """
    Dice Loss specifically designed for BraTS segmentation task.
    Calculates loss for the three tumor regions: ET, WT, and TC.
    
    BraTS label convention:
    - 0: Background
    - 1: Necrotic Tumor Core (NCR)
    - 2: Peritumoral Edema (ED)
    - 4: Enhancing Tumor (ET) (represented at index 3 in tensors)
    """
    def __init__(self, smooth=1.0, region_weights={'ET': 1.0, 'WT': 1.0, 'TC': 1.0}):
        super(BraTSDiceLoss, self).__init__()
        self.smooth = smooth
        self.region_weights = region_weights
        
    def forward(self, y_pred, y_true):
        # Get sigmoid activation for binary predictions
        if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
            probs = torch.sigmoid(y_pred)
        else:
            probs = y_pred
        
        batch_size = y_pred.size(0)
        device = y_pred.device
        
        # Initialize per-region losses
        et_loss = torch.tensor(0.0, device=device, requires_grad=True)
        wt_loss = torch.tensor(0.0, device=device, requires_grad=True)
        tc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Count valid samples for each region
        et_count = 0
        wt_count = 0
        tc_count = 0
        
        for b in range(batch_size):
            # ET (Enhancing Tumor) - Class 4 at index 3
            pred_et = probs[b, 3].reshape(-1)
            true_et = y_true[b, 3].reshape(-1)
            
            # Only calculate if there's something to predict
            if true_et.sum() > 0 or pred_et.sum() > 0:
                intersection_et = (pred_et * true_et).sum()
                dice_et = (2. * intersection_et + self.smooth) / (pred_et.sum() + true_et.sum() + self.smooth)
                et_loss = et_loss + (1. - dice_et)
                et_count += 1
            
            # WT (Whole Tumor) - Classes 1+2+4 (indices 1,2,3)
            # Combine predictions across class channels
            pred_wt = (probs[b, 1:4].sum(dim=0) > 0.5).float().reshape(-1)
            true_wt = (y_true[b, 1:4].sum(dim=0) > 0.5).float().reshape(-1)
            
            if true_wt.sum() > 0 or pred_wt.sum() > 0:
                intersection_wt = (pred_wt * true_wt).sum()
                dice_wt = (2. * intersection_wt + self.smooth) / (pred_wt.sum() + true_wt.sum() + self.smooth)
                wt_loss = wt_loss + (1. - dice_wt)
                wt_count += 1
            
            # TC (Tumor Core) - Classes 1+4 (indices 1,3)
            # Combine predictions for relevant classes
            pred_tc = ((probs[b, 1] + probs[b, 3]) > 0.5).float().reshape(-1)
            true_tc = ((y_true[b, 1] + y_true[b, 3]) > 0.5).float().reshape(-1)
            
            if true_tc.sum() > 0 or pred_tc.sum() > 0:
                intersection_tc = (pred_tc * true_tc).sum()
                dice_tc = (2. * intersection_tc + self.smooth) / (pred_tc.sum() + true_tc.sum() + self.smooth)
                tc_loss = tc_loss + (1. - dice_tc)
                tc_count += 1
        
        # Calculate average loss per region
        if et_count > 0:
            et_loss = et_loss / et_count
        if wt_count > 0:
            wt_loss = wt_loss / wt_count
        if tc_count > 0:
            tc_loss = tc_loss / tc_count
        
        # Apply region weights
        weighted_loss = (
            self.region_weights.get('ET', 1.0) * et_loss +
            self.region_weights.get('WT', 1.0) * wt_loss +
            self.region_weights.get('TC', 1.0) * tc_loss
        )
        
        # Calculate mean across regions that have samples
        valid_regions = (et_count > 0) + (wt_count > 0) + (tc_count > 0)
        if valid_regions > 0:
            return weighted_loss / valid_regions
        else:
            # No valid regions found
            return torch.tensor(0.0, requires_grad=True, device=device)

class BraTSCombinedLoss(nn.Module):
    """
    Combined loss function for BraTS segmentation task.
    Combines Dice loss, BCE loss, and Focal loss with configurable weights.
    
    BraTS label convention:
    - 0: Background
    - 1: Necrotic Tumor Core (NCR)
    - 2: Peritumoral Edema (ED)
    - 4: Enhancing Tumor (ET) (represented at index 3 in tensors)
    """
    def __init__(self, dice_weight=0.7, bce_weight=0.2, focal_weight=0.1, 
                 region_weights={'ET': 1.2, 'WT': 1.0, 'TC': 1.0}):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.brats_dice_loss = BraTSDiceLoss(region_weights=region_weights)
        
        # Set class weights for BCE - higher weight for tumor classes
        # For BraTS with channels 0,1,2,3 representing classes 0,1,2,4
        pos_weight = torch.ones(4)
        pos_weight[1:] = 5.0  # Higher weight for tumor classes (1,2,4)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda() if torch.cuda.is_available() else pos_weight)
        
        # Focal loss for handling class imbalance
        class_weights = torch.tensor([0.1, 1.0, 1.0, 1.2])
        self.focal_loss = nn.CrossEntropyLoss(
            weight=class_weights.cuda() if torch.cuda.is_available() else class_weights
        )

    def forward(self, y_pred, y_true):
        # Dice loss calculation
        dice = self.brats_dice_loss(y_pred, y_true)
        
        # BCE loss calculation
        bce = self.bce_loss(y_pred, y_true)
        
        # Focal loss calculation - convert multi-class mask to class indices
        # First create target with shape [B, H, W, D] with class indices
        target = y_true.argmax(dim=1)
        focal = self.focal_loss(y_pred, target)
        
        # Return weighted sum of losses
        return self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal



def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler=None):
    """
    Training epoch function with comprehensive BraTS metrics.
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
            dice_loss = BraTSDiceLoss()(outputs, masks)

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
                'mean_iou': iou_metrics['mean_iou'],
                'dice_et': dice_metrics.get('ET_mean', 0.0),
                'dice_wt': dice_metrics.get('WT_mean', 0.0),
                'dice_tc': dice_metrics.get('TC_mean', 0.0),
                'iou_et': iou_metrics.get('ET_mean', 0.0),
                'iou_wt': iou_metrics.get('WT_mean', 0.0),
                'iou_tc': iou_metrics.get('TC_mean', 0.0)
            })

            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Learning rate scheduling
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Update progress bar - now include tumor region metrics
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'WT': f"{dice_metrics.get('WT_mean', 0.0):.1f}%",
                'TC': f"{dice_metrics.get('TC_mean', 0.0):.1f}%",
                'ET': f"{dice_metrics.get('ET_mean', 0.0):.1f}%"
            })
            if batch_idx % 40 == 0:
                with torch.no_grad():
                    slice_indices = [38, 77, 124]
                    # הפעל את UNet3D וקבל את כל הפלטים הנדרשים
                    unet_out, mid_features, sam_embeddings, metadata = model.unet3d(images, use_full_decoder=True)
                    # הפעל את SAM2 עם ההקשר התלת-ממדי, תוך שימוש ב־mid_features וב־metadata
                    sam2_out = model.process_volume_with_3d_context(images, mid_features, metadata, device)
                    # קריאה לפונקציה המראה השוואה טקסטואלית (לפי הקוד שהגדרנו)
                    model.visualize_slice_comparison(images, unet_out, sam2_out, masks, slice_indices)

                
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
        'dice_loss': np.mean([m['dice_loss'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_et': np.mean([m['dice_et'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_wt': np.mean([m['dice_wt'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_tc': np.mean([m['dice_tc'] for m in all_metrics]) if all_metrics else 0.0,
        'iou_et': np.mean([m['iou_et'] for m in all_metrics]) if all_metrics else 0.0,
        'iou_wt': np.mean([m['iou_wt'] for m in all_metrics]) if all_metrics else 0.0,
        'iou_tc': np.mean([m['iou_tc'] for m in all_metrics]) if all_metrics else 0.0
    }

    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model with comprehensive BraTS metrics.
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
                dice_loss = BraTSDiceLoss()(outputs, masks)

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
                    'mean_iou': iou_metrics['mean_iou'],
                    'dice_et': dice_metrics.get('ET_mean', 0.0),
                    'dice_wt': dice_metrics.get('WT_mean', 0.0),
                    'dice_tc': dice_metrics.get('TC_mean', 0.0),
                    'iou_et': iou_metrics.get('ET_mean', 0.0),
                    'iou_wt': iou_metrics.get('WT_mean', 0.0),
                    'iou_tc': iou_metrics.get('TC_mean', 0.0)
                })
                                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'WT': f"{dice_metrics.get('WT_mean', 0.0):.1f}%",
                    'TC': f"{dice_metrics.get('TC_mean', 0.0):.1f}%",
                    'ET': f"{dice_metrics.get('ET_mean', 0.0):.1f}%"
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
            'dice_loss': np.mean([m['dice_loss'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_et': np.mean([m['dice_et'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_wt': np.mean([m['dice_wt'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_tc': np.mean([m['dice_tc'] for m in all_metrics]) if all_metrics else 0.0,
            'iou_et': np.mean([m['iou_et'] for m in all_metrics]) if all_metrics else 0.0,
            'iou_wt': np.mean([m['iou_wt'] for m in all_metrics]) if all_metrics else 0.0,
            'iou_tc': np.mean([m['iou_tc'] for m in all_metrics]) if all_metrics else 0.0
        }

        return avg_loss, avg_metrics

def visualize_batch(images, masks, outputs, epoch, prefix=""):
   """
   Visualize a batch of images, masks, and predictions with BraTS class conventions
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
   
   # Apply sigmoid if outputs are logits
   if torch.min(outputs) < 0 or torch.max(outputs) > 1:
       output_slice = torch.sigmoid(outputs[b, :, middle_idx]).cpu().detach().numpy()
   else:
       output_slice = outputs[b, :, middle_idx].cpu().detach().numpy()
   
   # Create figure
   fig, axes = plt.subplots(1, 3, figsize=(12, 4))
   
   # Show FLAIR image
   axes[0].imshow(image_slice[0], cmap='gray')
   axes[0].set_title('FLAIR')
   axes[0].axis('off')
   
   # Create RGB mask for ground truth
   # BraTS: 1=NCR (Blue), 2=ED (Green), 4=ET (Red)
   rgb_mask = np.zeros((mask_slice.shape[1], mask_slice.shape[2], 3))
   rgb_mask[mask_slice[1] > 0.5, :] = [0, 0, 1]  # NCR: Blue
   rgb_mask[mask_slice[2] > 0.5, :] = [0, 1, 0]  # ED: Green
   rgb_mask[mask_slice[3] > 0.5, :] = [1, 0, 0]  # ET: Red
   
   # Show ground truth
   axes[1].imshow(image_slice[0], cmap='gray')
   axes[1].imshow(rgb_mask, alpha=0.5)
   axes[1].set_title('Ground Truth')
   axes[1].axis('off')
   
   # Create RGB mask for prediction
   rgb_pred = np.zeros((output_slice.shape[1], output_slice.shape[2], 3))
   rgb_pred[output_slice[1] > 0.5, :] = [0, 0, 1]  # NCR: Blue
   rgb_pred[output_slice[2] > 0.5, :] = [0, 1, 0]  # ED: Green
   rgb_pred[output_slice[3] > 0.5, :] = [1, 0, 0]  # ET: Red
   
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
   Save training history plot with enhanced BraTS metrics
   """
   plt.figure(figsize=(16, 12))
   
   # Plot losses
   plt.subplot(3, 2, 1)
   plt.plot(history['train_loss'], label='Train Loss')
   plt.plot(history['val_loss'], label='Val Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.title('Training and Validation Loss')
   
   # Plot mean Dice scores
   plt.subplot(3, 2, 2)
   plt.plot(history['train_dice'], label='Train Mean Dice')
   plt.plot(history['val_dice'], label='Val Mean Dice')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Mean Dice Score')
   
   # Plot WT Dice scores
   plt.subplot(3, 2, 3)
   plt.plot(history['train_dice_wt'], label='Train WT')
   plt.plot(history['val_dice_wt'], label='Val WT')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Whole Tumor (WT) Dice Score')
   
   # Plot TC Dice scores
   plt.subplot(3, 2, 4)
   plt.plot(history['train_dice_tc'], label='Train TC')
   plt.plot(history['val_dice_tc'], label='Val TC')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Tumor Core (TC) Dice Score')
   
   # Plot ET Dice scores
   plt.subplot(3, 2, 5)
   plt.plot(history['train_dice_et'], label='Train ET')
   plt.plot(history['val_dice_et'], label='Val ET')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Enhancing Tumor (ET) Dice Score')
   
   # Plot Learning Rate
   plt.subplot(3, 2, 6)
   plt.plot(history['lr'])
   plt.xlabel('Epoch')
   plt.ylabel('Learning Rate')
   plt.yscale('log')
   plt.title('Learning Rate Schedule')
   
   plt.tight_layout()
   plt.savefig(f"results/{filename}")
   plt.close()
   
   # Create a second figure for IoU metrics
   plt.figure(figsize=(16, 12))
   
   # Plot mean IoU
   plt.subplot(2, 2, 1)
   plt.plot(history['train_iou'], label='Train Mean IoU')
   plt.plot(history['val_iou'], label='Val Mean IoU')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Mean IoU')
   
   # Plot WT IoU
   plt.subplot(2, 2, 2)
   plt.plot(history['train_iou_wt'], label='Train WT')
   plt.plot(history['val_iou_wt'], label='Val WT')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Whole Tumor (WT) IoU')
   
   # Plot TC IoU
   plt.subplot(2, 2, 3)
   plt.plot(history['train_iou_tc'], label='Train TC')
   plt.plot(history['val_iou_tc'], label='Val TC')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Tumor Core (TC) IoU')
   
   # Plot ET IoU
   plt.subplot(2, 2, 4)
   plt.plot(history['train_iou_et'], label='Train ET')
   plt.plot(history['val_iou_et'], label='Val ET')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Enhancing Tumor (ET) IoU')
   
   plt.tight_layout()
   plt.savefig(f"results/{filename.replace('.png', '_iou.png')}")
   plt.close()

def preprocess_batch(batch, device=None):
   """
   Preprocess batch for BraTS segmentation with class labels 0, 1, 2, 4
   """
   images, masks = batch
   
   # Convert binary masks to multi-class format if needed
   if masks.shape[1] == 1:
       # For binary masks, create proper BraTS format with classes 0, 1, 2, 4
       multi_class_masks = torch.zeros((masks.shape[0], 4, *masks.shape[2:]), dtype=torch.float32)
       
       # Class 0: Background (where mask is 0)
       multi_class_masks[:, 0] = (masks[:, 0] == 0).float()
       
       # If we have a binary tumor mask, distribute it to the three tumor classes
       # This is a simplified approach when we only have tumor/no-tumor labels
       if torch.sum(masks[:, 0]) > 0:
           # Use percentages of the tumor mask for each class
           # Class 1: NCR (Necrotic tumor core)
           multi_class_masks[:, 1] = (masks[:, 0] * (torch.rand_like(masks[:, 0]) < 0.3)).float()
           
           # Class 2: ED (Peritumoral edema)
           multi_class_masks[:, 2] = (masks[:, 0] * (torch.rand_like(masks[:, 0]) < 0.5)).float()
           
           # Class 4 (at index 3): ET (Enhancing tumor)
           multi_class_masks[:, 3] = (masks[:, 0] * (torch.rand_like(masks[:, 0]) < 0.2)).float()
       
       masks = multi_class_masks
   
   # Ensure mask values are within expected range
   masks = torch.clamp(masks, 0, 1)
   
   # Move to device if specified
   if device is not None:
       images = images.to(device)
       masks = masks.to(device)
   
   return images, masks

        


def train_model(data_path, batch_size=1, epochs=20, learning_rate=1e-3,
               use_mixed_precision=False, test_run=False, reset=True):
    """
    Optimized train function with BraTS-specific metrics
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
    
    # Define loss criterion - use our new BraTS-specific loss
    criterion = BraTSCombinedLoss(dice_weight=0.7, bce_weight=0.2, focal_weight=0.1,
                                region_weights={'ET': 1.2, 'WT': 1.0, 'TC': 1.0})
    
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
        cache_data=False, verbose=True
    )

    val_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=False,
        normalize=True, max_samples=max_samples, num_workers=4,
        filter_empty=False, use_augmentation=False,
        cache_data=False, verbose=True
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
    
    # Initialize history with all BraTS-specific metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_bce': [], 'val_bce': [],
        'train_dice_et': [], 'val_dice_et': [],
        'train_dice_wt': [], 'val_dice_wt': [],
        'train_dice_tc': [], 'val_dice_tc': [],
        'train_iou_et': [], 'val_iou_et': [],
        'train_iou_wt': [], 'val_iou_wt': [],
        'train_iou_tc': [], 'val_iou_tc': [],
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
        history['train_dice_et'].append(train_metrics.get('dice_et', 0.0))
        history['train_dice_wt'].append(train_metrics.get('dice_wt', 0.0))
        history['train_dice_tc'].append(train_metrics.get('dice_tc', 0.0))
        history['train_iou_et'].append(train_metrics.get('iou_et', 0.0))
        history['train_iou_wt'].append(train_metrics.get('iou_wt', 0.0))
        history['train_iou_tc'].append(train_metrics.get('iou_tc', 0.0))
        
        # Validate
        try:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
            print(f"Validation metrics: {val_metrics}")  # Debugging

            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_metrics.get('mean_dice', 0.0))
            history['val_iou'].append(val_metrics.get('mean_iou', 0.0))
            history['val_bce'].append(val_metrics.get('bce_loss', 0.0))
            history['val_dice_et'].append(val_metrics.get('dice_et', 0.0))
            history['val_dice_wt'].append(val_metrics.get('dice_wt', 0.0))
            history['val_dice_tc'].append(val_metrics.get('dice_tc', 0.0))
            history['val_iou_et'].append(val_metrics.get('iou_et', 0.0))
            history['val_iou_wt'].append(val_metrics.get('iou_wt', 0.0))
            history['val_iou_tc'].append(val_metrics.get('iou_tc', 0.0))

        except Exception as e:
            print(f"Error during validation: {e}")
            history['val_loss'].append(float('inf'))
            history['val_dice'].append(0.0)
            history['val_iou'].append(0.0)
            history['val_bce'].append(0.0)
            history['val_dice_et'].append(0.0)
            history['val_dice_wt'].append(0.0)
            history['val_dice_tc'].append(0.0)
            history['val_iou_et'].append(0.0)
            history['val_iou_wt'].append(0.0)
            history['val_iou_tc'].append(0.0)
                
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
        
        # Print BraTS-specific metrics
        print("\nRegional Dice Scores:")
        print(f"  ET: Train={train_metrics.get('dice_et', 0.0):.2f}%, Val={val_metrics.get('dice_et', 0.0):.2f}%")
        print(f"  WT: Train={train_metrics.get('dice_wt', 0.0):.2f}%, Val={val_metrics.get('dice_wt', 0.0):.2f}%")
        print(f"  TC: Train={train_metrics.get('dice_tc', 0.0):.2f}%, Val={val_metrics.get('dice_tc', 0.0):.2f}%")
        
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
    
    # Display final metrics breakdown
    print("\n" + "="*50)
    print("Final metrics breakdown:")
    print("="*50)
    
    # Load the best model metrics
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            
            # Display region-specific metrics
            print("\nDice Scores:")
            for region in ['ET', 'WT', 'TC']:
                if f'{region}_mean' in val_metrics:
                    mean_val = val_metrics.get(f'dice_{region.lower()}', 0.0)
                    median_val = val_metrics.get(f'{region}_median', 0.0)
                    std_val = val_metrics.get(f'{region}_std', 0.0)
                    iqr_val = val_metrics.get(f'{region}_iqr', 0.0)
                    print(f"{region}: Mean={mean_val:.2f}%, Median={median_val:.2f}%, Std={std_val:.2f}%, IQR={iqr_val:.2f}%")
            
            print("\nIoU Scores:")
            for region in ['ET', 'WT', 'TC']:
                if f'{region}_mean' in val_metrics:
                    mean_val = val_metrics.get(f'iou_{region.lower()}', 0.0)
                    median_val = val_metrics.get(f'{region}_median', 0.0)
                    std_val = val_metrics.get(f'{region}_std', 0.0)
                    iqr_val = val_metrics.get(f'{region}_iqr', 0.0)
                    print(f"{region}: Mean={mean_val:.2f}%, Median={median_val:.2f}%, Std={std_val:.2f}%, IQR={iqr_val:.2f}%")
    
    print(f"Training complete! Best overall Dice score: {best_dice:.4f}")

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
    
    args = parser.parse_args()
    
    # Set memory limit for GPU if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.memory_limit)
            print(f"Set GPU memory fraction to {args.memory_limit * 100:.1f}%")
        except Exception as e:
            print(f"Warning: Could not set GPU memory fraction: {e}")

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
