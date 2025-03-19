# Standard library imports
import os
import time
import gc
import logging
from collections import defaultdict

# External libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, binary_erosion, binary_dilation, label, distance_transform_edt

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

# For selecting exactly 30% of slices with center concentration:
def get_strategic_slices(depth, percentage=0.3):
    """
    Select strategic slices making up exactly the requested percentage of total depth
    with higher concentration in the center regions.
    
    Args:
        depth: Total number of slices in the volume
        percentage: Percentage of slices to select (0.0-1.0)
    
    Returns:
        List of selected slice indices
    """
    # Calculate number of slices to select (30% of total)
    num_slices = max(1, int(depth * percentage))
    
    # Create a distribution that favors the middle
    if num_slices <= 5:
        # For very few slices, use simple approach
        return [int(depth * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9][:num_slices]]
    
    # Create three regions with different densities
    center_region = 0.5  # 50% of slices in the center 40% of the volume
    sides_region = 0.3   # 30% of slices in the middle 40% of the volume 
    edges_region = 0.2   # 20% of slices in the outer 20% of the volume
    
    # Calculate slice counts for each region
    center_count = int(num_slices * center_region)
    sides_count = int(num_slices * sides_region)
    edges_count = num_slices - center_count - sides_count
    
    # Generate slice indices for center region (40-60% of depth)
    center_start = int(depth * 0.4)
    center_end = int(depth * 0.6)
    center_slices = []
    if center_count > 0:
        step = (center_end - center_start) / center_count
        center_slices = [int(center_start + i * step) for i in range(center_count)]
    
    # Generate slice indices for sides regions (20-40% and 60-80% of depth)
    side1_start = int(depth * 0.2)
    side1_end = int(depth * 0.4)
    side2_start = int(depth * 0.6)
    side2_end = int(depth * 0.8)
    
    sides_slices = []
    if sides_count > 0:
        sides_per_side = sides_count // 2
        remainder = sides_count % 2
        
        side1_step = (side1_end - side1_start) / (sides_per_side)
        side1_slices = [int(side1_start + i * side1_step) for i in range(sides_per_side)]
        
        side2_step = (side2_end - side2_start) / (sides_per_side + remainder)
        side2_slices = [int(side2_start + i * side2_step) for i in range(sides_per_side + remainder)]
        
        sides_slices = side1_slices + side2_slices
    
    # Generate slice indices for edge regions (0-20% and 80-100% of depth)
    edge1_start = 0
    edge1_end = int(depth * 0.2)
    edge2_start = int(depth * 0.8)
    edge2_end = depth
    
    edges_slices = []
    if edges_count > 0:
        edges_per_side = edges_count // 2
        remainder = edges_count % 2
        
        edge1_step = (edge1_end - edge1_start) / (edges_per_side)
        edge1_slices = [int(edge1_start + i * edge1_step) for i in range(edges_per_side)]
        
        edge2_step = (edge2_end - edge2_start) / (edges_per_side + remainder)
        edge2_slices = [int(edge2_start + i * edge2_step) for i in range(edges_per_side + remainder)]
        
        edges_slices = edge1_slices + edge2_slices
    
    # Combine all slices and sort
    all_slices = sorted(center_slices + sides_slices + edges_slices)
    
    # Ensure we don't have duplicates and stay within bounds
    all_slices = sorted(list(set([min(depth-1, max(0, idx)) for idx in all_slices])))
    
    return all_slices

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
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock3D(base_channels, base_channels)
        )
        
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

        self.dropout = nn.Dropout3d


    def forward(self, x, use_full_decoder=True):
        """
        Forward pass with flexible options and consistent axial orientation handling
        """
        # Get batch dimensions - using consistent naming to avoid confusion
        batch_size, channels, dim1, dim2, dim3 = x.shape
        
        # For axial view, the depth dimension is now at position 2 (after channels)
        # No need to identify it dynamically
        depth_dim_idx = 2
        depth = dim1  # First spatial dimension is depth for axial orientation
        
        # Ultra-defensive slice selection
        # Select only slices that are guaranteed to be within bounds
        max_slice_idx = depth - 1  # Maximum valid index
        
        # Generate slices based on the actual depth
        key_indices = get_strategic_slices(depth, percentage=0.3)
        key_indices.sort()
        
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
        dec_out1 = self.dropout(0.05)(dec_out1)
        
        dec_out2 = self.dec2(dec_out1, x3)
        dec_out2 = self.dropout(0.1)(dec_out2)
        
        # Generate SAM2 embeddings
        sam_embeddings = self.sam_projection(dec_out2)
        
        # Calculate downsampled indices safely
        downsampled_depth = max(1, depth // 4)  # Prevent divide by zero
        ds_key_indices = [min(idx // 4, downsampled_depth-1) for idx in key_indices]
        
        # Store metadata
        metadata = {
            "key_indices": key_indices,
            "ds_key_indices": ds_key_indices,
            "depth_dim_idx": depth_dim_idx,
            "mid_decoder_shape": dec_out2.shape
        }
        
        # If not using full decoder, return mid-decoder features
        if not use_full_decoder:
            return None, dec_out2, sam_embeddings, metadata
        
        # Late decoder stages
        dec_out3 = self.dec3(dec_out2, x2)
        dec_out3 = self.dropout(0.15)(dec_out3)
        
        dec_out4 = self.dec4(dec_out3, x1)
    
        # Final convolution
        logits = self.output_conv(dec_out4)
        
        # Apply sigmoid
        segmentation = F.softmax(logits, dim=1)
        
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
        if probability_maps.shape[1] >= 4:
            # Equal weighting of all tumor classes
            tumor_prob = torch.zeros((height, width), device=probability_maps.device)
            for c in range(1, probability_maps.shape[1]):
                tumor_prob += torch.sigmoid(probability_maps[0, c])
            # Normalize to 0-1 range and increase contrast
            tumor_prob = tumor_prob / (probability_maps.shape[1] - 1) 
            tumor_prob = tumor_prob * 1.5  # Boost signal
            tumor_prob = tumor_prob.cpu().detach().numpy()
        else:
            tumor_prob = torch.sigmoid(probability_maps[0, 1]).cpu().detach().numpy() * 1.5
        
        # Resize to target dimensions if necessary
        if tumor_prob.shape != (height, width):
            tumor_prob = zoom(tumor_prob, (height / tumor_prob.shape[0], width / tumor_prob.shape[1]), order=1)
        
        # Create binary mask from probability map
        binary_mask = tumor_prob > 0.4
        
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
        return volume[:, :, idx]
    
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
            # Extract original slice (always along dimension 2 for axial view)
            orig_slice = input_vol[:, :, slice_idx]
            
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
        
            # Generate point prompts for whole tumor detection
            points, labels, box, _ = self.prompt_generator.generate_prompts(
                enhanced_features, slice_idx, h, w
            )
            
            # Set image in SAM2
            self.sam2.set_image(rgb_image)
            
            # Call SAM2 with both point and box prompts to get whole tumor mask
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
                
                # Strategy 1: Anatomical distribution approach
                # Background (class 0) is inverse of tumor mask
                multi_class_mask[:, 0] = 1.0 - mask_tensor[:, 0]
                
                # Anatomical distribution for tumor regions
                # Class 2 (ED) - Most common, typically surrounds other classes
                multi_class_mask[:, 2] = mask_tensor[:, 0].clone() 
                
                # Get smaller "core" region for NCR (class 1) and ET (class 3)
                # Use erosion via max pooling on inverted mask to simulate
                kernel_size = max(3, int(min(h, w) * 0.05))
                padded_mask = F.pad(mask_tensor, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='constant', value=0)
                eroded_mask = F.max_pool2d(padded_mask, kernel_size=kernel_size, stride=1)
                
                # Further erosion for ET (class 3)
                kernel_size_small = max(3, int(min(h, w) * 0.025))
                padded_eroded = F.pad(eroded_mask, (kernel_size_small//2, kernel_size_small//2, kernel_size_small//2, kernel_size_small//2), mode='constant', value=0)
                double_eroded = F.max_pool2d(padded_eroded, kernel_size=kernel_size_small, stride=1)
                
                # Distribute to classes
                # NCR (class 1) is core minus inner core
                multi_class_mask[:, 1] = eroded_mask - double_eroded
                # ET (class 3) is the innermost region
                multi_class_mask[:, 3] = double_eroded
                
                # Make classes mutually exclusive
                # Enforce only one active class per pixel by taking the argmax
                indices = multi_class_mask.argmax(dim=1, keepdim=True)
                one_hot = torch.zeros_like(multi_class_mask)
                one_hot.scatter_(1, indices, 1.0)
                
                return one_hot
            else:
                # No valid masks found
                logger.warning(f"SAM2 failed to generate masks for slice {slice_idx}")
                return None
                    
        except Exception as e:
            logger.error(f"Error processing slice {slice_idx} with SAM2: {e}")
            return None
    
    def create_3d_from_slices(self, input_shape, sam2_slices, depth_dim_idx, device):
        """Create a 3D volume from 2D slice results with multi-class awareness"""
        # Create empty volume matching the expected output size
        batch_size = input_shape[0]
        output_shape = list(input_shape)
        output_shape[1] = self.num_classes  # Change channel dimension to match num_classes
        
        volume = torch.zeros(output_shape, device=device)
        
        # Insert each slice result into the appropriate position
        for slice_idx, mask in sam2_slices.items():
            if mask is not None:
                # For axial view, slices are always along dimension 2
                volume[:, :, slice_idx] = mask
        
        # Fill gaps between processed slices
        if len(sam2_slices) > 1:
            # Get sorted slice indices
            processed_indices = sorted(sam2_slices.keys())
            
            # For each gap between processed slices
            for i in range(len(processed_indices) - 1):
                current_idx = processed_indices[i]
                next_idx = processed_indices[i + 1]
                
                # Skip if slices are adjacent
                if next_idx - current_idx <= 1:
                    continue
                
                # Interpolate for each gap slice
                start_mask = volume[:, :, current_idx]
                end_mask = volume[:, :, next_idx]
                
                for j in range(current_idx + 1, next_idx):
                    # Calculate interpolation weight
                    alpha = (j - current_idx) / (next_idx - current_idx)
                    
                    # Linear interpolation between masks
                    interp_mask = (1 - alpha) * start_mask + alpha * end_mask
                    
                    # Make mutually exclusive if needed
                    if not self.training:
                        indices = interp_mask.argmax(dim=1, keepdim=True)
                        interp_mask = torch.zeros_like(interp_mask)
                        interp_mask.scatter_(1, indices, 1.0)
                    
                    # Assign interpolated mask
                    volume[:, :, j] = interp_mask
        
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
        slice_data = volume[:, :, idx]
        
        # If we have a batch, just take the first item
        if slice_data.shape[0] > 1:
            slice_data = slice_data[0:1]
        
        return slice_data

    def debug_mask_distribution(self, mask):
    """Debug helper to check mask class distribution"""
    if torch.is_tensor(mask):
        # Get percentage of each class
        total_pixels = torch.prod(torch.tensor(mask.shape[2:]))
        class_percentages = []
        
        for c in range(mask.shape[1]):
            pixels_in_class = torch.sum(mask[:, c] > 0.5)
            percentage = (pixels_in_class / total_pixels) * 100
            class_percentages.append(percentage.item())
        
        print(f"Class distribution: {class_percentages}")
    return mask

    
    def combine_results(self, unet_output, sam2_slices, depth_dim_idx, blend_weight=0.7):
        """Combines UNet and SAM2 outputs with proper multi-class handling"""
        # Start with UNet output 
        combined = unet_output.clone()
        
        # Debug the UNet output distribution
        self.debug_mask_distribution(combined)
        
        # For each slice processed by SAM2
        for slice_idx, mask in sam2_slices.items():
            if mask is not None:
                # Debug SAM2 output
                self.debug_mask_distribution(mask.unsqueeze(0))
                
                # Get UNet slice
                unet_slice = combined[:, :, slice_idx]
                
                # Blend each class separately
                for c in range(combined.shape[1]):
                    # Weighted average of probabilities (not argmax yet)
                    combined[:, c, slice_idx] = blend_weight * unet_slice[:, c] + (1 - blend_weight) * mask[:, c]
        
        # Apply softmax again to ensure valid probability distribution
        # Reshape for softmax
        b, c, *spatial_dims = combined.shape
        flat_shape = (b, c, -1)
        combined_flat = combined.reshape(flat_shape)
        
        # Apply softmax along class dimension
        combined_flat = F.softmax(combined_flat, dim=1)
        
        # Reshape back
        combined = combined_flat.reshape(combined.shape)
        
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
        """Forward pass with multi-class aware processing"""
        start_time = time.time()
        device = x.device
        
        # Flag for SAM2 usage tracking
        self.has_sam2_enabled = self.enable_sam2 and self.has_sam2
        
        # Process with UNet3D to get mid-decoder features
        unet_output, mid_features, sam_embeddings, metadata = self.unet3d(
            x, 
            use_full_decoder=self.enable_unet_decoder
        )
        
        # Store UNet output for use in SAM2 processing
        if self.enable_sam2:
            self.last_unet_output = unet_output
        
        # Mode 1: UNet3D only
        if not self.enable_sam2 or not self.has_sam2:
            self.performance_metrics["total_time"].append(time.time() - start_time)
            return unet_output
        
        # Process slices with SAM2 - now multi-class aware
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
            # Combine UNet output with SAM2 results
            final_output = self.combine_results(
                unet_output, sam2_results, metadata["depth_dim_idx"]
            )
        
        # Update timing metrics
        total_time = time.time() - start_time
        self.performance_metrics["total_time"].append(total_time)
    
        # Store results for visualization/debugging
        self.last_unet_output = unet_output
        self.last_sam2_slices = sam2_results
        self.last_combined_output = final_output
        
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


print("=== AUTOSAM2 WITH FLEXIBLE ARCHITECTURE LOADED SUCCESSFULLY ===")
