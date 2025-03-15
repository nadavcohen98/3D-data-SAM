#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
import logging
import os
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autosam2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoSAM2")

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

class ResidualBlock3D(nn.Module):
    """
    3D convolutional block with residual connections and group normalization.
    """
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
    """
    Encoder block that combines downsampling with residual convolutions.
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super(EncoderBlock3D, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResidualBlock3D(in_channels, out_channels, num_groups)
        )
    
    def forward(self, x):
        return self.encoder(x)

class DecoderBlock3D(nn.Module):
    """
    Decoder block with upsampling and residual convolutions.
    Includes option for trilinear upsampling or transposed convolutions.
    """
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

class EnhancedPartialDecoder3D(nn.Module):
    """
    Enhanced mini-decoder for AutoSAM2 that generates rich embeddings for SAM2.
    This version includes more skip connections and feature enhancement layers
    to better capture 3D context in the embeddings.
    """
    def __init__(self, base_channels=16, trilinear=True):
        super(EnhancedPartialDecoder3D, self).__init__()
        
        # First decoder block (from bottleneck)
        self.dec1 = DecoderBlock3D(base_channels * 16, base_channels * 4, trilinear=trilinear)  # 8 + 8 = 16
        
        # Second decoder block (partial upsampling)
        self.dec2 = DecoderBlock3D(base_channels * 8, base_channels * 2, trilinear=trilinear)   # 4 + 4 = 8
        
        # Feature enhancement layer before projection
        self.feature_enhancement = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism to emphasize important features
        self.channel_attention = ChannelAttention(base_channels * 2)
        
        # Projection to embedding format for SAM2 (256 channels)
        self.embedding_projection = nn.Conv3d(base_channels * 2, 256, kernel_size=1)
        
    def forward(self, x5, x4, x3):
        """
        Forward pass for the enhanced partial decoder
        
        Args:
            x5: Bottleneck features
            x4: Skip connection from encoder level 4
            x3: Skip connection from encoder level 3
            
        Returns:
            Enhanced embeddings ready for SAM2
        """
        # First upsampling
        x = self.dec1(x5, x4)
        
        # Second upsampling
        x = self.dec2(x, x3)
        
        # Apply feature enhancement
        x = self.feature_enhancement(x)
        
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Project to embedding space for SAM2
        embeddings = self.embedding_projection(x)
        
        return embeddings

class ChannelAttention(nn.Module):
    """
    Channel attention module to emphasize important feature channels
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP
        reduced_channels = max(8, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Average pooling
        avg_out = self.shared_mlp(self.avg_pool(x))
        
        # Max pooling
        max_out = self.shared_mlp(self.max_pool(x))
        
        # Fuse and apply sigmoid activation
        attention = self.sigmoid(avg_out + max_out)
        
        # Apply attention weights to input
        return x * attention

class SpatialAttention(nn.Module):
    """
    Spatial attention module to emphasize important spatial locations
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Calculate average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate average and max
        attention_input = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid activation
        attention = self.sigmoid(self.conv(attention_input))
        
        # Apply attention weights to input
        return x * attention

class UNet3DEncoder(nn.Module):
    """
    Enhanced UNet3D Encoder that creates features at different resolutions
    with improved ability to capture 3D context
    """
    def __init__(self, in_channels=4, base_channels=16, slice_interval=1):
        super(UNet3DEncoder, self).__init__()
        
        # Configuration
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.slice_interval = slice_interval  # This should be 1 to process all slices
        
        # Initial convolution block
        self.initial_conv = ResidualBlock3D(in_channels, base_channels)
        
        # Encoder pathway
        self.enc1 = EncoderBlock3D(base_channels, base_channels * 2)
        self.enc2 = EncoderBlock3D(base_channels * 2, base_channels * 4)
        self.enc3 = EncoderBlock3D(base_channels * 4, base_channels * 8)
        self.enc4 = EncoderBlock3D(base_channels * 8, base_channels * 8)  # Keep channel count at 128
        
        # Add spatial attention at different levels
        self.spatial_attention1 = SpatialAttention(kernel_size=7)
        self.spatial_attention2 = SpatialAttention(kernel_size=5)
        self.spatial_attention3 = SpatialAttention(kernel_size=3)
    
    def forward(self, x):
        # Get batch dimensions
        batch_size, channels, dim1, dim2, dim3 = x.shape
        
        # Identify depth dimension (smallest one)
        dims = [dim1, dim2, dim3]
        depth_idx = dims.index(min(dims))
        depth = dims[depth_idx]
        
        # Process ALL slices - we use slice_interval=1
        # Create a list of all indices along the depth dimension
        key_indices = list(range(depth))
        
        # Encoder pathway with attention
        x1 = self.initial_conv(x)  # Full resolution
        x1 = self.spatial_attention1(x1)
        
        x2 = self.enc1(x1)         # 1/2 resolution
        
        x3 = self.enc2(x2)         # 1/4 resolution
        x3 = self.spatial_attention2(x3)
        
        x4 = self.enc3(x3)         # 1/8 resolution
        
        x5 = self.enc4(x4)         # 1/16 resolution
        x5 = self.spatial_attention3(x5)
        
        # Calculate downsampled indices for bottleneck
        downsampled_depth = depth // 16  # After 4 encoder blocks
        ds_key_indices = [min(idx // 16, downsampled_depth-1) for idx in key_indices]
        
        # Store metadata for key slice processing
        metadata = {
            "key_indices": key_indices,
            "ds_key_indices": ds_key_indices,
            "depth_dim_idx": depth_idx,
            "original_shape": x.shape,
            "bottleneck_shape": x5.shape
        }
        
        # Return all encoder features and metadata
        return x1, x2, x3, x4, x5, metadata

class EnhancedEmbeddingProcessor(nn.Module):
    """
    Enhanced processor for converting 3D embeddings into 2D slices suitable for SAM2
    with improved feature extraction and normalization
    """
    def __init__(self, embedding_size=64):
        super(EnhancedEmbeddingProcessor, self).__init__()
        
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Projection to SAM2-compatible embedding format
        self.projection = nn.Conv2d(256, 256, kernel_size=1)
        
        # Adaptive pooling to ensure correct output size (64x64 for SAM2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((embedding_size, embedding_size))
        
        # Normalization for embeddings
        self.norm = nn.GroupNorm(32, 256)
        self.activation = nn.ReLU(inplace=True)
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )
    
    def extract_slice(self, volume, idx, depth_dim=2):
        """Extract a specific slice from a 3D volume"""
        if depth_dim == 0:
            return volume[:, :, idx]
        elif depth_dim == 1:
            return volume[:, :, :, idx]
        else:  # default to dim 2
            return volume[:, :, :, :, idx]
    
    def forward(self, embeddings_3d, slice_idx, depth_dim):
        """
        Process a single slice embedding for SAM2
        
        Args:
            embeddings_3d: 3D embeddings from partial decoder [B, C, D, H, W]
            slice_idx: Index of the slice to process
            depth_dim: Dimension representing depth
            
        Returns:
            Enhanced 2D embedding for SAM2 [B, C, 64, 64]
        """
        # Extract slice
        slice_embedding = self.extract_slice(embeddings_3d, slice_idx, depth_dim)
        
        # Apply multi-scale feature extraction
        feat = self.conv1(slice_embedding)
        feat = self.activation(feat)
        feat = self.conv2(feat)
        feat = self.activation(feat)
        
        # Apply channel attention
        channel_weights = self.channel_attn(feat)
        feat = feat * channel_weights
        
        # Apply projection
        slice_embedding = self.projection(feat)
        
        # Normalize
        slice_embedding = self.norm(slice_embedding)
        slice_embedding = self.activation(slice_embedding)
        
        # Ensure correct size with adaptive pooling
        slice_embedding = self.adaptive_pool(slice_embedding)
        
        return slice_embedding

class FullUNet3D(nn.Module):
    """
    Complete UNet3D model with full encoder and decoder
    Used as a fallback when SAM2 is not available
    """
    def __init__(self, in_channels=4, n_classes=4, base_channels=16, trilinear=True):
        super(FullUNet3D, self).__init__()
        
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
        self.dec1 = DecoderBlock3D(base_channels * 16, base_channels * 4, trilinear=trilinear)  # 8 + 8 = 16
        self.dec2 = DecoderBlock3D(base_channels * 8, base_channels * 2, trilinear=trilinear)   # 4 + 4 = 8
        self.dec3 = DecoderBlock3D(base_channels * 4, base_channels, trilinear=trilinear)       # 2 + 2 = 4
        self.dec4 = DecoderBlock3D(base_channels * 2, base_channels, trilinear=trilinear)       # 1 + 1 = 2
        
        # Final output layer
        self.output_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder pathway
        x1 = self.initial_conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        # Decoder pathway
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        
        # Final convolution
        x = self.output_conv(x)
        
        return x

class AutoSAM2(nn.Module):
    """
    Improved AutoSAM2: Adapting SAM2 for 3D Medical Image Segmentation
    
    This implementation includes:
    1. Enhanced embedding generation for better SAM2 integration
    2. Improved slice processing with multi-point prompts
    3. Better handling of SAM2 inputs with proper preprocessing
    4. Slice-by-slice processing with SAM2 across the entire volume
    5. Compatibility with the existing train.py
    """
    def __init__(self, num_classes=4, base_channels=16, slice_interval=1, 
                 trilinear=True, sam2_model_path=None, enable_sam2=True):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        # Set slice interval to 1 by default to process all slices
        self.slice_interval = 1  # Changed from slice_interval parameter to always use 1
        self.sam2_model_path = sam2_model_path
        self.enable_sam2 = enable_sam2
        
        # Create the UNet3D encoder with improved features
        self.encoder = UNet3DEncoder(
            in_channels=4,
            base_channels=base_channels,
            slice_interval=1  # Always use 1 to process all slices
        )
        
        # Create the enhanced partial decoder for better embeddings
        self.partial_decoder = EnhancedPartialDecoder3D(
            base_channels=base_channels,
            trilinear=trilinear
        )
        
        # Enhanced processor for converting 3D embeddings to 2D slices for SAM2
        self.embedding_processor = EnhancedEmbeddingProcessor(embedding_size=64)
        
        # Initialize SAM2
        self.initialize_sam2()
        
        # Create a fallback UNet3D model for when SAM2 is not available
        # This ensures compatibility with the training pipeline
        self.fallback_model = FullUNet3D(
            in_channels=4, 
            n_classes=num_classes,
            base_channels=base_channels,
            trilinear=trilinear
        )
        
        # Create compatibility with train.py
        # The train.py expects encoder and decoder properties
        self.decoder = lambda x_tuple: x_tuple[0] if isinstance(x_tuple, tuple) else x_tuple
        
        # For fallback compatibility - required by train.py
        self.has_sam2_enabled = False  # Will be set to True during forward pass if SAM2 is used
        
        # Performance tracking
        self.performance_metrics = {
            "sam2_processing_time": [],
            "sam2_slices_processed": 0,
            "encoder_time": [],
            "decoder_time": [],
            "total_time": []
        }
        
        logger.info(f"Initialized AutoSAM2 with {base_channels} base channels, "
                   f"processing ALL slices, SAM2 enabled: {self.has_sam2 if hasattr(self, 'has_sam2') else False}")
        
    def initialize_sam2(self):
        """Initialize SAM2 with improved approach"""
        self.has_sam2 = False
        self.sam2 = None
        
        if not HAS_SAM2 or not self.enable_sam2:
            if not HAS_SAM2:
                logger.warning("SAM2 package not available. Will run in fallback mode.")
            elif not self.enable_sam2:
                logger.info("SAM2 integration disabled by user. Will run in fallback mode.")
            return
            
        try:
            # Try using build_sam2_hf directly
            logger.info("Attempting to initialize SAM2 with build_sam2_hf")
            try:
                model_id = "facebook/sam2-hiera-large"
                logger.info(f"Building SAM2 with model_id: {model_id}")
                sam2_model = build_sam2_hf(model_id)
                self.sam2 = SAM2ImagePredictor(sam2_model)
                self.has_sam2 = True
                logger.info("SAM2 initialized successfully")
                return
            except Exception as e:
                logger.error(f"Failed to initialize SAM2 with build_sam2_hf: {e}")
                
        except Exception as e:
            logger.error(f"Error in SAM2 initialization: {e}")
            self.has_sam2 = False
            self.sam2 = None
            
        # Final logging of SAM2 status
        if self.has_sam2:
            logger.info("SAM2 initialized successfully")
        else:
            logger.warning("SAM2 initialization failed, will use fallback UNet3D")
    
    def generate_multi_point_prompts(self, img_shape):
        """
        Generate multiple point prompts for SAM2 based on the image shape
        
        Args:
            img_shape: Shape of the image (H, W)
            
        Returns:
            point_coords: Numpy array of point coordinates
            point_labels: Numpy array of point labels (1=foreground)
        """
        h, w = img_shape
        
        # Generate a grid of points (5x5 grid)
        grid_size = 5
        h_step = h // grid_size
        w_step = w // grid_size
        
        points = []
        # Center point
        points.append([w//2, h//2])
        
        # Grid points
        for i in range(1, grid_size):
            for j in range(1, grid_size):
                if (i == grid_size//2 and j == grid_size//2):
                    # Skip center point (already added)
                    continue
                # Add point only if it's not too close to the edge
                if i * h_step > h//10 and i * h_step < h*9//10 and j * w_step > w//10 and j * w_step < w*9//10:
                    points.append([j * w_step, i * h_step])
        
        # Convert to numpy arrays
        point_coords = np.array(points)
        point_labels = np.ones(len(points))  # All points are foreground
        
        return point_coords, point_labels
    
    def preprocess_slice_for_sam2(self, img_slice):
        """
        Preprocess a slice for SAM2 with improved normalization and contrast
        
        Args:
            img_slice: Input slice tensor [1, 1, H, W]
            
        Returns:
            Preprocessed numpy array [H, W, 3]
        """
        # Convert tensor to numpy
        img_np = img_slice[0, 0].detach().cpu().numpy()
        
        # Apply contrast enhancement
        p1, p99 = np.percentile(img_np, (1, 99))
        img_np = np.clip((img_np - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        # Convert to 3-channel image
        img_np_3ch = np.stack([img_np, img_np, img_np], axis=2)
        
        return img_np_3ch
    
    def process_slice_with_sam2(self, img_slice, embedding, slice_idx, device):
        """
        Process a single 2D slice with SAM2 using improved preprocessing and multi-point prompts
        
        Args:
            img_slice: Input slice [B, C, H, W]
            embedding: Embedding for SAM2 [B, C, 64, 64]
            slice_idx: Index of the slice for logging
            device: Computation device
            
        Returns:
            SAM2 segmentation mask
        """
        if not self.has_sam2:
            logger.warning(f"SAM2 not available for slice {slice_idx}. Using fallback.")
            # Return placeholder if SAM2 not available
            height, width = img_slice.shape[2:]
            return torch.zeros((1, self.num_classes, height, width), device=device)
            
        # Skip processing if the slice is empty (all zeros or nearly all zeros)
        if img_slice.sum() < 1e-6:
            logger.info(f"Skipping empty slice {slice_idx}")
            height, width = img_slice.shape[2:]
            return torch.zeros((1, self.num_classes, height, width), device=device)
            
        try:
            logger.info(f"Processing slice {slice_idx} with SAM2")
            start_time = time.time()
            
            # Enhanced preprocessing for SAM2
            img_np_3ch = self.preprocess_slice_for_sam2(img_slice)
            
            # Process with SAM2
            try:
                # 1. Set the image on the predictor
                self.sam2.set_image(img_np_3ch)
                
                # 2. Generate multi-point prompts
                point_coords, point_labels = self.generate_multi_point_prompts(img_np_3ch.shape[:2])
                
                # 3. Get prediction from SAM2
                masks, scores, low_res_logits = self.sam2.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True  # Get multiple masks
                )
                
                # 4. Select the best mask based on scores
                best_mask_idx = scores.argmax(0)
                best_mask = masks[best_mask_idx]
                
                # Convert results back to torch tensor
                mask = torch.from_numpy(best_mask).to(device)
                
                # Reshape to [1, 1, H, W] for consistency
                mask = mask.unsqueeze(0).unsqueeze(0)
                
            except Exception as e:
                logger.error(f"Error in SAM2 prediction: {e}")
                # Return empty mask on error
                height, width = img_slice.shape[2:]
                return torch.zeros((1, self.num_classes, height, width), device=device)
            
            # Reshape mask to match expected output format [B, C, H, W]
            height, width = mask.shape[2:]
            multi_class_mask = torch.zeros((1, self.num_classes, height, width), device=device)
            
            # Use the binary mask for all tumor classes (class 1, 2, 3)
            # This approach uses the same mask for all tumor classes
            # For a more advanced approach, we could use different SAM2 prompts for each class
            for c in range(1, self.num_classes):
                multi_class_mask[:, c] = mask[:, 0]
            
            # Track time
            process_time = time.time() - start_time
            self.performance_metrics["sam2_processing_time"].append(process_time)
            self.performance_metrics["sam2_slices_processed"] += 1
            
            logger.info(f"Slice {slice_idx} processed with SAM2 in {process_time:.4f}s")
            
            return multi_class_mask
        
        except Exception as e:
            logger.error(f"Error processing slice {slice_idx} with SAM2: {e}")
            height, width = img_slice.shape[2:]
            return torch.zeros((1, self.num_classes, height, width), device=device)
    
    def improved_volume_from_slices(self, sam2_masks, volume_shape, slice_indices, depth_dim_idx):
        """
        Create a full 3D volume from processed 2D slices with improved interpolation
        
        Args:
            sam2_masks: Dictionary of slice masks {slice_idx: mask}
            volume_shape: Shape of the output volume [B, C, D, H, W]
            slice_indices: List of slice indices that were processed
            depth_dim_idx: Index of the depth dimension
            
        Returns:
            Full 3D volume with segmentation masks
        """
        device = next(iter(sam2_masks.values())).device
        batch_size, num_classes, _, height, width = volume_shape
        depth = volume_shape[2 + depth_dim_idx]  # Get depth from the right dimension
        
        # Create empty volume
        volume = torch.zeros(volume_shape, device=device)
        
        # Fill in the slices we have
        for slice_idx, mask in sam2_masks.items():
            if depth_dim_idx == 0:
                volume[:, :, slice_idx, :, :] = mask
            elif depth_dim_idx == 1:
                volume[:, :, :, slice_idx, :] = mask
            else:  # default to dim 2
                volume[:, :, :, :, slice_idx] = mask
        
        # Improved interpolation between processed slices
        # Sort slice indices for proper interpolation
        sorted_indices = sorted(slice_indices)
        
        # For each pair of adjacent processed slices
        for i in range(len(sorted_indices) - 1):
            start_idx = sorted_indices[i]
            end_idx = sorted_indices[i + 1]
            
            # Skip if they're adjacent
            if end_idx - start_idx <= 1:
                continue
            
            # Get masks for start and end slices
            if depth_dim_idx == 0:
                start_mask = volume[:, :, start_idx, :, :]
                end_mask = volume[:, :, end_idx, :, :]
            elif depth_dim_idx == 1:
                start_mask = volume[:, :, :, start_idx, :]
                end_mask = volume[:, :, :, end_idx, :]
            else:  # default to dim 2
                start_mask = volume[:, :, :, :, start_idx]
                end_mask = volume[:, :, :, :, end_idx]
            
            # Compute distance weights for non-linear interpolation
            # This creates a sigmoid-like weighting instead of linear
            # to better preserve boundaries
            delta = end_idx - start_idx
            
            # Interpolate for each slice in between
            for j in range(start_idx + 1, end_idx):
                # Calculate interpolation weight with sigmoid curve
                # This gives smoother transitions at mask boundaries
                t = (j - start_idx) / delta  # normalized position
                alpha = 1 / (1 + np.exp(-10 * (t - 0.5)))  # sigmoid weighting
                
                # Apply weighted combination
                interp_mask = (1 - alpha) * start_mask + alpha * end_mask
                
                # Apply a threshold to prevent gradient artifacts
                # This ensures clean binary masks after interpolation
                interp_mask = (interp_mask > 0.5).float()
                
                # Assign to volume
                if depth_dim_idx == 0:
                    volume[:, :, j, :, :] = interp_mask
                elif depth_dim_idx == 1:
                    volume[:, :, :, j, :] = interp_mask
                else:  # default to dim 2
                    volume[:, :, :, :, j] = interp_mask
        
        return volume


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
