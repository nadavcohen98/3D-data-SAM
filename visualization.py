# Standard library imports
import os

# External libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import binary_dilation, binary_erosion

def create_custom_colormaps():
    """Create custom colormaps for BraTS segmentation visualization"""
    # Create a custom colormap for tumor regions
    # NCR: Blue (0, 0, 1), ED: Green (0, 1, 0), ET: Red (1, 0, 0)
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Blue, Green, Red
    tumor_cmap = LinearSegmentedColormap.from_list('tumor_cmap', colors, N=3)
    
    # Create a high-contrast colormap for uncertainty visualization
    uncertainty_colors = [(0.9, 0.9, 0.9), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]  # White to Yellow to Orange to Red
    uncertainty_cmap = LinearSegmentedColormap.from_list('uncertainty_cmap', uncertainty_colors, N=256)
    
    return tumor_cmap, uncertainty_cmap

def get_slice_indices(volume, num_slices=5):
    """
    Get strategic slice indices from a 3D volume.
    
    Assumes the input volume has shape [B, C, D, H, W] or similar.
    To be robust, we consider the smallest spatial dimension (among D, H, and W)
    as the 'depth' used for selecting representative slices.
    
    Args:
        volume (Tensor): A 3D volume with shape [B, C, D, H, W] (or similar).
        num_slices (int): Number of slices to select (default is 5).
    
    Returns:
        list: A list of selected slice indices based on percentiles.
    """
    # Extract the spatial dimensions (assumed to be dimensions 2, 3, and 4)
    spatial_dims = volume.shape[2:]
    # Choose the smallest dimension as the effective depth
    depth = min(spatial_dims)
    
    # Calculate slice indices using percentiles (skip boundaries)
    percentiles = [15, 30, 50, 70, 85]
    indices = [max(0, min(depth - 1, int(depth * p / 100))) for p in percentiles]
    
    # Ensure the middle slice is always included
    middle_idx = depth // 2
    if middle_idx not in indices:
        indices[len(indices) // 2] = middle_idx
    
    return indices


def visualize_mri_modalities(image_slice, fig, axs, mri_titles=None):
    """Visualize all MRI modalities for a single slice"""
    if mri_titles is None:
        mri_titles = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    num_modalities = min(4, image_slice.shape[0])
    
    for i in range(num_modalities):
        # Apply contrast enhancement for better visualization
        img = image_slice[i]
        p1, p99 = np.percentile(img, (1, 99))
        img_normalized = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        axs[i].imshow(img_normalized, cmap='gray')
        axs[i].set_title(mri_titles[i])
        axs[i].axis('off')

def create_segmentation_overlay(img_slice, seg_slice, alpha=0.7):
    """Create an RGB overlay of segmentation on grayscale image"""
    # Create RGB version of the grayscale image
    p1, p99 = np.percentile(img_slice, (1, 99))
    img_normalized = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb_img = np.stack([img_normalized]*3, axis=-1)
    
    # Create RGB mask
    h, w = seg_slice.shape[1:]
    rgb_mask = np.zeros((h, w, 3))
    
    # Apply BraTS color convention
    # Class 1 (NCR) - Blue
    if 1 < seg_slice.shape[0]:  # Make sure we have this channel
        rgb_mask[seg_slice[1] > 0.5, :] = [0, 0, 1]  # Blue
    
    # Class 2 (ED) - Green
    if 2 < seg_slice.shape[0]:
        rgb_mask[seg_slice[2] > 0.5, :] = [0, 1, 0]  # Green
    
    # Class 4 (ET) at index 3 - Red
    if 3 < seg_slice.shape[0]:
        rgb_mask[seg_slice[3] > 0.5, :] = [1, 0, 0]  # Red
    
    # Combine the image and mask
    combined = rgb_img * (1 - alpha * (rgb_mask.sum(axis=-1, keepdims=True) > 0)) + rgb_mask * alpha
    
    return combined

def create_difference_map(gt_slice, pred_slice, img_slice):
    """Create a map highlighting differences between ground truth and prediction"""
    # Extract tumor regions (any tumor vs no tumor)
    if gt_slice.shape[0] > 3 and pred_slice.shape[0] > 3:
        gt_any = (gt_slice[1:4].sum(axis=0) > 0.5).astype(float)
        pred_any = (pred_slice[1:4].sum(axis=0) > 0.5).astype(float)
    else:
        gt_any = (gt_slice[1:].sum(axis=0) > 0.5).astype(float)
        pred_any = (pred_slice[1:].sum(axis=0) > 0.5).astype(float)
    
    # Calculate difference map
    diff_map = np.zeros_like(gt_any)
    
    # True Positive: Both GT and Pred have tumor - Green
    true_positive = (gt_any > 0) & (pred_any > 0)
    
    # False Positive: Pred has tumor but GT doesn't - Red
    false_positive = (gt_any == 0) & (pred_any > 0)
    
    # False Negative: GT has tumor but Pred doesn't - Blue
    false_negative = (gt_any > 0) & (pred_any == 0)
    
    # Create RGB difference map
    h, w = diff_map.shape
    diff_rgb = np.zeros((h, w, 3))
    diff_rgb[true_positive] = [0, 1, 0]  # Green for TP
    diff_rgb[false_positive] = [1, 0, 0]  # Red for FP
    diff_rgb[false_negative] = [0, 0, 1]  # Blue for FN
    
    # Create background image
    p1, p99 = np.percentile(img_slice, (1, 99))
    img_normalized = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
    img_rgb = np.stack([img_normalized]*3, axis=-1)
    
    # Combine with background MRI
    alpha = 0.7
    combined = img_rgb * (1 - alpha * (diff_rgb.sum(axis=-1, keepdims=True) > 0)) + diff_rgb * alpha
    
    return combined

def calculate_uncertainty(outputs):
    """Calculate prediction uncertainty from softmax outputs"""
    # Apply softmax to get probabilities
    if torch.min(outputs) < 0 or torch.max(outputs) > 1:
        probs = torch.sigmoid(outputs)
    else:
        probs = outputs
    
    # Each pixel gets a tumor class with highest probability
    max_probs, _ = torch.max(probs[:, 1:], dim=1)  # Only tumor classes (1,2,4)
    
    # Uncertainty is inversely related to max probability
    # 1 - max_prob gives uncertainty (0 = certain, 1 = uncertain)
    uncertainty = 1 - max_probs
    
    return uncertainty.cpu().detach().numpy()

def get_boundary_pixels(mask, dilate=2):
    """Get boundary pixels of a binary mask"""
    from scipy.ndimage import binary_dilation, binary_erosion
    
    # Ensure mask is binary
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # Calculate boundary through dilation and erosion
    dilated = binary_dilation(binary_mask, iterations=dilate)
    eroded = binary_erosion(binary_mask, iterations=dilate)
    
    # Boundary is dilated minus eroded
    boundary = dilated.astype(float) - eroded.astype(float)
    boundary = np.clip(boundary, 0, 1)
    
    return boundary

def visualize_batch_comprehensive(images, masks, outputs, epoch, slice_indices=None, 
                                  mode="hybrid", output_dir="results", prefix=""):
    """
    Enhanced visualization of a batch with comprehensive views
    
    Args:
        images: Input MRI scans [B, C, D, H, W]
        masks: Ground truth masks [B, C, D, H, W]
        outputs: Model predictions [B, C, D, H, W]
        epoch: Current epoch number
        slice_indices: Specific slice indices to visualize (if None, will choose automatically)
        mode: Processing mode ("unet3d", "sam2", or "hybrid")
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/slices", exist_ok=True)
    
    # Get batch item (usually just one in 3D segmentation)
    b = 0
    
    # Convert outputs to probabilities if they are logits
    if torch.min(outputs) < 0 or torch.max(outputs) > 1:
        probs = torch.sigmoid(outputs)
    else:
        probs = outputs
    
    # Determine slice indices if not provided
    if slice_indices is None:
        slice_indices = get_slice_indices(images)
    
    # Process each slice
    for i, slice_idx in enumerate(slice_indices):
        # Extract slice data
        image_slice = images[b, :, slice_idx].cpu().detach().numpy()
        mask_slice = masks[b, :, slice_idx].cpu().detach().numpy()
        output_slice = probs[b, :, slice_idx].cpu().detach().numpy()
        
        # Create main figure (2x3 grid)
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Row 1: MRI Modalities (pick one modality) and segmentations
        ax1 = fig.add_subplot(gs[0, 0])  # FLAIR or T2 modality
        ax2 = fig.add_subplot(gs[0, 1])  # Ground Truth overlay
        ax3 = fig.add_subplot(gs[0, 2])  # Prediction overlay
        
        # Row 2: Difference map, boundary analysis, and uncertainty
        ax4 = fig.add_subplot(gs[1, 0])  # Difference map (TP, FP, FN)
        ax5 = fig.add_subplot(gs[1, 1])  # Boundary comparison
        ax6 = fig.add_subplot(gs[1, 2])  # Uncertainty heatmap
        
        # Show FLAIR MRI (assumed to be at index 3 in BraTS)
        flair_idx = min(3, image_slice.shape[0]-1)  # Use last channel if fewer than 4
        p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
        flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        ax1.imshow(flair_normalized, cmap='gray')
        ax1.set_title(f'FLAIR MRI (Slice {slice_idx})')
        ax1.axis('off')
        
        # Show ground truth overlay
        gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
        ax2.imshow(gt_overlay)
        ax2.set_title('Ground Truth Segmentation')
        ax2.axis('off')
        
        # Show prediction overlay
        pred_overlay = create_segmentation_overlay(image_slice[flair_idx], output_slice)
        ax3.imshow(pred_overlay)
        ax3.set_title(f'Prediction ({mode.upper()})')
        ax3.axis('off')
        
        # Show difference map
        diff_map = create_difference_map(mask_slice, output_slice, image_slice[flair_idx])
        ax4.imshow(diff_map)
        ax4.set_title('Difference Map (TP=Green, FP=Red, FN=Blue)')
        ax4.axis('off')
        
        # Show boundary analysis
        # Get tumor boundaries for GT and prediction
        if mask_slice.shape[0] > 3 and output_slice.shape[0] > 3:
            gt_tumor = (mask_slice[1:4].sum(axis=0) > 0.5).astype(float)
            pred_tumor = (output_slice[1:4].sum(axis=0) > 0.5).astype(float)
        else:
            gt_tumor = (mask_slice[1:].sum(axis=0) > 0.5).astype(float)
            pred_tumor = (output_slice[1:].sum(axis=0) > 0.5).astype(float)
            
        gt_boundary = get_boundary_pixels(gt_tumor)
        pred_boundary = get_boundary_pixels(pred_tumor)
        
        # Create boundary comparison visualization
        boundary_img = np.stack([flair_normalized]*3, axis=-1)
        boundary_img[gt_boundary > 0.5] = [0, 0, 1]  # Blue for GT boundary
        boundary_img[pred_boundary > 0.5] = [1, 0, 0]  # Red for prediction boundary
        # Where boundaries overlap, make purple
        boundary_img[(gt_boundary > 0.5) & (pred_boundary > 0.5)] = [1, 0, 1]  # Purple for overlap
        
        ax5.imshow(boundary_img)
        ax5.set_title('Boundary Analysis (Blue=GT, Red=Pred, Purple=Overlap)')
        ax5.axis('off')
        
        # Show uncertainty heatmap (based on confidence scores)
        if isinstance(outputs, torch.Tensor):
            uncertainty = calculate_uncertainty(outputs[b, :, slice_idx:slice_idx+1])
            uncertainty = uncertainty[0]  # Remove batch dimension
            
            # Create uncertainty visualization with colorbar
            im = ax6.imshow(uncertainty, cmap='hot', vmin=0, vmax=1)
            ax6.set_title('Prediction Uncertainty')
            ax6.axis('off')
            
            # Add colorbar
            divider = make_axes_locatable(ax6)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            # If outputs is not a tensor, show a placeholder
            ax6.text(0.5, 0.5, "Uncertainty data not available", 
                     ha='center', va='center', transform=ax6.transAxes)
            ax6.axis('off')
        
        # Add title with metrics if available
        # You would compute these metrics separately and pass them in
        plt.suptitle(f'Epoch {epoch} - {mode.upper()} Mode - Slice {slice_idx}', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        slice_filename = f"{output_dir}/slices/{prefix}_{mode}_epoch{epoch}_slice{slice_idx}.png"
        plt.savefig(slice_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Create a summary figure with all slices side by side
    summary_fig, summary_axs = plt.subplots(3, len(slice_indices), figsize=(4*len(slice_indices), 12))
    
    # Add a row for each view: MRI, Ground Truth, Prediction
    for i, slice_idx in enumerate(slice_indices):
        # Extract slice data
        image_slice = images[b, :, slice_idx].cpu().detach().numpy()
        mask_slice = masks[b, :, slice_idx].cpu().detach().numpy()
        output_slice = probs[b, :, slice_idx].cpu().detach().numpy()
        
        # Visualize FLAIR MRI (row 0)
        flair_idx = min(3, image_slice.shape[0]-1)
        p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
        flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
        summary_axs[0, i].imshow(flair_normalized, cmap='gray')
        summary_axs[0, i].set_title(f'Slice {slice_idx}')
        summary_axs[0, i].axis('off')
        
        # Visualize Ground Truth (row 1)
        gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
        summary_axs[1, i].imshow(gt_overlay)
        summary_axs[1, i].axis('off')
        
        # Visualize Prediction (row 2)
        pred_overlay = create_segmentation_overlay(image_slice[flair_idx], output_slice)
        summary_axs[2, i].imshow(pred_overlay)
        summary_axs[2, i].axis('off')
    
    # Add row titles
    summary_axs[0, 0].set_ylabel('FLAIR MRI', fontsize=14)
    summary_axs[1, 0].set_ylabel('Ground Truth', fontsize=14)
    summary_axs[2, 0].set_ylabel(f'Prediction ({mode})', fontsize=14)
    
    # Add title
    plt.suptitle(f'Epoch {epoch} - {mode.upper()} Mode - Overview', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    summary_filename = f"{output_dir}/{prefix}_{mode}_epoch{epoch}_summary.png"
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    plt.close(summary_fig)
    
    return {
        'summary_path': summary_filename,
        'slice_paths': [f"{output_dir}/slices/{prefix}_{mode}_epoch{epoch}_slice{slice_idx}.png" 
                        for slice_idx in slice_indices]
    }

def visualize_model_comparison(images, masks, unet_outputs, sam_outputs, hybrid_outputs, 
                              epoch, slice_indices=None, output_dir="results", prefix=""):
    """
    Visualize side-by-side comparison of different model outputs
    
    Args:
        images: Input MRI scans [B, C, D, H, W]
        masks: Ground truth masks [B, C, D, H, W]
        unet_outputs: UNet3D predictions [B, C, D, H, W]
        sam_outputs: SAM2 predictions [B, C, D, H, W]
        hybrid_outputs: Hybrid model predictions [B, C, D, H, W]
        epoch: Current epoch number
        slice_indices: Specific slice indices to visualize (if None, will choose automatically)
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/comparison", exist_ok=True)
    
    # Get batch item (usually just one in 3D segmentation)
    b = 0
    
    # Convert outputs to probabilities if they are logits
    if torch.is_tensor(unet_outputs) and (torch.min(unet_outputs) < 0 or torch.max(unet_outputs) > 1):
        unet_probs = torch.sigmoid(unet_outputs)
    else:
        unet_probs = unet_outputs
        
    if torch.is_tensor(sam_outputs) and (torch.min(sam_outputs) < 0 or torch.max(sam_outputs) > 1):
        sam_probs = torch.sigmoid(sam_outputs)
    else:
        sam_probs = sam_outputs
        
    if torch.is_tensor(hybrid_outputs) and (torch.min(hybrid_outputs) < 0 or torch.max(hybrid_outputs) > 1):
        hybrid_probs = torch.sigmoid(hybrid_outputs)
    else:
        hybrid_probs = hybrid_outputs
    
    # Determine slice indices if not provided
    if slice_indices is None:
        slice_indices = get_slice_indices(images)
    
    # Process each slice
    for i, slice_idx in enumerate(slice_indices):
        # Extract slice data
        image_slice = images[b, :, slice_idx].cpu().detach().numpy()
        mask_slice = masks[b, :, slice_idx].cpu().detach().numpy()
        
        # Extract outputs from each model (with error handling)
        unet_slice = unet_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(unet_probs) else None
        sam_slice = sam_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(sam_probs) else None
        hybrid_slice = hybrid_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(hybrid_probs) else None
        
        # Create figure (2x3 grid)
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Row 1: MRI input, Ground Truth, Difference maps
        ax1 = fig.add_subplot(gs[0, 0])  # FLAIR MRI
        ax2 = fig.add_subplot(gs[0, 1])  # Ground Truth
        ax3 = fig.add_subplot(gs[0, 2])  # Difference map (overlay all models)
        
        # Row 2: Model predictions (UNet3D, SAM2, Hybrid)
        ax4 = fig.add_subplot(gs[1, 0])  # UNet3D output
        ax5 = fig.add_subplot(gs[1, 1])  # SAM2 output
        ax6 = fig.add_subplot(gs[1, 2])  # Hybrid output
        
        # Show FLAIR MRI
        flair_idx = min(3, image_slice.shape[0]-1)  # Use last channel if fewer than 4
        p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
        flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        ax1.imshow(flair_normalized, cmap='gray')
        ax1.set_title(f'FLAIR MRI (Slice {slice_idx})')
        ax1.axis('off')
        
        # Show ground truth
        gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
        ax2.imshow(gt_overlay)
        ax2.set_title('Ground Truth Segmentation')
        ax2.axis('off')
        
        # Create difference map visualization
        # This will show where models agree/disagree with each other
        h, w = image_slice.shape[1:]
        diff_rgb = np.zeros((h, w, 3))
        
        # Add visualization for model comparison
        if unet_slice is not None and sam_slice is not None and hybrid_slice is not None:
            # Extract tumor masks (any tumor class)
            unet_tumor = (unet_slice[1:4].sum(axis=0) > 0.5).astype(bool)
            sam_tumor = (sam_slice[1:4].sum(axis=0) > 0.5).astype(bool)
            hybrid_tumor = (hybrid_slice[1:4].sum(axis=0) > 0.5).astype(bool)
            gt_tumor = (mask_slice[1:4].sum(axis=0) > 0.5).astype(bool)
            
            # Agreement regions between models
            all_agree = unet_tumor & sam_tumor & hybrid_tumor  # All models predict tumor
            all_agree_neg = ~unet_tumor & ~sam_tumor & ~hybrid_tumor  # All models predict background
            
            # Partial agreement regions
            unet_sam_agree = unet_tumor & sam_tumor & ~hybrid_tumor
            unet_hybrid_agree = unet_tumor & ~sam_tumor & hybrid_tumor
            sam_hybrid_agree = ~unet_tumor & sam_tumor & hybrid_tumor
            
            # Disagreement regions (only one model predicts tumor)
            only_unet = unet_tumor & ~sam_tumor & ~hybrid_tumor
            only_sam = ~unet_tumor & sam_tumor & ~hybrid_tumor
            only_hybrid = ~unet_tumor & ~sam_tumor & hybrid_tumor
            
            # Add colors for different agreement patterns
            diff_rgb[all_agree & gt_tumor] = [0, 1, 0]  # Green: All models correct
            diff_rgb[all_agree & ~gt_tumor] = [1, 0, 0]  # Red: All models wrong
            
            diff_rgb[only_unet & gt_tumor] = [0, 0, 1]  # Blue: Only UNet correct
            diff_rgb[only_sam & gt_tumor] = [1, 0.5, 0]  # Orange: Only SAM correct
            diff_rgb[only_hybrid & gt_tumor] = [1, 0, 1]  # Purple: Only Hybrid correct
            
            diff_rgb[all_agree_neg & ~gt_tumor] = [0.8, 0.8, 0.8]  # Light gray: All correctly predict background
            
            # Create background image
            img_rgb = np.stack([flair_normalized]*3, axis=-1)
            
            # Combine with background MRI
            alpha = 0.7
            combined = img_rgb * (1 - alpha * (diff_rgb.sum(axis=-1, keepdims=True) > 0)) + diff_rgb * alpha
            
            ax3.imshow(combined)
            ax3.set_title('Model Comparison')
            ax3.axis('off')
        else:
            ax3.text(0.5, 0.5, "Comparison data not available", 
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        # Show UNet3D prediction
        if unet_slice is not None:
            unet_overlay = create_segmentation_overlay(image_slice[flair_idx], unet_slice)
            ax4.imshow(unet_overlay)
            ax4.set_title('UNet3D Prediction')
            ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, "UNet3D output not available", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        # Show SAM2 prediction
        if sam_slice is not None:
            sam_overlay = create_segmentation_overlay(image_slice[flair_idx], sam_slice)
            ax5.imshow(sam_overlay)
            ax5.set_title('SAM2 Prediction')
            ax5.axis('off')
        else:
            ax5.text(0.5, 0.5, "SAM2 output not available", 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.axis('off')
        
        # Show Hybrid prediction
        if hybrid_slice is not None:
            hybrid_overlay = create_segmentation_overlay(image_slice[flair_idx], hybrid_slice)
            ax6.imshow(hybrid_overlay)
            ax6.set_title('Hybrid Prediction')
            ax6.axis('off')
        else:
            ax6.text(0.5, 0.5, "Hybrid output not available", 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.axis('off')
        
        # Add title with information and metrics
        plt.suptitle(f'Epoch {epoch} - Model Comparison - Slice {slice_idx}', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        comp_filename = f"{output_dir}/comparison/{prefix}_comparison_epoch{epoch}_slice{slice_idx}.png"
        plt.savefig(comp_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Create a summary grid of all slices comparing models
    fig, axs = plt.subplots(len(slice_indices), 4, figsize=(16, 4*len(slice_indices)))
    
    # If only one slice, wrap axes in a list for consistent indexing
    if len(slice_indices) == 1:
        axs = [axs]
    
    # For each slice, show GT, UNet, SAM2, and Hybrid results
    for i, slice_idx in enumerate(slice_indices):
        # Extract slice data
        image_slice = images[b, :, slice_idx].cpu().detach().numpy()
        mask_slice = masks[b, :, slice_idx].cpu().detach().numpy()
        
        flair_idx = min(3, image_slice.shape[0]-1)
        
        # Ground Truth (column 0)
        gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
        axs[i, 0].imshow(gt_overlay)
        axs[i, 0].set_title(f'GT (Slice {slice_idx})' if i == 0 else f'Slice {slice_idx}')
        axs[i, 0].axis('off')
        
        # UNet3D (column 1)
        if torch.is_tensor(unet_probs):
            unet_slice = unet_probs[b, :, slice_idx].cpu().detach().numpy()
            unet_overlay = create_segmentation_overlay(image_slice[flair_idx], unet_slice)
            axs[i, 1].imshow(unet_overlay)
        else:
            axs[i, 1].text(0.5, 0.5, "No Data", ha='center', va='center')
        axs[i, 1].set_title('UNet3D' if i == 0 else '')
        axs[i, 1].axis('off')
        
        # SAM2 (column 2)
        if torch.is_tensor(sam_probs):
            sam_slice = sam_probs[b, :, slice_idx].cpu().detach().numpy()
            sam_overlay = create_segmentation_overlay(image_slice[flair_idx], sam_slice)
            axs[i, 2].imshow(sam_overlay)
        else:
            axs[i, 2].text(0.5, 0.5, "No Data", ha='center', va='center')
        axs[i, 2].set_title('SAM2' if i == 0 else '')
        axs[i, 2].axis('off')
        
        # Hybrid (column 3)
        if torch.is_tensor(hybrid_probs):
            hybrid_slice = hybrid_probs[b, :, slice_idx].cpu().detach().numpy()
            hybrid_overlay = create_segmentation_overlay(image_slice[flair_idx], hybrid_slice)
            axs[i, 3].imshow(hybrid_overlay)
        else:
            axs[i, 3].text(0.5, 0.5, "No Data", ha='center', va='center')
        axs[i, 3].set_title('Hybrid' if i == 0 else '')
        axs[i, 3].axis('off')
    
    # Add overall title
    plt.suptitle(f'Epoch {epoch} - Model Comparison Overview', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    summary_comp_filename = f"{output_dir}/{prefix}_model_comparison_epoch{epoch}.png"
    plt.savefig(summary_comp_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'summary_path': summary_comp_filename,
        'slice_paths': [f"{output_dir}/comparison/{prefix}_comparison_epoch{epoch}_slice{slice_idx}.png" 
                      for slice_idx in slice_indices]
    }

def analyze_tumor_regions_by_class(images, masks, outputs, epoch, slice_indices=None, 
                                 output_dir="results", prefix=""):
    """
    Create visualizations focused on specific tumor regions (WT, TC, ET)
    
    Args:
        images: Input MRI scans [B, C, D, H, W]
        masks: Ground truth masks [B, C, D, H, W]
        outputs: Model predictions [B, C, D, H, W]
        epoch: Current epoch number
        slice_indices: Specific slice indices to visualize
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/regions", exist_ok=True)
    
    # Get batch item
    b = 0
    
    # Convert to probabilities if needed
    if torch.min(outputs) < 0 or torch.max(outputs) > 1:
        probs = torch.sigmoid(outputs)
    else:
        probs = outputs
    
    # Determine slice indices if not provided
    if slice_indices is None:
        slice_indices = get_slice_indices(images)
    
    # Create figures for each tumor region
    for i, slice_idx in enumerate(slice_indices):
        # Extract slice data
        image_slice = images[b, :, slice_idx].cpu().detach().numpy()
        mask_slice = masks[b, :, slice_idx].cpu().detach().numpy()
        output_slice = probs[b, :, slice_idx].cpu().detach().numpy()
        
        # Get FLAIR image for background
        flair_idx = min(3, image_slice.shape[0]-1)
        p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
        flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
        flair_rgb = np.stack([flair_normalized]*3, axis=-1)
        
        # Create figure with three rows (one for each tumor region)
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        
        # Row 1: WT (Whole Tumor) - Classes 1+2+4 (indices 1,2,3)
        # Create binary masks for WT
        gt_wt = (mask_slice[1:4].sum(axis=0) > 0.5).astype(float)
        pred_wt = (output_slice[1:4].sum(axis=0) > 0.5).astype(float)
        
        # WT Ground Truth
        wt_gt_rgb = flair_rgb.copy()
        wt_gt_rgb[gt_wt > 0.5] = [0, 1, 0]  # Green for WT
        axs[0, 0].imshow(wt_gt_rgb)
        axs[0, 0].set_title('WT Ground Truth')
        axs[0, 0].axis('off')
        
        # WT Prediction
        wt_pred_rgb = flair_rgb.copy()
        wt_pred_rgb[pred_wt > 0.5] = [0, 1, 0]  # Green for WT
        axs[0, 1].imshow(wt_pred_rgb)
        axs[0, 1].set_title('WT Prediction')
        axs[0, 1].axis('off')
        
        # WT Difference
        wt_diff_rgb = flair_rgb.copy()
        wt_tp = (gt_wt > 0.5) & (pred_wt > 0.5)  # True positive
        wt_fp = (gt_wt < 0.5) & (pred_wt > 0.5)  # False positive
        wt_fn = (gt_wt > 0.5) & (pred_wt < 0.5)  # False negative
        
        wt_diff_rgb[wt_tp] = [0, 1, 0]  # Green for TP
        wt_diff_rgb[wt_fp] = [1, 0, 0]  # Red for FP
        wt_diff_rgb[wt_fn] = [0, 0, 1]  # Blue for FN
        
        axs[0, 2].imshow(wt_diff_rgb)
        axs[0, 2].set_title('WT Difference')
        axs[0, 2].axis('off')
        
        # Row 2: TC (Tumor Core) - Classes 1+4 (indices 1,3)
        # Create binary masks for TC
        gt_tc = ((mask_slice[1] + mask_slice[3]) > 0.5).astype(float)
        pred_tc = ((output_slice[1] + output_slice[3]) > 0.5).astype(float)
        
        # TC Ground Truth
        tc_gt_rgb = flair_rgb.copy()
        tc_gt_rgb[gt_tc > 0.5] = [1, 0.7, 0]  # Orange for TC
        axs[1, 0].imshow(tc_gt_rgb)
        axs[1, 0].set_title('TC Ground Truth')
        axs[1, 0].axis('off')
        
        # TC Prediction
        tc_pred_rgb = flair_rgb.copy()
        tc_pred_rgb[pred_tc > 0.5] = [1, 0.7, 0]  # Orange for TC
        axs[1, 1].imshow(tc_pred_rgb)
        axs[1, 1].set_title('TC Prediction')
        axs[1, 1].axis('off')
        
        # TC Difference
        tc_diff_rgb = flair_rgb.copy()
        tc_tp = (gt_tc > 0.5) & (pred_tc > 0.5)  # True positive
        tc_fp = (gt_tc < 0.5) & (pred_tc > 0.5)  # False positive
        tc_fn = (gt_tc > 0.5) & (pred_tc < 0.5)  # False negative
        
        tc_diff_rgb[tc_tp] = [0, 1, 0]  # Green for TP
        tc_diff_rgb[tc_fp] = [1, 0, 0]  # Red for FP
        tc_diff_rgb[tc_fn] = [0, 0, 1]  # Blue for FN
        
        axs[1, 2].imshow(tc_diff_rgb)
        axs[1, 2].set_title('TC Difference')
        axs[1, 2].axis('off')
        
        # Row 3: ET (Enhancing Tumor) - Class 4 (index 3)
        # Create binary masks for ET
        gt_et = (mask_slice[3] > 0.5).astype(float)
        pred_et = (output_slice[3] > 0.5).astype(float)
        
        # ET Ground Truth
        et_gt_rgb = flair_rgb.copy()
        et_gt_rgb[gt_et > 0.5] = [1, 0, 0]  # Red for ET
        axs[2, 0].imshow(et_gt_rgb)
        axs[2, 0].set_title('ET Ground Truth')
        axs[2, 0].axis('off')
        
        # ET Prediction
        et_pred_rgb = flair_rgb.copy()
        et_pred_rgb[pred_et > 0.5] = [1, 0, 0]  # Red for ET
        axs[2, 1].imshow(et_pred_rgb)
        axs[2, 1].set_title('ET Prediction')
        axs[2, 1].axis('off')
        
        # ET Difference
        et_diff_rgb = flair_rgb.copy()
        et_tp = (gt_et > 0.5) & (pred_et > 0.5)  # True positive
        et_fp = (gt_et < 0.5) & (pred_et > 0.5)  # False positive
        et_fn = (gt_et > 0.5) & (pred_et < 0.5)  # False negative
        
        et_diff_rgb[et_tp] = [0, 1, 0]  # Green for TP
        et_diff_rgb[et_fp] = [1, 0, 0]  # Red for FP
        et_diff_rgb[et_fn] = [0, 0, 1]  # Blue for FN
        
        axs[2, 2].imshow(et_diff_rgb)
        axs[2, 2].set_title('ET Difference')
        axs[2, 2].axis('off')
        
        # Add title
        plt.suptitle(f'Epoch {epoch} - Tumor Region Analysis - Slice {slice_idx}', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        region_filename = f"{output_dir}/regions/{prefix}_regions_epoch{epoch}_slice{slice_idx}.png"
        plt.savefig(region_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return {
        'region_paths': [f"{output_dir}/regions/{prefix}_regions_epoch{epoch}_slice{slice_idx}.png" 
                       for slice_idx in slice_indices]
    }
