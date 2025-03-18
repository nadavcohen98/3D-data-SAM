import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import binary_dilation, binary_erosion

def get_slice_indices(volume, num_slices=5):
    """Get strategic slice indices from a 3D volume"""
    depth = volume.shape[2]  # Assuming shape is [B, C, D, H, W]
    
    # Calculate percentiles to get representative slices
    # Skip very beginning and end which often have less information
    percentiles = [15, 30, 50, 70, 85]
    indices = [max(0, min(depth-1, int(depth * p / 100))) for p in percentiles]
    
    # Always include middle slice
    middle_idx = depth // 2
    if middle_idx not in indices:
        indices[len(indices)//2] = middle_idx
    
    return indices

def create_pastel_colormaps():
    """Create custom pastel colormaps for medical visualization"""
    # Create a pastel colormap for tumor regions
    tumor_colors = [
        (0.6, 0.6, 0.95),  # Pastel blue for NCR
        (0.95, 0.95, 0.6),  # Pastel yellow for ED
        (0.95, 0.6, 0.95)   # Pastel pink for ET
    ]
    tumor_cmap = LinearSegmentedColormap.from_list('tumor_pastel', tumor_colors, N=3)
    
    # Difference map colormap (green, red, blue with more transparency)
    diff_colors = [
        (0.7, 0.9, 0.7),  # Pastel green for TP
        (0.9, 0.7, 0.7),  # Pastel red for FP
        (0.7, 0.7, 0.9)   # Pastel blue for FN
    ]
    diff_cmap = LinearSegmentedColormap.from_list('diff_pastel', diff_colors, N=3)
    
    # Boundary analysis colormap
    boundary_colors = [
        (0.7, 0.7, 0.9),  # Pastel blue for GT
        (0.9, 0.7, 0.7),  # Pastel red for Pred
        (0.9, 0.7, 0.9)   # Pastel purple for overlap
    ]
    boundary_cmap = LinearSegmentedColormap.from_list('boundary_pastel', boundary_colors, N=3)
    
    return tumor_cmap, diff_cmap, boundary_cmap

def create_segmentation_overlay(img_slice, seg_slice, alpha=0.6):
    """Create an RGB overlay of segmentation on grayscale image with pastel colors"""
    # Create RGB version of the grayscale image
    p1, p99 = np.percentile(img_slice, (1, 99))
    img_normalized = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb_img = np.stack([img_normalized]*3, axis=-1)
    
    # Create RGB mask with pastel colors
    h, w = seg_slice.shape[1:]
    rgb_mask = np.zeros((h, w, 3))
    
    # Apply BraTS color convention with pastel colors
    # Class 1 (NCR) - Pastel Blue
    if 1 < seg_slice.shape[0]:  # Make sure we have this channel
        rgb_mask[seg_slice[1] > 0.5, :] = [0.6, 0.6, 0.95]  # Pastel blue
    
    # Class 2 (ED) - Pastel Yellow
    if 2 < seg_slice.shape[0]:
        rgb_mask[seg_slice[2] > 0.5, :] = [0.95, 0.95, 0.6]  # Pastel yellow
    
    # Class 4 (ET) at index 3 - Pastel Pink
    if 3 < seg_slice.shape[0]:
        rgb_mask[seg_slice[3] > 0.5, :] = [0.95, 0.6, 0.95]  # Pastel pink
    
    # Combine the image and mask with softer blending
    mask_present = (rgb_mask.sum(axis=-1, keepdims=True) > 0)
    combined = rgb_img * (1 - alpha * mask_present) + rgb_mask * alpha
    
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
    
    # Calculate difference map with pastel colors
    h, w = gt_any.shape
    diff_rgb = np.zeros((h, w, 3))
    
    # True Positive - Pastel Green
    true_positive = (gt_any > 0) & (pred_any > 0)
    diff_rgb[true_positive] = [0.7, 0.9, 0.7]  # Pastel green
    
    # False Positive - Pastel Red
    false_positive = (gt_any == 0) & (pred_any > 0)
    diff_rgb[false_positive] = [0.9, 0.7, 0.7]  # Pastel red
    
    # False Negative - Pastel Blue
    false_negative = (gt_any > 0) & (pred_any == 0)
    diff_rgb[false_negative] = [0.7, 0.7, 0.9]  # Pastel blue
    
    # Create background image
    p1, p99 = np.percentile(img_slice, (1, 99))
    img_normalized = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
    img_rgb = np.stack([img_normalized]*3, axis=-1)
    
    # Combine with background MRI with softer blending
    alpha = 0.6
    diff_present = (diff_rgb.sum(axis=-1, keepdims=True) > 0)
    combined = img_rgb * (1 - alpha * diff_present) + diff_rgb * alpha
    
    return combined

def get_boundary_pixels(mask, dilate=2):
    """Get boundary pixels of a binary mask"""
    # Ensure mask is binary
    binary_mask = (mask > 0.5).astype(np.float32)
    
    # Calculate boundary through dilation and erosion
    dilated = binary_dilation(binary_mask, iterations=dilate)
    eroded = binary_erosion(binary_mask, iterations=dilate)
    
    # Boundary is dilated minus eroded
    boundary = dilated.astype(float) - eroded.astype(float)
    boundary = np.clip(boundary, 0, 1)
    
    return boundary

def visualize_batch_comprehensive(images, masks, outputs, epoch, mode="hybrid", prefix=""):
    """
    Enhanced visualization of a batch with comprehensive views
    
    Args:
        images: Input MRI scans [B, C, D, H, W]
        masks: Ground truth masks [B, C, D, H, W]
        outputs: Model predictions [B, C, D, H, W]
        epoch: Current epoch number
        mode: Processing mode ("unet3d", "sam2", or "hybrid")
        prefix: Prefix for saved files
    """
    # Create output directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/slices", exist_ok=True)
    
    # Get batch item (usually just one in 3D segmentation)
    b = 0
    
    # Convert outputs to probabilities if they are logits
    if torch.min(outputs) < 0 or torch.max(outputs) > 1:
        probs = torch.sigmoid(outputs)
    else:
        probs = outputs
    
    # Determine slice indices 
    slice_indices = get_slice_indices(images, num_slices=5)
    
    # Set up figure properties for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150
    })
    
    # Process each slice
    for i, slice_idx in enumerate(slice_indices):
        try:
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
            
            # Row 2: Difference map, boundary analysis, and class-specific view
            ax4 = fig.add_subplot(gs[1, 0])  # Difference map (TP, FP, FN)
            ax5 = fig.add_subplot(gs[1, 1])  # Boundary comparison
            ax6 = fig.add_subplot(gs[1, 2])  # ET class analysis
            
            # Show FLAIR MRI (assumed to be at index 3 in BraTS)
            flair_idx = min(3, image_slice.shape[0]-1)  # Use last channel if fewer than 4
            p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
            flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
            
            ax1.imshow(flair_normalized, cmap='gray')
            ax1.set_title(f'FLAIR MRI (Slice {slice_idx})')
            ax1.axis('off')
            
            # Show ground truth overlay with pastel colors
            gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
            ax2.imshow(gt_overlay)
            ax2.set_title('Ground Truth Segmentation')
            ax2.axis('off')
            
            # Show prediction overlay with pastel colors
            pred_overlay = create_segmentation_overlay(image_slice[flair_idx], output_slice)
            ax3.imshow(pred_overlay)
            ax3.set_title(f'Prediction ({mode.upper()})')
            ax3.axis('off')
            
            # Show difference map with pastel colors
            diff_map = create_difference_map(mask_slice, output_slice, image_slice[flair_idx])
            ax4.imshow(diff_map)
            ax4.set_title('Difference Map (TP=Green, FP=Red, FN=Blue)')
            ax4.axis('off')
            
            # Show boundary analysis with pastel colors
            # Get tumor boundaries for GT and prediction
            if mask_slice.shape[0] > 3 and output_slice.shape[0] > 3:
                gt_tumor = (mask_slice[1:4].sum(axis=0) > 0.5).astype(float)
                pred_tumor = (output_slice[1:4].sum(axis=0) > 0.5).astype(float)
            else:
                gt_tumor = (mask_slice[1:].sum(axis=0) > 0.5).astype(float)
                pred_tumor = (output_slice[1:].sum(axis=0) > 0.5).astype(float)
                
            gt_boundary = get_boundary_pixels(gt_tumor)
            pred_boundary = get_boundary_pixels(pred_tumor)
            
            # Create boundary comparison visualization with pastel colors
            boundary_img = np.stack([flair_normalized]*3, axis=-1)
            # Apply pastel colors and higher transparency
            boundary_img[gt_boundary > 0.5] = [0.7, 0.7, 0.9]  # Pastel blue for GT boundary
            boundary_img[pred_boundary > 0.5] = [0.9, 0.7, 0.7]  # Pastel red for prediction boundary
            # Where boundaries overlap, make pastel purple
            boundary_img[(gt_boundary > 0.5) & (pred_boundary > 0.5)] = [0.9, 0.7, 0.9]
            
            ax5.imshow(boundary_img)
            ax5.set_title('Boundary Analysis (Blue=GT, Red=Pred, Purple=Overlap)')
            ax5.axis('off')
            
            # Show class-specific view (ET class)
            # Focus on just the Enhancing Tumor class with pastel colors
            class_img = np.stack([flair_normalized]*3, axis=-1)
            if mask_slice.shape[0] > 3 and output_slice.shape[0] > 3:
                gt_et = (mask_slice[3] > 0.5).astype(float)
                pred_et = (output_slice[3] > 0.5).astype(float)
                
                # Overlay with pastel colors: GT=Blue, Pred=Red, Overlap=Purple
                class_img[gt_et > 0.5] = [0.7, 0.7, 0.9]  # Pastel blue for GT
                class_img[pred_et > 0.5] = [0.9, 0.7, 0.7]  # Pastel red for pred
                class_img[(gt_et > 0.5) & (pred_et > 0.5)] = [0.9, 0.7, 0.9]  # Pastel purple for overlap
                
                ax6.imshow(class_img)
                ax6.set_title('ET Class Analysis (Blue=GT, Red=Pred)')
                ax6.axis('off')
            else:
                ax6.text(0.5, 0.5, "ET class data not available", 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.axis('off')
            
            # Add title with metrics if available
            plt.suptitle(f'Epoch {epoch} - {mode.upper()} Mode - Slice {slice_idx}', fontsize=16)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
            slice_filename = f"results/slices/{prefix}_slice{slice_idx}_epoch{epoch}.png"
            plt.savefig(slice_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error visualizing slice {slice_idx}: {e}")
    
    # Create a summary figure with all slices side by side
    try:
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
            
            # Visualize Ground Truth (row 1) with pastel colors
            gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
            summary_axs[1, i].imshow(gt_overlay)
            summary_axs[1, i].axis('off')
            
            # Visualize Prediction (row 2) with pastel colors
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
        summary_filename = f"results/{prefix}_epoch{epoch}_summary.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close(summary_fig)
    except Exception as e:
        print(f"Error creating summary visualization: {e}")

# Additional visualization functions - not currently used in train.py
def analyze_tumor_regions_by_class(images, masks, outputs, epoch, slice_indices=None, 
                                 output_dir="results", prefix=""):
    """
    Create visualizations focused on specific tumor regions (WT, TC, ET)
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
    
    # Set up pastel colors palette
    pastel_blue = [0.7, 0.7, 0.95]    # Blue for ground truth
    pastel_green = [0.7, 0.95, 0.7]   # Green for true positives
    pastel_red = [0.95, 0.7, 0.7]     # Red for false positives
    pastel_orange = [0.95, 0.85, 0.6] # Orange for TC
    pastel_yellow = [0.95, 0.95, 0.6] # Yellow for WT
    pastel_pink = [0.95, 0.7, 0.95]   # Pink for ET
    
    # Create figures for each tumor region
    for i, slice_idx in enumerate(slice_indices):
        try:
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
            
            # WT Ground Truth - pastel yellow
            wt_gt_rgb = flair_rgb.copy()
            overlay_with_alpha(wt_gt_rgb, gt_wt > 0.5, pastel_yellow, 0.6)
            axs[0, 0].imshow(wt_gt_rgb)
            axs[0, 0].set_title('WT Ground Truth')
            axs[0, 0].axis('off')
            
            # WT Prediction - pastel yellow
            wt_pred_rgb = flair_rgb.copy()
            overlay_with_alpha(wt_pred_rgb, pred_wt > 0.5, pastel_yellow, 0.6)
            axs[0, 1].imshow(wt_pred_rgb)
            axs[0, 1].set_title('WT Prediction')
            axs[0, 1].axis('off')
            
            # WT Difference
            wt_diff_rgb = flair_rgb.copy()
            wt_tp = (gt_wt > 0.5) & (pred_wt > 0.5)  # True positive
            wt_fp = (gt_wt < 0.5) & (pred_wt > 0.5)  # False positive
            wt_fn = (gt_wt > 0.5) & (pred_wt < 0.5)  # False negative
            
            overlay_with_alpha(wt_diff_rgb, wt_tp, pastel_green, 0.6)
            overlay_with_alpha(wt_diff_rgb, wt_fp, pastel_red, 0.6)
            overlay_with_alpha(wt_diff_rgb, wt_fn, pastel_blue, 0.6)
            
            axs[0, 2].imshow(wt_diff_rgb)
            axs[0, 2].set_title('WT Difference')
            axs[0, 2].axis('off')
            
            # Row 2: TC (Tumor Core) - Classes 1+4 (indices 1,3)
            # Create binary masks for TC
            gt_tc = ((mask_slice[1] + mask_slice[3]) > 0.5).astype(float)
            pred_tc = ((output_slice[1] + output_slice[3]) > 0.5).astype(float)
            
            # TC Ground Truth - pastel orange
            tc_gt_rgb = flair_rgb.copy()
            overlay_with_alpha(tc_gt_rgb, gt_tc > 0.5, pastel_orange, 0.6)
            axs[1, 0].imshow(tc_gt_rgb)
            axs[1, 0].set_title('TC Ground Truth')
            axs[1, 0].axis('off')
            
            # TC Prediction - pastel orange
            tc_pred_rgb = flair_rgb.copy()
            overlay_with_alpha(tc_pred_rgb, pred_tc > 0.5, pastel_orange, 0.6)
            axs[1, 1].imshow(tc_pred_rgb)
            axs[1, 1].set_title('TC Prediction')
            axs[1, 1].axis('off')
            
            # TC Difference
            tc_diff_rgb = flair_rgb.copy()
            tc_tp = (gt_tc > 0.5) & (pred_tc > 0.5)  # True positive
            tc_fp = (gt_tc < 0.5) & (pred_tc > 0.5)  # False positive
            tc_fn = (gt_tc > 0.5) & (pred_tc < 0.5)  # False negative
            
            overlay_with_alpha(tc_diff_rgb, tc_tp, pastel_green, 0.6)
            overlay_with_alpha(tc_diff_rgb, tc_fp, pastel_red, 0.6)
            overlay_with_alpha(tc_diff_rgb, tc_fn, pastel_blue, 0.6)
            
            axs[1, 2].imshow(tc_diff_rgb)
            axs[1, 2].set_title('TC Difference')
            axs[1, 2].axis('off')
            
            # Row 3: ET (Enhancing Tumor) - Class 4 (index 3)
            # Create binary masks for ET
            gt_et = (mask_slice[3] > 0.5).astype(float)
            pred_et = (output_slice[3] > 0.5).astype(float)
            
            # ET Ground Truth - pastel pink
            et_gt_rgb = flair_rgb.copy()
            overlay_with_alpha(et_gt_rgb, gt_et > 0.5, pastel_pink, 0.6)
            axs[2, 0].imshow(et_gt_rgb)
            axs[2, 0].set_title('ET Ground Truth')
            axs[2, 0].axis('off')
            
            # ET Prediction - pastel pink
            et_pred_rgb = flair_rgb.copy()
            overlay_with_alpha(et_pred_rgb, pred_et > 0.5, pastel_pink, 0.6)
            axs[2, 1].imshow(et_pred_rgb)
            axs[2, 1].set_title('ET Prediction')
            axs[2, 1].axis('off')
            
            # ET Difference
            et_diff_rgb = flair_rgb.copy()
            et_tp = (gt_et > 0.5) & (pred_et > 0.5)  # True positive
            et_fp = (gt_et < 0.5) & (pred_et > 0.5)  # False positive
            et_fn = (gt_et > 0.5) & (pred_et < 0.5)  # False negative
            
            overlay_with_alpha(et_diff_rgb, et_tp, pastel_green, 0.6)
            overlay_with_alpha(et_diff_rgb, et_fp, pastel_red, 0.6)
            overlay_with_alpha(et_diff_rgb, et_fn, pastel_blue, 0.6)
            
            axs[2, 2].imshow(et_diff_rgb)
            axs[2, 2].set_title('ET Difference')
            axs[2, 2].axis('off')
            
            # Add title
            plt.suptitle(f'Epoch {epoch} - Tumor Region Analysis - Slice {slice_idx}', fontsize=16)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            region_filename = f"{output_dir}/regions/{prefix}_regions_epoch{epoch}_slice{slice_idx}.png"
            plt.savefig(region_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error analyzing tumor regions for slice {slice_idx}: {e}")

def overlay_with_alpha(img, mask, color, alpha=0.6):
    """Helper function to overlay colors with alpha transparency"""
    mask_3d = np.expand_dims(mask, axis=-1)
    img[mask] = img[mask] * (1-alpha) + np.array(color) * alpha
    return img

# Function to visualize results in a multi-model comparison format
def visualize_model_comparison(images, unet_outputs, sam_outputs, hybrid_outputs, masks, 
                             epoch, slice_indices=None, output_dir="results/comparison"):
    """
    Create visualizations comparing UNet3D, SAM2, and Hybrid modes
    
    Args:
        images: Input MRI scans [B, C, D, H, W]
        unet_outputs: UNet3D predictions [B, C, D, H, W]
        sam_outputs: SAM2 predictions [B, C, D, H, W]
        hybrid_outputs: Hybrid model predictions [B, C, D, H, W]
        masks: Ground truth masks [B, C, D, H, W]
        epoch: Current epoch number
        slice_indices: Specific slice indices to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get batch item
    b = 0
    
    # Convert to probabilities if needed
    if torch.is_tensor(unet_outputs) and torch.min(unet_outputs) < 0:
        unet_probs = torch.sigmoid(unet_outputs)
    else:
        unet_probs = unet_outputs
        
    if torch.is_tensor(sam_outputs) and torch.min(sam_outputs) < 0:
        sam_probs = torch.sigmoid(sam_outputs)
    else:
        sam_probs = sam_outputs
        
    if torch.is_tensor(hybrid_outputs) and torch.min(hybrid_outputs) < 0:
        hybrid_probs = torch.sigmoid(hybrid_outputs)
    else:
        hybrid_probs = hybrid_outputs
    
    # Determine slice indices if not provided
    if slice_indices is None:
        slice_indices = get_slice_indices(images)
    
    # Process each slice
        for slice_idx in slice_indices:
            try:
                # Extract slice data
                image_slice = images[b, :, slice_idx].cpu().detach().numpy()
                mask_slice = masks[b, :, slice_idx].cpu().detach().numpy()
                
                # Get outputs from each model
                unet_slice = unet_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(unet_probs) else None
                sam_slice = sam_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(sam_probs) else None
                hybrid_slice = hybrid_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(hybrid_probs) else None
                
                # Create figure (2x2 grid)
                fig, axs = plt.subplots(2, 2, figsize=(12, 12))
                
                # Show FLAIR MRI with ground truth overlay
                flair_idx = min(3, image_slice.shape[0]-1)
                p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
                flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
                
                # Ground Truth overlay (top left)
                gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
                axs[0, 0].imshow(gt_overlay)
                axs[0, 0].set_title('Ground Truth')
                axs[0, 0].axis('off')
                
                # UNet3D prediction (top right)
                if unet_slice is not None:
                    unet_overlay = create_segmentation_overlay(image_slice[flair_idx], unet_slice)
                    axs[0, 1].imshow(unet_overlay)
                    axs[0, 1].set_title('UNet3D Prediction')
                else:
                    axs[0, 1].imshow(flair_normalized, cmap='gray')
                    axs[0, 1].set_title('UNet3D (Not Available)')
                axs[0, 1].axis('off')
                
                # SAM2 prediction (bottom left)
                if sam_slice is not None:
                    sam_overlay = create_segmentation_overlay(image_slice[flair_idx], sam_slice)
                    axs[1, 0].imshow(sam_overlay)
                    axs[1, 0].set_title('SAM2 Prediction')
                else:
                    axs[1, 0].imshow(flair_normalized, cmap='gray')
                    axs[1, 0].set_title('SAM2 (Not Available)')
                axs[1, 0].axis('off')
                
                # Hybrid prediction (bottom right)
                if hybrid_slice is not None:
                    hybrid_overlay = create_segmentation_overlay(image_slice[flair_idx], hybrid_slice)
                    axs[1, 1].imshow(hybrid_overlay)
                    axs[1, 1].set_title('Hybrid Prediction')
                else:
                    axs[1, 1].imshow(flair_normalized, cmap='gray')
                    axs[1, 1].set_title('Hybrid (Not Available)')
                axs[1, 1].axis('off')
                
                # Add title
                plt.suptitle(f'Model Comparison - Epoch {epoch} - Slice {slice_idx}', fontsize=16)
                
                # Adjust layout and save
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                comparison_filename = f"{output_dir}/comparison_epoch{epoch}_slice{slice_idx}.png"
                plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Error creating comparison visualization for slice {slice_idx}: {e}")
