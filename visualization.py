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
            
            # Row 2: Difference map, boundary analysis, and uncertainty
            ax4 = fig.add_subplot(gs[1, 0])  # Difference map (TP, FP, FN)
            ax5 = fig.add_subplot(gs[1, 1])  # Boundary comparison
            ax6 = fig.add_subplot(gs[1, 2])  # Class-specific view
            
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
            
            # Show class-specific view (ET class)
            # Focus on just the Enhancing Tumor class
            class_img = np.stack([flair_normalized]*3, axis=-1)
            if mask_slice.shape[0] > 3 and output_slice.shape[0] > 3:
                gt_et = (mask_slice[3] > 0.5).astype(float)
                pred_et = (output_slice[3] > 0.5).astype(float)
                
                # Overlay: GT=Blue, Pred=Red, Overlap=Purple
                class_img[gt_et > 0.5] = [0, 0, 1]  # Blue for GT
                class_img[pred_et > 0.5] = [1, 0, 0]  # Red for pred
                class_img[(gt_et > 0.5) & (pred_et > 0.5)] = [1, 0, 1]  # Purple for overlap
                
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
            plt.savefig(slice_filename, dpi=150, bbox_inches='tight')
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
        summary_filename = f"results/{prefix}_epoch{epoch}_summary.png"
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
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
        except Exception as e:
            print(f"Error analyzing tumor regions for slice {slice_idx}: {e}")
