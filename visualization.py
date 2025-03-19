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

def create_segmentation_overlay(img_slice, seg_slice, alpha=0.6):
    """Create an RGB overlay of segmentation on grayscale image"""
    # Create RGB version of the grayscale image
    p1, p99 = np.percentile(img_slice, (1, 99))
    img_normalized = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
    rgb_img = np.stack([img_normalized]*3, axis=-1)
    
    # Create RGB mask with colors matching your existing visualization
    h, w = seg_slice.shape[1:]
    rgb_mask = np.zeros((h, w, 3))
    
    # Apply your color convention
    # Class 1 (NCR) - Blue
    if 1 < seg_slice.shape[0]:  # Make sure we have this channel
        ncr_mask = seg_slice[1] > 0.5
        rgb_mask[ncr_mask, 0] = 0.0  # R
        rgb_mask[ncr_mask, 1] = 0.0  # G
        rgb_mask[ncr_mask, 2] = 1.0  # B
    
    # Class 2 (ED) - Green
    if 2 < seg_slice.shape[0]:
        ed_mask = seg_slice[2] > 0.5
        rgb_mask[ed_mask, 0] = 0.0  # R
        rgb_mask[ed_mask, 1] = 1.0  # G
        rgb_mask[ed_mask, 2] = 0.0  # B
    
    # Class 4 (ET) at index 3 - Red
    if 3 < seg_slice.shape[0]:
        et_mask = seg_slice[3] > 0.5
        rgb_mask[et_mask, 0] = 1.0  # R
        rgb_mask[et_mask, 1] = 0.0  # G
        rgb_mask[et_mask, 2] = 0.0  # B
    
    # Combine the image and mask
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
    
    # Create background image
    p1, p99 = np.percentile(img_slice, (1, 99))
    img_normalized = np.clip((img_slice - p1) / (p99 - p1 + 1e-8), 0, 1)
    img_rgb = np.stack([img_normalized]*3, axis=-1)
    
    # Calculate difference map with existing colors
    h, w = gt_any.shape
    
    # Create a clean copy for drawing on
    diff_rgb = img_rgb.copy()
    
    # True Positive - Green
    true_positive = (gt_any > 0) & (pred_any > 0)
    diff_rgb[true_positive, 0] = 0.0  # R
    diff_rgb[true_positive, 1] = 1.0  # G
    diff_rgb[true_positive, 2] = 0.0  # B
    
    # False Positive - Red
    false_positive = (gt_any == 0) & (pred_any > 0)
    diff_rgb[false_positive, 0] = 1.0  # R
    diff_rgb[false_positive, 1] = 0.0  # G
    diff_rgb[false_positive, 2] = 0.0  # B
    
    # False Negative - Blue
    false_negative = (gt_any > 0) & (pred_any == 0)
    diff_rgb[false_negative, 0] = 0.0  # R
    diff_rgb[false_negative, 1] = 0.0  # G
    diff_rgb[false_negative, 2] = 1.0  # B
    
    return diff_rgb

def get_boundary_pixels(mask, dilate=1):
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
    # Create output directory - use absolute paths
    current_dir = os.path.abspath(os.getcwd())
    results_dir = os.path.join(current_dir, "results")
    slices_dir = os.path.join(results_dir, "slices")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(slices_dir, exist_ok=True)
    
    print(f"Saving visualizations to: {slices_dir}")
    
    # Get batch item (usually just one in 3D segmentation)
    b = 0
    
    # Convert outputs to probabilities if they are logits
    if torch.is_tensor(outputs):
        if torch.min(outputs) < 0 or torch.max(outputs) > 1:
            probs = torch.sigmoid(outputs)
        else:
            probs = outputs
    else:
        probs = outputs
    
    # Determine slice indices 
    slice_indices = get_slice_indices(images, num_slices=5)
    print(f"Visualizing slices at indices: {slice_indices}")
    
    # Set up figure properties for better visualization
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.titlesize': 12,
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
                
            # Get boundaries
            gt_boundary = get_boundary_pixels(gt_tumor)
            pred_boundary = get_boundary_pixels(pred_tumor)
            
            # Create boundary comparison visualization with blue(GT), red(pred), purple(overlap)
            p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
            flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
            boundary_img = np.stack([flair_normalized]*3, axis=-1)  # RGB
            
            # Apply GT boundary (blue)
            boundary_img[gt_boundary > 0.5, 0] = 0.0  # R
            boundary_img[gt_boundary > 0.5, 1] = 0.0  # G
            boundary_img[gt_boundary > 0.5, 2] = 1.0  # B
            
            # Apply prediction boundary (red)
            boundary_img[pred_boundary > 0.5, 0] = 1.0  # R
            boundary_img[pred_boundary > 0.5, 1] = 0.0  # G
            boundary_img[pred_boundary > 0.5, 2] = 0.0  # B
            
            # Overlap (purple)
            overlap = (gt_boundary > 0.5) & (pred_boundary > 0.5)
            boundary_img[overlap, 0] = 1.0  # R
            boundary_img[overlap, 1] = 0.0  # G
            boundary_img[overlap, 2] = 1.0  # B
            
            ax5.imshow(boundary_img)
            ax5.set_title('Boundary Analysis (Blue=GT, Red=Pred, Purple=Overlap)')
            ax5.axis('off')
            
            # Show class-specific view (ET class)
            # Focus on just the Enhancing Tumor class
            class_img = np.stack([flair_normalized]*3, axis=-1)
            if mask_slice.shape[0] > 3 and output_slice.shape[0] > 3:
                gt_et = (mask_slice[3] > 0.5).astype(float)
                pred_et = (output_slice[3] > 0.5).astype(float)
                
                # Overlay with colors: GT=Blue, Pred=Red, Overlap=Purple
                class_img[gt_et > 0.5, 0] = 0.0  # R
                class_img[gt_et > 0.5, 1] = 0.0  # G
                class_img[gt_et > 0.5, 2] = 1.0  # B
                
                class_img[pred_et > 0.5, 0] = 1.0  # R
                class_img[pred_et > 0.5, 1] = 0.0  # G
                class_img[pred_et > 0.5, 2] = 0.0  # B
                
                # Overlap in purple
                et_overlap = (gt_et > 0.5) & (pred_et > 0.5)
                class_img[et_overlap, 0] = 1.0  # R
                class_img[et_overlap, 1] = 0.0  # G
                class_img[et_overlap, 2] = 1.0  # B
                
                ax6.imshow(class_img)
                ax6.set_title('ET Class Analysis (Blue=GT, Red=Pred)')
                ax6.axis('off')
            else:
                ax6.text(0.5, 0.5, "ET class data not available", 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.axis('off')
            
            # Add title with metrics if available
            plt.suptitle(f'Epoch {epoch} - {mode.upper()} Mode - Slice {slice_idx}', fontsize=12)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
            slice_filename = os.path.join(slices_dir, f"{prefix}_slice{slice_idx}_epoch{epoch}.png")
            plt.savefig(slice_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved visualization to: {slice_filename}")
        except Exception as e:
            print(f"Error visualizing slice {slice_idx}: {e}")
            import traceback
            traceback.print_exc()
    
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
        summary_axs[0, 0].set_ylabel('FLAIR MRI', fontsize=10)
        summary_axs[1, 0].set_ylabel('Ground Truth', fontsize=10)
        summary_axs[2, 0].set_ylabel(f'Prediction ({mode})', fontsize=10)
        
        # Add title
        plt.suptitle(f'Epoch {epoch} - {mode.upper()} Mode - Overview', fontsize=12)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        summary_filename = f"results/{prefix}_epoch{epoch}_summary.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close(summary_fig)
    except Exception as e:
        print(f"Error creating summary visualization: {e}")

# Additional function to visualize a comparison of UNet3D and SAM2 outputs
def visualize_model_comparison(images, masks, unet_outputs, hybrid_outputs, epoch, slice_indices=None, output_dir="results/comparison"):
    """
    Create visualizations comparing UNet3D and Hybrid modes
    
    Args:
        images: Input MRI scans [B, C, D, H, W]
        masks: Ground truth masks [B, C, D, H, W]
        unet_outputs: UNet3D predictions [B, C, D, H, W]
        hybrid_outputs: Hybrid model predictions [B, C, D, H, W]
        epoch: Current epoch number
        slice_indices: Specific slice indices to visualize (optional)
        output_dir: Directory to save comparison visualizations
    """
    # Create output directory with absolute path
    current_dir = os.path.abspath(os.getcwd())
    comp_dir = os.path.join(current_dir, output_dir)
    os.makedirs(comp_dir, exist_ok=True)
    
    print(f"Saving comparison visualizations to: {comp_dir}")
    
    # Get batch item
    b = 0
    
    # Convert to probabilities if needed
    if torch.is_tensor(unet_outputs) and (torch.min(unet_outputs) < 0 or torch.max(unet_outputs) > 1):
        unet_probs = torch.sigmoid(unet_outputs)
    else:
        unet_probs = unet_outputs
        
    if torch.is_tensor(hybrid_outputs) and (torch.min(hybrid_outputs) < 0 or torch.max(hybrid_outputs) > 1):
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
            hybrid_slice = hybrid_probs[b, :, slice_idx].cpu().detach().numpy() if torch.is_tensor(hybrid_probs) else None
            
            # Create figure (2x2 grid)
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # Show FLAIR MRI with ground truth overlay
            flair_idx = min(3, image_slice.shape[0]-1)
            p1, p99 = np.percentile(image_slice[flair_idx], (1, 99))
            flair_normalized = np.clip((image_slice[flair_idx] - p1) / (p99 - p1 + 1e-8), 0, 1)
            
            # Ground Truth overlay (top left)
            gt_overlay = create_segmentation_overlay(image_slice[flair_idx], mask_slice)
            axs[0, 0].imshow(gt_overlay)
            axs[0, 0].set_title('Ground Truth', fontsize=12)
            axs[0, 0].axis('off')
            
            # UNet3D prediction (top right)
            if unet_slice is not None:
                unet_overlay = create_segmentation_overlay(image_slice[flair_idx], unet_slice)
                axs[0, 1].imshow(unet_overlay)
                axs[0, 1].set_title('UNet3D Prediction', fontsize=12)
            else:
                axs[0, 1].imshow(flair_normalized, cmap='gray')
                axs[0, 1].set_title('UNet3D (Not Available)', fontsize=12)
            axs[0, 1].axis('off')
            
            # Difference map for UNet3D (bottom left)
            if unet_slice is not None:
                unet_diff = create_difference_map(mask_slice, unet_slice, image_slice[flair_idx])
                axs[1, 0].imshow(unet_diff)
                axs[1, 0].set_title('UNet3D Difference Map (TP=Green, FP=Red, FN=Blue)', fontsize=12)
            else:
                axs[1, 0].imshow(flair_normalized, cmap='gray')
                axs[1, 0].set_title('UNet3D Difference (Not Available)', fontsize=12)
            axs[1, 0].axis('off')
            
            # Hybrid prediction (bottom right)
            if hybrid_slice is not None:
                hybrid_overlay = create_segmentation_overlay(image_slice[flair_idx], hybrid_slice)
                axs[1, 1].imshow(hybrid_overlay)
                axs[1, 1].set_title('Hybrid (70/30) Prediction', fontsize=12)
            else:
                axs[1, 1].imshow(flair_normalized, cmap='gray')
                axs[1, 1].set_title('Hybrid (Not Available)', fontsize=12)
            axs[1, 1].axis('off')
            
            # Add title
            plt.suptitle(f'Model Comparison - Epoch {epoch} - Slice {slice_idx}', fontsize=14)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            comparison_filename = os.path.join(comp_dir, f"compare_slice{slice_idx}_epoch{epoch}.png")
            plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved comparison to: {comparison_filename}")
            
        except Exception as e:
            print(f"Error creating comparison visualization for slice {slice_idx}: {e}")
            import traceback
            traceback.print_exc()
