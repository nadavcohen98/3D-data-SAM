# 3D-data-SAM

# AutoSAM2 for 3D Medical Image Segmentation

## Overview
Implementation of a hybrid 3D segmentation approach that integrates SAM2 (Segment Anything Model 2) with a 3D UNet for accurate multi-class tumor segmentation in volumetric medical images. This model specifically addresses the challenges of brain tumor segmentation in the BraTS dataset.

## Model Architecture

The architecture combines two powerful segmentation approaches:

1. **3D UNet Backbone**:
   - Extracts volumetric features from multi-modal MRI data
   - Mid-level decoder features are extracted for SAM2 integration
   - Can operate independently for full 3D segmentation

2. **SAM2 Integration**:
   - Strategic selection of axial slices (configurable percentage)
   - UNet to SAM2 bridge network with spatial and channel attention
   - Point and box prompt generation based on UNet predictions
   - Multi-class handling for tumor subregions

3. **Hybrid Fusion**:
   - Adaptive blending of UNet and SAM2 predictions
   - Class-specific weighting for optimal boundary precision

![Architecture Diagram](assets/architecture.png)

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `percentage` | Percentage of slices processed by SAM2 (0.0-1.0) | 0.3 |
| `enable_unet_decoder` | Whether to use UNet decoder pathway | True |
| `enable_sam2` | Whether to use SAM2 integration | True |
| `num_positive_points` | Number of positive points for SAM2 prompts | 10 |
| `bg_blend` | Background blending weight (UNet vs SAM2) | 0.9 |
| `tumor_blend` | Tumor region blending weight (UNet vs SAM2) | 0.5 |
| `test_run` | Use reduced dataset for testing | False |
| `learning_rate` | Initial learning rate | 1e-3 |

## Implementation Details

### Dependencies
- PyTorch 1.10+
- SAM2 (segment-anything-v2)
- nibabel
- SimpleITK
- matplotlib, numpy, scipy

### Data Handling
The implementation supports BraTS dataset format with T1, T1ce, T2, and FLAIR modalities, processing 3D volumes and handling multi-class segmentation for different tumor regions (ET, WT, TC).

## Usage

### Training
# Full training
python train.py --data_path /path/to/brats --epochs 15

# Test run with smaller dataset
python train.py --test_run --reset

## Model Configurations

# UNet only (no SAM2)
model = AutoSAM2(enable_sam2=False)

# SAM2 only (no UNet decoder)
model = AutoSAM2(enable_unet_decoder=False)

# Hybrid mode (default)
model = AutoSAM2(enable_unet_decoder=True, enable_sam2=True)

# Modify slice percentage (0%, 30%, 60%)
# Edit percentage parameter in model.py:
# def get_strategic_slices(depth, percentage=0.3): # Change to 0.0, 0.3, or 0.6



Results
Performance on BraTS dataset with different slice percentages:
| Slice % | Dice_ET | Dice_TC | Dice_WT | IoU_ET | IoU_TC | IoU_WT |
|---------|---------|---------|---------|--------|--------|--------|
| 0% (UNet only) | 65.2 | 74.3 | 82.7 | 4.1 | 3.8 | 5.7 |
| 30% | 70.4 | 79.1 | 86.7 | 7.3 | 6.1 | 8.9 |
| 60% | 72.1 | 81.5 | 87.9 | 8.5 | 6.6 | 9.1 |


References
This implementation builds upon:

[AutoSAM]([url](https://github.com/talshaharabany/AutoSAM)): Adapting SAM to Medical Images by Shaharabany et al.
[Segment Anything Model 2]([url](https://github.com/facebookresearch/sam2)) by Kirillov et al.
