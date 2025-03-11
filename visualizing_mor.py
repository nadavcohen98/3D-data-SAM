import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn.functional as F
from code_sam_unet import AutoSAM2  # Import your model

def load_nifti_image(filepath):
    """Load a NIfTI image and return as a NumPy array."""
    image = nib.load(filepath).get_fdata()
    return image

def load_multimodal_nifti(image_dir, base_filename):
    """Load all 4 modalities for a given sample and return as a single tensor."""
    modalities = []
    for i in range(4):  # Load FLAIR, T1, T1ce, T2
        file_path = os.path.join(image_dir, f"{base_filename}_000{i}.nii.gz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing modality: {file_path}")
        modalities.append(nib.load(file_path).get_fdata())
    
    return np.stack(modalities, axis=-1)  # Shape: (H, W, D, 4)

def preprocess_image(image):
    """Preprocess the MRI image (normalize all 4 modalities)."""
    if image.shape[-1] != 4:  # Ensure all 4 modalities are present
        raise ValueError(f"Expected 4 channels, but got {image.shape[-1]}")
    
    processed_modalities = []
    for i in range(4):  # Process each modality separately
        modality = image[..., i]  # Extract modality
        nonzero_mask = modality > 0
        if np.sum(nonzero_mask) > 0:  # Avoid empty scans
            mean, std = np.mean(modality[nonzero_mask]), np.std(modality[nonzero_mask])
            norm_modality = (modality - mean) / (std + 1e-8)  # Normalize
            norm_modality[~nonzero_mask] = 0  # Preserve background
        else:
            norm_modality = np.zeros_like(modality)
        
        processed_modalities.append(torch.tensor(norm_modality, dtype=torch.float32))
    
    return torch.stack(processed_modalities, dim=0).unsqueeze(0)  # Shape: [B, C, D, H, W]

def preprocess_mask(mask):
    """Preprocess the mask (convert multi-class to one-hot)."""
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
    return mask_tensor

def visualize_results(flair, ground_truth, prediction, save_path=None):
    """Visualize the FLAIR MRI, ground truth segmentation, and model prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # FLAIR Image (middle slice)
    axes[0].imshow(flair, cmap='gray')
    axes[0].set_title('FLAIR MRI')
    axes[0].axis('off')
    
    # Ground Truth Segmentation
    axes[1].imshow(ground_truth, cmap='jet', alpha=0.7)
    axes[1].set_title('Ground Truth Segmentation')
    axes[1].axis('off')
    
    # Model Prediction
    axes[2].imshow(prediction, cmap='jet', alpha=0.7)
    axes[2].set_title('Model Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()
    plt.close()

def infer_and_visualize(model_path, data_path, num_samples=10):
    """Loads the trained model, selects random test samples, and visualizes results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = AutoSAM2(num_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Get all test images
    images_dir = os.path.join(data_path, "imagesTr")
    labels_dir = os.path.join(data_path, "labelsTr")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')])
    mask_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    
    # Select random samples
    selected_files = random.sample(list(zip(image_files, mask_files)), num_samples)
    
    for img_file, mask_file in selected_files:
        base_filename = img_file.replace('_0000.nii.gz', '')  # Extract base patient ID
        img_path = os.path.join(images_dir, base_filename)
        mask_path = os.path.join(labels_dir, mask_file)
        
        # Load and preprocess
        image = load_multimodal_nifti(images_dir, base_filename)
        mask = load_nifti_image(mask_path)
        
        flair = preprocess_image(image).to(device)  # Preprocess multimodal MRI
        mask_tensor = preprocess_mask(mask).to(device)  # Preprocess ground truth mask
        
        with torch.no_grad():
            prediction = model(flair)
            prediction = F.softmax(prediction, dim=1)  # Convert logits to probabilities
            predicted_mask = (prediction > 0.5).float()
        
        # Convert tensors to numpy arrays for visualization
        middle_slice = flair.shape[2] // 2
        flair_np = flair[0, 0, middle_slice].cpu().numpy()  # Middle slice of FLAIR
        ground_truth_np = mask_tensor[0, middle_slice].cpu().numpy()  # Middle slice of mask
        predicted_seg_np = predicted_mask[0, 1, middle_slice].cpu().numpy()  # Tumor class
        
        # Save visualization
        save_path = f"results/{base_filename}.png"
        visualize_results(flair_np, ground_truth_np, predicted_seg_np, save_path)
    
    print("Inference and visualization complete!")

if __name__ == "__main__":
    MODEL_PATH = "checkpoints/best_autosam2_model.pth"
    DATA_PATH = "/home/erezhuberman/data/Task01_BrainTumour"  # Adjust dataset path
    infer_and_visualize(MODEL_PATH, DATA_PATH)
