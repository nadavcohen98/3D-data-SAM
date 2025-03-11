import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn.functional as F
from code_sam_unet import AutoSAM2  


def load_nifti_image(filepath):
    """Load a NIfTI image and return as a NumPy array."""
    image = nib.load(filepath).get_fdata()
    return image

def preprocess_image(image):
    """Preprocess the MRI image (normalize FLAIR modality)."""
    flair = image[..., 0]  # Extract FLAIR (assuming it's the first channel)
    mean, std = np.mean(flair[flair > 0]), np.std(flair[flair > 0])
    flair_norm = (flair - mean) / (std + 1e-8)
    flair_norm[flair == 0] = 0  # Preserve background
    return torch.tensor(flair_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [B, C, D, H, W]

def preprocess_mask(mask):
    """Preprocess the mask (convert multi-class to binary)."""
    mask_bin = (mask > 0).astype(np.float32)
    return torch.tensor(mask_bin, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [B, C, D, H, W]

def visualize_results(flair, ground_truth, prediction, save_path=None):
    """Visualize the FLAIR MRI, ground truth segmentation, and model prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # FLAIR Image
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Get all test images
    images_dir = os.path.join(data_path, "imagesTr")
    labels_dir = os.path.join(data_path, "labelsTr")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    
    # Select 10 random samples
    selected_files = random.sample(list(zip(image_files, mask_files)), num_samples)
    
    for img_file, mask_file in selected_files:
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(labels_dir, mask_file)
        
        # Load and preprocess
        image = load_nifti_image(img_path)
        mask = load_nifti_image(mask_path)
        
        flair = preprocess_image(image).to(device)  # Preprocess FLAIR
        mask_tensor = preprocess_mask(mask).to(device)  # Preprocess ground truth mask
        
        with torch.no_grad():
            prediction = model(flair)
            prediction = F.softmax(prediction, dim=1)  # Convert logits to probabilities
            predicted_mask = (prediction > 0.5).float()
        
        # Convert tensors to numpy arrays for visualization
        flair_np = flair[0, 0, flair.shape[2] // 2].cpu().numpy()  # Middle slice
        ground_truth_np = mask_tensor[0, 0, mask_tensor.shape[2] // 2].cpu().numpy()
        predicted_seg_np = predicted_mask[0, 1, predicted_mask.shape[2] // 2].cpu().numpy()  # Tumor class
        
        # Save visualization
        save_path = f"results/{img_file.replace('.nii.gz', '.png')}"
        visualize_results(flair_np, ground_truth_np, predicted_seg_np, save_path)
    
    print("Inference and visualization complete!")

if __name__ == "__main__":
    MODEL_PATH = "checkpoints/best_autosam2_model.pth"
    DATA_PATH = "/home/erezhuberman/data/Task01_BrainTumour"  # Change this to your dataset path
    infer_and_visualize(MODEL_PATH, DATA_PATH)
