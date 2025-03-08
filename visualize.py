import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import random
from nadav_code import UNet  # Replace with the actual filename (e.g., train or model)

# ==============================
# Inference Function
# ==============================
def load_model(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(in_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_segmentation(model, images):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    images = images.to(device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        pred_mask = torch.sigmoid(model(images)).squeeze(0)
    return (pred_mask > 0.5).float().cpu().numpy()

# ==============================
# Load and Preprocess MRI Scans
# ==============================
def load_and_preprocess_images(flair_path, t1_path, t1ce_path, t2_path):
    
    scans = {
        "flair": nib.load(flair_path).get_fdata(),
        "t1": nib.load(t1_path).get_fdata(),
        "t1ce": nib.load(t1ce_path).get_fdata(),
        "t2": nib.load(t2_path).get_fdata(),
    }

    # If 4D, take first modality
    for key in scans:
        if len(scans[key].shape) == 4:
            scans[key] = scans[key][0]

    # Normalize each scan
    for key in scans:
        scans[key] = (scans[key] - np.min(scans[key])) / (np.max(scans[key]) - np.min(scans[key]) + 1e-8)

    # Convert to PyTorch tensors
    scan_tensors = [torch.tensor(scans[key], dtype=torch.float32) for key in ["flair", "t1", "t1ce", "t2"]]

    # Extract middle slice
    middle_idx = scan_tensors[0].shape[2] // 2
    scan_tensors = [scan[:, :, middle_idx].unsqueeze(0) for scan in scan_tensors]

    # Stack all 4 modalities into a single 4-channel tensor
    input_tensor = torch.cat(scan_tensors, dim=0)

    return input_tensor

# ==============================
# Visualization Function
# ==============================
def visualize_segmentation(input_tensor, pred_mask, true_mask):
    
    flair_image = input_tensor[0].squeeze().cpu().numpy()  # Convert to NumPy
    pred_mask = pred_mask.squeeze()  # Ensure 2D mask
    true_mask = true_mask.squeeze()  # Ensure 2D mask

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    ax[0].imshow(flair_image, cmap="gray")
    ax[0].set_title("Input Scan")
    
    ax[1].imshow(true_mask, cmap="jet", alpha=0.5)
    ax[1].set_title("Ground Truth Segmentation")
    
    ax[2].imshow(pred_mask, cmap="jet", alpha=0.5)
    ax[2].set_title("Predicted Segmentation")
    
    ax[3].imshow(flair_image, cmap="gray")
    ax[3].imshow(pred_mask, cmap="jet", alpha=0.5)
    ax[3].set_title("Overlay of Prediction")

    plt.show()

# ==============================
# Run Inference on Random Validation Samples
# ==============================
def get_validation_cases(validation_dir, num_cases=1):
    patient_dirs = [os.path.join(validation_dir, d) for d in os.listdir(validation_dir) if os.path.isdir(os.path.join(validation_dir, d))]
    random.shuffle(patient_dirs)  # Shuffle the list randomly
    return patient_dirs[:num_cases]  # Select `num_cases` random directories

# Load trained model
model_path = "/Users/nadavcohen/Desktop/Universuty/deep_learning/project/code/best_unet_model.pth"
model = load_model(model_path)

# Define validation dataset path
VALIDATION_PATH = "/Users/nadavcohen/Desktop/Universuty/deep_learning/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Choose how many validation cases to test
num_cases_to_test = 7  # Change this number to test more cases

# Get the list of patient cases
validation_cases = get_validation_cases(VALIDATION_PATH, num_cases=num_cases_to_test)

for case in validation_cases:
    patient_id = os.path.basename(case)
    print(f"Running inference for {patient_id}...")

    # Define file paths
    flair_path = os.path.join(case, f"{patient_id}_flair.nii")
    t1_path = os.path.join(case, f"{patient_id}_t1.nii")
    t1ce_path = os.path.join(case, f"{patient_id}_t1ce.nii")
    t2_path = os.path.join(case, f"{patient_id}_t2.nii")
    seg_path = os.path.join(case, f"{patient_id}_seg.nii")  # Ground truth segmentation

    # Load and preprocess the images
    input_tensor = load_and_preprocess_images(flair_path, t1_path, t1ce_path, t2_path)

    # Load ground truth segmentation mask
    true_mask = nib.load(seg_path).get_fdata()
    middle_idx = true_mask.shape[2] // 2  # Extract middle slice
    true_mask = true_mask[:, :, middle_idx]
    true_mask = (true_mask > 0).astype(np.float32)  # Convert to binary mask
    true_mask = torch.tensor(true_mask, dtype=torch.float32)

    # Predict segmentation
    predicted_mask = predict_segmentation(model, input_tensor)

    # Visualize result
    visualize_segmentation(input_tensor, predicted_mask, true_mask)
