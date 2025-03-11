import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from dataset import get_brats_dataloader  # Importing dataloader from dataset.py
from model import AutoSAM2  # Importing model from model.py
import torch.nn.functional as F

def visualize_results(flair, ground_truth, prediction, save_path=None):
    """
    Visualize the FLAIR MRI, ground truth segmentation, and model prediction.
    """
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

def infer_and_visualize(model_path, data_path, num_samples=10, batch_size=1, target_shape=(64, 128, 128)):
    """
    Loads the trained model, selects random test samples, and visualizes results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = AutoSAM2(num_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test data loader
    test_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=False, target_shape=target_shape
    )
    
    # Randomly select 10 samples
    selected_indices = random.sample(range(len(test_loader.dataset)), num_samples)
    
    for idx in selected_indices:
        image, mask = test_loader.dataset[idx]
        image, mask = image.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(image)
            prediction = F.softmax(prediction, dim=1)  # Convert logits to probabilities
            predicted_mask = (prediction > 0.5).float()
        
        # Convert tensors to numpy arrays
        flair = image[0, 0].cpu().numpy()  # Extract FLAIR modality
        ground_truth = mask[0, 1:].cpu().numpy().sum(axis=0)  # Sum all tumor classes
        predicted_seg = predicted_mask[0, 1:].cpu().numpy().sum(axis=0)  # Sum all classes
        
        # Visualize
        save_path = f"results/test_sample_{idx}.png"
        visualize_results(flair, ground_truth, predicted_seg, save_path)
    
    print("Inference and visualization complete!")

if __name__ == "__main__":
    MODEL_PATH = "checkpoints/best_autosam2_model.pth"
    DATA_PATH = "/home/erezhuberman/data/Task01_BrainTumour"  # Change to your dataset path
    infer_and_visualize(MODEL_PATH, DATA_PATH)
