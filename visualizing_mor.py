import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from code_sam_unet import AutoSAM2, get_brats_dataloader

def visualize_predictions(model, dataloader, device, num_samples=10, save_dir="visualizations"):
    """Visualizes random samples from the dataset with ground truth and model predictions."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    selected_indices = random.sample(range(len(dataloader.dataset)), num_samples)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(selected_indices):
            image, mask = dataloader.dataset[sample_idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.cpu().numpy()
            
            # Run model prediction
            output = model(image)
            output = torch.sigmoid(output).cpu().numpy()  # Convert logits to probabilities
            
            # Extract middle slice
            depth = image.shape[2]
            middle_slice = depth // 2
            flair_slice = image[0, 0, middle_slice].cpu().numpy()
            gt_slice = mask[:, middle_slice]
            pred_slice = output[0, :, middle_slice]
            
            # Prepare visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Show FLAIR image
            axes[0].imshow(flair_slice, cmap='gray')
            axes[0].set_title("FLAIR Image")
            axes[0].axis('off')
            
            # Show ground truth segmentation
            gt_overlay = np.zeros((*gt_slice.shape[1:], 3))
            gt_overlay[gt_slice[1] > 0.5] = [1, 1, 0]  # Yellow for Edema
            gt_overlay[gt_slice[2] > 0.5] = [0, 1, 0]  # Green for Non-enhancing tumor
            gt_overlay[gt_slice[3] > 0.5] = [1, 0, 0]  # Red for Enhancing tumor
            
            axes[1].imshow(flair_slice, cmap='gray')
            axes[1].imshow(gt_overlay, alpha=0.5)
            axes[1].set_title("Ground Truth Segmentation")
            axes[1].axis('off')
            
            # Show model prediction
            pred_overlay = np.zeros((*pred_slice.shape[1:], 3))
            pred_overlay[pred_slice[1] > 0.5] = [1, 1, 0]
            pred_overlay[pred_slice[2] > 0.5] = [0, 1, 0]
            pred_overlay[pred_slice[3] > 0.5] = [1, 0, 0]
            
            axes[2].imshow(flair_slice, cmap='gray')
            axes[2].imshow(pred_overlay, alpha=0.5)
            axes[2].set_title("Model Prediction")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{idx + 1}.png"))
            plt.close()
            print(f"Saved visualization: sample_{idx + 1}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    model = AutoSAM2(num_classes=4).to(device)
    model.load_state_dict(torch.load("checkpoints/best_autosam2_model.pth", map_location=device)['model_state_dict'])
    
    # Load training dataset
    train_loader = get_brats_dataloader(
        root_dir="/home/erezhuberman/data/Task01_BrainTumour",
        batch_size=1,
        train=True,
        normalize=True,
        max_samples=None,
        num_workers=4,
        filter_empty=False,
        use_augmentation=False,
        target_shape=(64, 128, 128),
        cache_data=False,
        verbose=False
    )
    
    # Generate visualizations
    visualize_predictions(model, train_loader, device, num_samples=10)
