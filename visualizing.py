import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os

def visualize_batch(images, masks, outputs, epoch, prefix=""):
    """
    Visualize a batch of images, masks, and predictions.
    """
    os.makedirs("results", exist_ok=True)
    
    # Get middle slice of the first batch item
    b = 0
    depth = images.shape[2]
    middle_idx = depth // 2
    
    # Get slice data
    image_slice = images[b, :, middle_idx].cpu().detach().numpy()
    mask_slice = masks[b, :, middle_idx].cpu().detach().numpy()
    output_slice = F.softmax(outputs[b, :, middle_idx], dim=0).cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show FLAIR image
    axes[0].imshow(image_slice[0], cmap='gray')
    axes[0].set_title('Input Image (FLAIR)')
    axes[0].axis('off')
    
    # Create RGB mask for ground truth
    rgb_mask = np.zeros((mask_slice.shape[1], mask_slice.shape[2], 3))
    rgb_mask[mask_slice[1] > 0.5, :] = [1, 1, 0]  # Edema: Yellow
    rgb_mask[mask_slice[2] > 0.5, :] = [0, 1, 0]  # Non-enhancing: Green
    rgb_mask[mask_slice[3] > 0.5, :] = [1, 0, 0]  # Enhancing: Red
    
    # Show ground truth
    axes[1].imshow(image_slice[0], cmap='gray')
    axes[1].imshow(rgb_mask, alpha=0.5)
    axes[1].set_title('Ground Truth Segmentation')
    axes[1].axis('off')
    
    # Create RGB mask for prediction
    rgb_pred = np.zeros((output_slice.shape[1], output_slice.shape[2], 3))
    rgb_pred[output_slice[1] > 0.5, :] = [1, 1, 0]  # Edema: Yellow
    rgb_pred[output_slice[2] > 0.5, :] = [0, 1, 0]  # Non-enhancing: Green
    rgb_pred[output_slice[3] > 0.5, :] = [1, 0, 0]  # Enhancing: Red
    
    # Show prediction
    axes[2].imshow(image_slice[0], cmap='gray')
    axes[2].imshow(rgb_pred, alpha=0.5)
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/{prefix}_epoch{epoch}.png")
    plt.close()

def plot_training_history(history, filename):
    """
    Save a plot comparing training/validation loss and Dice scores.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    
    # Plot Dice scores
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Training vs Validation Dice Score')
    
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()
