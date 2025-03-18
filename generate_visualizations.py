import torch
import os
from model import AutoSAM2
from dataset import get_brats_dataloader
from visualization import visualize_batch_comprehensive, analyze_tumor_regions_by_class

# Load the best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/best_autosam2_model.pth"
model = AutoSAM2(num_classes=4).to(device)

# Load model weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get a validation batch
val_loader = get_brats_dataloader(
    "/home/erezhuberman/data/Task01_BrainTumour",  # Update this path
    batch_size=1, 
    train=False, 
    num_workers=4,
    max_samples=1  # Just get 1 sample
)

# Get the first batch
for batch in val_loader:
    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
    
    # Create comprehensive visualizations
    visualize_batch_comprehensive(
        images, masks, outputs, epoch=15,  # Use your final epoch number
        mode="hybrid", 
        output_dir="results/enhanced",
        prefix="final"
    )
    
    # Create region-specific visualizations
    analyze_tumor_regions_by_class(
        images, masks, outputs, epoch=15,
        output_dir="results/enhanced",
        prefix="final"
    )
    
    # Only process one batch
    break

print("Enhanced visualizations created in results/enhanced directory")
