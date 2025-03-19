import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.serialization
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import gc
import time
from datetime import datetime, timedelta
import random
import torch.nn.functional as F

# Local imports
from model import AutoSAM2
from dataset import get_brats_dataloader
from visualization import visualize_batch_comprehensive
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


def calculate_dice_score(y_pred, y_true, epsilon=1e-7):
    """
    Enhanced multi-class Dice score calculation for BraTS segmentation
    """
    # Apply sigmoid if input contains logits
    if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
        probs = torch.sigmoid(y_pred)
    else:
        probs = y_pred
    
    # Binary predictions using threshold
    preds = (probs > 0.5).float()
    
    # Ensure y_true and y_pred have same shape
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}")
    
    result = {}
    all_dice_scores = {
        'background': [],
        'NCR': [],      # Class 1
        'ED': [],       # Class 2
        'ET': [],       # Class 3 (index 3)
        'WT': [],       # Whole Tumor (NCR + ED + ET)
        'TC': []        # Tumor Core (NCR + ET)
    }
    
    batch_size = y_pred.size(0)
    
    for b in range(batch_size):
        # Background
        pred_bg = preds[b, 0].reshape(-1)
        true_bg = y_true[b, 0].reshape(-1)
        if true_bg.sum() > 0 or pred_bg.sum() > 0:
            intersection_bg = (pred_bg * true_bg).sum()
            dice_bg = (2.0 * intersection_bg / (pred_bg.sum() + true_bg.sum() + epsilon)).item() * 100
            all_dice_scores['background'].append(dice_bg)
        
        # NCR (Class 1)
        pred_ncr = preds[b, 1].reshape(-1)
        true_ncr = y_true[b, 1].reshape(-1)
        if true_ncr.sum() > 0 or pred_ncr.sum() > 0:
            intersection_ncr = (pred_ncr * true_ncr).sum()
            dice_ncr = (2.0 * intersection_ncr / (pred_ncr.sum() + true_ncr.sum() + epsilon)).item() * 100
            all_dice_scores['NCR'].append(dice_ncr)
        
        # ED (Class 2)
        pred_ed = preds[b, 2].reshape(-1)
        true_ed = y_true[b, 2].reshape(-1)
        if true_ed.sum() > 0 or pred_ed.sum() > 0:
            intersection_ed = (pred_ed * true_ed).sum()
            dice_ed = (2.0 * intersection_ed / (pred_ed.sum() + true_ed.sum() + epsilon)).item() * 100
            all_dice_scores['ED'].append(dice_ed)
        
        # ET (Class 3, index 3)
        pred_et = preds[b, 3].reshape(-1)
        true_et = y_true[b, 3].reshape(-1)
        if true_et.sum() > 0 or pred_et.sum() > 0:
            intersection_et = (pred_et * true_et).sum()
            dice_et = (2.0 * intersection_et / (pred_et.sum() + true_et.sum() + epsilon)).item() * 100
            all_dice_scores['ET'].append(dice_et)
        
        # Whole Tumor (WT): NCR + ED + ET
        pred_wt = (preds[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        true_wt = (y_true[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        if true_wt.sum() > 0 or pred_wt.sum() > 0:
            intersection_wt = (pred_wt * true_wt).sum()
            dice_wt = (2.0 * intersection_wt / (pred_wt.sum() + true_wt.sum() + epsilon)).item() * 100
            all_dice_scores['WT'].append(dice_wt)
        
        # Tumor Core (TC): NCR + ET
        pred_tc = ((preds[b, 1] + preds[b, 3]) > 0).float().reshape(-1)
        true_tc = ((y_true[b, 1] + y_true[b, 3]) > 0).float().reshape(-1)
        if true_tc.sum() > 0 or pred_tc.sum() > 0:
            intersection_tc = (pred_tc * true_tc).sum()
            dice_tc = (2.0 * intersection_tc / (pred_tc.sum() + true_tc.sum() + epsilon)).item() * 100
            all_dice_scores['TC'].append(dice_tc)
    
    # Calculate statistics for each category
    for key, scores in all_dice_scores.items():
        if scores:
            scores_tensor = torch.tensor(scores)
            result[f'{key}_mean'] = scores_tensor.mean().item()
            result[f'{key}_std'] = scores_tensor.std().item() if len(scores_tensor) > 1 else 0.0
            result[f'{key}_median'] = torch.median(scores_tensor).item()
            
            # Interquartile range
            if len(scores_tensor) > 1:
                q1, q3 = torch.tensor(scores_tensor.tolist()).quantile(torch.tensor([0.25, 0.75])).tolist()
                result[f'{key}_iqr'] = q3 - q1
            else:
                result[f'{key}_iqr'] = 0.0
        else:
            # Default values if no scores
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0
            result[f'{key}_median'] = 0.0
            result[f'{key}_iqr'] = 0.0
    
    # Overall mean Dice
    region_means = [result.get(f'{region}_mean', 0.0) for region in ['NCR', 'ED', 'ET']]
    result['mean'] = sum(region_means) / len(region_means) if region_means else 0.0
    
    return result

def calculate_iou(y_pred, y_true, threshold=0.5, epsilon=1e-7):
    """
    Enhanced multi-class IoU calculation for BraTS segmentation
    """
    # Apply sigmoid if input contains logits
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float() if torch.min(y_pred) < 0 else y_pred
    
    batch_size = y_pred.size(0)
    result = {}
    
    all_iou_values = {
        'background': [],
        'NCR': [],      # Class 1
        'ED': [],       # Class 2
        'ET': [],       # Class 3 (index 3)
        'WT': [],       # Whole Tumor (NCR + ED + ET)
        'TC': []        # Tumor Core (NCR + ET)
    }
    
    for b in range(batch_size):
        # IoU for each class follows same pattern as Dice calculation
        classes = [
            (0, 'background'),
            (1, 'NCR'),
            (2, 'ED'),
            (3, 'ET')
        ]
        
        for class_idx, class_name in classes:
            pred_class = y_pred_bin[b, class_idx].reshape(-1)
            true_class = y_true[b, class_idx].reshape(-1)
            
            intersection = (pred_class * true_class).sum()
            union = (pred_class + true_class).clamp(0, 1).sum()
            
            if union > epsilon:
                iou = (intersection / union).item() * 100
                all_iou_values[class_name].append(iou)
        
        # Whole Tumor and Tumor Core follow similar logic to Dice
        pred_wt = (y_pred_bin[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        true_wt = (y_true[b, 1:4].sum(dim=0) > 0).float().reshape(-1)
        
        if true_wt.sum() > 0 or pred_wt.sum() > 0:
            intersection_wt = (pred_wt * true_wt).sum()
            union_wt = (pred_wt + true_wt).clamp(0, 1).sum()
            
            iou_wt = (intersection_wt / union_wt).item() * 100 if union_wt > epsilon else 0.0
            all_iou_values['WT'].append(iou_wt)
        
        # Tumor Core
        pred_tc = ((y_pred_bin[b, 1] + y_pred_bin[b, 3]) > 0).float().reshape(-1)
        true_tc = ((y_true[b, 1] + y_true[b, 3]) > 0).float().reshape(-1)
        
        if true_tc.sum() > 0 or pred_tc.sum() > 0:
            intersection_tc = (pred_tc * true_tc).sum()
            union_tc = (pred_tc + true_tc).clamp(0, 1).sum()
            
            iou_tc = (intersection_tc / union_tc).item() * 100 if union_tc > epsilon else 0.0
            all_iou_values['TC'].append(iou_tc)
    
    # Calculate statistics for each category (similar to Dice score)
    for key, values in all_iou_values.items():
        if values:
            values_tensor = torch.tensor(values)
            result[f'{key}_mean'] = values_tensor.mean().item()
            result[f'{key}_std'] = values_tensor.std().item() if len(values_tensor) > 1 else 0.0
            result[f'{key}_median'] = torch.median(values_tensor).item()
            
            if len(values_tensor) > 1:
                q1, q3 = torch.tensor(values_tensor.tolist()).quantile(torch.tensor([0.25, 0.75])).tolist()
                result[f'{key}_iqr'] = q3 - q1
            else:
                result[f'{key}_iqr'] = 0.0
        else:
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0
            result[f'{key}_median'] = 0.0
            result[f'{key}_iqr'] = 0.0
    
    # Mean IoU
    region_means = [result.get(f'{region}_mean', 0.0) for region in ['NCR', 'ED', 'ET']]
    result['mean_iou'] = sum(region_means) / len(region_means) if region_means else 0.0
    
    return result

class BraTSDiceLoss(nn.Module):
    """
    Dice Loss specifically designed for BraTS segmentation task.
    """
    def __init__(self, smooth=1.0, region_weights={'ET': 1.0, 'WT': 1.0, 'TC': 1.0}):
        super(BraTSDiceLoss, self).__init__()
        self.smooth = smooth
        self.region_weights = region_weights
        
    def forward(self, y_pred, y_true):
        # Get sigmoid activation for binary predictions
        if torch.min(y_pred) < 0 or torch.max(y_pred) > 1:
            probs = torch.sigmoid(y_pred)
        else:
            probs = y_pred
        
        batch_size = y_pred.size(0)
        device = y_pred.device
        
        # Initialize losses for all regions
        losses = []
        
        for b in range(batch_size):
            # Compute dice loss for each class independently
            for class_idx in range(1, y_pred.size(1)):  # Skip background
                pred_class = probs[b, class_idx].reshape(-1)
                true_class = y_true[b, class_idx].reshape(-1)
                
                # Only calculate if there's something to predict
                if true_class.sum() > 0 or pred_class.sum() > 0:
                    intersection = (pred_class * true_class).sum()
                    dice = (2. * intersection + self.smooth) / (pred_class.sum() + true_class.sum() + self.smooth)
                    losses.append(1. - dice)
        
        # If no valid regions found
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        
        # Compute weighted mean loss
        losses_tensor = torch.stack(losses)
        
        # Apply regional weights if specified
        weighted_regions = {
            'ET': 1.3,   # More weight for enhancing tumor
            'WT': 1.0,   # Standard weight for whole tumor
            'TC': 1.2    # Slightly more weight for tumor core
        }
        
        return losses_tensor.mean()

class BraTSCombinedLoss(nn.Module):
    """
    Combined loss function for BraTS segmentation task.
    """
    def __init__(self, dice_weight=0.8, bce_weight=0.1, focal_weight=0.1, 
                 region_weights={'ET': 1.6, 'WT': 1.1, 'TC': 1.3}):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.brats_dice_loss = BraTSDiceLoss(region_weights=region_weights)
        
        # Set class weights for BCE - higher weight for tumor classes
        pos_weight = torch.ones(4)
        pos_weight[1:] = 5.0  # Higher weight for tumor classes (1,2,4)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda() if torch.cuda.is_available() else pos_weight)
        
        # Focal loss for handling class imbalance
        class_weights = torch.tensor([0.1, 1.0, 1.0, 1.2])
        self.focal_loss = nn.CrossEntropyLoss(
            weight=class_weights.cuda() if torch.cuda.is_available() else class_weights
        )

    def forward(self, y_pred, y_true):
        # Dice loss calculation - now multi-class aware
        dice = self.brats_dice_loss(y_pred, y_true)
        
        # BCE loss uses pos_weight to handle class imbalance
        bce = self.bce_loss(y_pred, y_true)
        
        # Focal loss using class indices
        target = y_true.argmax(dim=1)
        focal = self.focal_loss(y_pred, target)
        
        # Weighted sum of losses
        return self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler=None):
    """
    Training epoch function with comprehensive BraTS metrics.
    """
    model.train()
    total_loss = 0
    all_metrics = []
    batch_times = []

    epoch_start_time = time.time()
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Clear GPU cache if needed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Preprocess batch
            images, masks = preprocess_batch(batch, device=device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute BCE Loss and Dice Loss
            bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
            dice_loss = BraTSDiceLoss()(outputs, masks)

            # Compute combined loss
            loss = 0.7 * dice_loss + 0.3 * bce_loss
            
            # Compute IoU
            iou_metrics = calculate_iou(outputs, masks)

            # Compute Dice Score
            dice_metrics = calculate_dice_score(outputs.detach(), masks)
            
            # Store metrics
            all_metrics.append({
                'bce_loss': bce_loss.item(),
                'dice_loss': dice_loss.item(),
                'mean_dice': dice_metrics['mean'],
                'mean_iou': iou_metrics['mean_iou'],
                'dice_et': dice_metrics.get('ET_mean', 0.0),
                'dice_wt': dice_metrics.get('WT_mean', 0.0),
                'dice_tc': dice_metrics.get('TC_mean', 0.0),
                'iou_et': iou_metrics.get('ET_mean', 0.0),
                'iou_wt': iou_metrics.get('WT_mean', 0.0),
                'iou_tc': iou_metrics.get('TC_mean', 0.0)
            })

            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Learning rate scheduling
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # Update progress bar - now include tumor region metrics
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'WT': f"{dice_metrics.get('WT_mean', 0.0):.1f}%",
                'TC': f"{dice_metrics.get('TC_mean', 0.0):.1f}%",
                'ET': f"{dice_metrics.get('ET_mean', 0.0):.1f}%"
            })
            
            # Update total loss
            total_loss += loss.item()
            
            # Free memory
            del images, masks, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nCUDA OOM in batch {batch_idx}. Skipping batch...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                continue
            else:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                raise e
    
    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {
        'mean_dice': np.mean([m['mean_dice'] for m in all_metrics]) if all_metrics else 0.0,
        'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]) if all_metrics else 0.0,
        'bce_loss': np.mean([m['bce_loss'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_loss': np.mean([m['dice_loss'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_et': np.mean([m['dice_et'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_wt': np.mean([m['dice_wt'] for m in all_metrics]) if all_metrics else 0.0,
        'dice_tc': np.mean([m['dice_tc'] for m in all_metrics]) if all_metrics else 0.0,
        'iou_et': np.mean([m['iou_et'] for m in all_metrics]) if all_metrics else 0.0,
        'iou_wt': np.mean([m['iou_wt'] for m in all_metrics]) if all_metrics else 0.0,
        'iou_tc': np.mean([m['iou_tc'] for m in all_metrics]) if all_metrics else 0.0
    }

    epoch_time = time.time() - epoch_start_time
    avg_metrics['epoch_time'] = epoch_time
    avg_metrics['batch_time'] = epoch_time / len(train_loader)

    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model with comprehensive BraTS metrics.
    """
    model.eval()
    total_loss = 0
    all_metrics = []
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
  
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # Clear GPU cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Preprocess batch
                images, masks = preprocess_batch(batch, device=device)
                
                # Forward pass
                outputs = model(images)
                
                # Compute BCE Loss and Dice Loss
                bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
                dice_loss = BraTSDiceLoss()(outputs, masks)

                # Compute combined loss
                loss = 0.7 * dice_loss + 0.3 * bce_loss
                
                # Compute IoU
                iou_metrics = calculate_iou(outputs, masks)

                # Compute Dice Score
                dice_metrics = calculate_dice_score(outputs, masks)
                
                # Store metrics
                all_metrics.append({
                    'bce_loss': bce_loss.item(),
                    'dice_loss': dice_loss.item(),
                    'mean_dice': dice_metrics['mean'],
                    'mean_iou': iou_metrics['mean_iou'],
                    'dice_et': dice_metrics.get('ET_mean', 0.0),
                    'dice_wt': dice_metrics.get('WT_mean', 0.0),
                    'dice_tc': dice_metrics.get('TC_mean', 0.0),
                    'iou_et': iou_metrics.get('ET_mean', 0.0),
                    'iou_wt': iou_metrics.get('WT_mean', 0.0),
                    'iou_tc': iou_metrics.get('TC_mean', 0.0)
                })
                                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'WT': f"{dice_metrics.get('WT_mean', 0.0):.1f}%",
                    'TC': f"{dice_metrics.get('TC_mean', 0.0):.1f}%",
                    'ET': f"{dice_metrics.get('ET_mean', 0.0):.1f}%"
                })
                
                # Visualize batch
                if batch_idx in [1, 2, 3]:
                    print(f"Creating visualizations for validation batch {batch_idx}")
                    try:
                        # Add batch_idx to the prefix to create unique filenames
                        batch_prefix = f"val_batch{batch_idx}"
                        visualize_batch_comprehensive(images, masks, outputs, epoch, mode="hybrid", prefix=batch_prefix)
                        print(f"Visualization complete for batch {batch_idx}")
                    except Exception as e:
                        print(f"Error in visualization for batch {batch_idx}: {e}")
                        import traceback
                        traceback.print_exc()
               
                
                # Update total loss
                total_loss += loss.item()
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\nCUDA OOM in validation batch {batch_idx}. Skipping batch...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    print(f"\nError in validation batch {batch_idx}: {str(e)}")
                    raise e
        
        # Calculate average metrics
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {
            'mean_dice': np.mean([m['mean_dice'] for m in all_metrics]) if all_metrics else 0.0,
            'mean_iou': np.mean([m['mean_iou'] for m in all_metrics]) if all_metrics else 0.0,
            'bce_loss': np.mean([m['bce_loss'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_loss': np.mean([m['dice_loss'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_et': np.mean([m['dice_et'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_wt': np.mean([m['dice_wt'] for m in all_metrics]) if all_metrics else 0.0,
            'dice_tc': np.mean([m['dice_tc'] for m in all_metrics]) if all_metrics else 0.0,
'iou_et': np.mean([m['iou_et'] for m in all_metrics]) if all_metrics else 0.0,
            'iou_wt': np.mean([m['iou_wt'] for m in all_metrics]) if all_metrics else 0.0,
            'iou_tc': np.mean([m['iou_tc'] for m in all_metrics]) if all_metrics else 0.0
        }

        return avg_loss, avg_metrics

def save_training_history(history, filename):
   """
   Save training history plot with enhanced BraTS metrics
   """
   plt.figure(figsize=(16, 12))
   
   # Plot losses
   plt.subplot(3, 2, 1)
   plt.plot(history['train_loss'], label='Train Loss')
   plt.plot(history['val_loss'], label='Val Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.title('Training and Validation Loss')
   
   # Plot mean Dice scores
   plt.subplot(3, 2, 2)
   plt.plot(history['train_dice'], label='Train Mean Dice')
   plt.plot(history['val_dice'], label='Val Mean Dice')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Mean Dice Score')
   
   # Plot WT Dice scores
   plt.subplot(3, 2, 3)
   plt.plot(history['train_dice_wt'], label='Train WT')
   plt.plot(history['val_dice_wt'], label='Val WT')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Whole Tumor (WT) Dice Score')
   
   # Plot TC Dice scores
   plt.subplot(3, 2, 4)
   plt.plot(history['train_dice_tc'], label='Train TC')
   plt.plot(history['val_dice_tc'], label='Val TC')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Tumor Core (TC) Dice Score')
   
   # Plot ET Dice scores
   plt.subplot(3, 2, 5)
   plt.plot(history['train_dice_et'], label='Train ET')
   plt.plot(history['val_dice_et'], label='Val ET')
   plt.xlabel('Epoch')
   plt.ylabel('Dice Score (%)')
   plt.legend()
   plt.title('Enhancing Tumor (ET) Dice Score')
   
   # Plot Learning Rate
   plt.subplot(3, 2, 6)
   plt.plot(history['lr'])
   plt.xlabel('Epoch')
   plt.ylabel('Learning Rate')
   plt.yscale('log')
   plt.title('Learning Rate Schedule')
   
   plt.tight_layout()
   plt.savefig(f"results/{filename}")
   plt.close()
   
   # Create a second figure for IoU metrics
   plt.figure(figsize=(16, 12))
   
   # Plot mean IoU
   plt.subplot(2, 2, 1)
   plt.plot(history['train_iou'], label='Train Mean IoU')
   plt.plot(history['val_iou'], label='Val Mean IoU')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Mean IoU')
   
   # Plot WT IoU
   plt.subplot(2, 2, 2)
   plt.plot(history['train_iou_wt'], label='Train WT')
   plt.plot(history['val_iou_wt'], label='Val WT')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Whole Tumor (WT) IoU')
   
   # Plot TC IoU
   plt.subplot(2, 2, 3)
   plt.plot(history['train_iou_tc'], label='Train TC')
   plt.plot(history['val_iou_tc'], label='Val TC')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Tumor Core (TC) IoU')
   
   # Plot ET IoU
   plt.subplot(2, 2, 4)
   plt.plot(history['train_iou_et'], label='Train ET')
   plt.plot(history['val_iou_et'], label='Val ET')
   plt.xlabel('Epoch')
   plt.ylabel('IoU (%)')
   plt.legend()
   plt.title('Enhancing Tumor (ET) IoU')
   
   plt.tight_layout()
   plt.savefig(f"results/{filename.replace('.png', '_iou.png')}")
   plt.close()

    
    #running time plots
   plt.figure(figsize=(12, 6))
   
   plt.subplot(1, 2, 1)
   plt.plot(history['epoch_time'])
   plt.xlabel('Epoch')
   plt.ylabel('Time (seconds)')
   plt.title('Time per Epoch')
   
   plt.subplot(1, 2, 2)
   plt.plot(history['batch_time'])
   plt.xlabel('Epoch')
   plt.ylabel('Time (seconds)')
   plt.title('Average Time per Batch')
   
   plt.tight_layout()
   plt.savefig(f"results/{filename.replace('.png', '_timing.png')}")
   plt.close()
   
   # saving in text
   with open(f"results/{filename.replace('.png', '_timing_summary.txt')}", 'w') as f:
       f.write(f"Total training time: {sum(history['epoch_time']):.2f} seconds\n")
       f.write(f"Average epoch time: {np.mean(history['epoch_time']):.2f} seconds\n")
       f.write(f"Average batch time: {np.mean(history['batch_time']):.2f} seconds\n")
       f.write(f"First epoch time: {history['epoch_time'][0]:.2f} seconds\n")
       f.write(f"Last epoch time: {history['epoch_time'][-1]:.2f} seconds\n")

def preprocess_batch(batch, device=None):
    """
    Simplified preprocessing function that ensures consistent orientation
    between images and masks while preserving data integrity.
    """
    images, masks = batch
        
    # If shapes are different, we need to align them
    if images.shape != masks.shape:
        # Find which dimension in masks is 155 (standard BraTS depth)
        mask_155_dim = None
        for i in range(2, len(masks.shape)):
            if masks.shape[i] == 155:
                mask_155_dim = i
                break
                
        # Find which dimension in images is 155
        img_155_dim = None
        for i in range(2, len(images.shape)):
            if images.shape[i] == 155:
                img_155_dim = i
                break
        
        # If we found the 155 dimension in both, and they're different, permute masks
        if mask_155_dim is not None and img_155_dim is not None and mask_155_dim != img_155_dim:
            # Create a permutation that puts the mask's 155 dimension where the image has it
            perm = list(range(len(masks.shape)))
            perm[mask_155_dim], perm[img_155_dim] = perm[img_155_dim], perm[mask_155_dim]
            
            print(f"Permuting masks using order: {perm}")
            masks = masks.permute(*perm)
    
    # At this point, if shapes still don't match, it's likely due to a dimension order mismatch
    # Rather than create a new tensor, let's just ensure masks match images
    if images.shape != masks.shape:
        print(f"Warning: Image shape {images.shape} doesn't match mask shape {masks.shape}")
        print("Cannot safely reshape - using original masks")
        
    # Ensure mask values are within expected range
    masks = torch.clamp(masks, 0, 1)
    
    # Move to device if specified
    if device is not None:
        images = images.to(device)
        masks = masks.to(device)
    
    return images, masks


def train_model(data_path, batch_size=1, epochs=20, learning_rate=1e-3,
               use_mixed_precision=False, test_run=False, reset=True):
    """
    Optimized train function with BraTS-specific metrics
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
    
    # Make directories for results and checkpoints
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize AutoSAM2 model
    print("Initializing AutoSAM2 for multi-class segmentation")
    model = AutoSAM2(num_classes=4).to(device)
    
    # Check if model file exists to resume training
    model_path = "checkpoints/best_autosam2_model.pth"
    start_epoch = 0
    best_dice = 0.0
    
    if os.path.exists(model_path) and not reset:
        print(f"Found existing model checkpoint at {model_path}, loading...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            print(f"Resuming from epoch {start_epoch} with best Dice score: {best_dice:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch...")
            start_epoch = 0
            best_dice = 0.0
    else:
        print("No existing checkpoint found. Starting from epoch 0.")
    
    # Define loss criterion - use our new BraTS-specific loss
    criterion = BraTSCombinedLoss(dice_weight=0.75, bce_weight=0.15, focal_weight=0.1,
                                region_weights={'ET': 1.6, 'WT': 1.0, 'TC': 1.3})
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.encoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Load optimizer state if resuming
    if os.path.exists(model_path) and not reset and 'optimizer_state_dict' in locals().get('checkpoint', {}):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        except Exception as e:
            print(f"Error loading optimizer state: {e}. Using fresh optimizer.")
    
    # Get data loaders
    max_samples = 16 if test_run else None
    
    train_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=True,
        normalize=True, max_samples=max_samples, num_workers=4,
        filter_empty=False, use_augmentation=True,
        cache_data=False, verbose=True
    )

    val_loader = get_brats_dataloader(
        data_path, batch_size=batch_size, train=False,
        normalize=True, max_samples=max_samples, num_workers=4,
        filter_empty=False, use_augmentation=False,
        cache_data=False, verbose=True
    )
    
    # Set up OneCycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.4,
        div_factor=10,
        final_div_factor=200  
    )
    
    # Initialize history with all BraTS-specific metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_bce': [], 'val_bce': [],
        'train_dice_et': [], 'val_dice_et': [],
        'train_dice_wt': [], 'val_dice_wt': [],
        'train_dice_tc': [], 'val_dice_tc': [],
        'train_iou_et': [], 'val_iou_et': [],
        'train_iou_wt': [], 'val_iou_wt': [],
        'train_iou_tc': [], 'val_iou_tc': [],
        'lr': [],
        'epoch_time': [],
        'batch_time': []
    }
    
    # Early stopping parameters
    patience = 10
    counter = 0
    
    # Training loop
    training_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler)

        history['epoch_time'].append(train_metrics.get('epoch_time', 0.0))
        history['batch_time'].append(train_metrics.get('batch_time', 0.0))
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_metrics.get('mean_dice', 0.0))
        history['train_iou'].append(train_metrics.get('mean_iou', 0.0))
        history['train_bce'].append(train_metrics.get('bce_loss', 0.0))
        history['train_dice_et'].append(train_metrics.get('dice_et', 0.0))
        history['train_dice_wt'].append(train_metrics.get('dice_wt', 0.0))
        history['train_dice_tc'].append(train_metrics.get('dice_tc', 0.0))
        history['train_iou_et'].append(train_metrics.get('iou_et', 0.0))
        history['train_iou_wt'].append(train_metrics.get('iou_wt', 0.0))
        history['train_iou_tc'].append(train_metrics.get('iou_tc', 0.0))
        
        # Validate
        try:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
            
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_metrics.get('mean_dice', 0.0))
            history['val_iou'].append(val_metrics.get('mean_iou', 0.0))
            history['val_bce'].append(val_metrics.get('bce_loss', 0.0))
            history['val_dice_et'].append(val_metrics.get('dice_et', 0.0))
            history['val_dice_wt'].append(val_metrics.get('dice_wt', 0.0))
            history['val_dice_tc'].append(val_metrics.get('dice_tc', 0.0))
            history['val_iou_et'].append(val_metrics.get('iou_et', 0.0))
            history['val_iou_wt'].append(val_metrics.get('iou_wt', 0.0))
            history['val_iou_tc'].append(val_metrics.get('iou_tc', 0.0))
        
        except Exception as e:
            print(f"Error during validation: {e}")
            # Create default val_metrics when validation fails
            val_metrics = {
                'mean_dice': 0.0, 'mean_iou': 0.0, 'bce_loss': 0.0, 'dice_loss': 0.0,
                'dice_et': 0.0, 'dice_wt': 0.0, 'dice_tc': 0.0,
                'iou_et': 0.0, 'iou_wt': 0.0, 'iou_tc': 0.0
            }
            val_loss = float('inf')
            
            # Still update history with zeros
            history['val_loss'].append(val_loss)
            history['val_dice'].append(0.0)
            history['val_iou'].append(0.0)
            history['val_bce'].append(0.0)
            history['val_dice_et'].append(0.0)
            history['val_dice_wt'].append(0.0)
            history['val_dice_tc'].append(0.0)
            history['val_iou_et'].append(0.0)
            history['val_iou_wt'].append(0.0)
            history['val_iou_tc'].append(0.0)
                
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        estimated_total_time = elapsed_time / (epoch - start_epoch + 1) * (epochs - start_epoch) if epoch > start_epoch else epochs * epoch_time
        remaining_time = max(0, estimated_total_time - elapsed_time)
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Dice = {train_metrics.get('mean_dice', 0.0):.4f}, IoU = {train_metrics.get('mean_iou', 0.0):.4f}, BCE = {train_metrics.get('bce_loss', 0.0):.4f}")
        print(f"Epoch {epoch+1}: Val Dice = {val_metrics.get('mean_dice', 0.0):.4f}, IoU = {val_metrics.get('mean_iou', 0.0):.4f}, BCE = {val_metrics.get('bce_loss', 0.0):.4f}")
        print(f"Epoch Time: {timedelta(seconds=int(epoch_time))}, Remaining: {timedelta(seconds=int(remaining_time))}")
        
        # Print BraTS-specific metrics
        print("\nRegional Dice Scores:")
        print(f"  ET: Train={train_metrics.get('dice_et', 0.0):.2f}%, Val={val_metrics.get('dice_et', 0.0):.2f}%")
        print(f"  WT: Train={train_metrics.get('dice_wt', 0.0):.2f}%, Val={val_metrics.get('dice_wt', 0.0):.2f}%")
        print(f"  TC: Train={train_metrics.get('dice_tc', 0.0):.2f}%, Val={val_metrics.get('dice_tc', 0.0):.2f}%")
        
        # Save best model based on mean Dice score
        if val_metrics.get('mean_dice', 0.0) > best_dice:
            best_dice = val_metrics.get('mean_dice', 0.0)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'best_dice': best_dice,
            }, model_path)
            print(f"Saved new best model with Dice score: {best_dice:.4f}")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1  # Increment counter if no improvement
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, f"checkpoints/autosam2_model_epoch_{epoch}.pth")
            
            # Save training history
            save_training_history(history, f"training_history_epoch_{epoch}.png")
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Clean up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save final training history
    save_training_history(history, "final_training_history.png")
    
    # Display final metrics breakdown
    print("\n" + "="*50)
    print("Final metrics breakdown:")
    print("="*50)
    
    # Load the best model metrics
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            
            # Display region-specific metrics
            print("\nDice Scores:")
            for region in ['ET', 'WT', 'TC']:
                if f'{region}_mean' in val_metrics:
                    mean_val = val_metrics.get(f'dice_{region.lower()}', 0.0)
                    median_val = val_metrics.get(f'{region}_median', 0.0)
                    std_val = val_metrics.get(f'{region}_std', 0.0)
                    iqr_val = val_metrics.get(f'{region}_iqr', 0.0)
                    print(f"{region}: Mean={mean_val:.2f}%, Median={median_val:.2f}%, Std={std_val:.2f}%, IQR={iqr_val:.2f}%")
            
            print("\nIoU Scores:")
            for region in ['ET', 'WT', 'TC']:
                if f'{region}_mean' in val_metrics:
                    mean_val = val_metrics.get(f'iou_{region.lower()}', 0.0)
                    median_val = val_metrics.get(f'{region}_median', 0.0)
                    std_val = val_metrics.get(f'{region}_std', 0.0)
                    iqr_val = val_metrics.get(f'{region}_iqr', 0.0)
                    print(f"{region}: Mean={mean_val:.2f}%, Median={median_val:.2f}%, Std={std_val:.2f}%, IQR={iqr_val:.2f}%")
    
    print(f"Training complete! Best overall Dice score: {best_dice:.4f}")

    return model, history, best_metrics

def main():
    parser = argparse.ArgumentParser(description="Train AutoSAM2 for brain tumor segmentation")
    parser.add_argument('--data_path', type=str, default="/home/erezhuberman/data/Task01_BrainTumour",
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,  
                        help='Learning rate')
    parser.add_argument('--test_run', action='store_true',
                        help='Run with limited samples for testing')
    parser.add_argument('--reset', action='store_true', default=False,
                        help='Reset training from scratch, ignoring checkpoints')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--memory_limit', type=float, default=0.9,
                        help='Memory limit for GPU (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Set memory limit for GPU if available
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.memory_limit)
            print(f"Set GPU memory fraction to {args.memory_limit * 100:.1f}%")
        except Exception as e:
            print(f"Warning: Could not set GPU memory fraction: {e}")

    # Train the model
    try:
        model, history, best_metrics = train_model(
            data_path=args.data_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            use_mixed_precision=not args.no_mixed_precision,
            test_run=args.test_run,
            reset=args.reset,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return  # Exit if training fails

    # Before returning from main function
    try:
        if 'best_metrics' in locals() and best_metrics:
            print("\nFinal best metrics:")
            for key, value in best_metrics.items():
                print(f"{key}: {value:.4f}")
        else:
            print("\nNo final metrics were returned. Training completed anyway.")
    except:
        print("\nTraining completed, but could not display metrics.")

    # Print final metrics safely
    if best_metrics:
        print("\nFinal best metrics:")
        for key, value in best_metrics.items():
            print(f"{key}: {value:.4f}")
    else:
        print("\nNo final metrics were returned. Training may have failed.")

if __name__ == "__main__":
    main()
