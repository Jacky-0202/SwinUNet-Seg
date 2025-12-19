# utils/metrics.py

import torch
import torch.nn.functional as F

def calculate_dice(logits, targets, n_classes=1, ignore_index=255, exclude_background=True):
    """
    Calculate Mean Dice Score.
    Returns a Tensor (not a float) so operations remain in PyTorch graph if needed.
    """
    smooth = 1e-6
    
    # --- Binary Mode ---
    if n_classes == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        targets = targets.float()
        
        if ignore_index is not None:
            mask = (targets != ignore_index)
            preds = preds * mask
            targets = targets * mask

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice  # [Modified] Return Tensor

    # --- Multi-class Mode ---
    else:
        if logits.dim() == 4:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits

        dice_scores = []
        start_class = 1 if exclude_background else 0
        
        for c in range(start_class, n_classes):
            pred_binary = (preds == c)
            target_binary = (targets == c)
            
            if ignore_index is not None:
                valid_mask = (targets != ignore_index)
                pred_binary = pred_binary & valid_mask
                target_binary = target_binary & valid_mask

            intersection = (pred_binary & target_binary).sum().float()
            union = pred_binary.sum().float() + target_binary.sum().float()
            
            if union == 0:
                continue
            
            score = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(score) # Store Tensor
            
        if len(dice_scores) == 0:
            return torch.tensor(0.0, device=logits.device)
            
        # Stack tensors and take mean
        return torch.stack(dice_scores).mean()


def calculate_miou(logits, targets, n_classes=1, ignore_index=255, exclude_background=True):
    """
    Calculate Mean IoU.
    Returns a Tensor.
    """
    smooth = 1e-6

    # --- Binary Mode ---
    if n_classes == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        targets = targets.float()

        if ignore_index is not None:
            mask = (targets != ignore_index)
            preds = preds * mask
            targets = targets * mask
            
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou # Return Tensor

    # --- Multi-class Mode ---
    else:
        if logits.dim() == 4:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits

        iou_scores = []
        start_class = 1 if exclude_background else 0
        
        for c in range(start_class, n_classes):
            pred_binary = (preds == c)
            target_binary = (targets == c)
            
            if ignore_index is not None:
                valid_mask = (targets != ignore_index)
                pred_binary = pred_binary & valid_mask
                target_binary = target_binary & valid_mask

            intersection = (pred_binary & target_binary).sum().float()
            union = pred_binary.sum().float() + target_binary.sum().float() - intersection
            
            if union == 0:
                continue
            
            score = (intersection + smooth) / (union + smooth)
            iou_scores.append(score) # Store Tensor

        if len(iou_scores) == 0:
            return torch.tensor(0.0, device=logits.device)
            
        # Stack tensors and take mean
        return torch.stack(iou_scores).mean()