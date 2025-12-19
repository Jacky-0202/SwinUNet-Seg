# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes=1, smooth=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Robust Dice Loss that handles ignore_index safely.
        """
        # --- Binary Mode ---
        if self.n_classes == 1:
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)
            
            if self.ignore_index is not None:
                mask = (targets_flat != self.ignore_index)
                probs_flat = probs_flat[mask]
                targets_flat = targets_flat[mask]

            intersection = (probs_flat * targets_flat).sum()
            dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
            return 1 - dice

        # --- Multi-class Mode ---
        else:
            probs = torch.softmax(logits, dim=1) # (B, C, H, W)
            
            # [Critical Fix] Handle ignore_index (e.g., 255) to prevent crash
            if self.ignore_index is not None:
                # 1. Create valid pixel mask (B, H, W)
                valid_mask = (targets != self.ignore_index).float()
                
                # 2. Safe conversion: Temporarily change ignore_index to 0 to avoid one_hot error
                # (Since we will zero out these positions with valid_mask later, changing to 0 here is fine)
                targets_clamped = targets.clone()
                targets_clamped[targets == self.ignore_index] = 0
                
                # 3. Convert to One-Hot (B, C, H, W)
                targets_one_hot = F.one_hot(targets_clamped, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
                
                # 4. Apply mask: Zero out areas that were originally ignore_index (excluded from Loss calculation)
                targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)
                
                # Zero out corresponding positions in predictions so model doesn't learn from them
                probs = probs * valid_mask.unsqueeze(1) 

            else:
                targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

            # Flatten for calculation
            probs_flat = probs.contiguous().view(self.n_classes, -1)
            targets_flat = targets_one_hot.contiguous().view(self.n_classes, -1)
            
            # Calculate Dice for each class
            intersection = (probs_flat * targets_flat).sum(dim=1)
            union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
            
            # Calculate Mean Dice (add smooth term to prevent division by zero)
            dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
            
            return 1 - dice_per_class.mean()

class SegmentationLoss(nn.Module):
    def __init__(self, n_classes=1, weight_dice=0.5, weight_ce=0.5, ignore_index=None):
        """
        Combined Loss: CrossEntropy + Dice
        Args:
            n_classes (int): Number of classes.
            weight_dice (float): Weight for Dice Loss.
            weight_ce (float): Weight for CrossEntropy/BCE Loss.
            ignore_index (int, optional): Index to ignore in CrossEntropy. Default: None.
        """
        super(SegmentationLoss, self).__init__()
        self.n_classes = n_classes
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        
        # [Important] Must pass ignore_index here
        self.dice_loss = DiceLoss(n_classes=n_classes, ignore_index=ignore_index)
        
        if n_classes == 1:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            # CrossEntropy natively supports ignore_index, just set it here
            if ignore_index is not None:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # 1. Calculate CrossEntropy Loss (Pixel-level classification accuracy)
        if self.n_classes == 1:
            loss_ce = self.ce_loss(logits, targets.float())
        else:
            loss_ce = self.ce_loss(logits, targets.long())
            
        # 2. Calculate Dice Loss (Shape and Overlap)
        loss_dice = self.dice_loss(logits, targets)
        
        # 3. Weighted Sum
        return (self.weight_ce * loss_ce) + (self.weight_dice * loss_dice)