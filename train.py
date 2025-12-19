# train.py

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# --- Project Setup ---
# Ensure we can import from src/ and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Imports ---
import config as config
# [Modified] Import SwinUNet instead of ResUNet
from models.swinunet import SwinUNet 
from utils.dataset import MultiClassSegDataset, SynapseDataset
from utils.logger import CSVLogger
from utils.plot import plot_history
from utils.loss import SegmentationLoss
from utils.metrics import calculate_dice, calculate_miou

# --- 1. Helper Functions for Setup ---
def get_loaders():
    """Initializes and returns Train/Val DataLoaders."""
    print(f"📂 Dataset Root: {config.DATASET_ROOT}")
    
    # Update Dataset initialization to use specific paths from config
    # We now pass specific directories for images and masks directly
    
    # # --- Train Dataset ---
    # train_ds = MultiClassSegDataset(
    #     root_dir="", # Pass empty string or config.DATASET_ROOT (logic handled by img_folder/mask_folder)
    #     img_folder=config.TRAIN_IMG_DIR,   # Full path to training images
    #     mask_folder=config.TRAIN_MASK_DIR, # Full path to training masks
    #     mode='train',
    #     img_size=config.IMG_SIZE
    # )
    
    # # --- Val Dataset ---
    # val_ds = MultiClassSegDataset(
    #     root_dir="",
    #     img_folder=config.VAL_IMG_DIR,     # Full path to validation images
    #     mask_folder=config.VAL_MASK_DIR,   # Full path to validation masks
    #     mode='val',
    #     img_size=config.IMG_SIZE
    # )

# --- Train Dataset ---
    train_ds = SynapseDataset(
        root_dir=config.DATASET_ROOT,
        split='train',
        img_size=config.IMG_SIZE
    )
    
    # --- Val Dataset ---
    val_ds = SynapseDataset(
        root_dir=config.DATASET_ROOT,
        split='val',
        img_size=config.IMG_SIZE
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    print(f"✅ Data Loaded: Train={len(train_ds)}, Val={len(val_ds)}")
    return train_loader, val_loader

def get_model_components(device):
    """Initializes Model, Loss, Optimizer, Scaler, and Scheduler."""
    
    # Initialize SwinUNet
    # Make sure NUM_CLASSES matches the ADE20K dataset (151 classes)
    model = SwinUNet(
        n_classes=config.NUM_CLASSES, 
        img_size=config.IMG_SIZE,
        backbone_name=config.BACKBONE,
        pretrained=True
    ).to(device)
    
    loss_fn = SegmentationLoss(
        n_classes=config.NUM_CLASSES, 
        ignore_index=config.IGNORE_INDEX
    )
    
    # Use AdamW Optimizer (Recommended for Transformers)
    # weight_decay=1e-2 is standard for Swin Transformer
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    
    scaler = torch.amp.GradScaler()
    
    # Use Cosine Annealing with Warm Restarts Scheduler
    # T_0: Number of epochs for the first restart
    # T_mult: Factor to increase the cycle length after each restart
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    return model, loss_fn, optimizer, scaler, scheduler

# --- 2. Core Loops ---
def run_epoch(loader, model, optimizer, loss_fn, scaler, device, mode='train'):
    """
    Unified function for both Training and Validation loops.
    mode: 'train' or 'val' 
    """
    model.train() if mode == 'train' else model.eval()
    loop = tqdm(loader, desc=mode.capitalize(), leave=False)
    
    total_loss = 0.0
    total_dice = 0.0
    total_miou = 0.0
    
    # Enable gradient calculation only for training
    with torch.set_grad_enabled(mode == 'train'):
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            # Forward Pass (AMP)
            with torch.amp.autocast('cuda'):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # Backward Pass (Only for Train)
            if mode == 'train':
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Metrics (Get Tensors)
            preds_detached = predictions.detach()
            dice_tensor = calculate_dice(preds_detached, targets, n_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
            miou_tensor = calculate_miou(preds_detached, targets, n_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
            
            loss_val = loss.item()
            dice_val = dice_tensor.item()
            miou_val = miou_tensor.item()
            
            total_loss += loss_val
            total_dice += dice_val
            total_miou += miou_val

            loop.set_postfix(loss=f"{loss_val:.4f}", dice=f"{dice_val:.4f}", miou=f"{miou_val:.4f}")
            
    return total_loss / len(loader), total_dice / len(loader), total_miou / len(loader)

# --- 3. Main Execution ---
def main():
    print(f"--- SwinUNet with backbone: {config.BACKBONE}")
    print(f"--- Starting Training on {config.DEVICE} ---")
    print(f"--- Dataset: (Classes: {config.NUM_CLASSES}) ---")
    
    # A. Setup
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    logger = CSVLogger(save_dir=config.SAVE_DIR, filename='training_log.csv')
    
    # B. Load Data & Model
    train_loader, val_loader = get_loaders()
    model, loss_fn, optimizer, scaler, scheduler = get_model_components(config.DEVICE)

    # C. Training Loop
    best_miou = 0.0
    history = {'train_loss': [],
                'val_loss': [],
                'train_dice': [],
                'val_dice': [],
                'train_miou': [],
                'val_miou': []}

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        current_lr = optimizer.param_groups[0]['lr']

        # 1. Train & Validate
        train_loss, train_dice, train_miou = run_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, mode='train')
        val_loss, val_dice, val_miou = run_epoch(val_loader, model, None, loss_fn, None, config.DEVICE, mode='val')
        
        # Update Scheduler
        scheduler.step()

        # 2. Logging
        logger.log([epoch+1, current_lr, train_loss, train_dice, train_miou, val_loss, val_dice, val_miou])
        print(f"\tTrain Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | mIoU: {train_miou:.4f}")
        print(f"\tVal Loss:   {val_loss:.4f} | Dice: {val_dice:.4f} | mIoU: {val_miou:.4f}")
        
        # 3. Save History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)

        # 4. Save Checkpoints
        # Save best model based on mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best Model Saved! (mIoU: {best_miou:.4f})")
            
        # 5. Save the latest Checkpoints
        torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, f"last_model.pth"))

    print("\n🎉 Training Complete!")
    plot_history(
        history['train_loss'], history['val_loss'], 
        history['train_dice'], history['val_dice'], 
        history['train_miou'], history['val_miou'],
        save_dir=config.SAVE_DIR 
    )
    print(f"📈 Results saved to {config.SAVE_DIR}")

if __name__ == "__main__":
    main()