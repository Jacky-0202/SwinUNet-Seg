# utils/plot.py

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def to_cpu_numpy(data):
    """Helper to convert data to cpu numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return np.array([x.detach().cpu().numpy() for x in data])
        else:
            return np.array(data)
    return np.array(data)

def plot_history(train_losses, val_losses, train_dice, val_dice, train_miou, val_miou, save_dir):
    """
    Plots Loss, Dice, and mIoU curves.
    """
    # Convert all inputs
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)
    train_dice = to_cpu_numpy(train_dice)
    val_dice = to_cpu_numpy(val_dice)
    train_miou = to_cpu_numpy(train_miou)
    val_miou = to_cpu_numpy(val_miou)
    
    epochs = range(1, len(train_losses) + 1)

    # [Modified] Change figure size and add 3rd subplot
    plt.figure(figsize=(18, 5))

    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Dice
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_dice, 'b-', label='Train Dice')
    plt.plot(epochs, val_dice, 'g-', label='Val Dice')
    plt.title('Dice Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)
    
    # 3. mIoU [New]
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_miou, 'b-', label='Train mIoU')
    plt.plot(epochs, val_miou, 'm-', label='Val mIoU') # Magenta color
    plt.title('mIoU Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"📊 Training curves saved at: {save_path}")