import os
import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config as config
from models.swinunet import SwinUNet 

# --- Settings ---
TEST_DIR = os.path.join(config.DATASET_ROOT, 'test_vol_h5')
OUTPUT_DIR = 'test_results_synapse'
MODEL_PATH = os.path.join(config.SAVE_DIR, 'best_model.pth')

# Organ List (Synapse 9 classes)
CLASSES = ['Background', 'Aorta', 'Gallbladder', 'Kidney(L)', 'Kidney(R)', 'Liver', 'Pancreas', 'Spleen', 'Stomach']

def calculate_dice(pred, target, class_idx):
    """Calculate 3D Dice Score for a single organ"""
    pred_c = (pred == class_idx)
    target_c = (target == class_idx)
    
    if target_c.sum() == 0:
        return 1.0 if pred_c.sum() == 0 else 0.0
        
    intersection = (pred_c & target_c).sum()
    union = pred_c.sum() + target_c.sum()
    return (2. * intersection) / (union + 1e-5)

def inference_single_slice(image_slice, model, device):
    """Inference on a single slice"""
    h, w = image_slice.shape
    
    # 1. Preprocess
    # image_slice: (H, W) -> Tensor (1, 1, H, W)
    x = torch.from_numpy(image_slice).float().unsqueeze(0).unsqueeze(0)
    
    # Repeat channels (1->3) for Swin
    x = x.repeat(1, 3, 1, 1)
    
    # Resize to model input size (e.g., 224)
    if h != config.IMG_SIZE or w != config.IMG_SIZE:
        x = F.interpolate(x, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear', align_corners=False)
        
    x = x.to(device)
    
    # 2. Predict
    with torch.no_grad():
        out = model(x)
        out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
        
    # 3. Resize back to original size (Nearest Neighbor for Mask)
    if h != config.IMG_SIZE or w != config.IMG_SIZE:
        out = out.unsqueeze(0).unsqueeze(0).float()
        out = F.interpolate(out, size=(h, w), mode='nearest')
        out = out.squeeze().long()
        
    return out.cpu().numpy()

def save_visual_sample(image_vol, label_vol, pred_vol, case_name, save_dir):
    """
    Find the slice with the largest organ area and visualize it
    """
    # Find slice index with the most organ pixels
    sums = np.sum(label_vol > 0, axis=(0, 1)) # Sum along H, W
    best_slice_idx = np.argmax(sums)
    
    img = image_vol[:, :, best_slice_idx]
    gt = label_vol[:, :, best_slice_idx]
    pred = pred_vol[:, :, best_slice_idx]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title(f"{case_name} (Slice {best_slice_idx})\nOriginal"); plt.imshow(img, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(gt, cmap='tab10', vmin=0, vmax=8, interpolation='nearest'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Prediction"); plt.imshow(pred, cmap='tab10', vmin=0, vmax=8, interpolation='nearest'); plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{case_name}_vis.png"))
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE
    
    print(f"🚀 Loading Model: {MODEL_PATH}")
    model = SwinUNet(n_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE, backbone_name=config.BACKBONE, pretrained=False).to(device)
    
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in ckpt: model.load_state_dict(ckpt['model_state_dict'])
        else: model.load_state_dict(ckpt)
        model.eval()
    else:
        print("❌ Model not found!")
        return

    # 1. Find h5 files
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test dir not found: {TEST_DIR}")
        return
        
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.h5')])
    print(f"📂 Found {len(files)} test volumes.")
    
    total_dice = []

    for file_name in files:
        case_name = file_name.replace('.h5', '')
        file_path = os.path.join(TEST_DIR, file_name)
        
        # 2. Load Volume
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:] # (H, W, D) or (D, H, W), usually Synapse is (Slice, H, W)
            label = f['label'][:]
            
            # Check shape orientation
            # Assuming it is (Slice, H, W), we handle it accordingly
            if image.shape[0] > image.shape[1] and image.shape[0] > image.shape[2]:
                # Could be (Slice, H, W), common in medical data
                pass 
            else:
                # Ensure we iterate over the Slice dimension
                # Based on Synapse common format, usually (Slice, H, W)
                pass

        print(f"\n🧠 Processing {case_name} | Shape: {image.shape}")
        
        # 3. Slice-by-slice Inference
        pred_volume = np.zeros_like(label)
        
        # Assuming dimension is (Slice, H, W)
        num_slices = image.shape[0]
        
        for i in tqdm(range(num_slices), leave=False):
            slice_img = image[i, :, :]
            pred_mask = inference_single_slice(slice_img, model, device)
            pred_volume[i, :, :] = pred_mask
            
        # 4. Calculate Metrics (3D Dice)
        print(f"   📊 Metrics for {case_name}:")
        case_dices = []
        for i in range(1, 9): # Class 1 to 8 (Skip background)
            dice = calculate_dice(pred_volume, label, i)
            case_dices.append(dice)
            print(f"      - {CLASSES[i]}: {dice:.4f}")
            
        mean_dice = np.mean(case_dices)
        total_dice.append(mean_dice)
        print(f"   👉 Mean Dice: {mean_dice:.4f}")
        
        # 5. Visualize (Transpose back for plotting: Slice, H, W -> H, W, Slice)
        save_visual_sample(
            image.transpose(1, 2, 0), 
            label.transpose(1, 2, 0), 
            pred_volume.transpose(1, 2, 0), 
            case_name, OUTPUT_DIR
        )

    print(f"\n🏆 Final Average Dice (over {len(files)} cases): {np.mean(total_dice):.4f}")
    print(f"🖼️  Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()