# predict_ade.py

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config as config
from models.swinunet import SwinUNet 

# --- 1. Settings ---
INPUT_DIR = 'test_data'    
OUTPUT_DIR = 'test_results'  

MODEL_PATH = '/home/tec/Desktop/Project/SwinUNet-Seg/checkpoints/Swin_Tiny_ADE20K/best_model.pth' 

# --- 2. ADE20K Palette Generator ---
def get_ade20k_palette(num_classes=151):
    """
    Generate a stable color palette for ADE20K visualization.
    Uses a fixed random seed so colors remain consistent across runs.
    """
    np.random.seed(42) # Fixed seed for consistency
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    
    # Optional: Set class 0 (Background) to Black if desired
    # palette[0] = [0, 0, 0] 
    
    return palette

# Generate the palette once
ADE_PALETTE = get_ade20k_palette(config.NUM_CLASSES)

def process_image(img_path, model, device, transform):
    """
    Load Image -> Predict -> Resize back to original dimensions -> Return Mask
    """
    # 1. Load Image & Get Original Size
    original_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = original_img.size
    
    # 2. Preprocess
    # Swin needs a specific input size (e.g., 512x512)
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor)
        # output shape: (1, 151, H, W)
        pred_mask_fixed = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
    # 4. Resize Mask back to Original Size
    # Use Nearest Neighbor to keep integer class IDs correct
    pred_mask_orig = cv2.resize(pred_mask_fixed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
    return pred_mask_orig

def save_mask_visual(pred_mask, save_path):
    """
    Save the colorized mask
    """
    # Ensure mask values don't exceed our palette size
    # This protects against index errors if model predicts a weird class
    max_cls = pred_mask.max()
    if max_cls >= len(ADE_PALETTE):
        print(f"⚠️ Warning: Predicted class {max_cls} exceeds palette size {len(ADE_PALETTE)}")
        return

    # 1. Apply Color Map
    # pred_mask (H, W) -> color_mask (H, W, 3)
    color_mask = ADE_PALETTE[pred_mask]
    
    # 2. Save (Convert RGB -> BGR for OpenCV)
    cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

def main():
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"📁 Created '{INPUT_DIR}'. Please put your test images there.")
        return

    device = config.DEVICE
    print(f"🚀 Loading ADE20K Model from: {MODEL_PATH}")
    
    # Initialize Model
    # ADE20K usually has 150 classes + 1 background = 151
    model = SwinUNet(n_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE, pretrained=False).to(device)
    
    # Load Weights
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print("✅ Model weights loaded.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print(f"❌ Weight file not found: {MODEL_PATH}")
        return

    # Transform (Must match training)
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get Images
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_ext)]
    
    if not image_files:
        print(f"⚠️ No images in '{INPUT_DIR}'")
        return

    print(f"📂 Found {len(image_files)} images. Processing...")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, img_name)
        save_name = os.path.splitext(img_name)[0] + ".png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        try:
            mask = process_image(img_path, model, device, transform)
            save_mask_visual(mask, save_path)
        except Exception as e:
            print(f"⚠️ Error on {img_name}: {e}")

    print(f"\n✅ Done! Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()