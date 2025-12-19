# utils/dataset.py

import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class BinarySegDataset(Dataset):
    def __init__(self, root_dir, img_folder='im', mask_folder='gt', mode='train', img_size=320):
        """
        Generic Dataset class for Binary Segmentation with Data Augmentation.
        
        Args:
            root_dir (str): Root directory of the dataset.
            img_folder (str): Name of the folder containing images.
            mask_folder (str): Name of the folder containing masks.
            mode (str): 'train' or 'test'. Augmentation is applied only in 'train' mode.
            img_size (int): Target size for resizing images and masks.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        # 1. Construct full paths
        self.image_dir = os.path.join(root_dir, img_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Directories not found. Please check paths:\nImg: {self.image_dir}\nMask: {self.mask_dir}")

        # 2. Get all image filenames
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_ext)])
        
        if len(self.image_list) == 0:
            print(f"⚠️ Warning: No images found in {self.image_dir}!")

        # 3. Define Base Transforms (Resize & Normalization)
        # Note: We don't put random transforms here because we need to sync Image/Mask manually.
        
        # Image Normalization (ImageNet standards)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Resize allows us to handle input of any size
        self.resize = transforms.Resize((self.img_size, self.img_size))
        self.resize_mask = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 1. Get image filename
        img_name = self.image_list[idx]
        
        # 2. Handle corresponding Mask filename (Force .png extension)
        file_prefix = os.path.splitext(img_name)[0]
        mask_name = file_prefix + '.png'
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 3. Load Image and Mask
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L') # L = Grayscale
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Mask not found for image: {mask_path}")
        
        # 4. Data Augmentation (Only for Training)
        # SOTA Technique: Random Geometric Transformations
        if self.mode == 'train':
            # Random Horizontal Flip (50% prob)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # Random Vertical Flip (50% prob)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random Rotation (-10 to 10 degrees)
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # 5. Apply Resizing and ToTensor
        # Image: Resize -> ToTensor -> Normalize
        image = self.resize(image)
        image = TF.to_tensor(image)
        image = self.norm(image)
        
        # Mask: Resize -> ToTensor
        mask = self.resize_mask(mask)
        mask = TF.to_tensor(mask)
        
        # 6. Binarize Mask (0.0 / 1.0)
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        
        return image, mask
    
    
class MultiClassSegDataset(Dataset):
    def __init__(self, root_dir, img_folder='images', mask_folder='annotations', mode='train', img_size=512):
        """
        Generic Multi-class Dataset (Compatible with ADE20K & LaPa).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        self.image_dir = os.path.join(root_dir, img_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Directories not found:\nImg: {self.image_dir}\nMask: {self.mask_dir}")

        # Filter valid pairs
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        
        # Get all mask filenames (without extension) for quick lookup
        available_masks = set(os.path.splitext(f)[0] for f in os.listdir(self.mask_dir) if not f.startswith('.'))
        
        self.image_list = []
        for f in sorted(os.listdir(self.image_dir)):
            if f.lower().endswith(valid_ext):
                file_id = os.path.splitext(f)[0]
                if file_id in available_masks:
                    self.image_list.append(f)
        
        print(f"Dataset ({mode}): Found {len(self.image_list)} valid pairs.")

        # Transforms (ImageNet Normalization)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        file_id = os.path.splitext(img_name)[0]
        
        # ADE20K masks are usually .png
        mask_name = file_id + '.png'
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path) 
        except Exception as e:
            print(f"Error loading: {img_path} or {mask_path}")
            raise e
        
        # --- Data Augmentation ---
        if self.mode == 'train':
            # 1. Random Resized Crop
            if random.random() > 0.3:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.5, 1.0), ratio=(0.75, 1.33)
                )
                image = TF.resized_crop(image, i, j, h, w, (self.img_size, self.img_size), Image.BILINEAR)
                mask = TF.resized_crop(mask, i, j, h, w, (self.img_size, self.img_size), Image.NEAREST)
            else:
                image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
                mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
            
            # 2. Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # 3. Color Jitter (Only apply to image)
            if random.random() > 0.2:
                jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                image = jitter(image)
                
        else:
            # Validation: Simple Resize
            image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        # To Tensor & Normalize
        image = TF.to_tensor(image)
        image = self.norm(image)
        
        # Convert Mask to Numpy -> LongTensor
        mask_np = np.array(mask)
        
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]
        
        mask_tensor = torch.from_numpy(mask_np.copy()).long() 
        
        return image, mask_tensor
    
class SynapseDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        """
        Dataset for Synapse Multi-organ Segmentation.
        Reads .npz files from 'train_npz' folder.
        
        Args:
            root_dir (str): Path to Synapse folder (containing train_npz).
            split (str): 'train' or 'val'.
            img_size (int): Target size (usually 224 for Synapse).
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # 1. Find all .npz files
        npz_dir = os.path.join(root_dir, 'train_npz')
        
        if not os.path.exists(npz_dir):
            raise FileNotFoundError(f"❌ 'train_npz' folder not found: {npz_dir}")
            
        all_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        
        if len(all_files) == 0:
            raise RuntimeError(f"⚠️ No .npz files found in {npz_dir}!")

        # 2. Automatically split Train/Val (80% / 20%)
        # Use a fixed seed to ensure consistent train/val splits across runs.
        random.seed(1234) 
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * 0.8)
        
        if split == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        print(f"🧠 Synapse Dataset ({split}): Found {len(self.files)} slices.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # 1. Load .npz
        # Synapse .npz usually contains keys: 'image', 'label'
        data = np.load(file_path)
        image = data['image'] # (H, W) float
        label = data['label'] # (H, W) int (0~8)
        
        # 2. To Tensor
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # 3. Channel Expansion (1 -> 3)
        # Swin Transformer pretrained models require 3-channel input.
        # image shape: (H, W) -> (1, H, W) -> (3, H, W)
        image = image.unsqueeze(0).repeat(3, 1, 1)
        
        # 4. Resize
        # Note: Mask must use Nearest Neighbor interpolation to avoid introducing floating point values.
        if image.shape[-1] != self.img_size:
            image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
            # Mask needs to be unsqueezed to (1, H, W) before resizing, then squeezed back.
            label = TF.resize(label.unsqueeze(0), (self.img_size, self.img_size), interpolation=Image.NEAREST).squeeze(0)
            
        # 5. Data Augmentation (Training only)
        if self.split == 'train':
            # Random Rotation (-15 to 15 degrees)
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                image = TF.rotate(image, angle)
                label = TF.rotate(label.unsqueeze(0), angle).squeeze(0)
                
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                
            # Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

        return image, label