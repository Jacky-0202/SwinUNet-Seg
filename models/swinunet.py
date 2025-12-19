# models/swinunet.py

import torch.nn as nn
import timm
from models.blocks import Up, OutConv

class SwinUNet(nn.Module):
    def __init__(self, n_classes=151, img_size=224, backbone_name='swin_tiny_patch4_window7_224', pretrained=True):
        """
        Swin-UNet with selectable backbone.
        
        Args:
            n_classes (int): Number of output classes.
            img_size (int): Input image size (e.g., 224 or 384).
            backbone_name (str): Name of the timm Swin Transformer model.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(SwinUNet, self).__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        
        # 1. Create Backbone using timm
        # features_only=True returns the feature maps from intermediate layers
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size
        )
        
        # 2. Determine Channel Dimensions dynamically based on backbone type
        # Swin Tiny/Small start with 96 channels
        # Swin Base starts with 128 channels
        # Swin Large starts with 192 channels
        if 'tiny' in backbone_name or 'small' in backbone_name:
            embed_dim = 96
        elif 'base' in backbone_name:
            embed_dim = 128
        elif 'large' in backbone_name:
            embed_dim = 192
        else:
            # Fallback default (safe for tiny/small)
            print(f"⚠️ Warning: Unknown backbone '{backbone_name}', defaulting to embed_dim=96")
            embed_dim = 96

        # Calculate dims for 4 stages: [C, 2C, 4C, 8C]
        # e.g., Tiny: [96, 192, 384, 768]
        # e.g., Base: [128, 256, 512, 1024]
        dims = [embed_dim * (2 ** i) for i in range(4)]
        print(f"   -> Channel dims: {dims}")
        
        # -----------------------------------------------------------------
        # Decoder (U-Net Style with Skip Connections)
        # -----------------------------------------------------------------
        
        # Up 1: Input (Stage 3) + Skip (Stage 2) -> Output (Stage 2 dim)
        self.up1 = Up(dims[3], dims[2], dims[2]) 
        
        # Up 2: Input (Stage 2) + Skip (Stage 1) -> Output (Stage 1 dim)
        self.up2 = Up(dims[2], dims[1], dims[1])
        
        # Up 3: Input (Stage 1) + Skip (Stage 0) -> Output (Stage 0 dim)
        self.up3 = Up(dims[1], dims[0], dims[0])
        
        # -----------------------------------------------------------------
        # Final Upsampling & Head
        # -----------------------------------------------------------------
        # Stage 4: H/4 (Stage 0 dim) -> H/2 (Stage 0 dim / 2)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[0], dims[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.ReLU(inplace=True)
        ) 
        
        # Stage 5: H/2 -> H (Output resolution)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(dims[0] // 2, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 Conv to get class logits
        self.outc = OutConv(24, n_classes)

    def forward(self, x):
        """
        Forward pass of the network.
        x shape: (Batch_Size, 3, H, W)
        """
        
        # --- Encoder Path (Swin Transformer) ---
        features_raw = self.backbone(x)
        
        features = []
        for f in features_raw:
            # Swin Transformer usually outputs channels last (N, H, W, C).
            # We need to permute to (N, C, H, W) for the Decoder.
            if len(f.shape) == 4 and f.shape[1] != f.shape[-1]: 
               # Safe check: Permute to (N, C, H, W)
               f = f.permute(0, 3, 1, 2).contiguous()
            features.append(f)

        x0 = features[0] # Stage 0
        x1 = features[1] # Stage 1
        x2 = features[2] # Stage 2
        x3 = features[3] # Stage 3 (Bottleneck)
        
        # --- Decoder Path ---
        d1 = self.up1(x3, x2) 
        d2 = self.up2(d1, x1) 
        d3 = self.up3(d2, x0) 
        
        # --- Final Restoration ---
        d4 = self.up4(d3)     
        d_final = self.up5(d4)
        
        logits = self.outc(d_final)
        
        return logits