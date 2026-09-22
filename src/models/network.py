import torch
import torch.nn as nn
import timm
from monai.networks.nets import SwinUNETR

class LesionClassifier(nn.Module):
    """Vision Transformer for 7-class skin lesion diagnosis (Will be upgraded in Phase 3)."""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        return self.backbone.forward_features(x)

class BoundaryAttentionHead(nn.Module):
    """
    Custom 2-Conv layer head designed to predict high-frequency spatial boundaries.
    This represents the 'MPBA-Inspired' novelty for the thesis.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        boundary_logits = self.conv2(feat)
        return boundary_logits, self.sigmoid(boundary_logits)

class MPBASwinUNETR(nn.Module):
    """
    Primary Phase 2 Segmentation Architecture.
    Combines MONAI's SwinUNETR (Hybrid CNN-Transformer) with the custom Boundary Attention Head.
    """
    def __init__(self, img_size=(224, 224), in_channels=3, num_classes=1):
        super().__init__()
        # Official MONAI SwinUNETR backend. 
        # We output a 32-channel feature map instead of a final 1-channel mask.
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=32, 
            feature_size=24,
            spatial_dims=2
        )
        
        # Custom Boundary Head
        self.boundary_head = BoundaryAttentionHead(in_channels=32)
        
        # Final Segmentation Head
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 1. Base Hybrid Feature Extraction
        features = self.swin_unetr(x)
        
        # 2. Extract Boundary Map
        boundary_logits, boundary_probs = self.boundary_head(features)
        
        # 3. Attention Fusion: Enhance main features with boundary structural knowledge
        attended_features = features * (1 + boundary_probs)
        
        # 4. Final Lesion Mask Prediction
        mask_logits = self.seg_head(attended_features)
        
        # Both logits are returned so we can calculate a combined Dice/BCE loss during training
        return mask_logits, boundary_logits