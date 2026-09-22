import torch
import torch.nn as nn
import timm
from .multimodal_hybrid import CAFFM

class AdvancedMultimodalHybrid(nn.Module):
    def __init__(self, num_tabular_features=15, num_classes=9, dropout_rate=0.4):
        super().__init__()
        
        # EfficientNet configured to expose spatial features
        self.cnn = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0) 
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        
        img_dim = self.cnn.num_features + self.swin.num_features
        self.caffm = CAFFM(img_dim=img_dim, tab_dim=num_tabular_features, embed_dim=256)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_tab):
        # 1. Extract raw 7x7 spatial geometry from the CNN: (Batch, 1792, 7, 7)
        cnn_spatial = self.cnn.forward_features(x_img)
        
        # 2. Collapse the 1792 channels into a single heatmap: (Batch, 7, 7)
        # This represents EXACTLY where the network is looking
        attention_map = torch.mean(cnn_spatial, dim=1)
        
        # 3. Proceed with normal global pooling and multimodal fusion
        cnn_feat = self.cnn.global_pool(cnn_spatial)
        swin_feat = self.swin(x_img)
        
        img_feat = torch.cat((cnn_feat, swin_feat), dim=1)
        fused_feat = self.caffm(img_feat, x_tab)
        out = self.classifier(fused_feat)
        
        # We return both the final prediction and the spatial attention map
        return out, attention_map
