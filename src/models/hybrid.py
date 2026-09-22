import torch
import torch.nn as nn
import timm

class EfficientSwinHybrid(nn.Module):
    def __init__(self, num_classes=9, dropout_rate=0.4):
        super().__init__()
        
        # 1. CNN Branch: EfficientNet-B4 (Focuses on local texture, borders, and color variants)
        # num_classes=0 removes the final classification layer, returning raw 1792-dim features
        self.cnn = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0) 
        
        # 2. Transformer Branch: Swin-Tiny (Focuses on global context and geometry)
        # num_classes=0 returns raw 768-dim features
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        
        cnn_dim = self.cnn.num_features   # 1792
        swin_dim = self.swin.num_features # 768
        
        # 3. Multimodal Fusion Head (Combines 1792 + 768 = 2560 features)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + swin_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        swin_feat = self.swin(x)
        
        # Concatenate along the feature dimension
        fused_feat = torch.cat((cnn_feat, swin_feat), dim=1)
        out = self.fusion(fused_feat)
        return out

# Test Block
if __name__ == "__main__":
    print("Initializing EfficientNet-B4 + Swin-Tiny Hybrid Architecture...")
    model = EfficientSwinHybrid(num_classes=9)
    print("Testing forward pass with dummy tensor (Batch Size 2, 3 Channels, 224x224)...")
    
    # Simulate a batch of 2 images passing through the network
    dummy_input = torch.randn(2, 3, 224, 224) 
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape} (Expected: [2, 9])")
    print("Hybrid architecture successfully compiled and ready for training!")
