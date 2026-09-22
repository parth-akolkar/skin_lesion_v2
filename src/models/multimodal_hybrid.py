import torch
import torch.nn as nn
import timm

class CAFFM(nn.Module):
    def __init__(self, img_dim, tab_dim, embed_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        # Project both modalities to the exact same dimension for attention mathematical alignment
        self.img_proj = nn.Linear(img_dim, embed_dim)
        self.tab_proj = nn.Linear(tab_dim, embed_dim)
        
        # Bidirectional Cross-Attention
        # 1. Image queries the Tabular context
        self.cross_att_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # 2. Tabular queries the Image context
        self.cross_att_tab = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, img_feat, tab_feat):
        # Un-squeeze to create a sequence of length 1: Shape becomes (Batch, 1, Embed_Dim)
        img_q = self.img_proj(img_feat).unsqueeze(1)
        tab_kv = self.tab_proj(tab_feat).unsqueeze(1)
        
        # Stream 1: Image Attends to Tabular
        attn_img, _ = self.cross_att_img(query=img_q, key=tab_kv, value=tab_kv)
        img_out = self.norm1(img_q + attn_img).squeeze(1) 
        
        # Stream 2: Tabular Attends to Image
        attn_tab, _ = self.cross_att_tab(query=tab_kv, key=img_q, value=img_q)
        tab_out = self.norm2(tab_kv + attn_tab).squeeze(1)
        
        # Concatenate the context-aware features back together
        return torch.cat([img_out, tab_out], dim=1)

class MultimodalHybrid(nn.Module):
    def __init__(self, num_tabular_features=15, num_classes=9, dropout_rate=0.4):
        super().__init__()
        
        # 1. Visual Branch (EfficientNet-B4 + Swin-Tiny)
        self.cnn = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0) 
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        img_dim = self.cnn.num_features + self.swin.num_features # 1792 + 768 = 2560
        
        # 2. Fusion Branch
        self.caffm = CAFFM(img_dim=img_dim, tab_dim=num_tabular_features, embed_dim=256)
        
        # 3. Final Classification Head (256 + 256 = 512 dimensions)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_tab):
        cnn_feat = self.cnn(x_img)
        swin_feat = self.swin(x_img)
        img_feat = torch.cat((cnn_feat, swin_feat), dim=1)
        
        fused_feat = self.caffm(img_feat, x_tab)
        out = self.classifier(fused_feat)
        
        return out

# Test Block
if __name__ == "__main__":
    print("Initializing CAFFM Multimodal Architecture...")
    model = MultimodalHybrid(num_tabular_features=15, num_classes=9)
    dummy_img = torch.randn(2, 3, 224, 224) 
    dummy_tab = torch.randn(2, 15) 
    output = model(dummy_img, dummy_tab)
    print(f"Output shape: {output.shape} (Expected: [2, 9])")
    print("Multimodal network successfully compiled!")
