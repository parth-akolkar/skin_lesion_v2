import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Official XAI Library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.multimodal_dataset import ISIC2019MultimodalDataset
from src.models.multimodal_hybrid import MultimodalHybrid

# --- 1. Multimodal Wrapper ---
# Grad-CAM expects a model with a single image input. 
# This wrapper holds the clinical metadata in memory and passes it under the hood.
class MultimodalGradCamWrapper(torch.nn.Module):
    def __init__(self, base_model, metadata):
        super().__init__()
        self.base_model = base_model
        self.metadata = metadata

    def forward(self, img):
        return self.base_model(img, self.metadata)

def main():
    print("=== Phase 5: Grad-CAM Interpretability ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    print("Loading trained CAFFM architecture...")
    base_model = MultimodalHybrid(num_tabular_features=15, num_classes=9).to(device)
    base_model.load_state_dict(torch.load('weights/caffm_multimodal_best.pth', weights_only=True))
    base_model.eval()

    # In timm's EfficientNet-B4, 'conv_head' is the final spatial convolutional layer
    # This is where the richest spatial geometry is stored before it gets flattened.
    target_layers = [base_model.cnn.conv_head]

    # Dataset Setup (We only need eval transforms, no augmentations)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading test dataset samples...")
    dataset = ISIC2019MultimodalDataset(
        gt_csv='datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv',
        meta_csv='datasets/raw_images/archive/ISIC_2019_Training_Metadata.csv',
        abcd_csv='datasets/abcd_features.csv',
        img_dir='datasets/segmented_images',
        transform=eval_transform
    )

    # Grab a reproducible slice of the test set
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_idx = indices[train_size+val_size:]
    
    # Pull 5 interesting lesions to visualize
    sample_subset = Subset(dataset, test_idx[:5])
    loader = DataLoader(sample_subset, batch_size=1, shuffle=False)

    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    fig.suptitle('Multimodal AI Diagnostics: Original vs. Grad-CAM', fontsize=16)

    for i, (img, meta, label) in enumerate(loader):
        img = img.to(device)
        meta = meta.to(device)
        true_class = dataset.classes[label.item()]

        # Wrap the model for this specific patient's metadata
        wrapped_model = MultimodalGradCamWrapper(base_model, meta)
        
        # Initialize Grad-CAM
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        
        # We want to see what parts of the image triggered the model's ACTUAL prediction
        # Using None tells Grad-CAM to automatically target the highest scoring output class
        grayscale_cam = cam(input_tensor=img, targets=None)[0, :]

        # --- Un-normalize the image for plotting ---
        img_plot = img.squeeze().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_plot = std * img_plot + mean
        img_plot = np.clip(img_plot, 0, 1)

        # Generate the heatmap overlay
        visualization = show_cam_on_image(img_plot, grayscale_cam, use_rgb=True)

        # Output the model's actual prediction text
        with torch.no_grad():
            out = base_model(img, meta)
            pred_idx = out.argmax(dim=1).item()
            pred_class = dataset.classes[pred_idx]

        # Plotting
        axes[i, 0].imshow(img_plot)
        axes[i, 0].set_title(f"Original (True: {true_class})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"Grad-CAM (Pred: {pred_class})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    out_path = "gradcam_results.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSuccess! Heatmaps generated and saved to '{out_path}'.")

if __name__ == '__main__':
    main()
