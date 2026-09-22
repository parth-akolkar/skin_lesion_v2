import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.data.multimodal_dataset import ISIC2019MultimodalDataset
from src.models.multimodal_hybrid import MultimodalHybrid

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def main():
    print("=== Phase 4: Multimodal Model Training (CAFFM + Focal Loss) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    batch_size = 16 
    epochs = 15
    learning_rate = 1e-4

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets and extracting clinical geometry...")
    gt_csv = 'datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv'
    meta_csv = 'datasets/raw_images/archive/ISIC_2019_Training_Metadata.csv'
    abcd_csv = 'datasets/abcd_features.csv'
    img_dir = 'datasets/segmented_images'
    
    full_train_dataset = ISIC2019MultimodalDataset(gt_csv, meta_csv, abcd_csv, img_dir, transform=train_transform)
    full_eval_dataset = ISIC2019MultimodalDataset(gt_csv, meta_csv, abcd_csv, img_dir, transform=eval_transform)

    total_size = len(full_train_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_eval_dataset, val_idx)
    test_dataset = Subset(full_eval_dataset, test_idx)

    print("Configuring Data-Level Balancing (WeightedRandomSampler)...")
    df = pd.read_csv(gt_csv)
    train_df = df.iloc[train_idx]
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    class_counts = train_df[classes].sum().values
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    
    sample_weights = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mapping Sampler"):
        c_idx = next((i for i, c in enumerate(classes) if row[c] == 1.0), 8)
        sample_weights.append(class_weights[c_idx])
        
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("\nInitializing Multimodal Hybrid (CNN + Swin + CAFFM)...")
    # Initialize with 15 tabular features
    model = MultimodalHybrid(num_tabular_features=15, num_classes=9).to(device)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float('inf')
    os.makedirs('weights', exist_ok=True)
    best_model_path = 'weights/caffm_multimodal_best.pth'

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # TRAINING
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, metadata, labels in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, metadata, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)
                
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f">> New best validation loss! Saving checkpoint to '{best_model_path}'...")
            torch.save(model.state_dict(), best_model_path)

    print("\n=== Training Complete ===")
    print("Running final evaluation on the unseen Test Set...")
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, metadata, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            outputs = model(images, metadata)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"\nFINAL MULTIMODAL TEST ACCURACY: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
