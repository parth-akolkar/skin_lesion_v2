import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from src.data.advanced_dataset import AdvancedMultimodalDataset
from src.models.advanced_hybrid import AdvancedMultimodalHybrid
from train_multimodal import FocalLoss

def main():
    print("=== Phase 4.5: Ultimate Enforcer (Cropping + Fill + Attention Loss) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16 
    epochs = 15
    learning_rate = 1e-4
    lambda_attention = 0.5 # Balances classification vs. spatial forcing

    gt_csv = 'datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv'
    meta_csv = 'datasets/raw_images/archive/ISIC_2019_Training_Metadata.csv'
    abcd_csv = 'datasets/abcd_features.csv'
    img_dir = 'datasets/segmented_images'
    
    full_train_dataset = AdvancedMultimodalDataset(gt_csv, meta_csv, abcd_csv, img_dir, is_train=True)
    full_eval_dataset = AdvancedMultimodalDataset(gt_csv, meta_csv, abcd_csv, img_dir, is_train=False)

    total_size = len(full_train_dataset)
    train_size, val_size = int(0.70 * total_size), int(0.15 * total_size)
    
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42)).tolist()
    train_dataset = Subset(full_train_dataset, indices[:train_size])
    val_dataset = Subset(full_eval_dataset, indices[train_size:train_size+val_size])
    test_dataset = Subset(full_eval_dataset, indices[train_size+val_size:])

    # Setup Sampler
    train_df = pd.read_csv(gt_csv).iloc[indices[:train_size]]
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    class_counts = train_df[classes].sum().values
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    
    sample_weights = [class_weights[next((i for i, c in enumerate(classes) if row[c] == 1.0), 8)] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = AdvancedMultimodalHybrid().to(device)
    criterion_class = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_model_path = 'weights/ultimate_enforcer_best.pth'

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # TRAINING
        model.train()
        train_loss, train_cls_loss, train_att_loss = 0.0, 0.0, 0.0
        correct, total = 0, 0
        
        # Notice we now unpack the 4th item: target_mask
        for images, metadata, labels, target_masks in tqdm(train_loader, desc="Training"):
            images, metadata, labels, target_masks = images.to(device), metadata.to(device), labels.to(device), target_masks.to(device)
            
            optimizer.zero_grad()
            outputs, attention_maps = model(images, metadata)
            
            # 1. Classification Loss
            cls_loss = criterion_class(outputs, labels)
            
            # 2. Attention Guardrail Loss
            # Squish the CNN's attention output between 0 and 1, then calculate spatial error against the true tissue mask
            att_probs = torch.sigmoid(attention_maps)
            att_loss = F.mse_loss(att_probs, target_masks)
            
            # 3. Combined Dual Loss
            loss = cls_loss + (lambda_attention * att_loss)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_att_loss += att_loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        print(f"Train Acc: {100. * correct / total:.2f}% | Total Loss: {train_loss/len(train_loader):.4f} (Cls: {train_cls_loss/len(train_loader):.4f}, Att: {train_att_loss/len(train_loader):.4f})")
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, metadata, labels, target_masks in tqdm(val_loader, desc="Validating"):
                images, metadata, labels, target_masks = images.to(device), metadata.to(device), labels.to(device), target_masks.to(device)
                outputs, attention_maps = model(images, metadata)
                
                cls_loss = criterion_class(outputs, labels)
                att_loss = F.mse_loss(torch.sigmoid(attention_maps), target_masks)
                loss = cls_loss + (lambda_attention * att_loss)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Acc: {100. * val_correct / val_total:.2f}% | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    # FINAL TEST
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, metadata, labels, _ in tqdm(test_loader, desc="Testing"):
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs, _ = model(images, metadata)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    print(f"\nULTIMATE MULTIMODAL TEST ACCURACY: {100. * test_correct / test_total:.2f}%")

if __name__ == '__main__':
    main()
