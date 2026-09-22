import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.data.advanced_dataset import AdvancedMultimodalDataset
from src.models.advanced_hybrid import AdvancedMultimodalHybrid

# --- Custom Advanced TTA Transform ---
class AdvancedFiveCropTTA:
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img, mask):
        img = TF.resize(img, (224, 224))
        mask = TF.resize(mask, (7, 7), interpolation=Image.NEAREST)
        
        # Generate 5 explicit geometric variations
        img_crops = [
            img,                              # 0: Original
            TF.hflip(img),                    # 1: Horizontal Flip
            TF.vflip(img),                    # 2: Vertical Flip
            TF.rotate(img, 90),               # 3: Rotate 90 degrees
            TF.rotate(img, 180)               # 4: Rotate 180 degrees
        ]
        
        mask_crops = [
            mask,
            TF.hflip(mask),
            TF.vflip(mask),
            TF.rotate(mask, 90),
            TF.rotate(mask, 180)
        ]
        
        # Apply tensor conversion and normalization
        img_tensors = [self.normalize(TF.to_tensor(crop)) for crop in img_crops]
        mask_tensors = [TF.to_tensor(m).squeeze(0) for m in mask_crops]
        
        # Stack them into single tensor blocks
        return torch.stack(img_tensors), torch.stack(mask_tensors)

def main():
    print("=== Phase 4.6: Ultimate Enforcer TTA Evaluation ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Effective batch size is 8 * 5 = 40 images processed simultaneously
    batch_size = 8  
    
    print("Loading advanced datasets and applying strict test split...")
    gt_csv = 'datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv'
    meta_csv = 'datasets/raw_images/archive/ISIC_2019_Training_Metadata.csv'
    abcd_csv = 'datasets/abcd_features.csv'
    img_dir = 'datasets/segmented_images'
    
    # Load dataset and inject our custom 5-crop TTA transform
    full_dataset = AdvancedMultimodalDataset(gt_csv, meta_csv, abcd_csv, img_dir, is_train=False)
    full_dataset.transform = AdvancedFiveCropTTA()
    
    # Reproduce the exact test split used in training
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    test_idx = indices[train_size+val_size:]
    test_dataset = Subset(full_dataset, test_idx)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Initializing model and loading 'Ultimate Enforcer' weights...")
    model = AdvancedMultimodalHybrid(num_tabular_features=15, num_classes=9).to(device)
    model.load_state_dict(torch.load('weights/ultimate_enforcer_best.pth', weights_only=True))
    model.eval()

    test_correct = 0
    test_total = 0
    
    print("Running Advanced TTA Inference (Averaging 5 variations per image)...")
    with torch.no_grad():
        # We unpack 4 items because the advanced dataset returns the spatial mask
        for images, metadata, labels, _ in tqdm(test_loader, desc="Testing (TTA)"):
            
            # images shape: (Batch, 5, 3, 224, 224)
            B, N, C, H, W = images.shape
            
            # Collapse batch and TTA dimensions: (Batch*5, 3, 224, 224)
            images = images.view(B * N, C, H, W).to(device)
            
            # Repeat the clinical metadata 5 times
            metadata = metadata.repeat_interleave(N, dim=0).to(device)
            labels = labels.to(device)
            
            # Forward pass (unpacking the secondary attention map output)
            outputs, _ = model(images, metadata)
            
            # Convert raw output logits into true probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Separate the batch and TTA dimensions back out: (Batch, 5, Num_Classes)
            probs = probs.view(B, N, -1)
            
            # Average the 5 probability scores to get the consensus
            avg_probs = probs.mean(dim=1)
            
            # Get the final winning class
            _, predicted = avg_probs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"\nFINAL ULTIMATE TTA TEST ACCURACY: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
