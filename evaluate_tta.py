import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.data.multimodal_dataset import ISIC2019MultimodalDataset
from src.models.multimodal_hybrid import MultimodalHybrid

# --- Custom Test-Time Augmentation (TTA) Transform ---
class FiveCropTTA:
    def __init__(self):
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img):
        img = self.resize(img)
        
        # Generate 5 explicit geometric variations
        crops = [
            img,                              # 0: Original
            TF.hflip(img),                    # 1: Horizontal Flip
            TF.vflip(img),                    # 2: Vertical Flip
            TF.rotate(img, 90),               # 3: Rotate 90 degrees
            TF.rotate(img, 180)               # 4: Rotate 180 degrees
        ]
        
        # Apply tensor conversion and normalization to all 5 variations
        tensors = [self.normalize(self.to_tensor(crop)) for crop in crops]
        
        # Stack them into a single tensor block: Shape (5, 3, 224, 224)
        return torch.stack(tensors)

def main():
    print("=== Phase 4.1: Test-Time Augmentation (TTA) Evaluation ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Safe batch size to prevent OOM (Effective GPU batch size is 8 * 5 = 40)
    batch_size = 8  
    
    print("Loading datasets and applying strict test split...")
    gt_csv = 'datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv'
    meta_csv = 'datasets/raw_images/archive/ISIC_2019_Training_Metadata.csv'
    abcd_csv = 'datasets/abcd_features.csv'
    img_dir = 'datasets/segmented_images'
    
    # Apply the FiveCropTTA
    tta_transform = FiveCropTTA()
    full_dataset = ISIC2019MultimodalDataset(gt_csv, meta_csv, abcd_csv, img_dir, transform=tta_transform)
    
    # Reproduce the exact test split used in training to guarantee a fair evaluation
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    test_idx = indices[train_size+val_size:]
    test_dataset = Subset(full_dataset, test_idx)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Initializing model and loading best weights...")
    model = MultimodalHybrid(num_tabular_features=15, num_classes=9).to(device)
    # Using weights_only=True to silence the PyTorch warning from earlier
    model.load_state_dict(torch.load('weights/caffm_multimodal_best.pth', weights_only=True))
    model.eval()

    test_correct = 0
    test_total = 0
    
    print("Running TTA Inference (Averaging 5 variations per image)...")
    with torch.no_grad():
        for images, metadata, labels in tqdm(test_loader, desc="Testing (TTA)"):
            
            # images shape: (Batch, 5, 3, 224, 224)
            B, N, C, H, W = images.shape
            
            # Collapse the batch and TTA dimensions together to feed the network: (Batch*5, 3, 224, 224)
            images = images.view(B * N, C, H, W).to(device)
            
            # Repeat the clinical metadata 5 times to match the 5 image variations
            metadata = metadata.repeat_interleave(N, dim=0).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, metadata)
            
            # Convert raw output logits into true probabilities (0.0 to 1.0)
            probs = F.softmax(outputs, dim=1)
            
            # Separate the batch and TTA dimensions back out: (Batch, 5, Num_Classes)
            probs = probs.view(B, N, -1)
            
            # Average the 5 probability scores together to get the consensus
            avg_probs = probs.mean(dim=1)
            
            # Get the final winning class
            _, predicted = avg_probs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"\nFINAL MULTIMODAL TTA TEST ACCURACY: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
