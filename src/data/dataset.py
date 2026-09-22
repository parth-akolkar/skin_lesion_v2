import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class ISIC2019Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        
    def _get_class_name_and_label(self, row):
        for idx, c in enumerate(self.classes):
            if row[c] == 1.0:
                return c, idx
        return 'UNK', 8

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        img_id = row['image']
        cls_name, label = self._get_class_name_and_label(row)
        
        # Point to the isolated lesions we just generated
        img_path = os.path.join(self.img_dir, cls_name, f"{img_id}_segmented.jpg")
        
        # Fallback to raw image if a segmentation mask was totally empty
        if not os.path.exists(img_path):
            img_path = os.path.join('datasets/raw_images/archive', cls_name, f"{img_id}.jpg")
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Test Block
if __name__ == "__main__":
    print("Testing ISIC 2019 Dataset Loader...")
    dataset = ISIC2019Dataset(
        csv_file='datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv',
        img_dir='datasets/segmented_images'
    )
    print(f"Total dataset length: {len(dataset)}")
    img, label = dataset[0]
    print(f"First image size: {img.size} (Width, Height)")
    print(f"First image label index: {label} ({dataset.classes[label]})")
    print("Dataset loader is ready!")
