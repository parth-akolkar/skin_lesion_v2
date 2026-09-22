import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ISIC2019MultimodalDataset(Dataset):
    def __init__(self, gt_csv, meta_csv, abcd_csv, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        
        # 1. Load all three data streams
        gt_df = pd.read_csv(gt_csv)
        meta_df = pd.read_csv(meta_csv)
        abcd_df = pd.read_csv(abcd_csv)
        
        # 2. Merge them on the 'image' ID column
        df = pd.merge(gt_df, meta_df, on='image', how='left')
        self.data_frame = pd.merge(df, abcd_df, on='image', how='left')
        
        # Determine the correct column name for anatomy (ISIC 2019 uses 'anatom_site_general')
        self.anatomy_col = 'anatom_site_general' if 'anatom_site_general' in self.data_frame.columns else 'anatom_site_general_challenge'
        
        # Precompute categorical mappings for anatomy
        self.sites = self.data_frame[self.anatomy_col].dropna().unique().tolist()
        
        # 3. Z-Score Normalize the continuous numeric columns (Crucial for Neural Networks)
        self.numeric_cols = ['age_approx', 'asymmetry', 'border_irregularity', 'color_variegation', 'diameter']
        self.stats = {}
        for col in self.numeric_cols:
            mean = self.data_frame[col].mean()
            std = self.data_frame[col].std()
            self.stats[col] = {'mean': mean, 'std': std}
            # Apply normalization, filling NaNs with 0 (the mean)
            self.data_frame[col] = (self.data_frame[col] - mean) / (std + 1e-6)
            self.data_frame[col] = self.data_frame[col].fillna(0.0)
        
        # Calculate final metadata vector size (5 numeric + 2 sex + N anatomy sites)
        self.num_metadata_features = 5 + 2 + len(self.sites)
        
    def _get_class_name_and_label(self, row):
        for idx, c in enumerate(self.classes):
            if row[c] == 1.0:
                return c, idx
        return 'UNK', 8

    def _process_metadata(self, row):
        # Numeric Features (Already Z-Score Normalized)
        numeric_feats = [row[col] for col in self.numeric_cols]
        
        # Sex (One-hot encoding with safety check)
        sex = row.get('sex', '')
        sex = str(sex).lower() if not pd.isna(sex) else ''
        sex_feat = [1.0, 0.0] if sex == 'male' else [0.0, 1.0] if sex == 'female' else [0.0, 0.0]
        
        # Anatomy (One-hot encoding)
        site = row.get(self.anatomy_col, None)
        site_feat = [1.0 if site == s else 0.0 for s in self.sites]
        if pd.isna(site):
            site_feat = [0.0] * len(self.sites)
            
        return torch.tensor(numeric_feats + sex_feat + site_feat, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        img_id = row['image']
        cls_name, label = self._get_class_name_and_label(row)
        
        # Fetch Visual Data
        img_path = os.path.join(self.img_dir, cls_name, f"{img_id}_segmented.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join('datasets/raw_images/archive', cls_name, f"{img_id}.jpg")
            
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Fetch Tabular Data
        meta_data = self._process_metadata(row)
        
        return image, meta_data, label

# Test Block
if __name__ == "__main__":
    print("Testing Unified Multimodal Dataset Loader...")
    dataset = ISIC2019MultimodalDataset(
        gt_csv='datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv',
        meta_csv='datasets/raw_images/archive/ISIC_2019_Training_Metadata.csv',
        abcd_csv='datasets/abcd_features.csv',
        img_dir='datasets/segmented_images'
    )
    img, meta, label = dataset[0]
    print(f"Total dataset length: {len(dataset)}")
    print(f"Tabular Tensor Size: {meta.shape[0]} explicitly mapped features")
    print(f"Tabular Tensor Output: {meta}")
    print(f"Label: {label} ({dataset.classes[label]})")
    print("Data streams successfully unified!")
