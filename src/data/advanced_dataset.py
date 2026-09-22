import os
import torch
import pandas as pd
import numpy as np
import cv2
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

# Custom Transform to ensure image and mask stay geometrically aligned
class DualTransform:
    def __init__(self, is_train):
        self.is_train = is_train
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __call__(self, img, mask):
        img = TF.resize(img, (224, 224))
        mask = TF.resize(mask, (7, 7), interpolation=Image.NEAREST)
        
        if self.is_train:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
                
            angle = random.uniform(-45, 45)
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)
            img = self.color_jitter(img)
            
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = TF.to_tensor(mask).squeeze(0) # Target shape: (7, 7)
        return img, mask

class AdvancedMultimodalDataset(Dataset):
    def __init__(self, gt_csv, meta_csv, abcd_csv, img_dir, is_train=True):
        self.img_dir = img_dir
        self.transform = DualTransform(is_train=is_train)
        self.classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        
        gt_df, meta_df, abcd_df = pd.read_csv(gt_csv), pd.read_csv(meta_csv), pd.read_csv(abcd_csv)
        df = pd.merge(gt_df, meta_df, on='image', how='left')
        self.data_frame = pd.merge(df, abcd_df, on='image', how='left')
        
        self.anatomy_col = 'anatom_site_general' if 'anatom_site_general' in self.data_frame.columns else 'anatom_site_general_challenge'
        self.sites = self.data_frame[self.anatomy_col].dropna().unique().tolist()
        
        self.numeric_cols = ['age_approx', 'asymmetry', 'border_irregularity', 'color_variegation', 'diameter']
        for col in self.numeric_cols:
            mean, std = self.data_frame[col].mean(), self.data_frame[col].std()
            self.data_frame[col] = ((self.data_frame[col] - mean) / (std + 1e-6)).fillna(0.0)
        
    def _process_image_and_mask(self, img_path):
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        
        # 1. Bounding Box Crop
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_np = img_np[y:y+h, x:x+w]
            mask = mask[y:y+h, x:x+w]
            
        # 2. Mean Skin Fill (replaces remaining black corners)
        mean_color = cv2.mean(img_np, mask=mask)[:3]
        bg = np.full_like(img_np, mean_color, dtype=np.uint8)
        mask_3d = mask[:, :, None] / 255.0
        img_np = (img_np * mask_3d + bg * (1 - mask_3d)).astype(np.uint8)
        
        return Image.fromarray(img_np), Image.fromarray(mask)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        cls_name = next((c for c in self.classes if row[c] == 1.0), 'UNK')
        label = self.classes.index(cls_name)
        
        img_path = os.path.join(self.img_dir, cls_name, f"{row['image']}_segmented.jpg")
        img_pil, mask_pil = self._process_image_and_mask(img_path)
        
        image, spatial_mask = self.transform(img_pil, mask_pil)
        
        numeric = [row[col] for col in self.numeric_cols]
        sex = str(row.get('sex', '')).lower()
        sex_feat = [1.0, 0.0] if sex == 'male' else [0.0, 1.0] if sex == 'female' else [0.0, 0.0]
        site = row.get(self.anatomy_col, None)
        site_feat = [1.0 if site == s else 0.0 for s in self.sites]
        if pd.isna(site): site_feat = [0.0] * len(self.sites)
            
        meta_data = torch.tensor(numeric + sex_feat + site_feat, dtype=torch.float32)
        
        return image, meta_data, label, spatial_mask
