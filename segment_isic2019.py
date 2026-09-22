import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.models.network import MPBASwinUNETR

def get_class_name(row):
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    for c in classes:
        if row[c] == 1.0:
            return c
    return 'UNK'

def main():
    print("=== ISIC 2019 Batch Segmentation Pipeline ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("Loading MPBASwinUNETR architecture...")
    model = MPBASwinUNETR()
    
    weight_path = 'weights/swinunetr_autosave.pth'
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    
    # Safely extract model weights
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    raw_dir = 'datasets/raw_images/archive'
    out_dir = 'datasets/segmented_images'
    df = pd.read_csv(os.path.join(raw_dir, 'ISIC_2019_Training_GroundTruth.csv'))
    
    for c in ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)
        
    print(f"Starting cropping pipeline for all {len(df)} images...")
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
            img_id = row['image']
            cls_name = get_class_name(row)
            
            img_path = os.path.join(raw_dir, cls_name, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                continue
                
            orig_img = Image.open(img_path).convert('RGB')
            orig_size = orig_img.size 
            
            input_tensor = transform(orig_img).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            
            if isinstance(output, tuple):
                output = output[0]
                
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            
            mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
            img_array = np.array(orig_img)
            
            segmented_img_array = cv2.bitwise_and(img_array, img_array, mask=mask_resized)
            
            save_path = os.path.join(out_dir, cls_name, f"{img_id}_segmented.jpg")
            Image.fromarray(segmented_img_array).save(save_path)
            
    print("\nFull dataset segmentation complete! All lesions isolated successfully.")

if __name__ == '__main__':
    main()
