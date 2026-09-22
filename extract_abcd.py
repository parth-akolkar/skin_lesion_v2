import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_class_name(row):
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    for c in classes:
        if row[c] == 1.0:
            return c
    return 'UNK'

def calculate_abcd(image_path):
    # 1. Read segmented image
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, 0.0, 0.0, 0.0

    # 2. Generate mask from the segmented (black) background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0, 0.0

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area == 0:
        return 0.0, 0.0, 0.0, 0.0

    # --- A: Asymmetry ---
    # Distance between Center of Mass (Moments) and Center of Bounding Box, normalized
    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
        
    bb_cx, bb_cy = x + w / 2.0, y + h / 2.0
    diagonal = np.sqrt(w**2 + h**2)
    asymmetry = np.sqrt((cx - bb_cx)**2 + (cy - bb_cy)**2) / (diagonal + 1e-6)

    # --- B: Border Irregularity ---
    # Compactness formula: P^2 / (4 * pi * A). A perfect circle is 1.0. Jagged borders are > 1.0.
    perimeter = cv2.arcLength(c, True)
    border_irregular = (perimeter ** 2) / (4 * np.pi * area)

    # --- C: Color Variegation ---
    # Standard deviation of R, G, B channels exclusively within the lesion mask
    _, stddev = cv2.meanStdDev(img, mask=mask)
    color_variegation = np.mean(stddev)

    # --- D: Diameter ---
    # Maximum length of the minimum enclosing rectangle
    rect = cv2.minAreaRect(c)
    diameter = max(rect[1][0], rect[1][1])

    return float(asymmetry), float(border_irregular), float(color_variegation), float(diameter)

def main():
    print("=== Phase 4: OpenCV ABCD Feature Extraction ===")
    raw_dir = 'datasets/raw_images/archive'
    seg_dir = 'datasets/segmented_images'
    gt_csv = os.path.join(raw_dir, 'ISIC_2019_Training_GroundTruth.csv')
    
    df = pd.read_csv(gt_csv)
    results = []

    print(f"Extracting clinical geometry from {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        img_id = row['image']
        cls_name = get_class_name(row)
        
        # Point to the cleanly isolated lesions
        img_path = os.path.join(seg_dir, cls_name, f"{img_id}_segmented.jpg")
        
        if not os.path.exists(img_path):
            # Fallback to raw if segmentation failed/was skipped
            img_path = os.path.join(raw_dir, cls_name, f"{img_id}.jpg")
            
        a, b, c, d = calculate_abcd(img_path)
        
        results.append({
            'image': img_id,
            'asymmetry': a,
            'border_irregularity': b,
            'color_variegation': c,
            'diameter': d
        })

    out_df = pd.DataFrame(results)
    out_path = 'datasets/abcd_features.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nExtraction complete! Features saved to '{out_path}'.")

if __name__ == '__main__':
    main()
