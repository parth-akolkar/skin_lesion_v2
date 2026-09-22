import pandas as pd
import os

csv_path = "datasets/raw_images/archive/ISIC_2019_Training_GroundTruth.csv"
df = pd.read_csv(csv_path)
print(f"Total clinical records in ISIC 2019 ground truth: {len(df)}")
print("Class distribution:")

# ISIC 2019 columns represent one-hot encoded classes plus image
# Let's see the columns
print(df.head(2))
