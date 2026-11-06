
import pandas as pd
import os

folder = "data/raw/flood"

files = [
    "India_Flood_Inventory_v3.csv",
    "DFSI.csv",
    "District_FloodedArea.csv",
    "District_FloodImpact.csv"
]

for filename in files:

    path = os.path.join(folder, filename)

    print("\n" + "=" * 70)
    print("DATASET:", filename)
    print("=" * 70)

    if not os.path.exists(path):
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path, low_memory=False)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nData types:")
    print(df.dtypes.to_string())
