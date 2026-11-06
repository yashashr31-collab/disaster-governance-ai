
import pandas as pd
import os

folder = "data/raw/flood"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

# Load datasets
dfsi = pd.read_csv(
    os.path.join(folder, "DFSI.csv")
)

flood_area = pd.read_csv(
    os.path.join(folder, "District_FloodedArea.csv")
)

flood_impact = pd.read_csv(
    os.path.join(folder, "District_FloodImpact.csv")
)

# Remove unnecessary index columns
dfsi = dfsi.drop(columns=["Unnamed: 0"], errors="ignore")

# Standardize district names
def clean_name(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )

dfsi["Dist_Name"] = clean_name(dfsi["Unnamed: 0"]) if "Unnamed: 0" in dfsi.columns else clean_name(dfsi["State_Name"])

# The DFSI file's first column contains district names.
# Rename its columns explicitly.
dfsi = pd.read_csv(
    os.path.join(folder, "DFSI.csv")
)

dfsi.columns = ["Dist_Name", "State_Name", "DFSI"]

dfsi["Dist_Name"] = clean_name(dfsi["Dist_Name"])
flood_area["Dist_Name"] = clean_name(flood_area["Dist_Name"])
flood_impact["Dist_Name"] = clean_name(flood_impact["Dist_Name"])

# Merge district-level datasets
merged = pd.merge(
    flood_area,
    flood_impact,
    on="Dist_Name",
    how="inner"
)

merged = pd.merge(
    merged,
    dfsi[["Dist_Name", "State_Name", "DFSI"]],
    on="Dist_Name",
    how="left"
)

# Remove duplicate district records if any
merged = merged.drop_duplicates(subset=["Dist_Name"])

# Save
output_path = os.path.join(
    output_folder,
    "india_flood_districts.csv"
)

merged.to_csv(output_path, index=False)

print("Flood district dataset created!")
print("Shape:", merged.shape)
print("\nColumns:")
print(merged.columns.tolist())

print("\nFirst 10 rows:")
print(merged.head(10).to_string(index=False))

print("\nMissing values:")
print(merged.isnull().sum())

print("\nSaved to:", output_path)
