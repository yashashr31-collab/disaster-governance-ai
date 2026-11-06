
import pandas as pd
import numpy as np
import os

# Load processed flood dataset
df = pd.read_csv("data/processed/india_flood_districts.csv")

# Handle missing values
df["DFSI"] = df["DFSI"].fillna(df["DFSI"].median())
df["Mean_Flood_Duration"] = df["Mean_Flood_Duration"].fillna(
    df["Mean_Flood_Duration"].median()
)

# Normalize indicators using min-max scaling
def normalize(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(0, index=series.index)

    return (series - min_val) / (max_val - min_val)

df["DFSI_Score"] = normalize(df["DFSI"])
df["Flooded_Area_Score"] = normalize(
    df["Corrected_Percent_Flooded_Area"]
)
df["Fatality_Score"] = normalize(df["Human_fatality"])
df["Duration_Score"] = normalize(df["Mean_Flood_Duration"])

# Weighted historical risk indicator
df["Historical_Risk_Score"] = (
    0.40 * df["DFSI_Score"]
    + 0.30 * df["Flooded_Area_Score"]
    + 0.20 * df["Fatality_Score"]
    + 0.10 * df["Duration_Score"]
) * 100

# Rank districts
df = df.sort_values(
    "Historical_Risk_Score",
    ascending=False
)

# Assign descriptive categories
df["Risk_Category"] = pd.cut(
    df["Historical_Risk_Score"],
    bins=[-1, 25, 50, 75, 100],
    labels=["Low", "Moderate", "High", "Very High"]
)

# Save results
os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/flood_district_risk_ranking.csv",
    index=False
)

print("\nFlood Risk Analysis Completed!")
print("\nTop 15 districts by historical risk score:\n")

print(
    df[
        [
            "Dist_Name",
            "State_Name",
            "Historical_Risk_Score",
            "Risk_Category"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved to outputs/flood_district_risk_ranking.csv")
