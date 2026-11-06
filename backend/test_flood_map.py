
from flood_map import load_flood_map_data
import pandas as pd
from pathlib import Path

merged, matched, unmatched = load_flood_map_data()

ROOT = Path(__file__).resolve().parent.parent
risk = pd.read_csv(
    ROOT / "outputs" / "flood_district_risk_ranking.csv"
)

print("\nUNMATCHED GEOGRAPHIC DISTRICTS")
print("=" * 60)

unmatched_df = merged[
    merged["Historical_Risk_Score"].isna()
]

for state, group in unmatched_df.groupby("state_name"):
    print(f"\nSTATE: {state}")
    print(", ".join(sorted(group["district"].dropna().unique())))

print("\nDISTRICT NAMES IN FLOOD RISK DATA")
print("=" * 60)

for state, group in risk.groupby("State_Name"):
    print(f"\nSTATE: {state}")
    print(", ".join(sorted(group["Dist_Name"].dropna().unique())))

print("\nCURRENT MATCHING")
print("=" * 60)
print("Matched:", matched)
print("Unmatched:", unmatched)
