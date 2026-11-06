
from pathlib import Path
import geopandas as gpd
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

geo_path = ROOT / "data/raw/flood/india_districts.GeoJSON"
flood_path = ROOT / "outputs/flood_district_risk_ranking.csv"

geo = gpd.read_file(geo_path)
flood = pd.read_csv(flood_path)


def normalize(value):
    value = str(value).upper().strip()
    value = re.sub(r"[^A-Z0-9]", "", value)
    return value


# Normalize district and state names
geo["district_key"] = geo["district"].apply(normalize)
flood["district_key"] = flood["Dist_Name"].apply(normalize)

geo["state_key"] = geo["state_name"].apply(normalize)
flood["state_key"] = flood["State_Name"].apply(normalize)

# Correct known naming inconsistencies
state_aliases = {
    "ANDAMANANDNICOBARISLAND": "ANDAMANANDNICOBAR",
    "ARUNANCHALPRADESH": "ARUNACHALPRADESH",
}

geo["state_key"] = geo["state_key"].replace(state_aliases)

# Match using both district and state
geo["match_key"] = geo["district_key"] + "_" + geo["state_key"]
flood["match_key"] = flood["district_key"] + "_" + flood["state_key"]

matched = geo.merge(
    flood[["match_key", "Historical_Risk_Score", "Risk_Category"]],
    on="match_key",
    how="left"
)

matched_count = matched["Historical_Risk_Score"].notna().sum()
unmatched_count = matched["Historical_Risk_Score"].isna().sum()

print("\n========== DISTRICT MATCHING REPORT ==========")
print("GeoJSON districts:", len(geo))
print("Flood risk records:", len(flood))
print("Matched districts:", matched_count)
print("Unmatched GeoJSON districts:", unmatched_count)

print("\n========== UNMATCHED DISTRICTS ==========")

print(
    matched.loc[
        matched["Historical_Risk_Score"].isna(),
        ["district", "state_name"]
    ].to_string(index=False)
)

print("\n========== MATCHING COMPLETE ==========")
