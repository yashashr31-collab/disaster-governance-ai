
from pathlib import Path
import geopandas as gpd
import pandas as pd
from difflib import get_close_matches
import re

ROOT = Path(__file__).resolve().parent.parent

geo = gpd.read_file(
    ROOT / "data/raw/flood/india_districts.GeoJSON"
)

flood = pd.read_csv(
    ROOT / "outputs/flood_district_risk_ranking.csv"
)


def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


geo["district_key"] = geo["district"].apply(normalize)
geo["state_key"] = geo["state_name"].apply(normalize)

flood["district_key"] = flood["Dist_Name"].apply(normalize)
flood["state_key"] = flood["State_Name"].apply(normalize)

state_aliases = {
    "ANDAMANANDNICOBARISLAND": "ANDAMANANDNICOBAR",
    "ARUNANCHALPRADESH": "ARUNACHALPRADESH",
}

geo["state_key"] = geo["state_key"].replace(state_aliases)

flood_groups = {
    state: group
    for state, group in flood.groupby("state_key")
}

matched_keys = set(
    zip(flood["district_key"], flood["state_key"])
)

unmatched = geo[
    ~geo.apply(
        lambda row: (
            row["district_key"],
            row["state_key"]
        ) in matched_keys,
        axis=1
    )
]

print("\n========== SUGGESTED DISTRICT MATCHES ==========")

for _, row in unmatched.iterrows():

    state = row["state_key"]
    district = row["district_key"]

    candidates = flood_groups.get(state)

    if candidates is None:
        continue

    names = candidates["district_key"].unique()

    suggestions = get_close_matches(
        district,
        names,
        n=3,
        cutoff=0.65
    )

    print(f"\nGeoJSON: {row['district']} | {row['state_name']}")

    if suggestions:
        for suggestion in suggestions:
            original = candidates.loc[
                candidates["district_key"] == suggestion,
                "Dist_Name"
            ].iloc[0]

            print(f"  Possible match: {original}")
    else:
        print("  No close match found")

print("\n========== END OF SUGGESTIONS ==========")
