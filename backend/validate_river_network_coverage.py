
import pandas as pd
import geopandas as gpd
from pathlib import Path

geo_file = Path("data/raw/flood/india_districts.GeoJSON")
node_file = Path("outputs/final_river_network_nodes.csv")

print("Loading all district features...")
districts = gpd.read_file(geo_file)

print("Loading river graph nodes...")
nodes = pd.read_csv(node_file, dtype=str).fillna("")

# Feature IDs are based on the original GeoJSON row order.
districts = districts.reset_index(drop=True)
districts["district_feature_id"] = districts.index.astype(str)

network_ids = set(nodes["district_feature_id"].astype(str))

districts["in_river_network"] = (
    districts["district_feature_id"].isin(network_ids)
)

covered = districts[districts["in_river_network"]].copy()
not_covered = districts[~districts["in_river_network"]].copy()

print("\n========== COVERAGE REPORT ==========")
print("Total district features:", len(districts))
print("Features in river graph:", len(covered))
print("Features absent from river graph:", len(not_covered))
print(
    "Coverage percentage:",
    round(len(covered) / len(districts) * 100, 2),
    "%"
)

# Save complete coverage report.
districts.drop(columns="geometry").to_csv(
    "outputs/river_network_coverage_all_districts.csv",
    index=False
)

# Save districts absent from the network.
not_covered.drop(columns="geometry").to_csv(
    "outputs/river_network_unconnected_districts.csv",
    index=False
)

print("\nDISTRICTS ABSENT FROM THE GRAPH:")
print(
    not_covered[
        ["district_feature_id", "district", "state_name"]
    ].to_string(index=False)
)

print("\nCoverage files saved successfully.")
