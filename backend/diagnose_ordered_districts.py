
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RIVER_FILE = ROOT / "outputs/india_river_network.csv"
DISTRICT_FILE = ROOT / "data/raw/flood/india_districts.GeoJSON"

print("Loading datasets...")

rivers = pd.read_csv(RIVER_FILE)

districts = gpd.read_file(DISTRICT_FILE)

if districts.crs is None:
    raise ValueError("District GeoJSON has no CRS.")

metric_crs = districts.estimate_utm_crs()

if metric_crs is None:
    raise ValueError("Could not determine a suitable projected CRS.")

districts = districts.to_crs(metric_crs).reset_index(drop=True)

# Give each district a unique internal identifier.
districts["district_row_id"] = range(len(districts))

sample = rivers.sample(
    n=min(2000, len(rivers)),
    random_state=42
).copy()

sample["geometry"] = gpd.GeoSeries.from_wkt(
    sample["geometry_wkt"],
    crs="EPSG:4326"
)

river_gdf = gpd.GeoDataFrame(
    sample[["HYRIV_ID", "NEXT_DOWN"]],
    geometry=sample["geometry"],
    crs="EPSG:4326"
).to_crs(metric_crs).reset_index(drop=True)

print("Sampled river segments:", len(river_gdf))

# Spatial join using an unambiguous district identifier.
joined = gpd.sjoin(
    river_gdf,
    districts[
        [
            "district_row_id",
            "dtcode",
            "district",
            "state_name",
            "geometry"
        ]
    ],
    how="inner",
    predicate="intersects"
)

records = []

for river_index, group in joined.groupby(level=0):

    river = river_gdf.geometry.iloc[river_index]

    ordered = []

    for row in group.itertuples():

        district = districts.iloc[int(row.district_row_id)]

        intersection = river.intersection(district.geometry)

        if intersection.is_empty or intersection.length < 1:
            continue

        position = river.project(
            intersection.representative_point()
        )

        ordered.append({
            "position_m": position,
            "district_code": district["dtcode"],
            "district": district["district"],
            "state": district["state_name"],
            "intersection_length_m": intersection.length
        })

    # Sort districts in the direction of river geometry.
    ordered.sort(key=lambda x: x["position_m"])

    if len(ordered) > 1:

        records.append({
            "river_id": river_gdf.iloc[river_index]["HYRIV_ID"],
            "district_count": len(ordered),
            "district_sequence": " -> ".join(
                f"{item['district']} ({item['state']})"
                for item in ordered
            )
        })

result = pd.DataFrame(records)

print("\n========== ORDERED DISTRICT DIAGNOSTIC ==========")
print("Sampled river segments:", len(river_gdf))
print("Segments crossing multiple districts:", len(result))

if not result.empty:

    print("\nFirst 15 examples:")
    print(result.head(15).to_string(index=False))

    output = ROOT / "outputs/ordered_district_diagnostic.csv"
    result.to_csv(output, index=False)

    print("\nDiagnostic saved to:", output)

print("\nDiagnostic completed.")
