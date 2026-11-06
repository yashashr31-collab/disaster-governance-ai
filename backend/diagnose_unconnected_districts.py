
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkt

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DISTRICT_FILE = Path("data/raw/flood/india_districts.GeoJSON")
RIVER_FILE = Path("outputs/india_river_network.csv")
UNCOVERED_FILE = Path("outputs/river_network_unconnected_districts.csv")

OUTPUT_FILE = Path("outputs/unconnected_district_river_diagnostic.csv")

print("Loading district boundaries...")
districts = gpd.read_file(DISTRICT_FILE).reset_index(drop=True)

# Match the feature IDs used in our coverage validation.
districts["district_feature_id"] = districts.index.astype(str)

print("Loading the 57 districts absent from the graph...")
uncovered = pd.read_csv(
    UNCOVERED_FILE,
    dtype={"district_feature_id": str}
)

target_ids = set(uncovered["district_feature_id"])
target_districts = districts[
    districts["district_feature_id"].isin(target_ids)
].copy()

print("Districts to investigate:", len(target_districts))

# --------------------------------------------------
# LOAD RIVER SEGMENTS
# --------------------------------------------------

print("\nLoading HydroRIVERS segments...")

rivers = pd.read_csv(
    RIVER_FILE,
    usecols=["HYRIV_ID", "NEXT_DOWN", "geometry_wkt"],
    dtype=str
).fillna("")

rivers = rivers[rivers["geometry_wkt"] != ""].copy()

print("River segments loaded:", len(rivers))

print("\nConverting river geometries...")

rivers["geometry"] = rivers["geometry_wkt"].apply(wkt.loads)

river_gdf = gpd.GeoDataFrame(
    rivers.drop(columns=["geometry_wkt"]),
    geometry="geometry",
    crs="EPSG:4326"
)

# --------------------------------------------------
# ALIGN COORDINATE SYSTEMS
# --------------------------------------------------

if target_districts.crs is None:
    raise ValueError("District GeoJSON has no defined CRS.")

target_districts = target_districts.to_crs("EPSG:4326")

# --------------------------------------------------
# SPATIAL INTERSECTION
# --------------------------------------------------

print("\nChecking which river segments intersect each district...")

matches = gpd.sjoin(
    river_gdf[["HYRIV_ID", "NEXT_DOWN", "geometry"]],
    target_districts[
        ["district_feature_id", "district", "state_name", "geometry"]
    ],
    how="inner",
    predicate="intersects"
)

print("Total river-district intersections:", len(matches))

# --------------------------------------------------
# SUMMARIZE EACH DISTRICT
# --------------------------------------------------

summary = (
    matches.groupby("district_feature_id")
    .agg(
        intersecting_river_segments=("HYRIV_ID", "nunique"),
        river_intersection_records=("HYRIV_ID", "size"),
        segments_with_downstream_target=(
            "NEXT_DOWN",
            lambda x: x.ne("").sum()
        )
    )
    .reset_index()
)

result = uncovered.merge(
    summary,
    on="district_feature_id",
    how="left"
)

count_columns = [
    "intersecting_river_segments",
    "river_intersection_records",
    "segments_with_downstream_target"
]

result[count_columns] = result[count_columns].fillna(0).astype(int)

result["river_segment_status"] = result[
    "intersecting_river_segments"
].apply(
    lambda x: (
        "RIVER SEGMENTS FOUND"
        if x > 0
        else "NO INTERSECTING SEGMENTS FOUND"
    )
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

result.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

has_rivers = (
    result["intersecting_river_segments"] > 0
).sum()

no_rivers = (
    result["intersecting_river_segments"] == 0
).sum()

print("\n========== DISTRICT DIAGNOSTIC ==========")

print("Districts investigated:", len(result))
print("Districts with river segments:", has_rivers)
print("Districts without intersecting segments:", no_rivers)

print("\nDISTRICTS WITH RIVER SEGMENTS:")
print(
    result.loc[
        result["intersecting_river_segments"] > 0,
        [
            "district",
            "state_name",
            "intersecting_river_segments",
            "segments_with_downstream_target"
        ]
    ].to_string(index=False)
)

print("\nDISTRICTS WITHOUT INTERSECTING SEGMENTS:")
print(
    result.loc[
        result["intersecting_river_segments"] == 0,
        ["district", "state_name"]
    ].to_string(index=False)
)

print("\nSaved diagnostic:")
print(OUTPUT_FILE)

print("\nDIAGNOSTIC COMPLETED.")
