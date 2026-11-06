from pathlib import Path
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"
DATA = ROOT / "data" / "raw" / "flood" / "india_districts.GeoJSON"

NETWORK_FILE = OUTPUT / "full_directional_district_network.csv"
EXTERNAL_FILE = OUTPUT / "external_boundary_connections.csv"
MISSING_FILE = OUTPUT / "missing_downstream_targets.csv"

print("Loading directional network...")
network = pd.read_csv(NETWORK_FILE, dtype=str).fillna("")

print("Loading external-boundary records...")
external = pd.read_csv(EXTERNAL_FILE, dtype=str).fillna("")

# 1. Basic network counts
total_records = len(network)

required = [
    "upstream_district_code",
    "downstream_district_code"
]
for col in required:
    if col not in network.columns:
        raise ValueError(f"Missing required column: {col}")

network = network[
    (network["upstream_district_code"] != "") &
    (network["downstream_district_code"] != "")
].copy()

# 2. Aggregate directed district pairs
pair_columns = [
    "upstream_district_code",
    "downstream_district_code"
]

aggregations = {"upstream_district_code": "size"}

for col in [
    "upstream_district_name",
    "downstream_district_name",
    "upstream_state",
    "downstream_state"
]:
    if col in network.columns:
        aggregations[col] = "first"

if "upstream_river_id" in network.columns:
    aggregations["upstream_river_id"] = "nunique"

pairs = (
    network.groupby(pair_columns, dropna=False)
    .agg(aggregations)
    .rename(columns={
        "upstream_district_code": "record_count",
        "upstream_river_id": "unique_upstream_river_count"
    })
    .reset_index()
)

pairs = pairs.sort_values("record_count", ascending=False)

pairs.to_csv(
    OUTPUT / "directional_network_unique_pairs.csv",
    index=False
)

# 3. Find reciprocal district pairs
pair_set = set(
    zip(
        pairs["upstream_district_code"],
        pairs["downstream_district_code"]
    )
)

reciprocal = []
seen = set()

for a, b in pair_set:
    if a == b:
        continue

    if (b, a) in pair_set:
        key = tuple(sorted([a, b]))

        if key not in seen:
            seen.add(key)
            reciprocal.append({
                "district_code_a": a,
                "district_code_b": b
            })

reciprocal_df = pd.DataFrame(reciprocal)
reciprocal_df.to_csv(
    OUTPUT / "directional_network_reciprocal_pairs.csv",
    index=False
)

# 4. Review external-boundary records
external.to_csv(
    OUTPUT / "directional_network_external_review.csv",
    index=False
)

# 5. Compare network district codes with GeoJSON codes
print("Checking district codes against GeoJSON...")

districts = gpd.read_file(DATA)

if "dtcode" not in districts.columns:
    raise ValueError("GeoJSON does not contain the expected dtcode column.")

geojson_codes = set(
    districts["dtcode"]
    .dropna()
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(3)
)

network_codes = set(
    network["upstream_district_code"].astype(str).str.zfill(3)
) | set(
    network["downstream_district_code"].astype(str).str.zfill(3)
)

codes_in_geojson = network_codes.intersection(geojson_codes)
codes_not_in_geojson = network_codes - geojson_codes

# 6. Missing downstream target count
if MISSING_FILE.exists():
    missing = pd.read_csv(MISSING_FILE, dtype=str)
    missing_count = len(missing)
else:
    missing_count = 0

# 7. Summary
summary = {
    "total_cross_district_records": total_records,
    "valid_cross_district_records": len(network),
    "unique_directed_district_pairs": len(pairs),
    "repeated_records_beyond_unique_pairs": len(network) - len(pairs),
    "unique_reciprocal_district_pair_groups": len(reciprocal_df),
    "unique_upstream_districts": network["upstream_district_code"].nunique(),
    "unique_downstream_districts": network["downstream_district_code"].nunique(),
    "external_boundary_records": len(external),
    "network_district_codes": len(network_codes),
    "district_codes_present_in_geojson": len(codes_in_geojson),
    "district_codes_missing_from_geojson": len(codes_not_in_geojson),
    "missing_downstream_target_records": missing_count
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    OUTPUT / "directional_network_validation_summary.csv",
    index=False
)

# 8. Print results
print("\n========== DIRECTIONAL NETWORK VALIDATION ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nTOP 15 MOST FREQUENT DIRECTED DISTRICT PAIRS:")
print(pairs.head(15).to_string(index=False))

print("\nRECIPROCAL PAIR EXAMPLES:")
if len(reciprocal_df):
    print(reciprocal_df.head(20).to_string(index=False))
else:
    print("No reciprocal pairs found.")

print("\nEXTERNAL-BOUNDARY RECORD EXAMPLES:")
if len(external):
    display_columns = [
        c for c in [
            "upstream_district_code",
            "upstream_district_name",
            "upstream_state",
            "downstream_district_code",
            "downstream_district_name",
            "downstream_state"
        ]
        if c in external.columns
    ]
    print(external[display_columns].head(20).to_string(index=False))
else:
    print("No external-boundary records.")

print("\nDISTRICT CODES NOT FOUND IN GEOJSON:")
print(sorted(codes_not_in_geojson))

print("\nValidation files saved in outputs/")
print("====================================================")