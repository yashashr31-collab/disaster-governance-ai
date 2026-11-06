
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# FILES
# --------------------------------------------------

JUNCTION_FILE = Path("outputs/improved_river_junction_districts.csv")
MISSING_FILE = Path("outputs/river_network_unconnected_districts.csv")
OUTPUT_FILE = Path("outputs/missing_graph_node_diagnostic.csv")

print("Loading unconnected districts...")
missing = pd.read_csv(
    MISSING_FILE,
    dtype={"district_feature_id": str}
)

print("Loading improved river junctions...")
junctions = pd.read_csv(
    JUNCTION_FILE,
    dtype=str
).fillna("")

# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required = [
    "upstream_district_feature_id",
    "downstream_district_feature_id",
    "upstream_district",
    "upstream_state",
    "downstream_district",
    "downstream_state",
    "upstream_match_method",
    "downstream_match_method",
    "connection_type"
]

for col in required:
    if col not in junctions.columns:
        raise ValueError(f"Missing required column: {col}")

# --------------------------------------------------
# ANALYZE EACH MISSING DISTRICT
# --------------------------------------------------

results = []

for _, district in missing.iterrows():

    feature_id = str(district["district_feature_id"])

    upstream = junctions[
        junctions["upstream_district_feature_id"] == feature_id
    ]

    downstream = junctions[
        junctions["downstream_district_feature_id"] == feature_id
    ]

    involved = junctions[
        (junctions["upstream_district_feature_id"] == feature_id) |
        (junctions["downstream_district_feature_id"] == feature_id)
    ]

    same = involved[
        involved["connection_type"] == "same_district_continuation"
    ]

    cross = involved[
        involved["connection_type"] == "cross_district_transition"
    ]

    results.append({
        "district_feature_id": feature_id,
        "district": district.get("district", ""),
        "state": district.get("state_name", ""),
        "upstream_junction_records": len(upstream),
        "downstream_junction_records": len(downstream),
        "total_junction_records": len(involved),
        "same_district_records": len(same),
        "cross_district_records": len(cross),
        "strict_upstream_matches": (
            upstream["upstream_match_method"] == "strict_within"
        ).sum(),
        "strict_downstream_matches": (
            downstream["downstream_match_method"] == "strict_within"
        ).sum(),
        "fallback_upstream_matches": (
            upstream["upstream_match_method"] == "nearest_fallback"
        ).sum(),
        "fallback_downstream_matches": (
            downstream["downstream_match_method"] == "nearest_fallback"
        ).sum(),
        "endpoint_status": (
            "PRESENT IN JUNCTION DATA"
            if len(involved) > 0
            else "NO JUNCTION ASSIGNMENT"
        )
    })

# --------------------------------------------------
# SAVE AND DISPLAY
# --------------------------------------------------

result = pd.DataFrame(results)

result.to_csv(OUTPUT_FILE, index=False)

print("\n========== MISSING GRAPH NODE DIAGNOSTIC ==========")

print("Districts investigated:", len(result))
print(
    "Districts appearing in junction data:",
    (result["total_junction_records"] > 0).sum()
)
print(
    "Districts with no junction assignment:",
    (result["total_junction_records"] == 0).sum()
)
print(
    "Districts with cross-district transitions:",
    (result["cross_district_records"] > 0).sum()
)
print(
    "Districts with only same-district records:",
    (
        (result["same_district_records"] > 0) &
        (result["cross_district_records"] == 0)
    ).sum()
)

print("\nDISTRICT-BY-DISTRICT RESULTS:")

print(
    result[
        [
            "district",
            "state",
            "total_junction_records",
            "same_district_records",
            "cross_district_records",
            "endpoint_status"
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
print("DIAGNOSTIC COMPLETED.")
