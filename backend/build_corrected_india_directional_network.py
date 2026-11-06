from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

SOURCE = OUTPUT / "full_river_junction_districts.csv"

print("Loading original river junction records...")
df = pd.read_csv(SOURCE, dtype=str).fillna("")

required = [
    "upstream_river_id",
    "downstream_river_id",
    "upstream_district_code",
    "upstream_district",
    "upstream_state",
    "downstream_district_code",
    "downstream_district",
    "downstream_state",
    "connection_type"
]

for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Normalize text fields
text_columns = [
    "upstream_district",
    "upstream_state",
    "downstream_district",
    "downstream_state"
]

for column in text_columns:
    df[column] = df[column].str.strip()

# Create unique district identifiers using names and states.
df["upstream_district_id"] = (
    df["upstream_state"] + "|" + df["upstream_district"]
)

df["downstream_district_id"] = (
    df["downstream_state"] + "|" + df["downstream_district"]
)

# Identify Muzaffarabad-related records.
upstream_external = df["upstream_district"].str.casefold().eq("muzaffarabad")
downstream_external = df["downstream_district"].str.casefold().eq("muzaffarabad")

# Separate the records.
same_external = upstream_external & downstream_external
upstream_to_other = upstream_external & ~downstream_external
other_to_downstream = ~upstream_external & downstream_external

# India-network candidate records: neither endpoint is Muzaffarabad.
india_candidates = df[~(upstream_external | downstream_external)].copy()

# Keep cross-boundary records separate.
boundary_records = df[upstream_to_other | other_to_downstream].copy()

# Keep external-area continuations separate.
external_continuations = df[same_external].copy()

# Remove any same-district connections from the district-to-district network.
india_network = india_candidates[
    india_candidates["upstream_district_id"]
    != india_candidates["downstream_district_id"]
].copy()

# Aggregate repeated records into unique directed district pairs.
pair_summary = (
    india_network.groupby(
        ["upstream_district_id", "downstream_district_id"],
        dropna=False
    )
    .agg(
        record_count=("upstream_river_id", "size"),
        unique_upstream_rivers=("upstream_river_id", "nunique"),
        upstream_district=("upstream_district", "first"),
        upstream_state=("upstream_state", "first"),
        downstream_district=("downstream_district", "first"),
        downstream_state=("downstream_state", "first")
    )
    .reset_index()
    .sort_values("record_count", ascending=False)
)

# Save new files without overwriting previous outputs.
india_network.to_csv(
    OUTPUT / "corrected_india_directional_connections.csv",
    index=False
)

boundary_records.to_csv(
    OUTPUT / "corrected_external_boundary_review.csv",
    index=False
)

external_continuations.to_csv(
    OUTPUT / "external_area_river_continuations.csv",
    index=False
)

pair_summary.to_csv(
    OUTPUT / "corrected_india_directional_unique_pairs.csv",
    index=False
)

# Generate summary.
summary = {
    "original_junction_records": len(df),
    "external_area_continuations": len(external_continuations),
    "cross_boundary_records": len(boundary_records),
    "india_network_candidate_records": len(india_candidates),
    "same_district_records_removed": (
        len(india_candidates) - len(india_network)
    ),
    "corrected_india_directional_records": len(india_network),
    "unique_directed_district_pairs": len(pair_summary),
    "unique_upstream_districts": (
        india_network["upstream_district_id"].nunique()
    ),
    "unique_downstream_districts": (
        india_network["downstream_district_id"].nunique()
    )
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["metric", "value"]
)

summary_df.to_csv(
    OUTPUT / "corrected_india_network_summary.csv",
    index=False
)

print("\n========== CORRECTED NETWORK SUMMARY ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nTOP 15 DIRECTED DISTRICT PAIRS:")
print(pair_summary.head(15).to_string(index=False))

print("\nNew files saved in outputs/")
print("Original network files have not been modified.")
print("================================================")