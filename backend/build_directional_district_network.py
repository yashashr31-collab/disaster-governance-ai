
import os
import pandas as pd
import geopandas as gpd
from shapely import wkt

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RIVER_FILE = "outputs/india_river_network.csv"
DISTRICT_FILE = "data/raw/flood/india_districts.GeoJSON"

OUTPUT_FILE = "outputs/directional_district_network_test.csv"
SUMMARY_FILE = "outputs/directional_network_test_summary.csv"

SAMPLE_SIZE = 10000
RANDOM_STATE = 42


def main():
    print("Loading river and district datasets...")

    rivers = pd.read_csv(
        RIVER_FILE,
        dtype={"HYRIV_ID": str, "NEXT_DOWN": str}
    )

    districts = gpd.read_file(DISTRICT_FILE)

    # Remove invalid or missing district identifiers.
    districts = districts.dropna(
        subset=["dtcode", "district", "geometry"]
    ).copy()

    districts = districts.reset_index(drop=True)
    districts["district_row_id"] = districts.index

    # Convert river WKT into geometries.
    rivers["geometry"] = rivers["geometry_wkt"].apply(
        lambda value: wkt.loads(value) if isinstance(value, str) else None
    )

    rivers = rivers.dropna(subset=["geometry"]).copy()

    rivers = gpd.GeoDataFrame(
        rivers,
        geometry="geometry",
        crs="EPSG:4326"
    )

    # Test on a reproducible sample first.
    if len(rivers) > SAMPLE_SIZE:
        rivers = rivers.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).copy()

    print("River segments in test:", len(rivers))

    # Project both datasets to the same CRS for geometry operations.
    projected_crs = districts.estimate_utm_crs()

    if projected_crs is None:
        raise ValueError("Could not determine a projected CRS.")

    rivers = rivers.to_crs(projected_crs)
    districts = districts.to_crs(projected_crs)

    # Spatial join: identify districts intersected by each river.
    print("Finding river-district intersections...")

    joined = gpd.sjoin(
        rivers[["HYRIV_ID", "NEXT_DOWN", "geometry"]],
        districts[
            ["district_row_id", "dtcode", "district",
             "state_name", "geometry"]
        ],
        how="inner",
        predicate="intersects"
    )

    print("River-district associations:", len(joined))

    # Build ordered district sequences for each river segment.
    print("Calculating district traversal order...")

    district_sequences = {}

    district_lookup = districts.set_index("district_row_id")

    for river_index, group in joined.groupby(level=0):
        river = rivers.loc[river_index].geometry
        ordered = []

        for row in group.itertuples():
            district = district_lookup.loc[row.district_row_id]
            intersection = river.intersection(district.geometry)

            if intersection.is_empty:
                continue

            # Ignore point-only boundary touches.
            if intersection.length <= 0:
                continue

            position = river.project(intersection.representative_point())

            ordered.append({
                "position": position,
                "code": str(district["dtcode"]),
                "district": str(district["district"]),
                "state": str(district["state_name"])
            })

        ordered.sort(key=lambda item: item["position"])

        # Remove consecutive duplicate district entries.
        sequence = []

        for item in ordered:
            if not sequence or sequence[-1]["code"] != item["code"]:
                sequence.append(item)

        if sequence:
            river_id = str(rivers.loc[river_index, "HYRIV_ID"])
            next_down = rivers.loc[river_index, "NEXT_DOWN"]

            district_sequences[river_id] = {
                "next_down": str(next_down),
                "districts": sequence
            }

    print("Segments with district sequences:", len(district_sequences))

    # --------------------------------------------------
    # CREATE DIRECTIONAL DISTRICT CONNECTIONS
    # --------------------------------------------------

    print("Building directional district connections...")

    connections = []

    for river_id, data in district_sequences.items():
        sequence = data["districts"]

        # A river segment crossing multiple districts:
        # connect each consecutive district in its flow direction.
        for upstream, downstream in zip(sequence, sequence[1:]):
            connections.append({
                "upstream_river_id": river_id,
                "downstream_river_id": river_id,
                "upstream_district_code": upstream["code"],
                "upstream_district": upstream["district"],
                "upstream_state": upstream["state"],
                "downstream_district_code": downstream["code"],
                "downstream_district": downstream["district"],
                "downstream_state": downstream["state"],
                "connection_type": "within_segment_order",
                "evidence_level": "ordered_geometry"
            })

        # Connect the final district of this segment to the
        # first district of its NEXT_DOWN segment, if available.
        next_id = data["next_down"]

        if next_id in district_sequences:
            downstream_sequence = district_sequences[next_id]["districts"]

            if sequence and downstream_sequence:
                upstream = sequence[-1]
                downstream = downstream_sequence[0]

                if upstream["code"] != downstream["code"]:
                    connections.append({
                        "upstream_river_id": river_id,
                        "downstream_river_id": next_id,
                        "upstream_district_code": upstream["code"],
                        "upstream_district": upstream["district"],
                        "upstream_state": upstream["state"],
                        "downstream_district_code": downstream["code"],
                        "downstream_district": downstream["district"],
                        "downstream_state": downstream["state"],
                        "connection_type": "NEXT_DOWN_transition",
                        "evidence_level": "hydrology_plus_ordered_geometry"
                    })

    result = pd.DataFrame(connections)

    if result.empty:
        print("No directional connections were generated.")
        return

    result = result.drop_duplicates(
        subset=[
            "upstream_river_id",
            "downstream_river_id",
            "upstream_district_code",
            "downstream_district_code",
            "connection_type"
        ]
    )

    os.makedirs("outputs", exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    # Summary statistics.
    summary = pd.DataFrame([{
        "sampled_river_segments": len(rivers),
        "segments_with_district_sequences": len(district_sequences),
        "directional_connections": len(result),
        "unique_upstream_districts":
            result["upstream_district_code"].nunique(),
        "unique_downstream_districts":
            result["downstream_district_code"].nunique(),
        "within_segment_connections":
            int((result["connection_type"] == "within_segment_order").sum()),
        "NEXT_DOWN_connections":
            int((result["connection_type"] == "NEXT_DOWN_transition").sum())
    }])

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n========== DIRECTIONAL NETWORK TEST ==========")
    print(summary.to_string(index=False))
    print("\nFirst 15 connections:")
    print(result.head(15).to_string(index=False))
    print("\nSaved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)
    print("\nTest completed.")


if __name__ == "__main__":
    main()
