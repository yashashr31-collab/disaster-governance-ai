
import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from pyproj import Geod

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RIVER_FILE = "outputs/india_river_network.csv"
DISTRICT_FILE = "data/raw/flood/india_districts.GeoJSON"

OUTPUT_DIR = "outputs"

ALL_JUNCTIONS_FILE = os.path.join(
    OUTPUT_DIR, "full_river_junction_districts.csv"
)
NETWORK_FILE = os.path.join(
    OUTPUT_DIR, "full_directional_district_network.csv"
)
UNCERTAIN_FILE = os.path.join(
    OUTPUT_DIR, "uncertain_river_connections.csv"
)
EXTERNAL_FILE = os.path.join(
    OUTPUT_DIR, "external_boundary_connections.csv"
)
MISSING_FILE = os.path.join(
    OUTPUT_DIR, "missing_downstream_targets.csv"
)
SUMMARY_FILE = os.path.join(
    OUTPUT_DIR, "full_directional_network_summary.csv"
)

OFFSET_METRES = 100
BATCH_SIZE = 10000

geod = Geod(ellps="WGS84")


# ---------------------------------------------------------
# GEOMETRY FUNCTIONS
# ---------------------------------------------------------

def normalize_code(value):
    """Preserve district codes as text, including leading zeros."""

    if pd.isna(value):
        return ""

    code = str(value).strip()

    if code.endswith(".0"):
        code = code[:-2]

    return code.zfill(3)


def point_before_end(line, distance):
    """Create a point up to 100 metres before a river segment end."""

    coords = list(line.coords)

    if len(coords) < 2:
        return None

    lon1, lat1 = coords[-2][:2]
    lon2, lat2 = coords[-1][:2]

    azimuth, _, length = geod.inv(
        lon1, lat1, lon2, lat2
    )

    if length <= 0:
        return None

    lon, lat, _ = geod.fwd(
        lon2,
        lat2,
        azimuth + 180,
        min(distance, length / 2)
    )

    return Point(lon, lat)


def point_after_start(line, distance):
    """Create a point up to 100 metres after a river segment start."""

    coords = list(line.coords)

    if len(coords) < 2:
        return None

    lon1, lat1 = coords[0][:2]
    lon2, lat2 = coords[1][:2]

    azimuth, _, length = geod.inv(
        lon1, lat1, lon2, lat2
    )

    if length <= 0:
        return None

    lon, lat, _ = geod.fwd(
        lon1,
        lat1,
        azimuth,
        min(distance, length / 2)
    )

    return Point(lon, lat)


# ---------------------------------------------------------
# DISTRICT MATCHING
# ---------------------------------------------------------

def get_point_matches(joined):
    """Return unique district matches for every endpoint point."""

    match_lookup = {}

    for point_id, group in joined.groupby("point_id"):

        matches = group.dropna(
            subset=["dtcode", "district"]
        )

        unique = {}

        for row in matches.itertuples():

            code = normalize_code(row.dtcode)

            if not code:
                continue

            unique[code] = {
                "code": code,
                "name": str(row.district),
                "state": (
                    "" if pd.isna(row.state_name)
                    else str(row.state_name)
                )
            }

        match_lookup[point_id] = list(unique.values())

    return match_lookup


# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading river dataset...")

    rivers = pd.read_csv(
        RIVER_FILE,
        dtype={
            "HYRIV_ID": "string",
            "NEXT_DOWN": "string"
        }
    )

    rivers = rivers.dropna(
        subset=["HYRIV_ID", "geometry_wkt"]
    ).copy()

    rivers["HYRIV_ID"] = rivers["HYRIV_ID"].str.strip()
    rivers["NEXT_DOWN"] = rivers["NEXT_DOWN"].str.strip()

    print("River segments loaded:", len(rivers))

    print("Loading district boundaries...")

    districts = gpd.read_file(DISTRICT_FILE)

    districts = districts.dropna(
        subset=["dtcode", "district", "geometry"]
    ).copy()

    districts = districts.to_crs("EPSG:4326")

    print("District features loaded:", len(districts))

    # Keep one record per river ID for downstream lookup.
    river_lookup = (
        rivers.drop_duplicates("HYRIV_ID")
        .set_index("HYRIV_ID")["geometry_wkt"]
        .to_dict()
    )

    # Find river segments with a downstream target.
    has_target = (
        rivers["NEXT_DOWN"].notna()
        & (rivers["NEXT_DOWN"] != "")
    )

    missing_target = rivers[
        has_target
        & ~rivers["NEXT_DOWN"].isin(river_lookup)
    ][["HYRIV_ID", "NEXT_DOWN"]].copy()

    eligible = rivers[
        has_target
        & rivers["NEXT_DOWN"].isin(river_lookup)
    ].copy()

    print("River junctions available:", len(eligible))
    print("Downstream targets missing from extracted dataset:",
          len(missing_target))

    # Start new outputs without touching earlier diagnostics.
    output_files = [
        ALL_JUNCTIONS_FILE,
        NETWORK_FILE,
        UNCERTAIN_FILE,
        EXTERNAL_FILE
    ]

    for path in output_files:
        if os.path.exists(path):
            os.remove(path)

    missing_target.to_csv(MISSING_FILE, index=False)

    district_columns = [
        "dtcode", "district", "state_name", "geometry"
    ]

    total_batches = (
        len(eligible) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("\nStarting full-network processing...")

    for batch_number, start in enumerate(
        range(0, len(eligible), BATCH_SIZE), start=1
    ):

        batch = eligible.iloc[
            start:start + BATCH_SIZE
        ]

        point_records = []
        junction_metadata = {}

        print(
            f"\nProcessing batch {batch_number}/{total_batches}"
        )

        for local_index, row in enumerate(
            batch.itertuples(index=False)
        ):

            junction_id = start + local_index
            before_id = f"{junction_id}_before"
            after_id = f"{junction_id}_after"

            downstream_id = row.NEXT_DOWN

            try:
                upstream_line = wkt.loads(row.geometry_wkt)
                downstream_line = wkt.loads(
                    river_lookup[downstream_id]
                )

                before = point_before_end(
                    upstream_line, OFFSET_METRES
                )

                after = point_after_start(
                    downstream_line, OFFSET_METRES
                )

            except Exception:
                continue

            if before is None or after is None:
                continue

            junction_metadata[junction_id] = {
                "upstream_river_id": row.HYRIV_ID,
                "downstream_river_id": downstream_id,
                "before_id": before_id,
                "after_id": after_id
            }

            point_records.append({
                "point_id": before_id,
                "geometry": before
            })

            point_records.append({
                "point_id": after_id,
                "geometry": after
            })

        if not point_records:
            print("No valid endpoint points in this batch.")
            continue

        points = gpd.GeoDataFrame(
            point_records,
            geometry="geometry",
            crs="EPSG:4326"
        )

        joined = gpd.sjoin(
            points,
            districts[district_columns],
            how="left",
            predicate="within"
        )

        match_lookup = get_point_matches(joined)

        batch_results = []

        for junction_id, metadata in junction_metadata.items():

            before_matches = match_lookup.get(
                metadata["before_id"], []
            )

            after_matches = match_lookup.get(
                metadata["after_id"], []
            )

            before_count = len(before_matches)
            after_count = len(after_matches)

            before_match = (
                before_matches[0] if before_count == 1 else {}
            )

            after_match = (
                after_matches[0] if after_count == 1 else {}
            )

            upstream_code = before_match.get("code", "")
            upstream_name = before_match.get("name", "")
            upstream_state = before_match.get("state", "")

            downstream_code = after_match.get("code", "")
            downstream_name = after_match.get("name", "")
            downstream_state = after_match.get("state", "")

            external_names = (
                "muzaffarabad",
                "pakistan"
            )

            external = (
                any(x in upstream_name.lower()
                    for x in external_names)
                or
                any(x in downstream_name.lower()
                    for x in external_names)
                or
                any(x in upstream_state.lower()
                    for x in external_names)
                or
                any(x in downstream_state.lower()
                    for x in external_names)
            )

            if before_count != 1 or after_count != 1:
                status = "uncertain"
            elif external:
                status = "external_boundary"
            elif upstream_code == downstream_code:
                status = "same_district_continuation"
            else:
                status = "cross_district_transition"

            batch_results.append({
                "upstream_river_id":
                    metadata["upstream_river_id"],

                "downstream_river_id":
                    metadata["downstream_river_id"],

                "upstream_district_code": upstream_code,
                "upstream_district": upstream_name,
                "upstream_state": upstream_state,
                "upstream_match_count": before_count,

                "downstream_district_code": downstream_code,
                "downstream_district": downstream_name,
                "downstream_state": downstream_state,
                "downstream_match_count": after_count,

                "connection_type": status,
                "offset_metres": OFFSET_METRES
            })

        result = pd.DataFrame(batch_results)

        # Save all junctions.
        result.to_csv(
            ALL_JUNCTIONS_FILE,
            mode="a",
            header=not os.path.exists(ALL_JUNCTIONS_FILE),
            index=False
        )

        # Save confirmed India-only cross-district transitions.
        network = result[
            result["connection_type"]
            == "cross_district_transition"
        ]

        if not network.empty:
            network.to_csv(
                NETWORK_FILE,
                mode="a",
                header=not os.path.exists(NETWORK_FILE),
                index=False
            )

        # Save uncertain connections.
        uncertain = result[
            result["connection_type"] == "uncertain"
        ]

        if not uncertain.empty:
            uncertain.to_csv(
                UNCERTAIN_FILE,
                mode="a",
                header=not os.path.exists(UNCERTAIN_FILE),
                index=False
            )

        # Save external-boundary connections separately.
        external_rows = result[
            result["connection_type"] == "external_boundary"
        ]

        if not external_rows.empty:
            external_rows.to_csv(
                EXTERNAL_FILE,
                mode="a",
                header=not os.path.exists(EXTERNAL_FILE),
                index=False
            )

        print("Junctions processed:", len(result))
        print("Cross-district transitions:", len(network))
        print("Uncertain connections:", len(uncertain))
        print("External-boundary connections:", len(external_rows))

        del points, joined, result, batch_results

    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\nBuilding final summary...")

    if os.path.exists(ALL_JUNCTIONS_FILE):

        all_results = pd.read_csv(
            ALL_JUNCTIONS_FILE,
            dtype=str
        )

        counts = (
            all_results["connection_type"]
            .value_counts()
            .to_dict()
        )

        summary = pd.DataFrame([{
            "processed_junctions": len(all_results),
            "same_district_continuations":
                counts.get("same_district_continuation", 0),
            "cross_district_transitions":
                counts.get("cross_district_transition", 0),
            "uncertain_connections":
                counts.get("uncertain", 0),
            "external_boundary_connections":
                counts.get("external_boundary", 0),
            "missing_downstream_targets":
                len(missing_target),
            "unique_upstream_districts":
                all_results[
                    all_results["connection_type"]
                    == "cross_district_transition"
                ]["upstream_district_code"].nunique(),
            "unique_downstream_districts":
                all_results[
                    all_results["connection_type"]
                    == "cross_district_transition"
                ]["downstream_district_code"].nunique()
        }])

        summary.to_csv(SUMMARY_FILE, index=False)

        print("\n========== FULL NETWORK SUMMARY ==========")
        print(summary.to_string(index=False))

    else:
        print("No junction output was generated.")

    print("\nOutput files:")
    print(ALL_JUNCTIONS_FILE)
    print(NETWORK_FILE)
    print(UNCERTAIN_FILE)
    print(EXTERNAL_FILE)
    print(MISSING_FILE)
    print(SUMMARY_FILE)

    print("\nFull directional network processing completed.")


if __name__ == "__main__":
    main()
