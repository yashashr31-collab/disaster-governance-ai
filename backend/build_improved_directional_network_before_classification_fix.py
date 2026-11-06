import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from pyproj import Geod

# Improved directional river-to-district network.
# This script writes NEW "improved_" files and preserves all previous outputs.

RIVER_FILE = "outputs/india_river_network.csv"
DISTRICT_FILE = "data/raw/flood/india_districts.GeoJSON"
OUTPUT_DIR = "outputs"

ALL_FILE = os.path.join(OUTPUT_DIR, "improved_river_junction_districts.csv")
NETWORK_FILE = os.path.join(OUTPUT_DIR, "improved_directional_district_network.csv")
UNCERTAIN_FILE = os.path.join(OUTPUT_DIR, "improved_uncertain_river_connections.csv")
EXTERNAL_FILE = os.path.join(OUTPUT_DIR, "improved_external_boundary_connections.csv")
MISSING_FILE = os.path.join(OUTPUT_DIR, "improved_missing_downstream_targets.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "improved_directional_network_summary.csv")

OFFSET_METRES = 100
FALLBACK_MAX_METRES = 250
BATCH_SIZE = 10000
geod = Geod(ellps="WGS84")


def normalize_code(value):
    if pd.isna(value):
        return ""
    code = str(value).strip()
    if code.endswith(".0"):
        code = code[:-2]
    return code.zfill(3) if code else ""


def point_before_end(line, distance):
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    lon1, lat1 = coords[-2][:2]
    lon2, lat2 = coords[-1][:2]
    azimuth, _, length = geod.inv(lon1, lat1, lon2, lat2)
    if length <= 0:
        return None
    lon, lat, _ = geod.fwd(lon2, lat2, azimuth + 180, min(distance, length / 2))
    return Point(lon, lat)


def point_after_start(line, distance):
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    lon1, lat1 = coords[0][:2]
    lon2, lat2 = coords[1][:2]
    azimuth, _, length = geod.inv(lon1, lat1, lon2, lat2)
    if length <= 0:
        return None
    lon, lat, _ = geod.fwd(lon1, lat1, azimuth, min(distance, length / 2))
    return Point(lon, lat)


def match_records(joined):
    """Return matches keyed by point ID, preserving unique GeoJSON feature IDs."""
    lookup = {}
    if joined.empty:
        return lookup

    for point_id, group in joined.groupby("point_id"):
        valid = group.dropna(subset=["district_feature_id"])
        unique = {}
        for row in valid.itertuples():
            feature_id = int(row.district_feature_id)
            unique[feature_id] = {
                "feature_id": feature_id,
                "code": normalize_code(row.dtcode),
                "name": str(row.district),
                "state": "" if pd.isna(row.state_name) else str(row.state_name),
            }
        lookup[point_id] = list(unique.values())
    return lookup


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading river network...")
    rivers = pd.read_csv(
        RIVER_FILE,
        dtype={"HYRIV_ID": "string", "NEXT_DOWN": "string"}
    )
    rivers = rivers.dropna(subset=["HYRIV_ID", "geometry_wkt"]).copy()
    rivers["HYRIV_ID"] = rivers["HYRIV_ID"].str.strip()
    rivers["NEXT_DOWN"] = rivers["NEXT_DOWN"].fillna("").str.strip()
    print("River segments loaded:", len(rivers))

    print("\nLoading district boundaries...")
    districts = gpd.read_file(DISTRICT_FILE)
    districts = districts.dropna(subset=["dtcode", "district", "geometry"]).copy()
    if districts.crs is None:
        raise ValueError("District GeoJSON has no CRS. Stopping to avoid incorrect distance calculations.")

    districts = districts.reset_index(drop=True)
    districts["district_feature_id"] = districts.index.astype(int)

    # The supplied district layer uses EPSG:7755, which is projected and measured in metres.
    districts_metric = districts.to_crs("EPSG:7755")
    districts_wgs84 = districts.to_crs("EPSG:4326")

    print("District features:", len(districts))
    print("District source CRS:", districts.crs)
    print("Metric CRS:", districts_metric.crs)
    print("Duplicate dtcode features:", int(districts["dtcode"].duplicated(keep=False).sum()))

    river_lookup = (
        rivers.drop_duplicates("HYRIV_ID")
        .set_index("HYRIV_ID")["geometry_wkt"]
        .to_dict()
    )

    has_target = rivers["NEXT_DOWN"].ne("")
    missing = rivers[
        has_target & ~rivers["NEXT_DOWN"].isin(river_lookup)
    ][["HYRIV_ID", "NEXT_DOWN"]].copy()

    eligible = rivers[
        has_target & rivers["NEXT_DOWN"].isin(river_lookup)
    ].copy()

    print("River junctions available:", len(eligible))
    print("Downstream targets missing from extracted dataset:", len(missing))

    # Remove only previous improved outputs, never the original outputs.
    for path in [ALL_FILE, NETWORK_FILE, UNCERTAIN_FILE, EXTERNAL_FILE]:
        if os.path.exists(path):
            os.remove(path)
    missing.to_csv(MISSING_FILE, index=False)

    district_columns = [
        "district_feature_id", "dtcode", "district", "state_name", "geometry"
    ]

    strict_endpoint_matches = 0
    fallback_endpoint_matches = 0
    unresolved_endpoint_matches = 0
    skipped_geometry_records = 0
    processed_junctions = 0
    total_batches = (len(eligible) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_no, start in enumerate(range(0, len(eligible), BATCH_SIZE), 1):
        batch = eligible.iloc[start:start + BATCH_SIZE]
        print(f"\nProcessing batch {batch_no}/{total_batches}...")

        point_records = []
        metadata = {}

        for local_index, row in enumerate(batch.itertuples(index=False)):
            junction_id = start + local_index
            before_id = f"{junction_id}_before"
            after_id = f"{junction_id}_after"

            try:
                upstream_line = wkt.loads(row.geometry_wkt)
                downstream_line = wkt.loads(river_lookup[str(row.NEXT_DOWN)])
                before_point = point_before_end(upstream_line, OFFSET_METRES)
                after_point = point_after_start(downstream_line, OFFSET_METRES)
            except Exception:
                skipped_geometry_records += 1
                continue

            if before_point is None or after_point is None:
                skipped_geometry_records += 1
                continue

            metadata[junction_id] = {
                "upstream_river_id": str(row.HYRIV_ID),
                "downstream_river_id": str(row.NEXT_DOWN),
                "before_id": before_id,
                "after_id": after_id,
            }
            point_records.append({"point_id": before_id, "geometry": before_point})
            point_records.append({"point_id": after_id, "geometry": after_point})

        if not point_records:
            print("No valid endpoint points in this batch.")
            continue

        points = gpd.GeoDataFrame(point_records, geometry="geometry", crs="EPSG:4326")

        # Stage 1: strict point-in-polygon matching.
        strict_join = gpd.sjoin(
            points,
            districts_wgs84[district_columns],
            how="left",
            predicate="within"
        )
        strict_lookup = match_records(strict_join)

        # Stage 2: for zero-match points only, find nearest district within 250 metres.
        point_match_counts = {
            point_id: len(strict_lookup.get(point_id, []))
            for point_id in points["point_id"].tolist()
        }
        fallback_ids = [
            point_id for point_id, count in point_match_counts.items()
            if count == 0
        ]

        fallback_lookup = {}
        if fallback_ids:
            fallback_points = points[
                points["point_id"].isin(fallback_ids)
            ].to_crs("EPSG:7755")

            nearest_join = gpd.sjoin_nearest(
                fallback_points,
                districts_metric[district_columns],
                how="left",
                max_distance=FALLBACK_MAX_METRES,
                distance_col="nearest_distance_m"
            )

            for point_id, group in nearest_join.groupby("point_id"):
                valid = group.dropna(
                    subset=["district_feature_id", "nearest_distance_m"]
                )
                if valid.empty:
                    fallback_lookup[point_id] = []
                    continue

                nearest_distance = float(valid["nearest_distance_m"].min())
                # If two or more different polygons are effectively tied,
                # preserve ambiguity instead of selecting one arbitrarily.
                tied = valid[
                    valid["nearest_distance_m"] <= nearest_distance + 1.0
                ]
                unique = {}
                for row in tied.itertuples():
                    feature_id = int(row.district_feature_id)
                    unique[feature_id] = {
                        "feature_id": feature_id,
                        "code": normalize_code(row.dtcode),
                        "name": str(row.district),
                        "state": "" if pd.isna(row.state_name) else str(row.state_name),
                    }
                fallback_lookup[point_id] = list(unique.values())

        batch_results = []

        for junction_id, item in metadata.items():
            endpoint_matches = []

            for point_id in [item["before_id"], item["after_id"]]:
                strict_matches = strict_lookup.get(point_id, [])

                if len(strict_matches) == 1:
                    strict_endpoint_matches += 1
                    endpoint_matches.append((strict_matches[0], "strict_within", ""))
                elif len(strict_matches) > 1:
                    unresolved_endpoint_matches += 1
                    endpoint_matches.append(({}, "ambiguous_strict", ""))
                else:
                    nearby_matches = fallback_lookup.get(point_id, [])
                    if len(nearby_matches) == 1:
                        fallback_endpoint_matches += 1
                        endpoint_matches.append((nearby_matches[0], "nearest_fallback", ""))
                    else:
                        unresolved_endpoint_matches += 1
                        endpoint_matches.append(({}, "unmatched_or_ambiguous", ""))

            upstream, upstream_method, _ = endpoint_matches[0]
            downstream, downstream_method, _ = endpoint_matches[1]

            upstream_code = upstream.get("code", "")
            downstream_code = downstream.get("code", "")
            upstream_name = upstream.get("name", "")
            downstream_name = downstream.get("name", "")
            upstream_state = upstream.get("state", "")
            downstream_state = downstream.get("state", "")

            external_terms = ("muzaffarabad", "pakistan")
            external = any(
                term in value.lower()
                for value in [upstream_name, downstream_name, upstream_state, downstream_state]
                for term in external_terms
            )

            if not upstream or not downstream:
                status = "uncertain"
            elif external:
                status = "external_boundary"
            elif upstream["feature_id"] == downstream["feature_id"]:
                status = "same_district_continuation"
            else:
                status = "cross_district_transition"

            batch_results.append({
                "upstream_river_id": item["upstream_river_id"],
                "downstream_river_id": item["downstream_river_id"],
                "upstream_district_feature_id": upstream.get("feature_id", ""),
                "upstream_district_code": upstream_code,
                "upstream_district": upstream_name,
                "upstream_state": upstream_state,
                "upstream_match_method": upstream_method,
                "downstream_district_feature_id": downstream.get("feature_id", ""),
                "downstream_district_code": downstream_code,
                "downstream_district": downstream_name,
                "downstream_state": downstream_state,
                "downstream_match_method": downstream_method,
                "connection_type": status,
                "offset_metres": OFFSET_METRES,
                "fallback_limit_metres": FALLBACK_MAX_METRES,
            })

        result = pd.DataFrame(batch_results)
        processed_junctions += len(result)
        result.to_csv(
            ALL_FILE, mode="a",
            header=not os.path.exists(ALL_FILE),
            index=False
        )

        for status, output_path in [
            ("cross_district_transition", NETWORK_FILE),
            ("uncertain", UNCERTAIN_FILE),
            ("external_boundary", EXTERNAL_FILE),
        ]:
            subset = result[result["connection_type"] == status]
            if not subset.empty:
                subset.to_csv(
                    output_path, mode="a",
                    header=not os.path.exists(output_path),
                    index=False
                )

        print(
            "Junctions processed:", len(result),
            "| Cross-district:", int((result["connection_type"] == "cross_district_transition").sum()),
            "| Uncertain:", int((result["connection_type"] == "uncertain").sum()),
            "| Fallback endpoint matches so far:", fallback_endpoint_matches
        )

        del points, strict_join, result, batch_results

    if not os.path.exists(ALL_FILE):
        raise RuntimeError("No junction output was generated.")

    all_results = pd.read_csv(ALL_FILE, dtype=str).fillna("")
    counts = all_results["connection_type"].value_counts().to_dict()

    cross = all_results[
        all_results["connection_type"] == "cross_district_transition"
    ]

    summary = pd.DataFrame([{
        "eligible_junctions": len(eligible),
        "processed_junctions": processed_junctions,
        "skipped_geometry_records": skipped_geometry_records,
        "same_district_continuations": counts.get("same_district_continuation", 0),
        "cross_district_transitions": counts.get("cross_district_transition", 0),
        "uncertain_connections": counts.get("uncertain", 0),
        "external_boundary_connections": counts.get("external_boundary", 0),
        "missing_downstream_targets": len(missing),
        "strict_endpoint_matches": strict_endpoint_matches,
        "nearest_fallback_endpoint_matches": fallback_endpoint_matches,
        "unmatched_or_ambiguous_endpoints": unresolved_endpoint_matches,
        "fallback_limit_metres": FALLBACK_MAX_METRES,
        "unique_upstream_district_features": cross["upstream_district_feature_id"].nunique(),
        "unique_downstream_district_features": cross["downstream_district_feature_id"].nunique(),
    }])

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n========== IMPROVED NETWORK SUMMARY ==========")
    print(summary.to_string(index=False))
    print("\nNew outputs saved with 'improved_' prefixes.")
    print("Original network outputs have NOT been changed.")


if __name__ == "__main__":
    main()
