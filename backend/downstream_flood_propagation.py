
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RIVER_CSV = ROOT / "outputs/india_river_network.csv"
DISTRICT_GEOJSON = ROOT / "data/raw/flood/india_districts.GeoJSON"

OUTPUT_DIR = ROOT / "outputs"
DISTRICT_LINKS_FILE = OUTPUT_DIR / "river_district_connections.csv"
SUMMARY_FILE = OUTPUT_DIR / "downstream_engine_summary.csv"


def clean_id(series):
    """Normalize numeric river identifiers without changing their values."""
    return series.astype("string").str.strip()


def main():
    print("Loading river network CSV...", flush=True)

    rivers = pd.read_csv(
        RIVER_CSV,
        dtype={"HYRIV_ID": "string", "NEXT_DOWN": "string"}
    )

    required_river_fields = [
        "HYRIV_ID",
        "NEXT_DOWN",
        "geometry_wkt"
    ]

    for column in required_river_fields:
        if column not in rivers.columns:
            raise ValueError(f"Missing river field: {column}")

    rivers["HYRIV_ID"] = clean_id(rivers["HYRIV_ID"])
    rivers["NEXT_DOWN"] = clean_id(rivers["NEXT_DOWN"])

    rivers = rivers.dropna(subset=["HYRIV_ID", "geometry_wkt"])
    rivers = rivers.drop_duplicates(subset=["HYRIV_ID"])

    print("River segments loaded:", len(rivers), flush=True)

    print("Loading district boundaries...", flush=True)

    districts = gpd.read_file(DISTRICT_GEOJSON)

    if districts.crs is None:
        raise ValueError("District GeoJSON has no CRS.")

    if "dtcode" not in districts.columns:
        raise ValueError("District GeoJSON is missing dtcode.")

    district_name_field = (
        "district" if "district" in districts.columns else None
    )

    state_field = (
        "state_name"
        if "state_name" in districts.columns
        else "state"
        if "state" in districts.columns
        else None
    )

    if district_name_field is None or state_field is None:
        raise ValueError(
            "Could not identify district/state fields in GeoJSON."
        )

    districts = districts[
        ["dtcode", district_name_field, state_field, "geometry"]
    ].copy()

    districts = districts.rename(
        columns={
            "dtcode": "district_code",
            district_name_field: "district_name",
            state_field: "state_name"
        }
    )

    districts["district_code"] = (
        districts["district_code"].astype("string").str.strip()
    )

    # Convert river WKT strings into geometries.
    print("Building river geometries...", flush=True)

    river_geometry = gpd.GeoSeries.from_wkt(
        rivers["geometry_wkt"],
        crs="EPSG:4326"
    )

    river_gdf = gpd.GeoDataFrame(
        rivers[["HYRIV_ID", "NEXT_DOWN"]].copy(),
        geometry=river_geometry,
        crs="EPSG:4326"
    )

    # Put both datasets in the same CRS.
    river_gdf = river_gdf.to_crs(districts.crs)

    print("Associating river segments with districts...", flush=True)

    # Spatial index is used internally by GeoPandas.
    joined = gpd.sjoin(
        river_gdf,
        districts,
        how="inner",
        predicate="intersects"
    )

    joined = joined[
        ["HYRIV_ID", "NEXT_DOWN", "district_code",
         "district_name", "state_name"]
    ].drop_duplicates()

    print("River-to-district associations:", len(joined), flush=True)

    # A river segment may intersect more than one district.
    # Keep all associations, but do not treat them as confirmed flood impacts.
    river_districts = {}

    for row in joined.itertuples(index=False):
        river_districts.setdefault(row.HYRIV_ID, []).append(
            (
                row.district_code,
                row.district_name,
                row.state_name
            )
        )

    print("Building downstream district connections...", flush=True)

    links = []
    seen = set()

    # NEXT_DOWN means the next downstream river segment.
    for row in rivers[["HYRIV_ID", "NEXT_DOWN"]].itertuples(index=False):
        upstream_id = row.HYRIV_ID
        downstream_id = row.NEXT_DOWN

        if pd.isna(downstream_id) or downstream_id in ("", "0", "-1"):
            continue

        upstream_districts = river_districts.get(upstream_id, [])
        downstream_districts = river_districts.get(downstream_id, [])

        if not upstream_districts or not downstream_districts:
            continue

        for upstream in upstream_districts:
            for downstream in downstream_districts:

                upstream_code, upstream_name, upstream_state = upstream
                downstream_code, downstream_name, downstream_state = downstream

                # Same-district segment transitions are not district links.
                if upstream_code == downstream_code:
                    continue

                key = (
                    upstream_id,
                    downstream_id,
                    upstream_code,
                    downstream_code
                )

                if key in seen:
                    continue

                seen.add(key)

                links.append({
                    "upstream_river_id": upstream_id,
                    "downstream_river_id": downstream_id,
                    "upstream_district_code": upstream_code,
                    "upstream_district": upstream_name,
                    "upstream_state": upstream_state,
                    "downstream_district_code": downstream_code,
                    "downstream_district": downstream_name,
                    "downstream_state": downstream_state,
                    "connection_type": "hydrorivers_next_down",
                    "evidence_level": "river_segment_direction",
                    "district_association": "spatial_intersection",
                    "arrival_time_estimate": "not_available"
                })

    links_df = pd.DataFrame(links)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    links_df.to_csv(DISTRICT_LINKS_FILE, index=False)

    # Diagnostics: check how many downstream IDs resolve in the selected CSV.
    available_ids = set(rivers["HYRIV_ID"].dropna())
    nonzero_next = rivers[
        rivers["NEXT_DOWN"].notna()
        & ~rivers["NEXT_DOWN"].isin(["", "0", "-1"])
    ]

    resolved = nonzero_next["NEXT_DOWN"].isin(available_ids).sum()
    unresolved = len(nonzero_next) - resolved

    summary = pd.DataFrame([
        {"metric": "river_segments", "value": len(rivers)},
        {"metric": "river_district_associations", "value": len(joined)},
        {"metric": "downstream_ids_checked", "value": len(nonzero_next)},
        {"metric": "downstream_ids_resolved_in_india_csv", "value": int(resolved)},
        {"metric": "downstream_ids_not_in_india_csv", "value": int(unresolved)},
        {"metric": "directed_district_connections", "value": len(links_df)},
        {"metric": "unique_upstream_districts", "value": (
            links_df["upstream_district_code"].nunique()
            if not links_df.empty else 0
        )},
        {"metric": "unique_downstream_districts", "value": (
            links_df["downstream_district_code"].nunique()
            if not links_df.empty else 0
        )}
    ])

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nDOWNSTREAM ENGINE BUILD COMPLETE", flush=True)
    print("District connections:", DISTRICT_LINKS_FILE, flush=True)
    print("Summary:", SUMMARY_FILE, flush=True)
    print("\n", summary.to_string(index=False), flush=True)

    if links_df.empty:
        print(
            "\nWARNING: No district-level connections were generated. "
            "We must inspect the spatial associations and downstream IDs.",
            flush=True
        )
    else:
        print("\nSample downstream connections:")
        print(links_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
