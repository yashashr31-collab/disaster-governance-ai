
import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from pyproj import Geod

RIVER_FILE = "outputs/india_river_network.csv"
DISTRICT_FILE = "data/raw/flood/india_districts.GeoJSON"
OUTPUT_FILE = "outputs/endpoint_district_diagnostic.csv"

OFFSET_METRES = 100
SAMPLE_SIZE = 10000
RANDOM_STATE = 42

geod = Geod(ellps="WGS84")


def point_before_end(line, distance):
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    lon1, lat1 = coords[-2][:2]
    lon2, lat2 = coords[-1][:2]

    azimuth, _, length = geod.inv(lon1, lat1, lon2, lat2)
    if length <= 0:
        return None

    lon, lat, _ = geod.fwd(
        lon2, lat2, azimuth + 180, min(distance, length / 2)
    )
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

    lon, lat, _ = geod.fwd(
        lon1, lat1, azimuth, min(distance, length / 2)
    )
    return Point(lon, lat)


def main():
    print("Loading datasets...")

    rivers = pd.read_csv(
        RIVER_FILE,
        dtype={"HYRIV_ID": str, "NEXT_DOWN": str}
    )

    districts = gpd.read_file(DISTRICT_FILE)

    districts = districts.dropna(
        subset=["dtcode", "district", "geometry"]
    ).copy()

    districts = districts.to_crs("EPSG:4326")

    sample = rivers.sample(
        n=min(SAMPLE_SIZE, len(rivers)),
        random_state=RANDOM_STATE
    )

    lookup = rivers.set_index("HYRIV_ID", drop=False)

    records = []

    print("Finding river junctions...")

    for row in sample.itertuples():
        next_id = row.NEXT_DOWN

        if pd.isna(next_id) or next_id not in lookup.index:
            continue

        downstream = lookup.loc[next_id]

        if isinstance(downstream, pd.DataFrame):
            continue

        upstream_line = wkt.loads(row.geometry_wkt)
        downstream_line = wkt.loads(downstream["geometry_wkt"])

        before = point_before_end(upstream_line, OFFSET_METRES)
        after = point_after_start(downstream_line, OFFSET_METRES)

        if before is None or after is None:
            continue

        records.append({
            "upstream_river_id": row.HYRIV_ID,
            "downstream_river_id": next_id,
            "side": "before_junction",
            "geometry": before
        })

        records.append({
            "upstream_river_id": row.HYRIV_ID,
            "downstream_river_id": next_id,
            "side": "after_junction",
            "geometry": after
        })

    print("River links inspected:", len(records) // 2)

    points = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326"
    )

    print("Assigning district codes and names...")

    joined = gpd.sjoin(
        points,
        districts[["dtcode", "district", "state_name", "geometry"]],
        how="left",
        predicate="within"
    )

    # Keep the actual district code as a string.
    joined["district_code"] = joined["dtcode"].astype("string")

    output = []

    for (up_id, down_id, side), group in joined.groupby(
        ["upstream_river_id", "downstream_river_id", "side"]
    ):
        matches = group.dropna(subset=["district_code"])
        matches = matches.drop_duplicates(subset=["district_code"])

        if len(matches) == 0:
            code = ""
            name = "UNMATCHED"
            state = ""
            count = 0
        elif len(matches) == 1:
            code = matches.iloc[0]["district_code"]
            name = matches.iloc[0]["district"]
            state = matches.iloc[0]["state_name"]
            count = 1
        else:
            code = ";".join(matches["district_code"].astype(str))
            name = ";".join(matches["district"].astype(str))
            state = ";".join(matches["state_name"].astype(str))
            count = len(matches)

        output.append({
            "upstream_river_id": up_id,
            "downstream_river_id": down_id,
            "side": side,
            "district_match_count": count,
            "district_code": code,
            "district_name": name,
            "state_name": state
        })

    result = pd.DataFrame(output)

    os.makedirs("outputs", exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== ENDPOINT DISTRICT DIAGNOSTIC ==========")
    print("River links inspected:", len(records) // 2)
    print("Endpoint points:", len(result))
    print("Exactly one district:", (result["district_match_count"] == 1).sum())
    print("Unmatched:", (result["district_match_count"] == 0).sum())
    print("Multiple matches:", (result["district_match_count"] > 1).sum())

    print("\nFirst 10 results:")
    print(result.head(10).to_string(index=False))

    print("\nSaved:", OUTPUT_FILE)
    print("Diagnostic completed.")


if __name__ == "__main__":
    main()
