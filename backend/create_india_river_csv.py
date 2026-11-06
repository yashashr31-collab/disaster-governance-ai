
from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RIVER_FILE = ROOT / "data/raw/hydro/HydroRIVERS_Asia/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp"
INDIA_FILE = ROOT / "data/raw/flood/india_districts.GeoJSON"
OUTPUT_FILE = ROOT / "outputs/india_river_network.csv"


def main():
    print("Loading India boundaries...", flush=True)
    india = gpd.read_file(INDIA_FILE)

    print("Loading HydroRIVERS...", flush=True)
    rivers = gpd.read_file(RIVER_FILE)

    if india.crs is None or rivers.crs is None:
        raise ValueError("Missing CRS information.")

    # Convert India boundaries to river CRS.
    india = india.to_crs(rivers.crs)

    # Build one India boundary geometry.
    india_boundary = india.geometry.union_all()

    print("Building spatial index...", flush=True)
    spatial_index = rivers.sindex

    print("Finding candidate river segments...", flush=True)

    # Bounding-box query uses the spatial index.
    candidate_positions = spatial_index.query(
        india_boundary,
        predicate="intersects"
    )

    candidates = rivers.iloc[candidate_positions].copy()

    print("Candidate segments:", len(candidates), flush=True)

    # Preserve complete river geometries and original NEXT_DOWN IDs.
    # This avoids expensive clipping and keeps river segment topology.
    print("Preparing CSV...", flush=True)

    if "HYRIV_ID" not in candidates.columns:
        raise ValueError("HYRIV_ID field missing.")

    if "NEXT_DOWN" not in candidates.columns:
        raise ValueError("NEXT_DOWN field missing.")

    candidates["geometry_wkt"] = candidates.geometry.to_wkt()

    result = pd.DataFrame(
        candidates.drop(columns=[candidates.geometry.name])
    )

    result["selection_method"] = "india_boundary_intersection"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Writing CSV...", flush=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSUCCESS!", flush=True)
    print("Output:", OUTPUT_FILE, flush=True)
    print("River segments:", len(result), flush=True)
    print("Columns:", list(result.columns), flush=True)

    print("\nSample downstream links:")
    print(
        result[["HYRIV_ID", "NEXT_DOWN"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
