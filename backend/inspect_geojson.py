from pathlib import Path
import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent

GEOJSON_FILE = ROOT / "data" / "raw" / "flood" / "india_districts.geojson"

print("\nChecking GeoJSON file...")

if not GEOJSON_FILE.exists():
    print("ERROR: GeoJSON file not found!")
    print("Expected location:", GEOJSON_FILE)
else:
    districts = gpd.read_file(GEOJSON_FILE)

    print("\nGEOJSON LOADED SUCCESSFULLY")
    print("--------------------------------")
    print("Number of district features:", len(districts))

    print("\nAvailable columns:")
    for column in districts.columns:
        print("-", column)

    print("\nFirst 5 district records:")
    print(districts.drop(columns="geometry").head().to_string())

    print("\nGeometry types:")
    print(districts.geometry.geom_type.value_counts())

    print("\nCoordinate reference system:")
    print(districts.crs)