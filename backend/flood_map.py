from pathlib import Path
import json
import re

import geopandas as gpd
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parent.parent

# Use the actual GeoJSON filename, including uppercase extension.
GEOJSON_FILE = ROOT / "data/raw/flood/india_districts.GeoJSON"

# Use the verified district mapping instead of rematching names.
MAPPING_FILE = ROOT / "outputs/verified_district_risk_mapping.csv"


def normalize_name(value):
    """Normalize district and state names for consistent matching."""

    if pd.isna(value):
        return ""

    return re.sub(r"[^a-z0-9]", "", str(value).lower().strip())


def load_flood_map_data():

    if not GEOJSON_FILE.exists():
        raise FileNotFoundError(
            f"District GeoJSON not found: {GEOJSON_FILE}"
        )

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Verified district mapping not found: {MAPPING_FILE}"
        )

    # Load district boundaries.
    districts = gpd.read_file(GEOJSON_FILE)

    if districts.crs is None:
        raise ValueError("District GeoJSON has no coordinate reference system.")

    districts = districts.to_crs(epsg=4326)

    # Check required GeoJSON fields.
    required_geo_fields = {"district", "state_name", "dtcode"}

    missing_geo_fields = required_geo_fields - set(districts.columns)

    if missing_geo_fields:
        raise ValueError(
            f"Missing GeoJSON fields: {missing_geo_fields}"
        )

    # Load the verified mapping.
    mapping = pd.read_csv(MAPPING_FILE)

    required_mapping_fields = {
        "district",
        "state",
        "district_code",
        "historical_district",
        "match_method",
        "Historical_Risk_Score",
        "Risk_Category",
    }

    missing_mapping_fields = (
        required_mapping_fields - set(mapping.columns)
    )

    if missing_mapping_fields:
        raise ValueError(
            f"Missing mapping fields: {missing_mapping_fields}"
        )

    # Create normalized matching keys.
    districts["district_key"] = (
        districts["district"].apply(normalize_name)
    )

    districts["state_key"] = (
        districts["state_name"].apply(normalize_name)
    )

    mapping["district_key"] = (
        mapping["district"].apply(normalize_name)
    )

    mapping["state_key"] = (
        mapping["state"].apply(normalize_name)
    )

    districts["map_key"] = (
        districts["state_key"] + "|" + districts["district_key"]
    )

    mapping["map_key"] = (
        mapping["state_key"] + "|" + mapping["district_key"]
    )

    # Ensure each geographical district has one mapping record.
    if districts["map_key"].duplicated().any():
        duplicates = districts.loc[
            districts["map_key"].duplicated(keep=False),
            ["district", "state_name", "dtcode"]
        ]

        raise ValueError(
            "Duplicate district/state combinations in GeoJSON:\n"
            + duplicates.to_string(index=False)
        )

    if mapping["map_key"].duplicated().any():
        duplicates = mapping.loc[
            mapping["map_key"].duplicated(keep=False),
            ["district", "state", "district_code"]
        ]

        raise ValueError(
            "Duplicate district/state combinations in verified mapping:\n"
            + duplicates.to_string(index=False)
        )

    # Keep only the fields required from the verified mapping.
    mapping_columns = [
        "map_key",
        "historical_district",
        "match_method",
        "Historical_Risk_Score",
        "Risk_Category",
    ]

    # Join district boundaries with verified historical risk scores.
    merged = districts.merge(
        mapping[mapping_columns],
        on="map_key",
        how="left",
        validate="one_to_one",
    )

    # Calculate matching statistics.
    matched = int(
        merged["Historical_Risk_Score"].notna().sum()
    )

    unmatched = int(
        merged["Historical_Risk_Score"].isna().sum()
    )

    return merged, matched, unmatched


def create_flood_map(merged, selected_state="All States"):

    map_data = merged.copy()

    # Filter by selected state.
    if selected_state != "All States":
        map_data = map_data[
            map_data["state_name"] == selected_state
        ].copy()

    # Only display districts with verified historical risk scores.
    map_data = map_data[
        map_data["Historical_Risk_Score"].notna()
    ].copy()

    if map_data.empty:
        return None

    # Convert district geometries into GeoJSON.
    geojson = json.loads(map_data.to_json())

    # Create district-level historical flood-risk map.
    fig = px.choropleth_map(
        map_data,
        geojson=geojson,
        locations="map_key",
        featureidkey="properties.map_key",
        color="Historical_Risk_Score",
        hover_name="district",
        hover_data={
            "state_name": True,
            "historical_district": True,
            "Historical_Risk_Score": ":.2f",
            "Risk_Category": True,
            "match_method": True,
            "map_key": False,
        },
        color_continuous_scale="YlOrRd",
        range_color=(0, 100),
        map_style="carto-positron",
        center={
            "lat": 22.5,
            "lon": 82.0,
        },
        zoom=3.2,
        opacity=0.75,
        title="India District-Level Historical Flood Risk",
    )

    fig.update_layout(
        height=700,
        margin={
            "r": 0,
            "t": 50,
            "l": 0,
            "b": 0,
        },
    )

    return fig