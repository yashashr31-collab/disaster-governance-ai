
"""
District Connectivity Engine
-----------------------------
Builds a geographical adjacency graph from India's district GeoJSON.

IMPORTANT:
Shared boundaries indicate geographical adjacency only.
They do NOT prove upstream/downstream river connectivity.
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import networkx as nx


ROOT = Path(__file__).resolve().parent.parent

GEOJSON_PATH = ROOT / "data" / "raw" / "flood" / "india_districts.GeoJSON"


def load_district_boundaries():
    """Load district boundaries from GeoJSON."""

    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"District GeoJSON not found: {GEOJSON_PATH}"
        )

    gdf = gpd.read_file(GEOJSON_PATH)

    required = {"district", "state", "geometry"}

    missing = required - set(gdf.columns)

    if missing:
        raise ValueError(
            f"Missing required GeoJSON fields: {missing}"
        )

    if gdf.crs is None:
        raise ValueError("GeoJSON CRS is missing.")

    gdf = gdf.to_crs(epsg=4326)

    gdf["district"] = gdf["district"].astype(str).str.strip()
    gdf["state"] = gdf["state"].astype(str).str.strip()

    gdf = gdf[
        gdf.geometry.notna() &
        ~gdf.geometry.is_empty
    ].copy()

    return gdf


def build_adjacency_graph():
    """
    Build an undirected graph.

    Each node represents a district.
    Each edge represents a shared geographical boundary.
    """

    gdf = load_district_boundaries()

    graph = nx.Graph()

    # Add district nodes.
    for index, row in gdf.iterrows():

        node_id = str(row["dtcode"]) if "dtcode" in gdf.columns else str(index)

        graph.add_node(
            node_id,
            district=row["district"],
            state=row["state"],
        )

    # Use a projected CRS for reliable spatial operations.
    projected = gdf.to_crs(epsg=6933)

    spatial_index = projected.sindex

    for i, row in projected.iterrows():

        geom = row.geometry

        possible_neighbors = spatial_index.query(
            geom,
            predicate="intersects"
        )

        for j in possible_neighbors:

            if i >= j:
                continue

            other = projected.iloc[j]

            # A shared boundary must have positive length.
            shared_boundary = geom.boundary.intersection(
                other.geometry.boundary
            )

            if shared_boundary.length > 0:

                node_a = str(row["dtcode"])
                node_b = str(other["dtcode"])

                graph.add_edge(
                    node_a,
                    node_b,
                    connection_type="geographical_adjacency",
                    shared_boundary_length_m=float(
                        shared_boundary.length
                    ),
                )

    return graph


def get_connected_districts(graph, district_code, max_hops=1):
    """
    Return geographically connected districts.

    max_hops=1 means direct neighbours.
    max_hops=2 includes neighbours of neighbours.

    This is NOT a flood propagation prediction.
    """

    district_code = str(district_code)

    if district_code not in graph:
        raise ValueError(
            f"District code {district_code} not found."
        )

    distances = nx.single_source_shortest_path_length(
        graph,
        district_code,
        cutoff=max_hops
    )

    results = []

    for node_id, distance in distances.items():

        if node_id == district_code:
            continue

        data = graph.nodes[node_id]

        results.append({
            "district_code": node_id,
            "district": data["district"],
            "state": data["state"],
            "geographical_hops": distance,
            "connection_type": "geographical_adjacency",
        })

    return pd.DataFrame(results)


def export_adjacency(graph):
    """Export the adjacency relationships to CSV."""

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    records = []

    for source, target, data in graph.edges(data=True):

        records.append({
            "district_a_code": source,
            "district_a": graph.nodes[source]["district"],
            "state_a": graph.nodes[source]["state"],
            "district_b_code": target,
            "district_b": graph.nodes[target]["district"],
            "state_b": graph.nodes[target]["state"],
            "connection_type": data["connection_type"],
            "shared_boundary_length_m": data[
                "shared_boundary_length_m"
            ],
        })

    output_path = output_dir / "district_adjacency.csv"

    pd.DataFrame(records).to_csv(output_path, index=False)

    return output_path
if __name__ == "__main__":
    print("Loading district boundaries...")

    graph = build_adjacency_graph()

    print(f"Total districts: {graph.number_of_nodes()}")
    print(f"Total connections: {graph.number_of_edges()}")

    print("Exporting adjacency data...")

    output_path = export_adjacency(graph)

    print(f"Successfully saved: {output_path}")
