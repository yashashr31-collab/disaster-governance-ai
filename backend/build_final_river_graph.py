
from pathlib import Path
import json
import pandas as pd
import networkx as nx


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_FILE = Path("outputs/improved_directional_district_network.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# HELPER: FIND COLUMN NAMES
# ---------------------------------------------------------

def find_column(df, possible_names, description):
    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------
# LOAD VALIDATED RIVER CONNECTIONS
# ---------------------------------------------------------

print("Loading validated river connections...")

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print("Input transitions:", len(df))

up_id_col = find_column(
    df,
    [
        "upstream_district_feature_id",
        "upstream_feature_id",
        "upstream_district_id"
    ],
    "upstream district feature ID"
)

down_id_col = find_column(
    df,
    [
        "downstream_district_feature_id",
        "downstream_feature_id",
        "downstream_district_id"
    ],
    "downstream district feature ID"
)

up_name_col = find_column(
    df,
    ["upstream_district"],
    "upstream district name"
)

down_name_col = find_column(
    df,
    ["downstream_district"],
    "downstream district name"
)

up_river_col = find_column(
    df,
    ["upstream_river_id"],
    "upstream river ID"
)

down_river_col = find_column(
    df,
    ["downstream_river_id"],
    "downstream river ID"
)

# State columns are optional.
up_state_col = next(
    (c for c in ["upstream_state", "upstream_state_name"]
     if c in df.columns),
    None
)

down_state_col = next(
    (c for c in ["downstream_state", "downstream_state_name"]
     if c in df.columns),
    None
)


# ---------------------------------------------------------
# CLEAN AND STANDARDIZE
# ---------------------------------------------------------

df["upstream_node"] = df[up_id_col].astype(str).str.strip()
df["downstream_node"] = df[down_id_col].astype(str).str.strip()

df = df[
    (df["upstream_node"] != "") &
    (df["downstream_node"] != "")
].copy()

print("Valid transitions:", len(df))


# ---------------------------------------------------------
# BUILD DIRECTED MULTI-LEVEL NETWORK
# ---------------------------------------------------------

# One graph edge per individual river connection.
# Multiple river links between the same district features
# are preserved as separate edges in this graph.

multi_graph = nx.MultiDiGraph()

for _, row in df.iterrows():

    upstream = row["upstream_node"]
    downstream = row["downstream_node"]

    multi_graph.add_node(
        upstream,
        district=str(row[up_name_col]),
        state=str(row[up_state_col]) if up_state_col else ""
    )

    multi_graph.add_node(
        downstream,
        district=str(row[down_name_col]),
        state=str(row[down_state_col]) if down_state_col else ""
    )

    multi_graph.add_edge(
        upstream,
        downstream,
        upstream_river_id=str(row[up_river_col]),
        downstream_river_id=str(row[down_river_col])
    )


# ---------------------------------------------------------
# BUILD AGGREGATED DIRECTED DISTRICT GRAPH
# ---------------------------------------------------------

# One edge per unique directed district-feature pair.
# Repeated river connections are counted.

graph = nx.DiGraph()

for node, attrs in multi_graph.nodes(data=True):
    graph.add_node(node, **attrs)

grouped = df.groupby(
    ["upstream_node", "downstream_node"],
    dropna=False
)

edge_records = []

for (upstream, downstream), group in grouped:

    unique_upstream_rivers = group[up_river_col].nunique()
    unique_downstream_rivers = group[down_river_col].nunique()

    graph.add_edge(
        upstream,
        downstream,
        river_link_count=int(len(group)),
        unique_upstream_rivers=int(unique_upstream_rivers),
        unique_downstream_rivers=int(unique_downstream_rivers)
    )

    edge_records.append({
        "upstream_node": upstream,
        "upstream_district": group[up_name_col].iloc[0],
        "downstream_node": downstream,
        "downstream_district": group[down_name_col].iloc[0],
        "river_link_count": int(len(group)),
        "unique_upstream_rivers": int(unique_upstream_rivers),
        "unique_downstream_rivers": int(unique_downstream_rivers)
    })


# ---------------------------------------------------------
# CREATE NODE TABLE
# ---------------------------------------------------------

node_records = []

for node, attrs in graph.nodes(data=True):

    node_records.append({
        "district_feature_id": node,
        "district": attrs.get("district", ""),
        "state": attrs.get("state", ""),
        "in_degree": graph.in_degree(node),
        "out_degree": graph.out_degree(node),
        "total_degree": graph.degree(node)
    })

nodes_df = pd.DataFrame(node_records)
edges_df = pd.DataFrame(edge_records)


# ---------------------------------------------------------
# NETWORK STATISTICS
# ---------------------------------------------------------

weak_components = list(nx.weakly_connected_components(graph))
strong_components = list(nx.strongly_connected_components(graph))

largest_weak_component = max(
    (len(c) for c in weak_components),
    default=0
)

largest_strong_component = max(
    (len(c) for c in strong_components),
    default=0
)

top_downstream = nodes_df.sort_values(
    "in_degree",
    ascending=False
).head(10)

top_upstream = nodes_df.sort_values(
    "out_degree",
    ascending=False
).head(10)

summary = {
    "input_transition_records": int(len(df)),
    "individual_river_links": int(multi_graph.number_of_edges()),
    "district_feature_nodes": int(graph.number_of_nodes()),
    "unique_directed_district_connections": int(graph.number_of_edges()),
    "weakly_connected_components": int(len(weak_components)),
    "strongly_connected_components": int(len(strong_components)),
    "largest_weak_component_nodes": int(largest_weak_component),
    "largest_strong_component_nodes": int(largest_strong_component),
    "self_loops": int(nx.number_of_selfloops(graph)),
    "graph_type": "Directed district-feature river connectivity graph",
    "direction_source": "HydroRIVERS NEXT_DOWN"
}


# ---------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------

nodes_df.to_csv(
    OUTPUT_DIR / "final_river_network_nodes.csv",
    index=False
)

edges_df.to_csv(
    OUTPUT_DIR / "final_river_network_edges.csv",
    index=False
)

df.to_csv(
    OUTPUT_DIR / "final_river_network_all_links.csv",
    index=False
)

with open(
    OUTPUT_DIR / "final_river_network_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

# Save the aggregated directed graph.
nx.write_graphml(
    graph,
    OUTPUT_DIR / "final_river_network.graphml"
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n========== FINAL RIVER NETWORK ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nTOP DOWNSTREAM-CONNECTED DISTRICTS")
print(
    top_downstream[
        ["district", "state", "in_degree"]
    ].to_string(index=False)
)

print("\nTOP UPSTREAM-CONNECTED DISTRICTS")
print(
    top_upstream[
        ["district", "state", "out_degree"]
    ].to_string(index=False)
)

print("\nOUTPUT FILES CREATED:")
print("1. outputs/final_river_network_nodes.csv")
print("2. outputs/final_river_network_edges.csv")
print("3. outputs/final_river_network_all_links.csv")
print("4. outputs/final_river_network_summary.json")
print("5. outputs/final_river_network.graphml")

print("\nNETWORK BUILD COMPLETED.")
