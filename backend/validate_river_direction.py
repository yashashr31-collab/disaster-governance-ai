
import pandas as pd
import math
from shapely import wkt

print("Loading river network...")

rivers = pd.read_csv("outputs/india_river_network.csv")

# Keep connections whose downstream segment is also in the dataset
valid = rivers[rivers["NEXT_DOWN"].isin(rivers["HYRIV_ID"])]

sample = valid.sample(n=min(500, len(valid)), random_state=42)

# Load geometries for sampled upstream and downstream segments
ids = set(sample["HYRIV_ID"].astype(int))
ids.update(sample["NEXT_DOWN"].astype(int))

selected = rivers[rivers["HYRIV_ID"].isin(ids)]

geometries = {
    int(row.HYRIV_ID): wkt.loads(row.geometry_wkt)
    for row in selected.itertuples()
}

endpoint_results = []

for row in sample.itertuples():
    upstream_id = int(row.HYRIV_ID)
    downstream_id = int(row.NEXT_DOWN)

    upstream = list(geometries[upstream_id].coords)
    downstream = list(geometries[downstream_id].coords)

    distances = {
        "start-start": math.hypot(
            upstream[0][0] - downstream[0][0],
            upstream[0][1] - downstream[0][1]
        ),
        "start-end": math.hypot(
            upstream[0][0] - downstream[-1][0],
            upstream[0][1] - downstream[-1][1]
        ),
        "end-start": math.hypot(
            upstream[-1][0] - downstream[0][0],
            upstream[-1][1] - downstream[0][1]
        ),
        "end-end": math.hypot(
            upstream[-1][0] - downstream[-1][0],
            upstream[-1][1] - downstream[-1][1]
        )
    }

    closest = min(distances, key=distances.get)

    endpoint_results.append({
        "connection": closest,
        "distance": distances[closest]
    })

results = pd.DataFrame(endpoint_results)

print("\n========== VALIDATION RESULTS ==========")
print("Sampled connections:", len(results))

print("\nClosest endpoint combinations:")
print(results["connection"].value_counts().to_string())

print(
    "\nExact joins:",
    (results["distance"] < 0.000001).sum()
)

print(
    "Joins within 0.001 degrees:",
    (results["distance"] < 0.001).sum()
)

print(
    "Median closest distance:",
    results["distance"].median()
)

print(
    "Maximum closest distance:",
    results["distance"].max()
)

print("\nValidation completed.")
