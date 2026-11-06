
import pandas as pd
from shapely import wkt

FILE = "outputs/india_river_network.csv"

df = pd.read_csv(
    FILE,
    dtype={"HYRIV_ID": str, "NEXT_DOWN": str}
)

ids = ["41399786", "41399963", "41399532", "41399497"]

print("\n========== RIVER ENDPOINT INSPECTION ==========")

for river_id in ids:
    row = df[df["HYRIV_ID"] == river_id]

    if row.empty:
        print("\nRiver not found:", river_id)
        continue

    row = row.iloc[0]
    line = wkt.loads(row["geometry_wkt"])

    print("\nRiver ID:", river_id)
    print("NEXT_DOWN:", row["NEXT_DOWN"])
    print("Length (km):", row["LENGTH_KM"])
    print("Start coordinate:", line.coords[0])
    print("End coordinate:", line.coords[-1])

print("\nInspection completed.")
