
import pandas as pd

river_file = "outputs/india_river_network.csv"
network_file = "outputs/directional_district_network_test.csv"

rivers = pd.read_csv(
    river_file,
    dtype={"HYRIV_ID": str, "NEXT_DOWN": str}
)

network = pd.read_csv(network_file, dtype=str)

sample = rivers.sample(n=10000, random_state=42)

sample_ids = set(sample["HYRIV_ID"])

valid = sample[
    sample["NEXT_DOWN"].isin(sample_ids)
].copy()

transitions = network[
    network["connection_type"] == "NEXT_DOWN_transition"
]

print("\n========== NEXT_DOWN LINK INSPECTION ==========")
print("Available NEXT_DOWN links:", len(valid))
print("Generated district transitions:", len(transitions))

print("\nFirst 15 available river links:")
print(
    valid[["HYRIV_ID", "NEXT_DOWN"]]
    .head(15)
    .to_string(index=False)
)

print("\nGenerated district transitions:")
if len(transitions):
    print(
        transitions[
            [
                "upstream_river_id",
                "downstream_river_id",
                "upstream_district",
                "downstream_district",
                "connection_type"
            ]
        ].to_string(index=False)
    )

print("\nInspection completed.")
