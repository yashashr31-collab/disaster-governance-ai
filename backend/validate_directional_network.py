
import pandas as pd

RIVER_FILE = "outputs/india_river_network.csv"
NETWORK_FILE = "outputs/directional_district_network_test.csv"

rivers = pd.read_csv(
    RIVER_FILE,
    dtype={"HYRIV_ID": str, "NEXT_DOWN": str}
)

network = pd.read_csv(NETWORK_FILE, dtype=str)

# Reproduce the exact sample used by the network builder.
sample = rivers.sample(n=10000, random_state=42).copy()

sample_ids = set(sample["HYRIV_ID"])

valid_next = sample[
    sample["NEXT_DOWN"].notna()
    & (sample["NEXT_DOWN"] != "")
    & (sample["NEXT_DOWN"] != "0")
]

next_targets_in_sample = valid_next[
    valid_next["NEXT_DOWN"].isin(sample_ids)
]

next_targets_in_full_dataset = valid_next[
    valid_next["NEXT_DOWN"].isin(set(rivers["HYRIV_ID"]))
]

print("\n========== NEXT_DOWN COVERAGE ==========")
print("Sampled segments:", len(sample))
print("Segments with downstream IDs:", len(valid_next))
print("Downstream IDs found in full dataset:",
      len(next_targets_in_full_dataset))
print("Downstream IDs also in 10,000 sample:",
      len(next_targets_in_sample))
print("Expected possible sample transitions:",
      len(next_targets_in_sample))

print("\n========== NETWORK CONNECTIONS ==========")
print("Total connections:", len(network))
print("Within-segment connections:",
      (network["connection_type"] == "within_segment_order").sum())
print("NEXT_DOWN transitions:",
      (network["connection_type"] == "NEXT_DOWN_transition").sum())

print("\n========== RECIPROCAL DISTRICT PAIRS ==========")

pairs = network[
    ["upstream_district_code", "downstream_district_code"]
].drop_duplicates()

pairs = pairs.rename(columns={
    "upstream_district_code": "A",
    "downstream_district_code": "B"
})

pair_set = set(zip(pairs["A"], pairs["B"]))

reciprocal = sum(
    1 for a, b in pair_set if (b, a) in pair_set
)

print("Unique directed district pairs:", len(pair_set))
print("Pairs with a reverse counterpart:", reciprocal)

print("\n========== CONNECTION TYPE DETAILS ==========")
print(network["connection_type"].value_counts().to_string())

print("\nValidation diagnostic completed.")
