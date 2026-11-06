
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

flood_path = ROOT / "outputs" / "flood_district_risk_ranking.csv"

flood = pd.read_csv(flood_path)

states = [
    "ANDHRA PRADESH",
    "ASSAM",
    "BIHAR",
    "CHHATTISGARH",
    "JHARKHAND",
    "KARNATAKA",
    "MADHYA PRADESH",
    "MAHARASHTRA",
    "PUDUCHERRY",
    "SIKKIM",
    "TELANGANA",
    "UTTAR PRADESH",
    "UTTARAKHAND",
    "WEST BENGAL",
]

flood["State_Name"] = flood["State_Name"].astype(str).str.upper()

for state in states:
    print("\n==========", state, "==========")

    names = sorted(
        flood.loc[
            flood["State_Name"] == state,
            "Dist_Name"
        ].dropna().astype(str).unique()
    )

    for name in names:
        print(name)

print("\n========== COMPLETE ==========")
