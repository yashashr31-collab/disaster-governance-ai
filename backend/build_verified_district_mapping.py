from pathlib import Path
import re
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

GEO_PATH = ROOT / "data/raw/flood/india_districts.GeoJSON"
RISK_PATH = ROOT / "outputs/flood_district_risk_ranking.csv"
OUTPUT_PATH = ROOT / "outputs/verified_district_risk_mapping.csv"


def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


# Explicitly reviewed candidate aliases.
# Key: (GeoJSON district, GeoJSON state)
# Value: district name in the historical flood dataset.
#
# These are naming equivalents to review, not proof that
# historical administrative boundaries are identical.

ALIASES = {
    ("NARMADAPURAM", "MADHYAPRADESH"): "HOSHANGABAD",
    ("KHARGONE", "MADHYAPRADESH"): "WEST NIMAR",
    ("NARSINGHPUR", "MADHYAPRADESH"): "NARSIMHAPUR",

    ("PONDICHERRY", "PUDUCHERRY"): "PUDUCHERRY",
    ("JALORE", "RAJASTHAN"): "JALOR",
    ("JHUNJHUNU", "RAJASTHAN"): "JHUNJHUNUN",

    ("DARJEELING", "WESTBENGAL"): "DARJILING",
    ("DINAJPURDAKSHIN", "WESTBENGAL"): "DAKSHIN DINAJPUR",
    ("DINAJPURUTTAR", "WESTBENGAL"): "UTTAR DINAJPUR",
    ("MEDINIPUREAST", "WESTBENGAL"): "PURBA MEDINIPUR",
    ("PURULIA", "WESTBENGAL"): "PURULIYA",

    ("VISAKHAPATANAM", "ANDHRAPRADESH"): "VISAKHAPATNAM",
    ("MARIGAON", "ASSAM"): "MORIGAON",
    ("KAMRUPMETRO", "ASSAM"): "KAMRUP METROPOLITAN",
    ("PURBICHAMPARAN", "BIHAR"): "PURBA CHAMPARAN",

    ("DANTEWADA", "CHHATTISGARH"): "DAKSHIN BASTAR DANTEWADA",
    ("KANKER", "CHHATTISGARH"): "UTTAR BASTAR KANKER",
    ("KABIRDHAM", "CHHATTISGARH"): "KABEERDHAM",
    ("KOREA", "CHHATTISGARH"): "KORIYA",
    ("GARIYABAND", "CHHATTISGARH"): "GARIABAND",
    ("BEMETARA", "CHHATTISGARH"): "BAMETARA",

    ("DANG", "GUJARAT"): "THE DANGS",
    ("CHHOTAUDEPUR", "GUJARAT"): "CHOTA UDAIPUR",
    ("ARVALLI", "GUJARAT"): "ARAVALLI",

    ("EASTSINGHBUM", "JHARKHAND"): "PURBI SINGHBHUM",
    ("WESTSINGHBHUM", "JHARKHAND"): "PASHCHIMI SINGHBHUM",
    ("KODERMA", "JHARKHAND"): "KODARMA",
    ("SAHEBGANJ", "JHARKHAND"): "SAHIBGANJ",

    ("BUDGAM", "JAMMUKASHMIR"): "BADGAM",
    ("BARAMULLA", "JAMMUKASHMIR"): "BARAMULA",
    ("POONCH", "JAMMUKASHMIR"): "PUNCH",
    ("BANDIPORA", "JAMMUKASHMIR"): "BANDIPORE",
    ("SHOPIAN", "JAMMUKASHMIR"): "SHUPIYAN",

    ("DAVANGERE", "KARNATAKA"): "DAVANAGERE",
    ("LAKSHADWEEPDISTRICT", "LAKSHADWEEP"): "LAKSHADWEEP",

    ("AHMEDNAGAR", "MAHARASHTRA"): "AHMADNAGAR",
    ("BULDHANA", "MAHARASHTRA"): "BULDANA",
    ("GONDIA", "MAHARASHTRA"): "GONDIYA",
    ("BEED", "MAHARASHTRA"): "BID",

    ("BOUDH", "ODISHA"): "BAUDH",
    ("DEOGARH", "ODISHA"): "DEBAGARH",
    ("NABARANGPUR", "ODISHA"): "NABARANGAPUR",

    ("FEROZEPUR", "PUNJAB"): "FIROZPUR",

    ("CHITTORGARH", "RAJASTHAN"): "CHITTAURGARH",
    ("DHOLPUR", "RAJASTHAN"): "DHAULPUR",

    ("KANCHIPURAM", "TAMILNADU"): "KANCHEEPURAM",
    ("VILLUPURAM", "TAMILNADU"): "VILUPPURAM",
    ("TUTICORIN", "TAMILNADU"): "THOOTHUKKUDI",
    ("LAHULANDSPITI", "HIMACHALPRADESH"): "LAHUL & SPITI",

    ("SEPAHIJALA", "TRIPURA"): "SIPAHIJALA",
    ("UNAKOTI", "TRIPURA"): "UNOKOTI",

    ("HARIDWAR", "UTTARAKHAND"): "HARDWAR",
    ("UDAMSINGHNAGAR", "UTTARAKHAND"): "UDHAM SINGH NAGAR",

    ("AYODHYA", "UTTARPRADESH"): "FAIZABAD",
    ("MAHARAJGANJ", "UTTARPRADESH"): "MAHRAJGANJ",
    ("SANTKABEERNAGAR", "UTTARPRADESH"): "SANT KABIR NAGAR",
    ("SHRAVASTI", "UTTARPRADESH"): "SHRAWASTI",
}


# State-name spelling corrections.
STATE_ALIASES = {
    "ARUNANCHALPRADESH": "ARUNACHALPRADESH",
    "ANDAMANANDNICOBARISLAND": "ANDAMANANDNICOBAR",
}


def main():

    geo = gpd.read_file(GEO_PATH)
    risk = pd.read_csv(RISK_PATH)

    geo["district_key"] = geo["district"].apply(normalize)
    geo["state_key"] = geo["state_name"].apply(normalize)

    risk["district_key"] = risk["Dist_Name"].apply(normalize)
    risk["state_key"] = risk["State_Name"].apply(normalize)

    geo["state_key"] = geo["state_key"].replace(STATE_ALIASES)

    risk_lookup = {}

    for _, row in risk.iterrows():
        key = (row["district_key"], row["state_key"])

        if key in risk_lookup:
            raise ValueError(f"Duplicate historical risk key: {key}")

        risk_lookup[key] = row

    # Prevent one historical record from being assigned to
    # multiple geographical districts.
    used_historical_keys = set()
    results = []

    for _, row in geo.iterrows():

        district_key = row["district_key"]
        state_key = row["state_key"]

        exact_key = (district_key, state_key)

        matched_key = None
        match_method = "unmatched"

        if exact_key in risk_lookup:
            matched_key = exact_key
            match_method = "exact"

        else:
            alias_name = ALIASES.get(exact_key)

            if alias_name:
                alias_key = (normalize(alias_name), state_key)

                if alias_key in risk_lookup:
                    matched_key = alias_key
                    match_method = "manual_alias"

        result = {
            "district": row["district"],
            "state": row["state_name"],
            "district_code": row["dtcode"],
            "historical_district": None,
            "historical_state": None,
            "match_method": match_method,
            "Historical_Risk_Score": None,
            "Risk_Category": "No historical match",
        }

        if matched_key:

            if matched_key in used_historical_keys:
                raise ValueError(
                    f"Historical record assigned more than once: {matched_key}"
                )

            used_historical_keys.add(matched_key)

            historical = risk_lookup[matched_key]

            result["historical_district"] = historical["Dist_Name"]
            result["historical_state"] = historical["State_Name"]
            result["Historical_Risk_Score"] = historical["Historical_Risk_Score"]
            result["Risk_Category"] = historical["Risk_Category"]

        results.append(result)

    output = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\n========== VERIFIED DISTRICT MAPPING ==========")
    print("GeoJSON districts:", len(output))
    print("Exact matches:", (output["match_method"] == "exact").sum())
    print("Manual alias matches:", (output["match_method"] == "manual_alias").sum())
    print("Unmatched districts:", (output["match_method"] == "unmatched").sum())
    print("Total matched:", output["Historical_Risk_Score"].notna().sum())

    print("\nUnmatched districts:")
    print(
        output.loc[
            output["match_method"] == "unmatched",
            ["district", "state"]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()