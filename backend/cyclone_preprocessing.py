from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/cyclone/ibtracs.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "india_cyclones.csv"

REQUIRED_COLUMNS = [
    "SID", "SEASON", "BASIN", "SUBBASIN", "NAME", "ISO_TIME",
    "LAT", "LON", "WMO_WIND", "WMO_PRES", "WMO_AGENCY"
]

def classify_wmo_wind_knots(wind):
    """Initial wind bands in knots; validate against IMD conventions before official use."""
    if pd.isna(wind):
        return "Unknown"
    if wind < 17:
        return "Below Depression"
    if wind <= 27:
        return "Depression"
    if wind <= 33:
        return "Deep Depression"
    if wind <= 47:
        return "Cyclonic Storm"
    if wind <= 63:
        return "Severe Cyclonic Storm"
    if wind <= 89:
        return "Very Severe Cyclonic Storm"
    if wind <= 119:
        return "Extremely Severe Cyclonic Storm"
    return "Super Cyclonic Storm"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. First run: python scripts\\download_ibtracs.py"
        )

    print("Loading IBTrACS dataset...")
    df = pd.read_csv(
        INPUT_FILE,
        skiprows=[1],
        low_memory=False,
        na_values=["", " "],
    )

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Expected columns missing from source file: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    print("Global observations:", len(df))

    # North Indian Ocean. IBTrACS subbasin codes include BB and AS.
    df["BASIN"] = df["BASIN"].astype("string").str.strip()
    df["SUBBASIN"] = df["SUBBASIN"].astype("string").str.strip()
    df = df[
        (df["BASIN"] == "NI")
        & (df["SUBBASIN"].isin(["BB", "AS"]))
    ].copy()
    print("North Indian Ocean observations:", len(df))

    for col in ["SEASON", "LAT", "LON", "WMO_WIND", "WMO_PRES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)

    # Keep rows with a valid storm ID, timestamp and coordinates.
    df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"]).copy()

    # IBTrACS longitude may use 0..360. Retain source longitude and add -180..180.
    df = df[df["LAT"].between(-90, 90) & df["LON"].between(0, 360)].copy()
    df["LONGITUDE_180"] = ((df["LON"] + 180) % 360) - 180

    # Keep wind missing values as missing; do not substitute a different agency's estimate.
    df = df[df["WMO_WIND"].isna() | df["WMO_WIND"].between(0, 200)].copy()

    df["YEAR"] = df["ISO_TIME"].dt.year
    df["MONTH"] = df["ISO_TIME"].dt.month
    df["WIND_KMH"] = df["WMO_WIND"] * 1.852
    df["INTENSITY_CATEGORY"] = df["WMO_WIND"].apply(classify_wmo_wind_knots)

    df = df.rename(columns={
        "SID": "STORM_ID",
        "ISO_TIME": "DATE_UTC",
        "LAT": "LATITUDE",
        "LON": "LONGITUDE_0_360",
        "WMO_WIND": "WIND_KNOTS",
        "WMO_PRES": "PRESSURE_MB",
        "WMO_AGENCY": "WIND_REPORTING_AGENCY",
    })

    df = df.sort_values(["STORM_ID", "DATE_UTC"]).reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing complete.")
    print("Cleaned observations:", len(df))
    print("Unique storms:", df["STORM_ID"].nunique())
    print("\nSubbasin counts:\n", df["SUBBASIN"].value_counts().to_string())
    print("\nIntensity label counts:\n", df["INTENSITY_CATEGORY"].value_counts().to_string())
    print("\nMissing WMO wind values:", df["WIND_KNOTS"].isna().sum())
    print("\nSaved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
