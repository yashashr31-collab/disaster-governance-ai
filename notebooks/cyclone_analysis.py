from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("data/processed/india_cyclones.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run backend\\cyclone_preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["DATE_UTC"])

    # 1. Distribution of observation-level intensity labels.
    # Note: this counts track observations, not unique cyclones.
    counts = df["INTENSITY_CATEGORY"].value_counts()
    ax = counts.plot(kind="bar", figsize=(12, 5), title="IBTrACS North Indian Ocean observations by wind band")
    ax.set_xlabel("Wind band")
    ax.set_ylabel("Number of observations")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cyclone_intensity_observations.png", dpi=160)
    plt.close()

    # 2. Track point map (static scatter); color uses WMO wind knots.
    plot_df = df.dropna(subset=["WIND_KNOTS"])
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        plot_df["LONGITUDE_180"], plot_df["LATITUDE"],
        c=plot_df["WIND_KNOTS"], cmap="turbo", s=10, alpha=0.65
    )
    plt.colorbar(scatter, label="WMO wind (knots)")
    plt.title("Historical North Indian Ocean cyclone observations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cyclone_track_points.png", dpi=160)
    plt.close()

    # 3. Unique storm count by season, not observation count.
    storms = df.dropna(subset=["YEAR"]).groupby("YEAR")["STORM_ID"].nunique()
    ax = storms.plot(kind="bar", figsize=(13, 5), title="Unique North Indian Ocean storms by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique storms")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cyclones_by_year.png", dpi=160)
    plt.close()

    print("Saved plots in outputs/")

if __name__ == "__main__":
    main()
