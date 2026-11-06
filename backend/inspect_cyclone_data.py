from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/cyclone/ibtracs.csv")

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run: python scripts\\download_ibtracs.py"
        )

    print("Loading IBTrACS...")
    # IBTrACS CSV row 2 contains units; skip it so row 1 remains the header.
    df = pd.read_csv(INPUT_FILE, skiprows=[1], low_memory=False)

    print("\nDataset shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nFirst five records:\n", df.head().to_string())
    print("\nMissing values (first 20 columns):\n")
    print(df.isna().sum().head(20).to_string())

if __name__ == "__main__":
    main()
