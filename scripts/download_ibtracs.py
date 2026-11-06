from pathlib import Path
import requests

URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
DEST = Path("data/raw/cyclone/ibtracs.csv")

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading NOAA IBTrACS v04r01...")
    print("This may take a few minutes depending on your connection.")
    with requests.get(URL, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with DEST.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\rDownloaded {downloaded / 1024**2:.1f} / "
                            f"{total / 1024**2:.1f} MB",
                            end="",
                        )
    print(f"\nSaved dataset to: {DEST}")

if __name__ == "__main__":
    main()
