# Disaster Governance AI — starter project

A hackathon starter for an India-focused cyclone and flood hazard governance dashboard.

## Included in this ZIP
- IBTrACS cyclone CSV download helper
- Cyclone dataset inspection and preprocessing
- Basic exploratory visualizations
- Streamlit dashboard starter
- Project folders, requirements, and Git ignore rules

**The raw IBTrACS dataset is not bundled** because it is a large external dataset and its contents/version can change. Download it using the included script.

## Requirements
- Windows 10/11
- Python 3.10 or newer
- VS Code

## Setup (PowerShell, from this folder)

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 1. Download IBTrACS
```powershell
python scripts\download_ibtracs.py
```

The file is saved to:
`data/raw/cyclone/ibtracs.csv`

Source: NOAA NCEI IBTrACS v04r01 CSV directory:
https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/

## 2. Inspect the raw dataset
```powershell
python backend\inspect_cyclone_data.py
```

## 3. Preprocess North Indian Ocean observations
```powershell
python backend\cyclone_preprocessing.py
```

Output:
`data/processed/india_cyclones.csv`

The script filters IBTrACS basin `NI`, then keeps Bay of Bengal (`BB`) and Arabian Sea (`AS`) subbasins when those codes are present. It preserves agency and original wind fields. It uses `WMO_WIND` for the initial wind-based category only; missing WMO wind values are not silently replaced with another agency's wind estimate.

**Important scientific limitation:** IBTrACS is a historical best-track archive, not a real-time forecast feed. The prepared CSV is a historical observation dataset, not a future cyclone prediction model. Wind averaging conventions and agency reporting practices can differ. Validate the chosen wind field and category thresholds against the India Meteorological Department (IMD) conventions before presenting results as official IMD intensity categories.

## 4. Create exploratory plots
```powershell
python notebooks\cyclone_analysis.py
```
Plots are saved under `outputs/`.

## 5. Launch dashboard starter
```powershell
streamlit run dashboard\app.py
```

## GitHub
```powershell
git init
git add .
git commit -m "Add disaster governance AI starter"
```

Do not commit the raw downloaded dataset or virtual environment. `.gitignore` excludes them.

## Next development stages
1. Validate cyclone labels and time-series features.
2. Add rainfall, elevation, slope, river, land-cover, and historical flood data.
3. Build and evaluate flood susceptibility model.
4. Add exposure, vulnerability, fairness, uncertainty, and governance prioritization layers.
5. Integrate models into the dashboard.

Do not use this prototype as an operational warning or evacuation system. Real-world decisions require authoritative data, validation, and review by disaster-management authorities.
