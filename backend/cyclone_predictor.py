from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "cyclone_sequence_model.pkl"
DATA_FILE = ROOT / "data" / "processed" / "india_cyclones.csv"


def load_cyclone_data():
    """Load cyclone observations and normalize required data types."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cyclone data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df["DATE_UTC"] = pd.to_datetime(
        df["DATE_UTC"], utc=True, errors="coerce"
    )

    for column in ["WIND_KNOTS", "LATITUDE", "LONGITUDE_180"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def load_cyclone_model():
    """Load the trained model and its exact training feature order."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_FILE}")

    saved = joblib.load(MODEL_FILE)
    return saved["model"], saved["features"]


def predict_next_wind(storm_df, model, features):
    """
    Estimate the next recorded wind observation.

    The next location is unknown. This prototype assumes zero movement
    from the latest recorded position. It is not an operational forecast.
    """
    required = [
        "DATE_UTC", "WIND_KNOTS", "LATITUDE", "LONGITUDE_180"
    ]
    history = storm_df.dropna(subset=required).copy()
    history = history.sort_values("DATE_UTC")

    if len(history) < 2:
        raise ValueError(
            "At least two valid observations are required for this cyclone."
        )

    previous = history.iloc[-2]
    latest = history.iloc[-1]

    hours_since_prev = (
        latest["DATE_UTC"] - previous["DATE_UTC"]
    ).total_seconds() / 3600

    if not 1 <= hours_since_prev <= 12:
        raise ValueError(
            "The latest two valid observations are not 1–12 hours apart. "
            "Choose another cyclone with a suitable observation interval."
        )

    # Approximate the next observation time by repeating the latest interval.
    next_date = latest["DATE_UTC"] + pd.Timedelta(
        hours=hours_since_prev
    )

    month = next_date.month
    year = next_date.year

    row = {
        "PREV_WIND": float(latest["WIND_KNOTS"]),
        "PREV2_WIND": float(previous["WIND_KNOTS"]),
        "WIND_CHANGE": float(
            latest["WIND_KNOTS"] - previous["WIND_KNOTS"]
        ),
        "LATITUDE": float(latest["LATITUDE"]),
        "LONGITUDE_180": float(latest["LONGITUDE_180"]),
        # Future position is unavailable; use a stationary-position assumption.
        "LAT_CHANGE": 0.0,
        "LON_CHANGE": 0.0,
        "HOURS_SINCE_PREV": float(hours_since_prev),
        "MONTH_SIN": float(np.sin(2 * np.pi * month / 12)),
        "MONTH_COS": float(np.cos(2 * np.pi * month / 12)),
        "YEAR": int(year),
    }

    X = pd.DataFrame([row]).reindex(columns=features)

    if X.isna().any().any():
        raise ValueError(
            "The model feature list contains an unexpected or missing feature."
        )

    predicted_knots = float(model.predict(X)[0])
    predicted_knots = max(0.0, predicted_knots)

    return {
        "previous_wind_knots": float(latest["WIND_KNOTS"]),
        "previous_wind_kmh": float(latest["WIND_KNOTS"] * 1.852),
        "predicted_wind_knots": predicted_knots,
        "predicted_wind_kmh": predicted_knots * 1.852,
        "next_observation_time": next_date,
        "hours_since_prev": float(hours_since_prev),
        "latitude": float(latest["LATITUDE"]),
        "longitude": float(latest["LONGITUDE_180"]),
        "assumption": "Next location assumed unchanged from latest observation.",
    }
