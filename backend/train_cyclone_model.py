
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "india_cyclones.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading cyclone dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Keep only observations with a valid target and location
df["WIND_KNOTS"] = pd.to_numeric(df["WIND_KNOTS"], errors="coerce")
df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
df["LONGITUDE_180"] = pd.to_numeric(
    df["LONGITUDE_180"], errors="coerce"
)

df["DATE_UTC"] = pd.to_datetime(
    df["DATE_UTC"], errors="coerce", utc=True
)

df = df.dropna(
    subset=[
        "STORM_ID",
        "WIND_KNOTS",
        "LATITUDE",
        "LONGITUDE_180",
        "DATE_UTC"
    ]
).copy()

# Basic validity checks
df = df[
    df["LATITUDE"].between(-90, 90)
    & df["LONGITUDE_180"].between(-180, 180)
    & df["WIND_KNOTS"].between(0, 200)
].copy()

# Extract time features
df["YEAR"] = df["DATE_UTC"].dt.year
df["MONTH"] = df["DATE_UTC"].dt.month

# Encode month as a cyclical feature
df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

FEATURES = [
    "LATITUDE",
    "LONGITUDE_180",
    "YEAR",
    "MONTH_SIN",
    "MONTH_COS"
]

TARGET = "WIND_KNOTS"

X = df[FEATURES]
y = df[TARGET]
groups = df["STORM_ID"]

print(f"Valid observations: {len(df)}")
print(f"Unique storms: {groups.nunique()}")

# Split by storm, not by individual observation.
# This prevents observations from the same storm
# appearing in both training and testing sets.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print(f"MAE:  {mae:.2f} knots")
print(f"RMSE: {rmse:.2f} knots")
print(f"R2:   {r2:.4f}")

# Save trained model
joblib.dump(
    {"model": model, "features": FEATURES},
    MODEL_DIR / "cyclone_intensity_model.pkl"
)

# Save evaluation metrics
metrics = {
    "model": "Random Forest Regressor",
    "target": TARGET,
    "features": FEATURES,
    "valid_observations": int(len(df)),
    "unique_storms": int(groups.nunique()),
    "training_observations": int(len(X_train)),
    "testing_observations": int(len(X_test)),
    "MAE_knots": float(mae),
    "RMSE_knots": float(rmse),
    "R2": float(r2)
}

with open(OUTPUT_DIR / "cyclone_model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save sample predictions
results = pd.DataFrame({
    "ACTUAL_WIND_KNOTS": y_test,
    "PREDICTED_WIND_KNOTS": predictions
})

results.to_csv(
    OUTPUT_DIR / "cyclone_test_predictions.csv",
    index=False
)

# Feature importance
importance = pd.DataFrame({
    "FEATURE": FEATURES,
    "IMPORTANCE": model.feature_importances_
}).sort_values("IMPORTANCE", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

print("\nModel saved successfully!")
print("Location: models/cyclone_intensity_model.pkl")
print("Metrics: outputs/cyclone_model_metrics.json")
