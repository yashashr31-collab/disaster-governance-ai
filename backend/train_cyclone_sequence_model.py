
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "india_cyclones.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

print("Loading cyclone dataset...")

df = pd.read_csv(DATA, low_memory=False)

df["DATE_UTC"] = pd.to_datetime(
    df["DATE_UTC"],
    utc=True,
)

for col in ["WIND_KNOTS", "LATITUDE", "LONGITUDE_180"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "STORM_ID",
        "DATE_UTC",
        "WIND_KNOTS",
        "LATITUDE",
        "LONGITUDE_180",
    ]
).copy()

df = df[
    df["WIND_KNOTS"].between(0, 200)
    & df["LATITUDE"].between(-90, 90)
    & df["LONGITUDE_180"].between(-180, 180)
].copy()

df = df.sort_values(["STORM_ID", "DATE_UTC"])

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

g = df.groupby("STORM_ID", sort=False)

df["PREV_WIND"] = g["WIND_KNOTS"].shift(1)
df["PREV2_WIND"] = g["WIND_KNOTS"].shift(2)

df["PREV_LAT"] = g["LATITUDE"].shift(1)
df["PREV_LON"] = g["LONGITUDE_180"].shift(1)

df["PREV_DATE"] = g["DATE_UTC"].shift(1)

df["HOURS_SINCE_PREV"] = (
    df["DATE_UTC"] - df["PREV_DATE"]
).dt.total_seconds() / 3600

# Only use observations separated by 1 to 12 hours.
df = df[
    df["HOURS_SINCE_PREV"].between(1, 12)
].copy()

df["WIND_CHANGE"] = (
    df["PREV_WIND"] - df["PREV2_WIND"]
)

df["LAT_CHANGE"] = (
    df["LATITUDE"] - df["PREV_LAT"]
)

df["LON_CHANGE"] = (
    df["LONGITUDE_180"] - df["PREV_LON"]
)

df["MONTH"] = df["DATE_UTC"].dt.month
df["YEAR"] = df["DATE_UTC"].dt.year

df["MONTH_SIN"] = np.sin(
    2 * np.pi * df["MONTH"] / 12
)

df["MONTH_COS"] = np.cos(
    2 * np.pi * df["MONTH"] / 12
)

FEATURES = [
    "PREV_WIND",
    "PREV2_WIND",
    "WIND_CHANGE",
    "LATITUDE",
    "LONGITUDE_180",
    "LAT_CHANGE",
    "LON_CHANGE",
    "HOURS_SINCE_PREV",
    "MONTH_SIN",
    "MONTH_COS",
    "YEAR",
]

df = df.dropna(subset=FEATURES).copy()

# --------------------------------------------------
# GROUP-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

X = df[FEATURES]
y = df["WIND_KNOTS"]
groups = df["STORM_ID"]

print("\nDATASET INFORMATION")
print("-------------------")
print("Observations:", len(df))
print("Unique storms:", groups.nunique())

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_storms = set(groups.iloc[train_idx])
test_storms = set(groups.iloc[test_idx])

print("\nSPLIT INFORMATION")
print("-----------------")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training storms:", len(train_storms))
print("Testing storms:", len(test_storms))
print(
    "Storm overlap:",
    len(train_storms.intersection(test_storms)),
)

# --------------------------------------------------
# TRAIN RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# MODEL PREDICTIONS
# --------------------------------------------------

pred = model.predict(X_test)

# Persistence baseline:
# Predict the next wind speed will equal the previous wind speed.
baseline_pred = X_test["PREV_WIND"].to_numpy()

# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": float(
            r2_score(actual, predicted)
        ),
    }


model_metrics = calculate_metrics(y_test, pred)

baseline_metrics = calculate_metrics(
    y_test,
    baseline_pred,
)

print("\nRANDOM FOREST RESULTS")
print("---------------------")

for key, value in model_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nPERSISTENCE BASELINE RESULTS")
print("----------------------------")

for key, value in baseline_metrics.items():
    print(f"{key}: {value:.4f}")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
    },
    MODEL_DIR / "cyclone_sequence_model.pkl",
)

# --------------------------------------------------
# SAVE VALIDATION PREDICTIONS
# --------------------------------------------------

test_rows = df.iloc[test_idx].copy()

validation_df = pd.DataFrame({
    "STORM_ID": test_rows["STORM_ID"].values,
    "DATE_UTC": test_rows["DATE_UTC"].astype(str).values,
    "ACTUAL_WIND_KNOTS": y_test.to_numpy(),
    "PREDICTED_WIND_KNOTS": pred,
    "BASELINE_WIND_KNOTS": baseline_pred,
})

validation_df["MODEL_ERROR"] = (
    validation_df["ACTUAL_WIND_KNOTS"]
    - validation_df["PREDICTED_WIND_KNOTS"]
)

validation_df["ABS_MODEL_ERROR"] = (
    validation_df["MODEL_ERROR"].abs()
)

validation_df["BASELINE_ERROR"] = (
    validation_df["ACTUAL_WIND_KNOTS"]
    - validation_df["BASELINE_WIND_KNOTS"]
)

validation_df["ABS_BASELINE_ERROR"] = (
    validation_df["BASELINE_ERROR"].abs()
)

validation_df.to_csv(
    OUTPUT_DIR / "cyclone_validation_predictions.csv",
    index=False,
)

# --------------------------------------------------
# SAVE EVALUATION METRICS
# --------------------------------------------------

metrics = {
    "model": "Random Forest Regressor",
    "split_method": "GroupShuffleSplit",
    "test_size": 0.20,
    "random_state": 42,
    "observations": int(len(df)),
    "training_observations": int(len(X_train)),
    "testing_observations": int(len(X_test)),
    "total_storms": int(groups.nunique()),
    "training_storms": int(len(train_storms)),
    "testing_storms": int(len(test_storms)),
    "storm_overlap": int(
        len(train_storms.intersection(test_storms))
    ),
    "random_forest": model_metrics,
    "persistence_baseline": baseline_metrics,
}

with open(
    OUTPUT_DIR / "cyclone_sequence_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

importance.to_csv(
    OUTPUT_DIR / "cyclone_feature_importance.csv",
    index=False,
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

print("\nFILES SAVED")
print("-----------")
print("models/cyclone_sequence_model.pkl")
print("outputs/cyclone_sequence_metrics.json")
print("outputs/cyclone_validation_predictions.csv")
print("outputs/cyclone_feature_importance.csv")

print("\nTRAINING AND VALIDATION COMPLETED SUCCESSFULLY.")
