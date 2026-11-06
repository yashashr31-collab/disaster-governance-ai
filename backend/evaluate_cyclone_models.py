
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv(
    "data/processed/india_cyclones.csv",
    low_memory=False
)

df["DATE_UTC"] = pd.to_datetime(df["DATE_UTC"])

df = df.dropna(subset=["WIND_KNOTS"])
df = df.sort_values(["STORM_ID", "DATE_UTC"])

# Create sequence features
df["PREV_WIND"] = df.groupby("STORM_ID")["WIND_KNOTS"].shift(1)
df["PREV2_WIND"] = df.groupby("STORM_ID")["WIND_KNOTS"].shift(2)

df["PREV_DATE"] = df.groupby("STORM_ID")["DATE_UTC"].shift(1)

df["HOURS_SINCE_PREV"] = (
    df["DATE_UTC"] - df["PREV_DATE"]
).dt.total_seconds() / 3600

df = df[df["HOURS_SINCE_PREV"].between(1, 12)]
df = df.dropna(subset=["PREV_WIND", "PREV2_WIND"])

# Recreate model features
df["WIND_CHANGE"] = df["PREV_WIND"] - df["PREV2_WIND"]

df["LAT_CHANGE"] = (
    df.groupby("STORM_ID")["LATITUDE"].diff()
)

df["LON_CHANGE"] = (
    df.groupby("STORM_ID")["LONGITUDE_180"].diff()
)

df["MONTH"] = df["DATE_UTC"].dt.month
df["YEAR"] = df["DATE_UTC"].dt.year

df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

features = [
    "LATITUDE",
    "LONGITUDE_180",
    "PREV_WIND",
    "PREV2_WIND",
    "WIND_CHANGE",
    "LAT_CHANGE",
    "LON_CHANGE",
    "HOURS_SINCE_PREV",
    "MONTH_SIN",
    "MONTH_COS",
    "YEAR"
]

df = df.dropna(subset=features)

X = df[features]
y = df["WIND_KNOTS"]
groups = df["STORM_ID"]

# Same storm-level split for both approaches
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Load trained sequence model
saved_data = joblib.load("models/cyclone_sequence_model.pkl")

model = saved_data["model"]
trained_features = saved_data["features"]

# Use exactly the features used during training
X_test = X_test[trained_features]

predictions = model.predict(X_test)

# Persistence baseline
persistence_predictions = X_test["PREV_WIND"].values

# Calculate metrics
def evaluate(name, actual, predicted):
    print(f"\n{name}")
    print("MAE:", round(mean_absolute_error(actual, predicted), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 3))
    print("R2:", round(r2_score(actual, predicted), 3))

evaluate("Sequence Model", y_test, predictions)
evaluate("Persistence Baseline", y_test, persistence_predictions)

print("\nTest observations:", len(test_idx))
print("Test storms:", groups.iloc[test_idx].nunique())
