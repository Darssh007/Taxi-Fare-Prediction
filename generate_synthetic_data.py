"""Create a clearly labeled synthetic NYC-style taxi fare dataset.

This generator exists because the Kaggle competition files were not present in
the workspace. Synthetic fares follow a simplified meter-like formula plus noise.
Do not present models trained on this file as real-world NYC performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RANDOM_STATE, SYNTHETIC_DATA_PATH, SYNTHETIC_N_ROWS


def generate_synthetic_taxi_fares(n_rows: int = SYNTHETIC_N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pickup_lat = rng.uniform(40.64, 40.86, n_rows)
    pickup_lon = rng.uniform(-74.05, -73.78, n_rows)
    lat_shift = rng.normal(0, 0.04, n_rows)
    lon_shift = rng.normal(0, 0.05, n_rows)
    dropoff_lat = np.clip(pickup_lat + lat_shift, 40.60, 40.90)
    dropoff_lon = np.clip(pickup_lon + lon_shift, -74.10, -73.75)

    start = pd.Timestamp("2013-01-01")
    pickup_datetime = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n_rows), unit="m")
    passenger_count = rng.choice([1, 2, 3, 4, 5, 6], size=n_rows, p=[0.62, 0.20, 0.08, 0.06, 0.03, 0.01])

    from src.feature_engineering import haversine_distance

    distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    hour = pickup_datetime.hour.to_numpy()
    weekday = pickup_datetime.dayofweek.to_numpy()
    night = ((hour >= 20) | (hour <= 5)).astype(float)
    rush = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(float)
    weekend = (weekday >= 5).astype(float)

    fare = (
        2.50
        + 1.85 * distance_km
        + 0.90 * night
        + 1.10 * rush
        + 0.40 * weekend
        + 0.15 * (passenger_count - 1)
        + rng.normal(0, 1.25, n_rows)
    )
    fare = np.clip(fare, 2.5, 180.0)

    frame = pd.DataFrame(
        {
            "key": [f"synthetic_{i}" for i in range(n_rows)],
            "fare_amount": np.round(fare, 2),
            "pickup_datetime": pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
            "passenger_count": passenger_count,
            "data_source": "synthetic_demo_not_real_nyc_trips",
        }
    )

    # Inject a small number of invalid rows so cleaning can be demonstrated.
    n_bad = max(40, n_rows // 80)
    bad = frame.sample(n=n_bad, random_state=seed).copy()
    bad.iloc[: n_bad // 5, bad.columns.get_loc("fare_amount")] = -3.0
    bad.iloc[n_bad // 5 : 2 * n_bad // 5, bad.columns.get_loc("passenger_count")] = 0
    bad.iloc[2 * n_bad // 5 : 3 * n_bad // 5, bad.columns.get_loc("pickup_latitude")] = 0.0
    bad.iloc[3 * n_bad // 5 : 4 * n_bad // 5, bad.columns.get_loc("pickup_longitude")] = 0.0
    frame = pd.concat([frame, bad, frame.head(15)], ignore_index=True)
    return frame


def save_synthetic_dataset(path=SYNTHETIC_DATA_PATH, n_rows: int = SYNTHETIC_N_ROWS) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_taxi_fares(n_rows=n_rows)
    frame.to_csv(path, index=False)
    readme = DATA_DIR / "README.md"
    readme.write_text(
        "# Data folder\n\n"
        "Preferred real dataset: Kaggle *New York City Taxi Fare Prediction* `train.csv`.\n\n"
        "Source: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data\n\n"
        "`synthetic_taxi_fares.csv` is a **labeled synthetic demo file**. "
        "It is not NYC TLC data. Metrics from this file must not be reported as "
        "real-world model performance.\n\n"
        "To use the Kaggle data, download `train.csv` into this folder and rerun training.\n",
        encoding="utf-8",
    )
    return str(path)


if __name__ == "__main__":
    output = save_synthetic_dataset()
    print(f"Wrote synthetic demo dataset to {output}")
