"""Feature engineering shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURE_COLUMNS, TARGET_COLUMN


EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: np.ndarray | pd.Series | float,
    lon1: np.ndarray | pd.Series | float,
    lat2: np.ndarray | pd.Series | float,
    lon2: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Return great-circle distance in kilometers using the Haversine formula."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def parse_pickup_datetime(series: pd.Series) -> pd.Series:
    """Parse pickup timestamps to timezone-naive pandas datetime."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def add_trip_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add Haversine trip distance in kilometers."""
    out = df.copy()
    out["trip_distance_km"] = haversine_distance(
        out["pickup_latitude"].to_numpy(),
        out["pickup_longitude"].to_numpy(),
        out["dropoff_latitude"].to_numpy(),
        out["dropoff_longitude"].to_numpy(),
    )
    return out


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, weekday, and weekend flags from pickup time."""
    out = df.copy()
    pickup_dt = parse_pickup_datetime(out["pickup_datetime"])
    out["pickup_datetime"] = pickup_dt
    out["pickup_hour"] = pickup_dt.dt.hour.astype("Int64")
    out["pickup_day"] = pickup_dt.dt.day.astype("Int64")
    out["pickup_month"] = pickup_dt.dt.month.astype("Int64")
    out["day_of_week"] = pickup_dt.dt.dayofweek.astype("Int64")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("Int64")
    return out


def engineer_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Create the exact feature set used by the trained pipeline."""
    required = {
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

    out = add_datetime_features(df)
    out = add_trip_distance(out)

    columns = list(MODEL_FEATURE_COLUMNS)
    if include_target and TARGET_COLUMN in out.columns:
        columns = columns + [TARGET_COLUMN]
    return out[columns]


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the fare target without leaking identifiers."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for training.")
    engineered = engineer_features(df, include_target=True)
    x = engineered[list(MODEL_FEATURE_COLUMNS)].astype(float)
    y = engineered[TARGET_COLUMN].astype(float)
    return x, y
