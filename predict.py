"""Load the saved pipeline and score new taxi trips."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    MAX_FARE,
    MAX_PASSENGERS,
    MAX_TRIP_DISTANCE_KM,
    MIN_FARE,
    MIN_PASSENGERS,
    MIN_TRIP_DISTANCE_KM,
    MODEL_FEATURE_COLUMNS,
    MODEL_PATH,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
)
from src.feature_engineering import engineer_features, haversine_distance


class PredictionError(ValueError):
    """Raised when inputs are invalid or the trained model is unavailable."""


def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise PredictionError(
            f"Trained model not found at {path}. Run `python -m src.train_model` first."
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        raise PredictionError(f"Could not load model from {path}: {exc}") from exc


def validate_trip_input(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    passenger_count: int,
    pickup_datetime: str,
) -> None:
    """Validate raw user inputs before feature engineering."""
    try:
        pickup_dt = pd.to_datetime(pickup_datetime, utc=True)
        if pd.isna(pickup_dt):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise PredictionError("Pickup date and time could not be parsed.") from exc

    coords = {
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
    }
    for name, value in coords.items():
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"{name} must be a number.") from exc

    if not (NYC_LAT_MIN <= float(pickup_latitude) <= NYC_LAT_MAX):
        raise PredictionError(
            f"Pickup latitude must be between {NYC_LAT_MIN} and {NYC_LAT_MAX} (NYC area)."
        )
    if not (NYC_LON_MIN <= float(pickup_longitude) <= NYC_LON_MAX):
        raise PredictionError(
            f"Pickup longitude must be between {NYC_LON_MIN} and {NYC_LON_MAX} (NYC area)."
        )
    if not (NYC_LAT_MIN <= float(dropoff_latitude) <= NYC_LAT_MAX):
        raise PredictionError(
            f"Drop-off latitude must be between {NYC_LAT_MIN} and {NYC_LAT_MAX} (NYC area)."
        )
    if not (NYC_LON_MIN <= float(dropoff_longitude) <= NYC_LON_MAX):
        raise PredictionError(
            f"Drop-off longitude must be between {NYC_LON_MIN} and {NYC_LON_MAX} (NYC area)."
        )

    try:
        passengers = int(passenger_count)
    except (TypeError, ValueError) as exc:
        raise PredictionError("Passenger count must be an integer.") from exc
    if passengers < MIN_PASSENGERS or passengers > MAX_PASSENGERS:
        raise PredictionError(
            f"Passenger count must be between {MIN_PASSENGERS} and {MAX_PASSENGERS}."
        )

    distance = float(
        haversine_distance(
            float(pickup_latitude),
            float(pickup_longitude),
            float(dropoff_latitude),
            float(dropoff_longitude),
        )
    )
    if distance < MIN_TRIP_DISTANCE_KM:
        raise PredictionError(
            "Pickup and drop-off points are too close to form a valid taxi trip."
        )
    if distance > MAX_TRIP_DISTANCE_KM:
        raise PredictionError(
            "Trip distance is outside the range this model was trained to estimate."
        )


def build_feature_frame(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    passenger_count: int,
    pickup_datetime: str,
) -> pd.DataFrame:
    """Create the same columns used during training."""
    raw = pd.DataFrame(
        [
            {
                "pickup_datetime": pickup_datetime,
                "pickup_longitude": float(pickup_longitude),
                "pickup_latitude": float(pickup_latitude),
                "dropoff_longitude": float(dropoff_longitude),
                "dropoff_latitude": float(dropoff_latitude),
                "passenger_count": int(passenger_count),
            }
        ]
    )
    features = engineer_features(raw, include_target=False)
    missing = set(MODEL_FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise PredictionError(f"Engineered features are incomplete: {sorted(missing)}")
    return features[list(MODEL_FEATURE_COLUMNS)].astype(float)


def predict_fare(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    passenger_count: int,
    pickup_datetime: str,
    model=None,
    model_path: str | Path = MODEL_PATH,
) -> dict:
    """Validate inputs, engineer features, and return a fare estimate."""
    validate_trip_input(
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,
        passenger_count,
        pickup_datetime,
    )
    features = build_feature_frame(
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,
        passenger_count,
        pickup_datetime,
    )
    pipeline = model if model is not None else load_model(model_path)
    estimate = float(pipeline.predict(features)[0])
    estimate = float(max(MIN_FARE, min(MAX_FARE, estimate)))
    distance = float(features["trip_distance_km"].iloc[0])
    return {
        "predicted_fare_usd": round(estimate, 2),
        "trip_distance_km": round(distance, 3),
        "disclaimer": (
            "This is a model estimate, not an official taxi quote. It does not "
            "include live traffic, tolls, wait time, or current NYC tariff rules."
        ),
    }


if __name__ == "__main__":
    demo = predict_fare(
        pickup_latitude=40.7580,
        pickup_longitude=-73.9855,
        dropoff_latitude=40.6413,
        dropoff_longitude=-73.7781,
        passenger_count=1,
        pickup_datetime="2015-06-15 08:30:00",
    )
    print(demo)
