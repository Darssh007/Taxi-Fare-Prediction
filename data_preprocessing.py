"""Load, inspect, and clean taxi trip records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CANDIDATE_DATASETS,
    CLEANING_REPORT_PATH,
    DATA_DIR,
    ID_COLUMNS,
    MAX_FARE,
    MAX_KAGGLE_SAMPLE_ROWS,
    MAX_PASSENGERS,
    MAX_TRIP_DISTANCE_KM,
    MIN_FARE,
    MIN_PASSENGERS,
    MIN_TRIP_DISTANCE_KM,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
    RANDOM_STATE,
    SYNTHETIC_DATA_PATH,
    TARGET_COLUMN,
)
from src.feature_engineering import add_trip_distance, parse_pickup_datetime


def find_dataset_path(explicit_path: str | Path | None = None) -> Path:
    """Locate a CSV dataset. Prefer a real Kaggle file if present."""
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    for candidate in CANDIDATE_DATASETS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No dataset found. Place the Kaggle train.csv in the data/ folder "
        f"or generate synthetic data at {SYNTHETIC_DATA_PATH}."
    )


def load_dataset(explicit_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the trip CSV and downsample very large Kaggle extracts."""
    path = find_dataset_path(explicit_path)
    df = pd.read_csv(path)
    if len(df) > MAX_KAGGLE_SAMPLE_ROWS and path.name != SYNTHETIC_DATA_PATH.name:
        df = df.sample(n=MAX_KAGGLE_SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    return df, path


def is_synthetic_dataset(path: Path) -> bool:
    """Return True when the loaded file is the labeled synthetic demo set."""
    return path.resolve() == SYNTHETIC_DATA_PATH.resolve() or "synthetic" in path.name.lower()


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Collect the standard first-pass dataset diagnostics."""
    numeric_describe = df.describe(include=[np.number]).to_dict()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(val) for col, val in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target": TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
        "head": df.head().to_dict(orient="records"),
        "numeric_describe": numeric_describe,
    }


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print head, shape, dtypes, info, describe, missing values, and duplicates."""
    print("First five rows:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\ninfo():")
    df.info()
    print("\ndescribe():")
    print(df.describe(include="all"))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    input_features = [col for col in df.columns if col not in (TARGET_COLUMN, *ID_COLUMNS)]
    print("\nTarget variable:", TARGET_COLUMN)
    print("Input features:", input_features)


def _in_nyc_box(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (
        lat.between(NYC_LAT_MIN, NYC_LAT_MAX)
        & lon.between(NYC_LON_MIN, NYC_LON_MAX)
    )


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply domain-aware cleaning and return a structured decision log.

    Cleaning decisions:
    - Drop rows missing coordinates, datetime, passenger count, or fare.
    - Drop exact duplicate rows (repeated uploads / logging errors).
    - Remove fares below the NYC-style minimum flag drop and above a domain cap
      used to exclude extreme data errors, not typical long trips.
    - Keep high fares that remain inside the fare cap when distance is plausible.
    - Restrict coordinates to a generous NYC bounding box.
    - Restrict passenger_count to 1-6 (standard NYC taxi occupancy).
    - Remove near-zero and implausibly long Haversine distances.
    """
    working = df.copy()
    report: dict = {"initial_rows": int(len(working)), "steps": []}

    def _log(step: str, before: int) -> None:
        after = int(len(working))
        report["steps"].append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": before - after,
            }
        )

    before = len(working)
    required_cols = [
        TARGET_COLUMN,
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    missing_required = [col for col in required_cols if col not in working.columns]
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")

    working = working.dropna(subset=required_cols)
    _log("drop_missing_required_fields", before)

    before = len(working)
    working = working.drop_duplicates()
    _log("drop_duplicate_rows", before)

    working["pickup_datetime"] = parse_pickup_datetime(working["pickup_datetime"])
    before = len(working)
    working = working.dropna(subset=["pickup_datetime"])
    _log("drop_unparseable_pickup_datetime", before)

    before = len(working)
    working = working[(working[TARGET_COLUMN] >= MIN_FARE) & (working[TARGET_COLUMN] <= MAX_FARE)]
    _log(
        f"keep_fares_between_{MIN_FARE}_and_{MAX_FARE}",
        before,
    )

    before = len(working)
    working = working[
        _in_nyc_box(working["pickup_latitude"], working["pickup_longitude"])
        & _in_nyc_box(working["dropoff_latitude"], working["dropoff_longitude"])
    ]
    _log("restrict_coordinates_to_nyc_bounding_box", before)

    before = len(working)
    working = working[
        working["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
    ]
    _log("keep_passenger_count_1_to_6", before)

    working = add_trip_distance(working)
    before = len(working)
    working = working[
        working["trip_distance_km"].between(MIN_TRIP_DISTANCE_KM, MAX_TRIP_DISTANCE_KM)
    ]
    _log("keep_plausible_haversine_distance", before)

    # Statistical outlier screen: very high fares with tiny distances are errors.
    before = len(working)
    fare_per_km = working[TARGET_COLUMN] / working["trip_distance_km"]
    q1, q3 = fare_per_km.quantile(0.25), fare_per_km.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    likely_error = (fare_per_km > upper) & (working["trip_distance_km"] < 1.0)
    working = working.loc[~likely_error]
    _log("drop_short_trips_with_extreme_fare_per_km", before)

    working = working.reset_index(drop=True)
    report["final_rows"] = int(len(working))
    report["rows_removed_total"] = report["initial_rows"] - report["final_rows"]
    return working, report


def save_cleaning_report(report: dict, path: Path = CLEANING_REPORT_PATH) -> None:
    """Persist the cleaning decision log next to trained artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_and_clean(explicit_path: str | Path | None = None) -> tuple[pd.DataFrame, dict, Path]:
    """Convenience wrapper used by training and EDA scripts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df, path = load_dataset(explicit_path)
    cleaned, report = clean_dataset(df)
    report["source_path"] = str(path)
    report["synthetic"] = is_synthetic_dataset(path)
    save_cleaning_report(report)
    return cleaned, report, path


if __name__ == "__main__":
    frame, cleaning_report, source = load_and_clean()
    print(f"Loaded data from: {source}")
    print_dataset_overview(frame)
    print("\nCleaning report:")
    print(json.dumps(cleaning_report, indent=2))
