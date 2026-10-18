"""Project paths, column names, and domain constraints."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"

# Preferred real Kaggle filename, then common aliases, then synthetic demo data.
CANDIDATE_DATASETS = (
    DATA_DIR / "train.csv",
    DATA_DIR / "nyc_taxi_fare.csv",
    DATA_DIR / "new-york-city-taxi-fare-prediction.csv",
    DATA_DIR / "synthetic_taxi_fares.csv",
)

SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_taxi_fares.csv"
MODEL_PATH = MODELS_DIR / "taxi_fare_model.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_results.json"
CLEANING_REPORT_PATH = MODELS_DIR / "cleaning_report.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

TARGET_COLUMN = "fare_amount"
ID_COLUMNS = ("key",)

RAW_FEATURE_COLUMNS = (
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
)

MODEL_FEATURE_COLUMNS = (
    "trip_distance_km",
    "pickup_hour",
    "pickup_day",
    "pickup_month",
    "day_of_week",
    "is_weekend",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
)

# Approximate NYC operating box used for coordinate validation.
NYC_LAT_MIN = 40.50
NYC_LAT_MAX = 41.00
NYC_LON_MIN = -74.30
NYC_LON_MAX = -73.70

MIN_FARE = 2.50
MAX_FARE = 200.0
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6
MIN_TRIP_DISTANCE_KM = 0.20
MAX_TRIP_DISTANCE_KM = 60.0

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
MAX_KAGGLE_SAMPLE_ROWS = 80_000
SYNTHETIC_N_ROWS = 12_000
