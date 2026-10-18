"""Exploratory visualizations and business-insight notes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COLUMN
from src.data_preprocessing import load_and_clean
from src.feature_engineering import engineer_features


GRAPH_INSIGHTS = {
    "fare_distribution": (
        "The fare histogram and KDE show how prices concentrate around typical "
        "city trips and whether a long right tail of expensive rides exists. "
        "A right-skewed distribution is expected in taxi data because most trips "
        "are short while a smaller share of airport or long-distance trips costs more."
    ),
    "fare_boxplot": (
        "The boxplot highlights the central fare range and statistical outliers. "
        "Outliers are not automatically errors; long trips to airports can be valid. "
        "They do, however, motivate robust metrics such as MAE in addition to RMSE."
    ),
    "passenger_count": (
        "Passenger counts show occupancy demand. Solo riders usually dominate NYC "
        "yellow-cab traffic. This feature is retained because shared rides can "
        "correlate weakly with fare through trip purpose, even if the meter is "
        "distance- and time-based."
    ),
    "fare_vs_distance": (
        "Fare versus trip distance is the core pricing relationship. A roughly "
        "linear trend with scatter is expected: distance drives the meter, while "
        "traffic, wait time, and surcharges add residual variation."
    ),
    "fare_by_hour": (
        "Average fare by pickup hour captures rush-hour and late-night effects. "
        "Higher late-night fares can reflect longer empty-return trips, airport "
        "runs, or surcharge patterns rather than a simple volume effect."
    ),
    "fare_by_weekday": (
        "Weekday patterns separate commuting from weekend leisure demand. "
        "Differences in average fare can inform time-based features such as "
        "day_of_week and is_weekend."
    ),
    "correlation_heatmap": (
        "The correlation heatmap shows linear associations among numeric features. "
        "Trip distance should be the strongest correlate of fare. Weak correlations "
        "do not mean a feature is useless for tree models, which can capture "
        "non-linear effects."
    ),
}


def _prepare_eda_frame(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(df, include_target=True)


def _save_fig(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def plot_fare_distribution(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(9, 5))
    sns.histplot(df[TARGET_COLUMN], bins=40, kde=True, color="#1f4e79")
    plt.title("Taxi Fare Amount Distribution")
    plt.xlabel("Fare amount (USD)")
    plt.ylabel("Trip count")
    return _save_fig("fare_distribution")


def plot_fare_boxplot(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[TARGET_COLUMN], color="#7aa2c5")
    plt.title("Fare Amount Boxplot")
    plt.xlabel("Fare amount (USD)")
    return _save_fig("fare_boxplot")


def plot_passenger_count(df: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["passenger_count"], color="#2e7d4f")
    plt.title("Passenger Count Distribution")
    plt.xlabel("Passenger count")
    plt.ylabel("Trip count")
    return _save_fig("passenger_count")


def plot_fare_vs_distance(df: pd.DataFrame) -> Path:
    sample = df.sample(n=min(len(df), 4000), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=sample,
        x="trip_distance_km",
        y=TARGET_COLUMN,
        alpha=0.35,
        s=18,
        color="#c46b1a",
    )
    plt.title("Fare Amount vs Trip Distance")
    plt.xlabel("Trip distance (km)")
    plt.ylabel("Fare amount (USD)")
    return _save_fig("fare_vs_distance")


def plot_fare_by_hour(df: pd.DataFrame) -> Path:
    hourly = df.groupby("pickup_hour")[TARGET_COLUMN].mean().reset_index()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=hourly, x="pickup_hour", y=TARGET_COLUMN, marker="o", color="#5b2c6f")
    plt.title("Average Fare by Pickup Hour")
    plt.xlabel("Pickup hour")
    plt.ylabel("Average fare (USD)")
    return _save_fig("fare_by_hour")


def plot_fare_by_weekday(df: pd.DataFrame) -> Path:
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df.groupby("day_of_week")[TARGET_COLUMN].mean().reset_index()
    weekly["weekday"] = weekly["day_of_week"].map(lambda i: labels[int(i)])
    plt.figure(figsize=(8, 5))
    sns.barplot(data=weekly, x="weekday", y=TARGET_COLUMN, color="#1f6f8b")
    plt.title("Average Fare by Day of Week")
    plt.xlabel("Day of week")
    plt.ylabel("Average fare (USD)")
    return _save_fig("fare_by_weekday")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    numeric_cols = [
        TARGET_COLUMN,
        "trip_distance_km",
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "day_of_week",
        "is_weekend",
        "passenger_count",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric Features")
    return _save_fig("correlation_heatmap")


def run_eda(df: pd.DataFrame | None = None) -> dict:
    """Generate all EDA figures and write insight notes beside them."""
    if df is None:
        df, _, _ = load_and_clean()
    eda_df = _prepare_eda_frame(df)
    sns.set_theme(style="whitegrid")

    outputs = {
        "fare_distribution": str(plot_fare_distribution(eda_df)),
        "fare_boxplot": str(plot_fare_boxplot(eda_df)),
        "passenger_count": str(plot_passenger_count(eda_df)),
        "fare_vs_distance": str(plot_fare_vs_distance(eda_df)),
        "fare_by_hour": str(plot_fare_by_hour(eda_df)),
        "fare_by_weekday": str(plot_fare_by_weekday(eda_df)),
        "correlation_heatmap": str(plot_correlation_heatmap(eda_df)),
        "insights": GRAPH_INSIGHTS,
    }
    notes_path = FIGURES_DIR / "eda_insights.json"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    results = run_eda()
    print("Saved EDA figures:")
    for key, value in results.items():
        if key != "insights":
            print(f"  {key}: {value}")
    print("\nBusiness insights:")
    for name, text in GRAPH_INSIGHTS.items():
        print(f"\n[{name}]\n{text}")
