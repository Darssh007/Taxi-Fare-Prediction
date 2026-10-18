"""Regression metrics, comparison plots, and overfitting checks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_PATH, MODELS_DIR


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and R² from actual predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison table from nested metric dictionaries."""
    rows = []
    for model_name, payload in results.items():
        row = {"model": model_name}
        row.update(payload)
        rows.append(row)
    frame = pd.DataFrame(rows)
    preferred = [
        "model",
        "cv_mae",
        "train_mae",
        "test_mae",
        "test_mse",
        "test_rmse",
        "test_r2",
        "overfit_gap_mae",
    ]
    ordered = [col for col in preferred if col in frame.columns]
    remaining = [col for col in frame.columns if col not in ordered]
    return frame[ordered + remaining]


def select_best_model(table: pd.DataFrame) -> str:
    """Select the model with the lowest cross-validated MAE on training data.

    Test metrics are reported later and are not used for model selection.
    """
    if "cv_mae" not in table.columns:
        raise ValueError("cv_mae is required for leakage-safe model selection.")
    ranked = table.sort_values(["cv_mae", "cv_mae_std"], ascending=True)
    return str(ranked.iloc[0]["model"])


def plot_model_comparison(table: pd.DataFrame) -> Path:
    """Bar chart of test MAE and RMSE for every trained model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = table.melt(
        id_vars="model",
        value_vars=[col for col in ("test_mae", "test_rmse") if col in table.columns],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    models = plot_df["model"].unique()
    metrics = plot_df["metric"].unique()
    x = np.arange(len(models))
    width = 0.35
    for i, metric in enumerate(metrics):
        values = [
            plot_df[(plot_df["model"] == model) & (plot_df["metric"] == metric)]["value"].iloc[0]
            for model in models
        ]
        plt.bar(x + i * width, values, width, label=metric.replace("_", " ").upper())
    plt.xticks(x + width / 2, models, rotation=20, ha="right")
    plt.ylabel("Error (USD)")
    plt.title("Model Comparison on Held-out Test Set")
    plt.legend()
    plt.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, s=16, color="#1f4e79")
    limits = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(limits, limits, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Actual fare (USD)")
    plt.ylabel("Predicted fare (USD)")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.tight_layout()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def save_evaluation_results(payload: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_evaluation_results(path: Path = METRICS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No evaluation file found at {path}. Train a model first with "
            "`python -m src.train_model`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def print_evaluation_summary(payload: dict | None = None) -> None:
    payload = payload or load_evaluation_results()
    table = pd.DataFrame(payload["comparison_table"])
    print("Validation strategy:", payload.get("validation_strategy"))
    print("\nModel comparison:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSelected model:", payload.get("best_model"))
    print("Selection rule:", payload.get("selection_rule"))
    if payload.get("synthetic"):
        print(
            "\nNOTE: Metrics were computed on labeled SYNTHETIC data. "
            "They are workflow demonstrations, not real-world NYC taxi performance."
        )
    print("\nFinal test metrics:")
    for key, value in payload.get("final_test_metrics", {}).items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    if not METRICS_PATH.exists():
        print(f"No metrics found at {METRICS_PATH}. Run `python -m src.train_model` first.")
    else:
        print_evaluation_summary()
        print(f"\nArtifacts directory: {MODELS_DIR}")
