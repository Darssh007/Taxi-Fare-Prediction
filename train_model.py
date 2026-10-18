"""Train, compare, tune, and persist the taxi fare regression pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CV_FOLDS,
    FEATURE_NAMES_PATH,
    MODEL_FEATURE_COLUMNS,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_preprocessing import is_synthetic_dataset, load_and_clean
from src.evaluate_model import (
    compute_metrics,
    metrics_table,
    plot_actual_vs_predicted,
    plot_model_comparison,
    save_evaluation_results,
    select_best_model,
)
from src.exploratory_data_analysis import run_eda
from src.feature_engineering import split_features_and_target


def _optional_xgboost():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
        )
    except Exception:
        return None


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features. Fit this transformer only on training data."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(MODEL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def candidate_models() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            max_iter=120,
            learning_rate=0.08,
            max_depth=8,
        ),
    }
    xgb_model = _optional_xgboost()
    if xgb_model is not None:
        models["XGBoost"] = xgb_model
    return models


def param_distributions(model_name: str) -> dict:
    if model_name == "Random Forest":
        return {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 4],
        }
    if model_name == "Gradient Boosting":
        return {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.05, 0.08, 0.12],
        }
    if model_name == "HistGradientBoosting":
        return {
            "model__max_iter": [100, 150, 200],
            "model__learning_rate": [0.05, 0.08, 0.12],
            "model__max_depth": [6, 8, 12],
        }
    if model_name == "XGBoost":
        return {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.08, 0.12],
        }
    return {
        "model__max_depth": [6, 10, 14, None],
        "model__min_samples_leaf": [1, 2, 5],
    }


def evaluate_candidates(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Fit each model, score with CV on train, and report untouched test metrics."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = {}
    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator)
        cv_scores = -cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        pipeline.fit(x_train, y_train)
        train_pred = pipeline.predict(x_train)
        test_pred = pipeline.predict(x_test)
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)
        records[name] = {
            "cv_mae": float(np.mean(cv_scores)),
            "cv_mae_std": float(np.std(cv_scores)),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "overfit_gap_mae": train_metrics["mae"] - test_metrics["mae"],
        }
        print(f"{name}: CV MAE={records[name]['cv_mae']:.4f} | test MAE={records[name]['test_mae']:.4f}")
    return metrics_table(records)


def tune_model(model_name: str, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict]:
    """Randomized search with cross-validation on training data only."""
    base = candidate_models()[model_name]
    pipeline = build_pipeline(base)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions(model_name),
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    best_cv_mae = float(-search.best_score_)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "tuned_cv_mae": best_cv_mae,
    }


def save_pipeline(pipeline: Pipeline, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    FEATURE_NAMES_PATH.write_text(
        json.dumps(list(MODEL_FEATURE_COLUMNS), indent=2),
        encoding="utf-8",
    )
    return path


def train(run_eda_plots: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned, cleaning_report, source_path = load_and_clean()
    synthetic = is_synthetic_dataset(source_path)

    if run_eda_plots:
        run_eda(cleaned)

    x, y = split_features_and_target(cleaned)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    comparison = evaluate_candidates(x_train, y_train, x_test, y_test)
    selected_name = select_best_model(comparison)
    tuned_pipeline, tuning_details = tune_model(selected_name, x_train, y_train)

    final_train_pred = tuned_pipeline.predict(x_train)
    final_test_pred = tuned_pipeline.predict(x_test)
    final_train = compute_metrics(y_train, final_train_pred)
    final_test = compute_metrics(y_test, final_test_pred)

    comparison_plot = plot_model_comparison(comparison)
    residual_plot = plot_actual_vs_predicted(y_test, final_test_pred, selected_name)
    model_path = save_pipeline(tuned_pipeline)

    payload = {
        "synthetic": synthetic,
        "source_path": str(source_path),
        "n_rows_after_cleaning": int(len(cleaned)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "validation_strategy": (
            "80/20 hold-out split with random_state=42. Model selection uses "
            f"{CV_FOLDS}-fold CV MAE on the training set only. RandomizedSearchCV "
            "tunes the selected model on training folds. The test set is scored once "
            "after tuning."
        ),
        "selection_rule": "Lowest cross-validated MAE on training data; CV MAE standard deviation is the tie-breaker. The test set is not used for selection.",
        "comparison_table": comparison.to_dict(orient="records"),
        "best_model": selected_name,
        "tuning": tuning_details,
        "final_train_metrics": final_train,
        "final_test_metrics": final_test,
        "overfitting_check": {
            "train_mae": final_train["mae"],
            "test_mae": final_test["mae"],
            "mae_gap_test_minus_train": final_test["mae"] - final_train["mae"],
            "comment": (
                "A much lower train MAE than test MAE suggests overfitting. "
                "A small gap supports generalization on this dataset."
            ),
        },
        "model_path": str(model_path),
        "comparison_plot": str(comparison_plot),
        "actual_vs_predicted_plot": str(residual_plot),
        "cleaning_report": cleaning_report,
        "disclaimer": (
            "Predictions are statistical estimates from historical trip attributes. "
            "They are not official taxi quotes and do not include live traffic, tolls, "
            "or current tariff rules unless those signals are added as features."
        ),
    }
    if synthetic:
        payload["synthetic_warning"] = (
            "This run used labeled synthetic data. Metrics demonstrate the workflow "
            "only and must not be reported as real-world NYC taxi model performance."
        )

    save_evaluation_results(payload)
    print("\nSelected model:", selected_name)
    print("Saved pipeline:", model_path)
    print("Final test MAE: {:.4f} | RMSE: {:.4f} | R2: {:.4f}".format(
        final_test["mae"], final_test["rmse"], final_test["r2"]
    ))
    if synthetic:
        print("SYNTHETIC DATA: do not treat these scores as real-world performance.")
    return payload


if __name__ == "__main__":
    train()
