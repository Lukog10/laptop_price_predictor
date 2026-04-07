"""Model training utilities for the laptop price predictor."""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    LOG_TRANSFORM,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_MAX_FEATURES,
    RF_MAX_SAMPLES,
    SPLIT_RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.preprocess import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def build_pipeline(model: Any) -> Pipeline:
    """Build a preprocessing and regression pipeline for a supplied model.

    Parameters
    ----------
    model:
        A scikit-learn compatible regression model.

    Returns
    -------
    Pipeline
        Complete preprocessing and model pipeline.
    """
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS)
        ],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessor", transformer),
        ("model", model),
    ])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a trained model using R² on log scale and INR errors.

    Parameters
    ----------
    model:
        Trained pipeline.
    X_test:
        Test features.
    y_test:
        Test target values, log-transformed when configured.

    Returns
    -------
    dict[str, float]
        Dictionary containing R², MAE, and RMSE.
    """
    y_pred = model.predict(X_test)
    r2_value = r2_score(y_test, y_pred)

    if LOG_TRANSFORM:
        y_true_original = np.exp(y_test)
        y_pred_original = np.exp(y_pred)
    else:
        y_true_original = y_test
        y_pred_original = y_pred

    mae_value = mean_absolute_error(y_true_original, y_pred_original)
    rmse_value = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    return {
        "R2": float(r2_value),
        "MAE": float(mae_value),
        "RMSE": float(rmse_value),
    }


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Train the notebook's core models and compare their performance.

    Parameters
    ----------
    X_train, X_test, y_train, y_test:
        Train-test split produced from the preprocessed dataset.

    Returns
    -------
    pd.DataFrame
        Comparison table of model performance.
    """
    model_registry = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.001),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_samples=RF_MAX_SAMPLES,
            max_features=RF_MAX_FEATURES,
            max_depth=RF_MAX_DEPTH,
        ),
    }

    rows = []
    for model_name, estimator in model_registry.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        rows.append({"Model": model_name, **metrics})

    comparison = pd.DataFrame(rows).sort_values(by="R2", ascending=False).reset_index(drop=True)
    return comparison


def save_best_model(pipeline: Pipeline, path: str) -> None:
    """Persist the best-performing pipeline to disk using pickle.

    Parameters
    ----------
    pipeline:
        Trained pipeline to serialize.
    path:
        Destination file path.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file_pointer:
        pickle.dump(pipeline, file_pointer)


def main() -> None:
    """Execute the full training workflow and save the random forest model."""
    LOGGER.info("Loading dataset from %s", DATA_PATH)
    raw_df = pd.read_csv(DATA_PATH)
    processed_df = run_pipeline(raw_df)

    X = processed_df.drop(columns=[TARGET_COLUMN])
    y = np.log(processed_df[TARGET_COLUMN]) if LOG_TRANSFORM else processed_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )

    comparison = train_all_models(X_train, X_test, y_train, y_test)
    LOGGER.info("\n%s", comparison.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    best_pipeline = build_pipeline(
        RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_samples=RF_MAX_SAMPLES,
            max_features=RF_MAX_FEATURES,
            max_depth=RF_MAX_DEPTH,
        )
    )
    best_pipeline.fit(X_train, y_train)
    save_best_model(best_pipeline, str(MODEL_PATH))
    LOGGER.info("Saved best model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
