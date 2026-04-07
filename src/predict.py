"""Inference helpers for laptop price prediction."""

from __future__ import annotations

import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import FINAL_FEATURE_COLUMNS, LOG_TRANSFORM


def load_model(path: str) -> Any:
    """Load a pickled scikit-learn pipeline from disk.

    Parameters
    ----------
    path:
        File path to the serialized model.

    Returns
    -------
    Any
        Loaded model pipeline.
    """
    with open(path, "rb") as file_pointer:
        return pickle.load(file_pointer)


def predict_price(model: Any, input_dict: Dict[str, Any]) -> float:
    """Predict laptop price in INR from a single engineered input record.

    Parameters
    ----------
    model:
        Trained scikit-learn compatible model or pipeline.
    input_dict:
        Dictionary containing the engineered model features.

    Returns
    -------
    float
        Predicted laptop price in INR.

    Raises
    ------
    ValueError
        If required input fields are missing.
    """
    missing_columns = [column for column in FINAL_FEATURE_COLUMNS if column not in input_dict]
    if missing_columns:
        raise ValueError(
            "Missing required input fields: " + ", ".join(missing_columns)
        )

    inference_frame = pd.DataFrame([{column: input_dict[column] for column in FINAL_FEATURE_COLUMNS}])
    prediction = model.predict(inference_frame)[0]
    if LOG_TRANSFORM:
        prediction = np.exp(prediction)
    return float(prediction)
