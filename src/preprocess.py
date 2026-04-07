"""Data cleaning and feature engineering utilities for laptop price prediction."""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import FINAL_FEATURE_COLUMNS, TARGET_COLUMN


RAW_COLUMNS = [
    "Company",
    "TypeName",
    "Inches",
    "ScreenResolution",
    "Cpu",
    "Ram",
    "Memory",
    "Gpu",
    "OpSys",
    "Weight",
    TARGET_COLUMN,
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe by removing unnamed columns and resetting the index.

    Parameters
    ----------
    df:
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe restricted to expected raw columns when available.
    """
    cleaned = df.copy()
    cleaned = cleaned.loc[:, ~cleaned.columns.str.contains(r"^Unnamed", case=False)]
    existing_columns = [column for column in RAW_COLUMNS if column in cleaned.columns]
    if existing_columns:
        cleaned = cleaned[existing_columns]
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def clean_ram_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Convert RAM and weight columns from strings to numeric values.

    Parameters
    ----------
    df:
        Input dataframe containing `Ram` and `Weight` columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with numeric RAM and weight values.
    """
    cleaned = df.copy()
    cleaned["Ram"] = (
        cleaned["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
    )
    cleaned["Weight"] = pd.to_numeric(
        cleaned["Weight"].astype(str).str.replace("kg", "", regex=False),
        errors="coerce",
    )
    return cleaned


def _extract_resolution_values(screen_resolution: str) -> Tuple[int, int]:
    """Parse X and Y resolution values from a screen resolution string."""
    parts = str(screen_resolution).split("x", maxsplit=1)
    if len(parts) != 2:
        return 0, 0
    x_match = re.findall(r"(\d+\.?\d+)", parts[0].replace(",", ""))
    y_match = re.findall(r"(\d+\.?\d+)", parts[1].replace(",", ""))
    x_res = int(float(x_match[0])) if x_match else 0
    y_res = int(float(y_match[0])) if y_match else 0
    return x_res, y_res


def extract_screen_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create screen-related features including touch support, IPS, and PPI.

    Parameters
    ----------
    df:
        Input dataframe containing `ScreenResolution` and `Inches`.

    Returns
    -------
    pd.DataFrame
        Dataframe with screen features engineered and raw resolution columns removed.
    """
    engineered = df.copy()
    screen_text = engineered["ScreenResolution"].astype(str)
    engineered["TouchScreen"] = screen_text.str.contains(
        "touchscreen", case=False
    ).astype(int)
    engineered["IPS"] = screen_text.str.contains("IPS", case=False).astype(int)

    resolutions = screen_text.apply(_extract_resolution_values)
    engineered[["X_res", "Y_res"]] = pd.DataFrame(
        resolutions.tolist(), index=engineered.index
    )
    engineered["PPI"] = np.sqrt(
        engineered["X_res"] ** 2 + engineered["Y_res"] ** 2
    ) / engineered["Inches"].astype(float)
    engineered = engineered.drop(
        columns=["ScreenResolution", "Inches", "X_res", "Y_res"]
    )
    return engineered


def _map_cpu_category(cpu_name: str) -> str:
    """Map a CPU descriptor to the notebook-compatible CPU bucket."""
    leading_tokens = " ".join(str(cpu_name).split()[:3])
    if leading_tokens in {"Intel Core i7", "Intel Core i5", "Intel Core i3"}:
        return leading_tokens
    if str(leading_tokens).startswith("Intel"):
        return "Other Intel Processor"
    return "AMD Processor"


def extract_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create CPU category features and drop the raw CPU column.

    Parameters
    ----------
    df:
        Input dataframe containing `Cpu`.

    Returns
    -------
    pd.DataFrame
        Dataframe with a categorical `CPU_name` feature.
    """
    engineered = df.copy()
    engineered["CPU_name"] = engineered["Cpu"].apply(_map_cpu_category)
    engineered = engineered.drop(columns=["Cpu"])
    return engineered


def _parse_memory_part(part: str) -> Tuple[str, int]:
    """Parse a memory token into its storage type and capacity in gigabytes."""
    normalized = str(part).replace(".0", "")
    normalized = normalized.replace("Flash Storage", "FlashStorage")
    normalized = normalized.replace("GB", "")
    normalized = normalized.replace("TB", "000")
    normalized = normalized.strip()
    digits = re.sub(r"\D", "", normalized)
    capacity = int(digits) if digits else 0

    storage_mapping: Dict[str, str] = {
        "HDD": "HDD",
        "SSD": "SSD",
        "Hybrid": "Hybrid",
        "FlashStorage": "Flash_Storage",
    }
    for raw_label, output_label in storage_mapping.items():
        if raw_label in normalized:
            return output_label, capacity
    return "Unknown", capacity


def extract_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split the memory column into storage-type capacities.

    Parameters
    ----------
    df:
        Input dataframe containing the `Memory` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with HDD and SSD retained, while negligible storage features are dropped.
    """
    engineered = df.copy()
    engineered["HDD"] = 0
    engineered["SSD"] = 0
    engineered["Hybrid"] = 0
    engineered["Flash_Storage"] = 0

    for index, memory_value in engineered["Memory"].astype(str).items():
        for part in memory_value.split("+"):
            storage_type, capacity = _parse_memory_part(part)
            if storage_type in {"HDD", "SSD", "Hybrid", "Flash_Storage"}:
                engineered.at[index, storage_type] += capacity

    engineered = engineered.drop(columns=["Memory", "Hybrid", "Flash_Storage"])
    return engineered


def extract_gpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a GPU brand feature and remove unsupported ARM rows.

    Parameters
    ----------
    df:
        Input dataframe containing `Gpu`.

    Returns
    -------
    pd.DataFrame
        Dataframe with `Gpu brand` replacing the raw GPU text.
    """
    engineered = df.copy()
    engineered["Gpu brand"] = engineered["Gpu"].astype(str).str.split().str[0]
    engineered = engineered[engineered["Gpu brand"] != "ARM"].copy()
    engineered = engineered.drop(columns=["Gpu"]).reset_index(drop=True)
    return engineered


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing workflow and return model-ready data.

    Parameters
    ----------
    df:
        Raw laptop dataframe.

    Returns
    -------
    pd.DataFrame
        Final preprocessed dataframe ordered for training and inference.
    """
    processed = clean_dataframe(df)
    processed = clean_ram_weight(processed)
    processed = extract_screen_features(processed)
    processed = extract_cpu_features(processed)
    processed = extract_memory_features(processed)
    processed = extract_gpu_features(processed)

    ordered_columns = FINAL_FEATURE_COLUMNS.copy()
    if TARGET_COLUMN in processed.columns:
        ordered_columns.append(TARGET_COLUMN)
    processed = processed[[column for column in ordered_columns if column in processed.columns]]
    return processed.reset_index(drop=True)
