"""Streamlit application for interactive laptop price prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import DATA_PATH, MODEL_PATH
from src.predict import load_model, predict_price


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Load the raw laptop dataset for UI option generation."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_prediction_model():
    """Load the trained model for repeated Streamlit inference."""
    return load_model(str(MODEL_PATH))


def calculate_ppi(width: int, height: int, inches: float) -> float:
    """Calculate pixels per inch for a chosen screen setup."""
    return float(np.sqrt(width ** 2 + height ** 2) / inches)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")
    st.title("Laptop Price Predictor")
    st.caption("Predict laptop price in INR using the refactored Random Forest model.")

    raw_df = load_reference_data()

    resolution_presets = {
        "1366 x 768": (1366, 768),
        "1600 x 900": (1600, 900),
        "1920 x 1080": (1920, 1080),
        "2160 x 1440": (2160, 1440),
        "2560 x 1600": (2560, 1600),
        "2880 x 1800": (2880, 1800),
        "3200 x 1800": (3200, 1800),
        "3840 x 2160": (3840, 2160),
    }

    st.sidebar.header("Laptop Specifications")
    company = st.sidebar.selectbox("Company", sorted(raw_df["Company"].dropna().unique()))
    type_name = st.sidebar.selectbox("Type", sorted(raw_df["TypeName"].dropna().unique()))
    op_sys = st.sidebar.selectbox("Operating System", sorted(raw_df["OpSys"].dropna().unique()))
    ram = st.sidebar.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.sidebar.slider("Weight (kg)", min_value=0.7, max_value=5.0, value=2.0, step=0.01)
    inches = st.sidebar.slider("Screen Size (inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
    resolution_choice = st.sidebar.selectbox("Screen Resolution", list(resolution_presets.keys()))
    touch_screen = st.sidebar.selectbox("Touch Screen", [0, 1], format_func=lambda value: "Yes" if value else "No")
    ips = st.sidebar.selectbox("IPS Panel", [0, 1], format_func=lambda value: "Yes" if value else "No")
    cpu_name = st.sidebar.selectbox(
        "CPU Category",
        ["Intel Core i3", "Intel Core i5", "Intel Core i7", "Other Intel Processor", "AMD Processor"],
    )
    ssd = st.sidebar.number_input("SSD Capacity (GB)", min_value=0, max_value=2048, value=256, step=128)
    hdd = st.sidebar.number_input("HDD Capacity (GB)", min_value=0, max_value=2048, value=0, step=128)
    gpu_brand = st.sidebar.selectbox("GPU Brand", ["AMD", "Intel", "Nvidia"])

    width, height = resolution_presets[resolution_choice]
    ppi = calculate_ppi(width, height, inches)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("R²", "0.88")
    col3.metric("Calculated PPI", f"{ppi:.2f}")

    input_payload = {
        "Company": company,
        "TypeName": type_name,
        "Ram": int(ram),
        "OpSys": op_sys,
        "Weight": float(weight),
        "TouchScreen": int(touch_screen),
        "IPS": int(ips),
        "PPI": float(ppi),
        "CPU_name": cpu_name,
        "HDD": int(hdd),
        "SSD": int(ssd),
        "Gpu brand": gpu_brand,
    }

    st.subheader("Selected Input")
    st.json(input_payload)

    if st.button("Predict Price", type="primary"):
        if not MODEL_PATH.exists():
            st.error("Model file not found. Run `python -m src.train` first.")
            return

        model = load_prediction_model()
        prediction = predict_price(model, input_payload)
        st.success(f"Estimated Laptop Price: ₹ {prediction:,.2f}")


if __name__ == "__main__":
    main()
