"""Project configuration constants for the laptop price predictor."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "laptop_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"
TARGET_COLUMN = "Price"
TEST_SIZE = 0.15
RANDOM_STATE = 3
SPLIT_RANDOM_STATE = 2
N_ESTIMATORS = 100
RF_MAX_SAMPLES = 0.5
RF_MAX_FEATURES = 0.75
RF_MAX_DEPTH = 15
LOG_TRANSFORM = True
FINAL_FEATURE_COLUMNS = [
    "Company",
    "TypeName",
    "Ram",
    "OpSys",
    "Weight",
    "TouchScreen",
    "IPS",
    "PPI",
    "CPU_name",
    "HDD",
    "SSD",
    "Gpu brand",
]
CATEGORICAL_COLS = [0, 1, 3, 8, 11]
