"""Unit tests for preprocessing utilities."""

from __future__ import annotations

import unittest

import pandas as pd

from src.preprocess import clean_dataframe, run_pipeline


class TestPreprocess(unittest.TestCase):
    """Test suite for notebook-to-module preprocessing logic."""

    def setUp(self) -> None:
        """Create a miniature raw dataset for preprocessing tests."""
        self.raw_df = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1, 1],
                "Company": ["Dell", "HP", "HP"],
                "TypeName": ["Notebook", "Ultrabook", "Ultrabook"],
                "Inches": [15.6, 13.3, 13.3],
                "ScreenResolution": [
                    "Full HD 1920x1080",
                    "IPS Panel Touchscreen 2560x1440",
                    "IPS Panel Touchscreen 2560x1440",
                ],
                "Cpu": [
                    "Intel Core i5 7200U 2.5GHz",
                    "AMD Ryzen 5 2500U 2.0GHz",
                    "AMD Ryzen 5 2500U 2.0GHz",
                ],
                "Ram": ["8GB", "16GB", "16GB"],
                "Memory": ["256GB SSD", "512GB SSD + 1TB HDD", "512GB SSD + 1TB HDD"],
                "Gpu": ["Nvidia GTX 1050", "AMD Radeon Vega 8", "AMD Radeon Vega 8"],
                "OpSys": ["Windows 10", "Windows 10", "Windows 10"],
                "Weight": ["2.1kg", "1.3kg", "1.3kg"],
                "Price": [50000.0, 75000.0, 75000.0],
            }
        )

    def test_clean_dataframe_removes_unnamed_columns(self) -> None:
        """The cleaning step should remove unnamed columns and preserve row count."""
        cleaned = clean_dataframe(self.raw_df)
        self.assertNotIn("Unnamed: 0", cleaned.columns)
        self.assertEqual(len(cleaned), 3)

    def test_run_pipeline_creates_expected_features(self) -> None:
        """The full pipeline should create model-ready engineered features."""
        processed = run_pipeline(self.raw_df)
        expected_columns = {
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
            "Price",
        }
        self.assertEqual(set(processed.columns), expected_columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(processed["Ram"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed["Weight"]))
        self.assertNotIn("ScreenResolution", processed.columns)
        self.assertNotIn("Cpu", processed.columns)
        self.assertNotIn("Memory", processed.columns)
        self.assertNotIn("Gpu", processed.columns)


if __name__ == "__main__":
    unittest.main()
