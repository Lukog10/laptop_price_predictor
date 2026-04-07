"""Unit tests for inference helpers."""

from __future__ import annotations

import pickle
import tempfile
import unittest

import numpy as np

from src.predict import load_model, predict_price


class StubModel:
    """Simple prediction stub that returns a fixed log-price prediction."""

    def predict(self, frame):
        """Return a fixed log-space prediction for the supplied frame."""
        return np.array([np.log(50000.0)])


class TestPredict(unittest.TestCase):
    """Test suite for model loading and price prediction."""

    def setUp(self) -> None:
        """Prepare a valid engineered input payload for inference tests."""
        self.valid_input = {
            "Company": "Dell",
            "TypeName": "Notebook",
            "Ram": 8,
            "OpSys": "Windows 10",
            "Weight": 2.0,
            "TouchScreen": 0,
            "IPS": 1,
            "PPI": 141.21,
            "CPU_name": "Intel Core i5",
            "HDD": 0,
            "SSD": 512,
            "Gpu brand": "Nvidia",
        }

    def test_load_model_returns_deserialized_object(self) -> None:
        """The load helper should deserialize a pickled model object."""
        stub_model = StubModel()
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            pickle.dump(stub_model, tmp_file)
            tmp_file.flush()
            loaded_model = load_model(tmp_file.name)
        self.assertIsInstance(loaded_model, StubModel)

    def test_predict_price_returns_inr_value(self) -> None:
        """Predictions should be converted back from log space to INR."""
        prediction = predict_price(StubModel(), self.valid_input)
        self.assertAlmostEqual(prediction, 50000.0, places=2)

    def test_predict_price_raises_for_missing_fields(self) -> None:
        """Missing required fields should raise a ValueError."""
        invalid_input = {key: value for key, value in self.valid_input.items() if key != "SSD"}
        with self.assertRaises(ValueError):
            predict_price(StubModel(), invalid_input)


if __name__ == "__main__":
    unittest.main()
