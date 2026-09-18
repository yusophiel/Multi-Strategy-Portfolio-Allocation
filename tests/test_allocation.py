import unittest

import pandas as pd

from src.allocation import compute_turnover, normalize_with_caps
from src.config import load_config


class AllocationTest(unittest.TestCase):
    def test_normalize_with_caps(self):
        config = load_config(None)
        raw = pd.Series(
            {
                "equity_beta": 100.0,
                "trend_following": 1.0,
                "value_quality": 1.0,
                "carry_macro": 1.0,
                "defensive_low_vol": 1.0,
                "cash": 1.0,
            }
        )
        weights = normalize_with_caps(raw, config)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertLessEqual(float(weights.max()), config["allocation"]["max_weight"] + 1e-9)

    def test_turnover(self):
        old = pd.Series({"a": 0.5, "b": 0.5})
        new = pd.Series({"a": 0.2, "b": 0.8})
        self.assertAlmostEqual(compute_turnover(new, old), 0.3)


if __name__ == "__main__":
    unittest.main()
