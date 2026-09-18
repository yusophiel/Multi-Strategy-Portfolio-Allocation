import math
import unittest

import pandas as pd

from src.metrics import compute_performance_metrics, max_drawdown


class MetricsTest(unittest.TestCase):
    def test_max_drawdown(self):
        returns = pd.Series([0.10, -0.20, 0.05, -0.10])
        self.assertAlmostEqual(max_drawdown(returns), -0.244, places=3)

    def test_performance_metrics_are_finite_for_basic_series(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.0, 0.03])
        metrics = compute_performance_metrics(returns, periods_per_year=12)
        self.assertTrue(math.isfinite(metrics["annual_return"]))
        self.assertTrue(math.isfinite(metrics["annual_volatility"]))
        self.assertIn("sharpe", metrics)


if __name__ == "__main__":
    unittest.main()
