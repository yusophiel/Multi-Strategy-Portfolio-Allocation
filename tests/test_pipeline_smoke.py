import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.backtest import run_experiment_suite
from src.config import deep_update, load_config
from src.data import generate_synthetic_dataset
from src.features import build_feature_frame
from src.report import write_suite_outputs


class PipelineSmokeTest(unittest.TestCase):
    def test_feature_target_is_next_period_return(self):
        config = load_config(None)
        dataset = generate_synthetic_dataset(config)
        features = build_feature_frame(dataset, config)
        row = features[(features["date"] == dataset.returns.index[10]) & (features["sleeve"] == "equity_beta")].iloc[0]
        expected = dataset.returns.loc[dataset.returns.index[11], "equity_beta"]
        self.assertAlmostEqual(float(row["target_next_return"]), float(expected))
        self.assertEqual(row["next_date"], dataset.returns.index[11])

    def test_run_suite_and_write_outputs(self):
        config = deep_update(
            load_config(None),
            {
                "data": {"start": "2014-01-31", "end": "2021-12-31", "seed": 7},
                "experiment": {
                    "train_window": 24,
                    "validation_window": 6,
                    "execution_lag_periods": 1,
                    "min_training_rows": 80,
                    "enabled": ["equal_weight", "full_workflow"],
                },
            },
        )
        suite = run_experiment_suite(config)
        self.assertIn("full_workflow", suite.experiments)
        self.assertFalse(suite.summary.empty)
        frame = suite.experiments["full_workflow"].returns
        self.assertTrue((frame["return_date"] > frame["decision_date"]).all())
        self.assertTrue((frame["execution_lag_periods"] == 1).all())
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_suite_outputs(suite, Path(tmp))
            self.assertTrue(paths["dashboard"].exists())
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["research_summary"].exists())

    def test_adaptive_universe_preserves_t_plus_one(self):
        config = deep_update(
            load_config(None),
            {
                "data": {"start": "2018-01-31", "end": "2022-12-31", "seed": 8},
                "experiment": {
                    "train_window": 18,
                    "validation_window": 6,
                    "execution_lag_periods": 1,
                    "adaptive_universe": True,
                    "min_sleeve_history": 8,
                    "min_training_rows": 50,
                    "enabled": ["equal_weight"],
                },
            },
        )
        suite = run_experiment_suite(config)
        frame = suite.experiments["equal_weight"].returns
        self.assertFalse(frame.empty)
        self.assertTrue((frame["return_date"] > frame["information_date"]).all())

    def test_experiment_checkpoints_resume_completed_runs(self):
        config = deep_update(
            load_config(None),
            {
                "data": {"start": "2016-01-31", "end": "2021-12-31", "seed": 9},
                "experiment": {
                    "train_window": 18,
                    "validation_window": 6,
                    "execution_lag_periods": 1,
                    "min_training_rows": 50,
                    "enabled": ["equal_weight", "full_workflow"],
                },
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            first = run_experiment_suite(config, checkpoint_dir=checkpoint_dir)
            second = run_experiment_suite(config, checkpoint_dir=checkpoint_dir)
            self.assertEqual(len(list(checkpoint_dir.glob("*.pkl"))), 2)
            pd.testing.assert_frame_equal(first.summary, second.summary)


if __name__ == "__main__":
    unittest.main()
