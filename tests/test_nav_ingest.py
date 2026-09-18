import unittest
from pathlib import Path

import pandas as pd

from src.nav_ingest import excel_engine_for_path, nav_series_from_frame


class NavIngestTest(unittest.TestCase):
    def test_excel_engine_is_explicit_for_legacy_xls_files(self):
        self.assertEqual(excel_engine_for_path(Path("fund.xlsx")), "openpyxl")
        self.assertEqual(excel_engine_for_path(Path("fund.xls")), "xlrd")

    def test_labeled_nav_sheet_scores_above_unlabeled_inferred_sheet(self):
        path = Path("示例基金A.xlsx")
        labeled = pd.DataFrame(
            {
                "产品名称": ["示例基金A"] * 4,
                "日期": pd.date_range("2025-02-07", periods=4, freq="W-FRI"),
                "单位净值": [1.0, 1.01, 1.02, 1.03],
                "累计净值": [1.2, 1.21, 1.22, 1.23],
            }
        )
        unlabeled = pd.DataFrame(
            {
                "2025-01-01": pd.date_range("2025-01-01", periods=20, freq="W-FRI"),
                "1.0": [1.0 + i * 0.001 for i in range(20)],
                "1.1": [1.1 + i * 0.001 for i in range(20)],
            }
        )
        labeled_series = nav_series_from_frame(labeled, path, "Sheet1")
        unlabeled_series = nav_series_from_frame(unlabeled, path, "Sheet2")
        self.assertIsNotNone(labeled_series)
        self.assertIsNotNone(unlabeled_series)
        self.assertGreater(labeled_series.quality_score, unlabeled_series.quality_score)


if __name__ == "__main__":
    unittest.main()
