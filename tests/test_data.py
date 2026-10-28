"""Offline tests of the exact analytic-sample gate."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raman.data import (
    CLEVELAND_COLUMNS, FRAMINGHAM_COLUMNS, SOURCES, DataValidationError,
    clean_dataset, download, sha256, split_xy,
)


class DataTests(unittest.TestCase):
    def test_source_fingerprints_are_sha256(self):
        for _, _, fingerprint in SOURCES.values():
            self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")

    def test_cleveland_missing_and_binarization(self):
        frame = pd.DataFrame([[1] * 14 for _ in range(303)], columns=CLEVELAND_COLUMNS)
        frame["num"] = [4] * 137 + [0] * 166
        frame["ca"] = frame["ca"].astype(object)
        frame.loc[297:, "ca"] = "?"
        cleaned = clean_dataset("cleveland", frame)
        self.assertEqual(len(cleaned), 297)
        x, y = split_xy("cleveland", cleaned)
        self.assertEqual(x.shape[1], 13)
        self.assertEqual(y.sum(), 137)
        self.assertNotIn("num", x)
        self.assertNotIn("target", x)

    def framingham(self):
        frame = pd.DataFrame([[1] * 16 for _ in range(4240)], columns=FRAMINGHAM_COLUMNS)
        frame["age"] = range(4240)
        frame["TenYearCHD"] = [1] * 557 + [0] * (4240 - 557)
        frame.loc[3658:, "glucose"] = float("nan")
        return frame

    def test_framingham_complete_case_then_deduplicate(self):
        frame = self.framingham()
        frame.loc[3658] = frame.loc[1000]
        cleaned = clean_dataset("framingham", frame)
        self.assertEqual(len(cleaned), 3658)
        x, y = split_xy("framingham", cleaned)
        self.assertEqual(x.shape[1], 15)
        self.assertEqual(y.sum(), 557)

    def test_counts_and_positives_are_hard_gate(self):
        frame = self.framingham()
        with self.assertRaises(DataValidationError):
            clean_dataset("framingham", frame.iloc[:-1])
        frame.loc[0, "TenYearCHD"] = 0
        with self.assertRaises(DataValidationError):
            clean_dataset("framingham", frame)
        frame = self.framingham()
        frame.loc[0, "glucose"] = float("nan")
        with self.assertRaises(DataValidationError):
            clean_dataset("framingham", frame)

    def test_schema_and_target_validation(self):
        frame = self.framingham()
        with self.assertRaises(DataValidationError):
            clean_dataset("framingham", frame.drop(columns="age"))
        frame.loc[0, "TenYearCHD"] = 2
        with self.assertRaises(DataValidationError):
            clean_dataset("framingham", frame)

    def test_bad_cache_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "processed.cleveland.data"
            path.write_text("invalid input")
            with patch("raman.data.urlopen") as request:
                with self.assertRaises(DataValidationError):
                    download(directory)
                request.assert_not_called()
            self.assertEqual(len(sha256(path)), 64)


if __name__ == "__main__":
    unittest.main()
