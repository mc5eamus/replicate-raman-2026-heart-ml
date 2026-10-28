#!/usr/bin/env python3
"""Download public data and stop on any analytic-sample discrepancy."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raman.data import DataValidationError, data_summary, load_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        frames = load_datasets(args.data_dir)
    except DataValidationError as exc:
        parser.exit(1, f"DATA GATE FAILED: {exc}\n")
    summary = data_summary(frames, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "data_checks.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
