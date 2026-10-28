#!/usr/bin/env python3
"""Stratified 80/20 reproduction; defaults to both fixed and tuned models."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raman.analysis import run_analysis
from raman.data import load_datasets
from raman.models import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=["cleveland", "framingham"],
                        default=["cleveland", "framingham"])
    parser.add_argument("--approaches", nargs="+", choices=["paper", "tuned"],
                        default=["paper", "tuned"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    datasets = load_datasets(args.data_dir)
    run_analysis({name: datasets[name] for name in args.datasets}, args.output_dir,
                 models=args.models, approaches=args.approaches, seeds=[args.seed],
                 balances=(True,), include_cv=False, n_jobs=args.n_jobs,
                 reproduction=True, explain=not args.skip_shap)


if __name__ == "__main__":
    main()
