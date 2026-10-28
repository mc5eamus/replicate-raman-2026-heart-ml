#!/usr/bin/env python3
"""500 holdouts plus 5x10-fold outer CV, with nested five-fold tuning."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raman.analysis import parse_seeds, run_analysis
from raman.data import load_datasets
from raman.models import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--seeds", help="Comma-separated seeds or half-open range, e.g. 0:500")
    count.add_argument("--splits", type=int, help="Use seeds 0 through N-1; default 500")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=["cleveland", "framingham"],
                        default=["cleveland", "framingham"])
    parser.add_argument("--approaches", nargs="+", choices=["paper", "tuned"],
                        default=["paper", "tuned"])
    parser.add_argument("--balancing", choices=["both", "balanced", "none"], default="both")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    try:
        seeds = parse_seeds(args.seeds) if args.seeds else list(
            range(args.splits if args.splits is not None else 500))
        if not seeds:
            raise ValueError("--splits must be positive")
        if args.cv_folds < 2 or args.cv_repeats < 1:
            raise ValueError("CV requires at least two folds and one repeat")
    except ValueError as error:
        parser.error(str(error))
    datasets = load_datasets(args.data_dir)
    balances = {"both": (True, False), "balanced": (True,), "none": (False,)}[args.balancing]
    run_analysis({name: datasets[name] for name in args.datasets}, args.output_dir,
                 models=args.models, approaches=args.approaches, seeds=seeds,
                 balances=balances, include_cv=not args.skip_cv,
                 cv_folds=args.cv_folds, cv_repeats=args.cv_repeats,
                 cv_seed=args.cv_seed, n_jobs=args.n_jobs, explain=False)


if __name__ == "__main__":
    main()
