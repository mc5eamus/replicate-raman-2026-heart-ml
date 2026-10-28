# replicate-raman-2026-heart-ml
Independent code reproduction of Raman et al. 2026 (Epidemiologia, PMID 42346163): ML for coronary heart disease on Framingham and UCI Cleveland, plus a repeated-split stability analysis.

## Guided, standalone notebook

Open [`notebooks/raman2026_reproduction.ipynb`](notebooks/raman2026_reproduction.ipynb)
on GitHub to read the narrated walkthrough with executed tables and plots, or
open it in a Jupyter-capable editor (for example VS Code, selecting the environment
below). It contains all analysis code and uses no repository Python modules.
It follows [the paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13298444/), labels
unreported implementation choices as assumptions, and computes both the
paper-versus-ours comparison and its conclusions from predictions.

With **Python 3.12**, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r notebooks/requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 notebooks/raman2026_reproduction.ipynb
```

The requirements include CPU-only PyTorch and Jupyter execution tooling; a
graphical Jupyter server is optional. The first run downloads two public datasets
over HTTPS, verifies and prints their SHA-256 fingerprints, and caches them under
`notebooks/data/` (ignored by Git). Later runs can use that cache offline. Do not
commit raw patient records; Framingham mirror redistribution terms are unclear.

The single configuration cell sets `SEED = 42` and `N_SPLITS = 50`. The default
run targets **under 15 minutes on a laptop CPU**, with one computational thread,
and reports its measured elapsed time in the final output. Timings depend on the
machine and network. Increasing `N_SPLITS` increases runtime proportionally.
The [separate 500-split analysis in PR #2](https://github.com/mc5eamus/replicate-raman-2026-heart-ml/pull/2)
is not imported or required.

The **Standalone notebook** Actions workflow installs only these requirements,
runs focused protocol tests, and executes the entire default notebook with a
15-minute execution budget. Local protocol/output checks:

```bash
python -m unittest discover -s tests -p test_notebook.py -v
```

These are historical teaching datasets with different outcomes; the notebook is
a methodological reproduction, **not clinical advice or a deployable risk tool**.

## Command-line reproduction

Paper: [doi:10.3390/epidemiologia7030075](https://doi.org/10.3390/epidemiologia7030075),
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13298444/).
This is a methods reproduction, not a clinical tool.

### Run

Python 3.11+ is required. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/01_data_checks.py
.venv/bin/python scripts/02_reproduce.py
.venv/bin/python scripts/03_stability.py
```

The last command requests **500 seeds (0–499), both datasets, all five models,
both fixed-paper and five-fold tuned settings, with and without balancing,
plus 5 repeats of 10-fold outer stratified CV**. Tuning is nested within every
outer training partition. This is a substantial CPU workload, not a quick test.
`--n-jobs` controls parallel grid-search workers; model-level workers stay at one.
For predictable resource use, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
and `MKL_NUM_THREADS=1` before running.

For a short, explicitly limited validation run:

```bash
.venv/bin/python scripts/03_stability.py --splits 2 --skip-cv \
  --approaches paper --output-dir /tmp/raman-smoke
```

All scripts accept `--data-dir` and `--output-dir`. Reproduction defaults to
seed 42 and includes SHAP for both fixed and tuned XGBoost. Stability accepts
`--seeds 0:500`, `--models`, `--datasets`, `--approaches`, and `--balancing`.
Use `--help` for the complete interface. Defaults resolve relative to the
script's repository, not the current working directory.

Completed fits are appended to CSV. Re-running the **identical command** in the
same output directory resumes it; changed settings require a separate directory.
Do not run simultaneous writers against the same directory. Summary tables and
AUROC plots are produced after the requested run finishes. Keep partial/smoke
runs separate from full results, and check sample counts before interpretation.

### Outputs and tests

- [`data/README.md`](data/README.md): raw-file sources, SHA-256, cleaning, licences.
- `results/data_checks.csv`: validated cohort counts and provenance.
- `reproduction_raw.csv`, `stability_raw.csv`: per-partition seven-metric results
  and tuned best parameters.
- `*_summary.csv`: mean, sample SD, and empirical 2.5/97.5 percentiles.
- `*_rankings.csv`: paired AUROC win fractions (equal credit for exact ties).
- `*_paper_comparison.csv`: paper values and interval-containment flags.
- `shap_*.csv`: XGBoost feature importance by mean absolute held-out SHAP.
- `*.png`: AUROC distributions, with red dashed paper targets.
- `*_settings.json`: exact run selection and input fingerprints.

See [`REPORT.md`](REPORT.md) for executed results, assumptions, and limitations.
Checked-in results include the complete single-split reproduction,
`results/stability_paper/` (500 splits + 5×10 CV, fixed settings, both balancing
variants), and `results/stability_smoke/` (five splits + 1×2 CV, fixed and tuned).
The full 500-split **tuned** experiment has not been executed.
Raw patient records, fitted models, and virtual environments are not committed.

Tests use standard-library `unittest` and synthetic data; no network is needed:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
