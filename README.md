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
