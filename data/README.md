# Data provenance

Downloaded and checked on 2026-10-28. Raw files are deliberately not committed.
Run `python scripts/01_data_checks.py` from the repository root to download into
`data/raw/`. Existing files and downloads must match these SHA-256 fingerprints;
a mismatch stops execution rather than silently accepting a different extract.

| Dataset | Raw file SHA-256 | Raw rows | Clean rows | Positive |
|---|---|---:|---:|---:|
| Cleveland | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` | 303 | 297 | 137 (46.13%) |
| Framingham | `2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018` | 4240 | 3658 | 557 (15.23%) |

## Sources and cleaning

- Cleveland: <https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data>.
  UCI Heart Disease, donated by Janosi, Steinbrunn, Pfisterer, and Detrano,
  DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X), CC BY 4.0.
  Headerless file; `?` is missing. Remove incomplete rows (six), then replace
  `num` with `target = int(num > 0)`. Keep all 13 predictors.
- Framingham teaching extract:
  <https://raw.githubusercontent.com/sta210-sp20/datasets/master/framingham.csv>.
  Distribution page: <https://github.com/sta210-sp20/datasets/blob/master/framingham.csv>.
  Redistribution licence for this extract has not been established here.
  Remove rows missing any field (582), then remove duplicate complete records
  (zero in these bytes). Keep all 15 predictors and binary `TenYearCHD`.

Both datasets must pass the gate before any model can fit. Source row order is
preserved. These are different outcomes: prevalent disease in Cleveland versus
10-year CHD in Framingham; their metrics are not a common clinical endpoint.
