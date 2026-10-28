# Independent reproduction of Raman et al. (2026)

Reference: Raman et al., *Epidemiologia* 7(3):75,
[doi:10.3390/epidemiologia7030075](https://doi.org/10.3390/epidemiologia7030075),
[full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13298444/).
No author code was available. No authors were contacted. No clinical claims
are made; these experiments concern reproducibility on two small public extracts.

## Data gate

Both supplied sources pass the pre-specified gate:

| Dataset | Raw | Complete, non-duplicate analytic sample | Positive | Features | Train/test |
|---|---:|---:|---:|---:|---:|
| Cleveland | 303 | 297 | 137 (46.13%) | 13 | 237/60 |
| Framingham | 4240 | 3658 | 557 (15.23%) | 15 | 2926/732 |

Cleveland removes incomplete records and binarizes `num > 0`; Framingham removes
incomplete records then duplicates. The actual Framingham extract has no complete
duplicate records. Source bytes and SHA-256 are in [data/README.md](data/README.md).
Any fingerprint, schema, count, or positive-count discrepancy stops modelling.

## Protocol and assumptions where the paper is silent

1. **Split:** stratified 80/20 using scikit-learn, seed 42 for the single-split
   reproduction. Preserve source row order; no search for a seed matching the
   paper. Stability uses pre-specified seeds 0–499, without cherry-picking.
2. **Inputs:** retain all 15 Framingham and 13 Cleveland predictors. Cleveland
   categorical codes are numeric/ordinal, without one-hot encoding. Education
   is numeric in Framingham. This is an assumption, not a claim about the
   authors' encoding.
3. **Balancing order:** imbalanced-learn pipeline: SMOTE, random undersampling,
   standardization, estimator. All learn only from each training partition.
   The same pipeline is inside GridSearchCV; neither held-out patients nor
   inner validation folds are used for sampling or scaling.
4. **Ratios:** SMOTE uses its default five neighbors and balances to 1:1;
   RandomUnderSampler then targets 1:1, so it removes no additional observations.
   The unspecified intermediate ratio cannot be inferred. Synthetic interpolation
   treats binary/categorical inputs as numeric and can create fractional codes.
   Scaling follows resampling as requested, even though unscaled nearest-neighbor
   distances can be dominated by large-magnitude predictors.
5. **Fixed settings:** the four classical estimators use the exact dataset-specific
   paper hyperparameters from Section 2.3. Remaining estimator settings are
   library defaults except fixed seeds, LR `max_iter=2000`, XGBoost binary
   logistic objective/log loss, and one model worker.
6. **Tuning:** separately evaluate a compact, explicitly specified grid because
   the original search space is unpublished. LR varies C over 0.001/0.01/0.1;
   RF varies trees over 75/200; KNN varies uniform/distance weighting; XGBoost
   varies learning rate 0.01/0.1 and minimum child weight 1/3. Other settings
   stay at each dataset's reported values. The reported point is included.
   Optimize AUROC over shuffled, stratified five-fold training CV, seed equal
   to the outer seed, and refit the winner on the outer training sample.
   These grids are not asserted to recover the authors' search.
7. **NN:** CPU PyTorch MLP, input–32–16–1, ReLU hidden layers, output logits,
   BCEWithLogitsLoss, Adam, 100 full-batch epochs, learning rate 0.001, no weight
   decay/dropout/early stopping. Initialization is seeded. The tuned variant
   compares learning rates 0.001/0.01 using the same inner CV. There is no
   reported NN best-parameter point; “paper” means this fixed surrogate.
8. **Decisions:** probability >= 0.5 predicts positive; no threshold tuning or
   post-hoc calibration. Undefined precision/recall/F1 use zero. AUROC and Brier
   use continuous positive-class probabilities.
9. **SHAP:** explain fixed and tuned XGBoost on up to the first 200 held-out
   observations using TreeExplainer, tree-path-dependent background, raw-margin
   output, and mean absolute attribution. No test-set selection feeds back into
   fitting. Features retain their original names after standardization.
10. **Stability:** refit models and preprocessing for every holdout, both with
    and without balancing. Tuned settings are selected afresh inside every
    outer training partition. Repeated CV is 10 folds repeated five times,
    fixed outer seed 42, also with nested tuning and the balancing ablation.
    No pooled resubstitution predictions are used.
11. **Intervals:** sample SD (`ddof=1`) and empirical 2.5/97.5 percentiles describe
    split sensitivity, not a confidence interval for population performance.
    Splits share patients; repeated-CV fold scores are also dependent and use
    smaller test folds than 80/20 holdouts. Interpret the two phases separately.
12. **Ranking:** rank AUROC only within the same dataset, phase, approach,
    balancing setting, and split. Split credit equally among exact ties.
    “Clear majority” operationally means a win fraction strictly above 50%.
    Rankings on a selected subset of models do not establish the five-model
    ranking. No majority is reported as “statistically indistinguishable”
    under the issue's operational definition, **not** as a formal equivalence
    test or proof of equal performance.
13. **Confirmation:** use the issue's inclusive interval-containment rule for
    paper values, applied to the completed 500-split holdout analysis. A
    single-split match, smoke-run interval, ablation, or CV interval does not
    by itself confirm reproduction. Code emits descriptive containment flags
    for each phase; their sample count and context must always be checked.
14. **Software:** direct scientific dependencies are pinned in `requirements.txt`;
    transitive dependencies/platform math libraries are not fully locked.
    Fixed random seeds do not promise bitwise identity across hardware/library
    builds. These independent implementation choices can change model rankings.

## Executed analyses

Data validation and the complete single-split reproduction finished using
Python 3.12.3 and the pinned requirements on CPU. `results/reproduction_raw.csv`
contains 20 fits: two datasets × five models × fixed/tuned settings, with
training-only balancing and seed 42. The exact train/test sizes match the table
above. All seven metrics are in the CSV; the AUROC comparison is:

| Dataset | Model | Paper AUROC | Fixed-paper AUROC | Tuned AUROC |
|---|---|---:|---:|---:|
| Cleveland | RF | 0.9275 | 0.9408 | 0.9408 |
| Cleveland | LR | 0.9475 | 0.9509 | 0.9509 |
| Cleveland | KNN | 0.9531 | 0.9554 | 0.9581 |
| Cleveland | XGB | 0.9364 | 0.9330 | 0.9330 |
| Cleveland | NN | 0.8867 | 0.9487 | 0.9487 |
| Framingham | RF | 0.6929 | 0.6810 | 0.6810 |
| Framingham | LR | 0.7234 | 0.6581 | 0.6695 |
| Framingham | KNN | 0.6741 | 0.6295 | 0.6304 |
| Framingham | XGB | 0.6882 | 0.6898 | 0.6898 |
| Framingham | NN | 0.6625 | 0.6456 | 0.6456 |

Other requested fixed-setting comparisons:

| Dataset/model | Metric | Paper | This split |
|---|---|---:|---:|
| Cleveland LR | Accuracy | 0.8667 | 0.9000 |
| Cleveland LR | MCC | 0.7321 | 0.8000 |
| Cleveland KNN | Brier | 0.0942 | 0.0923 |
| Framingham RF | Brier | 0.1750 | 0.1704 |
| Framingham NN | Accuracy | 0.7719 | 0.6325 |

KNN remains the Cleveland AUROC winner on this split. XGBoost, not LR, wins
Framingham. These are **single-split observations, not stability conclusions**.
The NN is an independently chosen surrogate, not an exact architecture
reproduction. Similar rounded numbers are not sufficient for confirmation.

### SHAP observations

The fixed and tuned models selected identical XGBoost settings on this split,
and consequently yield the same importance rankings:

- Cleveland top five: `thal`, `ca`, `cp`, `chol`, `slope`. Three of the paper's
  four named leading features appear in our top five; `age` does not.
- Framingham top five: `age`, `sysBP`, `education`, `cigsPerDay`, `diaBP`.
  `sysBP` overlaps with the paper; smoking status (`currentSmoker`) and
  hypertension (`prevalentHyp`) are not top-five here. `cigsPerDay` is smoking
  intensity, not the same feature as smoking status.

See the four `results/shap_*.csv` tables for every feature and attribution.
Feature-importance differences are reported rather than tuned away.

### Stability execution status

The default script requests the full 500-split experiment and nested repeated
CV. Execution coverage and limitations are reported separately below; merely
requesting a run is not evidence that it has completed.

## Interpretation limits

Framingham predicts a 10-year event; Cleveland classifies prevalent disease.
Complete-case selection, ordinal treatment of categories, synthetic fractional
codes, balancing-induced prior shift, and small test sets constrain inference.
Brier is overall probabilistic error, not calibration alone; an unbalanced
baseline can improve Brier simply through lower event prevalence. No external
validation or clinical utility assessment is performed.
