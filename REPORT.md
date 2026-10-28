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

The checked-in `results/stability_smoke/` run completed **280 fits**:
seeds 0–4, both datasets, all five models, both approaches, both balancing
settings, and **one repeat of two-fold CV**. It took approximately 182 seconds.
The complete 20-fit single-split run took approximately 22 seconds.
These measurements exclude dependency installation.

Exploratory balanced, fixed-setting holdout means, **five splits only**:

| Dataset | Model | Mean AUROC | AUROC SD | Mean accuracy | Mean Brier | AUROC win fraction |
|---|---|---:|---:|---:|---:|---:|
| Cleveland | RF | 0.9048 | 0.0144 | 0.8300 | 0.1265 | 0.0 |
| Cleveland | LR | 0.9080 | 0.0166 | 0.8533 | 0.2140 | 0.2 |
| Cleveland | KNN | 0.9064 | 0.0166 | 0.8433 | 0.1218 | 0.4 |
| Cleveland | XGB | 0.9056 | 0.0141 | 0.8300 | 0.1236 | 0.2 |
| Cleveland | NN | 0.9058 | 0.0149 | 0.8267 | 0.1223 | 0.2 |
| Framingham | RF | 0.6894 | 0.0228 | 0.7533 | 0.1690 | 0.0 |
| Framingham | LR | 0.6853 | 0.0280 | 0.6631 | 0.2128 | 0.0 |
| Framingham | KNN | 0.6315 | 0.0316 | 0.6628 | 0.2158 | 0.0 |
| Framingham | XGB | 0.7039 | 0.0208 | 0.7128 | 0.2225 | 0.8 |
| Framingham | NN | 0.6891 | 0.0300 | 0.6653 | 0.2084 | 0.2 |

The smoke directory includes raw scores, mean/SD/percentile summaries for all
seven metrics, paired rankings, paper comparisons, and AUROC distribution
figures for each phase/approach/balancing setting. The CV portion exercises the
nested evaluation path but **is not the requested repeated 5×10-fold estimate**.
No scientific conclusion follows from the smoke-run ranking or containment flags.

### Full fixed-parameter stability: completed

`results/stability_paper/` contains **11,000 fits**: 500 held-out splits and
50 outer CV folds × two datasets × five models × two balancing settings,
using the fixed paper parameters (and the specified NN surrogate). Four
parallel seed blocks plus CV completed in approximately **495 seconds**.
The consolidated result has one row per fit, with duplicate seed-zero
holdouts from the CV job removed. Its manifest supports resumption through
the ordinary CLI; intermediate job outputs are not committed.

Balanced 500-split results:

| Dataset | Model | Mean AUROC ± SD | AUROC 2.5–97.5% | Mean accuracy | Mean Brier | AUROC #1 |
|---|---|---|---|---:|---:|---:|
| Cleveland | RF | 0.8999 ± 0.0352 | 0.8237–0.9618 | 0.8201 | 0.1287 | 10.7% |
| Cleveland | LR | 0.9027 ± 0.0350 | 0.8275–0.9626 | 0.8324 | 0.2136 | 18.8% |
| Cleveland | KNN | 0.9048 ± 0.0349 | 0.8315–0.9626 | 0.8304 | 0.1222 | 23.2% |
| Cleveland | XGB | 0.9060 ± 0.0347 | 0.8381–0.9654 | 0.8269 | 0.1238 | 33.0% |
| Cleveland | NN | 0.8999 ± 0.0355 | 0.8242–0.9604 | 0.8294 | 0.1264 | 14.3% |
| Framingham | RF | 0.6831 ± 0.0237 | 0.6389–0.7293 | 0.7475 | 0.1717 | 2.4% |
| Framingham | LR | 0.6906 ± 0.0236 | 0.6452–0.7433 | 0.6663 | 0.2122 | 21.3% |
| Framingham | KNN | 0.6336 ± 0.0243 | 0.5885–0.6835 | 0.6623 | 0.2142 | 0.4% |
| Framingham | XGB | 0.6982 ± 0.0239 | 0.6521–0.7428 | 0.7235 | 0.2225 | 59.1% |
| Framingham | NN | 0.6892 ± 0.0243 | 0.6422–0.7367 | 0.6606 | 0.2103 | 16.8% |

`stability_summary.csv` also includes SD and percentiles for accuracy and Brier
(and all remaining metrics), separately for every ablation and CV context.
Wins split exact ties; percentages therefore need not represent integer wins.

**Interval-containment findings, fixed balanced approach:** all ten paper
AUROC values satisfy the issue's reproduction criterion. Cleveland LR accuracy
and MCC, Cleveland KNN Brier, and Framingham RF Brier also fall inside their
500-split intervals. **Framingham NN accuracy 0.7719 does not:** its interval
is 0.6216–0.6974. This is a material mismatch for our assumed architecture.
Of the additional full-table targets, Framingham RF/KNN recall and NN
precision/recall/F1 also fall outside their respective intervals. All Cleveland
full-table targets are contained. These are descriptive containment results,
not proof that the authors used the same implementation.

**Ranking:** the paper's Cleveland winner KNN wins only 23.2% of splits, and
no Cleveland model wins a majority. The models are operationally
“statistically indistinguishable” under the issue's ranking criterion, not a
formal equivalence result. The paper's Framingham winner LR wins only 21.3%;
its top-model ranking is likewise unsupported. Instead, XGBoost wins 59.1%,
a strict majority under our chosen protocol. Thus reproducing the paper's
AUROC ranges does **not** reproduce its model ranking.

### Repeated 5×10-fold CV and balancing ablation: completed for fixed settings

Balanced outer-fold AUROC estimates (50 folds, not 50 independent samples):

| Dataset | Model | Mean AUROC ± SD |
|---|---|---|
| Cleveland | RF | 0.9007 ± 0.0636 |
| Cleveland | LR | 0.9033 ± 0.0618 |
| Cleveland | KNN | 0.9087 ± 0.0623 |
| Cleveland | XGB | 0.9061 ± 0.0629 |
| Cleveland | NN | 0.9019 ± 0.0630 |
| Framingham | RF | 0.6856 ± 0.0390 |
| Framingham | LR | 0.6918 ± 0.0345 |
| Framingham | KNN | 0.6388 ± 0.0335 |
| Framingham | XGB | 0.7012 ± 0.0386 |
| Framingham | NN | 0.6856 ± 0.0335 |

Removing resampling barely changes Cleveland mean AUROC (all model differences
below 0.0011). Framingham changes substantially: unbalanced LR mean AUROC
is **0.7300** versus 0.6906 balanced, and LR wins **85.2%** of unbalanced splits.
Unbalanced RF mean AUROC is 0.7168 versus 0.6831 balanced. The model ranking
is therefore sensitive to balancing. Framingham Brier scores also decrease
without resampling (e.g. LR 0.1189 versus 0.2122); training prior shift and
the imbalanced outcome distribution matter. High accuracy without balancing
is not sufficient evidence of good disease detection.

### Remaining execution

The **fully tuned** 500-split/5×10-fold analysis remains unexecuted. Five-fold
tuning was executed for the complete single-split reproduction and the smoke
analysis, but not repeated across all 500 seeds. Extrapolation from the smoke
workload suggests roughly four CPU-hours for both approaches and ablations
on this runner; this is an estimate. Consequently full-stability conclusions
above apply only to fixed reported parameters and our fixed NN, not to
per-split hyperparameter selection.

To reproduce the completed fixed-setting design:

```bash
python scripts/03_stability.py --approaches paper --output-dir results/stability_paper
```

To execute the full design including nested tuning:

```bash
python scripts/03_stability.py --output-dir results/full
```

Re-run the same command to resume interrupted fits. The full command is
deliberately separate from both completed result directories. Once it completes,
use `results/full/stability_paper_comparison.csv` **holdout** rows (`n=500`) to
apply the requested reproduction rule, and `stability_rankings.csv` to assess
whether the paper's top model wins a strict majority of splits among all five.
Consider fixed and tuned approaches separately; ablations are sensitivity
analyses, not substitutes for the balanced primary analysis.

**Confirmation status:** all paper AUROCs meet the issue's interval rule under
the fixed-setting reproduction, but neither dataset's paper top-model ranking
is supported. The Framingham NN accuracy is not reproduced. Confirmation under
the independently tuned protocol remains undetermined.

## Interpretation limits

Framingham predicts a 10-year event; Cleveland classifies prevalent disease.
Complete-case selection, ordinal treatment of categories, synthetic fractional
codes, balancing-induced prior shift, and small test sets constrain inference.
Brier is overall probabilistic error, not calibration alone; an unbalanced
baseline can improve Brier simply through lower event prevalence. No external
validation or clinical utility assessment is performed.

## Validation

Six offline data-check tests and eleven model/protocol tests pass. Tests cover
missingness, deduplication, hard count/target/hash gates, exact paper settings,
all five estimators, fold-local resampling, NN determinism and tuning, metrics,
paired ties, empirical summaries, SHAP, repeated CV, and resumable output.
The model tests run without a `PYTHONPATH` override. Actual CLI runs and
resume-without-refitting checks succeeded for the saved analyses.
`pip check` found no dependency conflicts. The independent code review found
no significant issues; dependency advisory checks, secret scanning, and CodeQL
reported no findings.
