"""Reproduction, nested stability evaluation, and descriptive comparisons."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .models import MODELS, make_estimator

METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "brier")
PAPER_METRICS = {
    "framingham": {
        "RF": (0.7391, 0.2872, 0.4865, 0.3612, 0.2222, 0.6929, 0.1750),
        "LR": (0.6803, 0.2764, 0.6847, 0.3938, 0.2698, 0.7234, 0.2138),
        "KNN": (0.6489, 0.2448, 0.6306, 0.3526, 0.2079, 0.6741, 0.2271),
        "XGB": (0.6899, 0.2456, 0.5045, 0.3304, 0.1762, 0.6882, 0.2255),
        "NN": (0.7719, 0.2812, 0.3243, 0.3013, 0.1664, 0.6625, 0.2048),
    },
    "cleveland": {
        "RF": (0.8333, 0.8462, 0.7857, 0.8148, 0.6652, 0.9275, 0.1122),
        "LR": (0.8667, 0.8571, 0.8571, 0.8571, 0.7321, 0.9475, 0.2096),
        "KNN": (0.8333, 0.8462, 0.7857, 0.8148, 0.6652, 0.9531, 0.0942),
        "XGB": (0.8333, 0.8750, 0.7500, 0.8077, 0.6683, 0.9364, 0.1087),
        "NN": (0.8500, 0.8276, 0.8571, 0.8421, 0.6997, 0.8867, 0.1360),
    },
}
PAPER_TARGETS = {
    dataset: {model: dict(zip(METRICS, values)) for model, values in models.items()}
    for dataset, models in PAPER_METRICS.items()
}
GROUPS = ["dataset", "phase", "balanced", "approach", "model"]
KEYS = GROUPS + ["split", "seed"]
FIELDS = KEYS + ["train_n", "test_n", "best_params"] + list(METRICS)


def evaluate(y, probability):
    probability = np.asarray(probability)
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "mcc": matthews_corrcoef(y, prediction),
        "auroc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
    }


def summarize(results):
    """SD uses ddof=1; a single observation has undefined (NaN) SD."""
    rows = []
    for key, group in results.groupby(GROUPS, dropna=False):
        for metric in METRICS:
            values = group[metric]
            row = dict(zip(GROUPS, key))
            row.update(metric=metric, n=len(values), mean=values.mean(),
                       sd=values.std(ddof=1), lower=values.quantile(0.025),
                       upper=values.quantile(0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def rank_wins(results):
    """Paired AUROC wins, splitting credit equally among exact ties.

    Only complete model panels are counted; ranking is among selected models,
    not evidence about excluded models. A clear majority means strictly >50%.
    """
    contexts = ["dataset", "phase", "balanced", "approach"]
    rows = []
    for key, group in results.groupby(contexts):
        models = sorted(group.model.unique())
        panel = group.pivot(index=["split", "seed"], columns="model", values="auroc")
        panel = panel.reindex(columns=models).dropna()
        if panel.empty:
            continue
        winner = panel.eq(panel.max(axis=1), axis=0)
        shares = winner.div(winner.sum(axis=1), axis=0).sum(axis=0)
        for model in models:
            fraction = float(shares[model] / len(panel))
            row = dict(zip(contexts, key))
            row.update(model=model, splits=len(panel), models_compared=len(models),
                       fractional_wins=float(shares[model]), win_fraction=fraction,
                       clear_majority=fraction > 0.5,
                       paper_top_model={"cleveland": "KNN", "framingham": "LR"}[key[0]])
            rows.append(row)
    return pd.DataFrame(rows)


def compare_paper(summary):
    rows = []
    for record in summary.to_dict("records"):
        target = PAPER_TARGETS[record["dataset"]][record["model"]].get(record["metric"])
        if target is not None:
            record.update(paper=target, mean_minus_paper=record["mean"] - target,
                          reproduced=record["lower"] <= target <= record["upper"])
            rows.append(record)
    return pd.DataFrame(rows)


def write_summaries(results, output_dir, prefix="stability"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    rank_wins(results).to_csv(output_dir / f"{prefix}_rankings.csv", index=False)
    compare_paper(summary).to_csv(output_dir / f"{prefix}_paper_comparison.csv", index=False)
    plot_auroc(results, output_dir, prefix)
    return summary


def plot_auroc(results, output_dir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for context, group in results.groupby(["dataset", "phase", "balanced", "approach"]):
        dataset, phase, balanced, approach = context
        models = [model for model in MODELS if model in group.model.unique()]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot([group.loc[group.model == model, "auroc"].to_numpy()
                    for model in models], tick_labels=models, showfliers=False)
        for position, model in enumerate(models, 1):
            ax.plot([position - 0.3, position + 0.3],
                    [PAPER_TARGETS[dataset][model]["auroc"]] * 2,
                    color="red", linestyle="--",
                    label="Paper AUROC" if position == 1 else None)
        ax.set(title=f"{dataset}: {phase}, balanced={balanced}, {approach}",
               ylabel="Held-out AUROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"{prefix}_{dataset}_{phase}_{balanced}_{approach}.png",
                    dpi=150)
        plt.close(fig)


def shap_features(estimator, X_test, output_path, max_samples=200):
    """Explain transformed held-out XGB inputs; support old/new SHAP shapes.

    Some older SHAP releases cannot read XGBoost 3's bracketed base_score.
    Explain a copied booster with its mathematically identical scalar score.
    """
    import shap

    pipeline = getattr(estimator, "best_estimator_", estimator)
    sample = X_test.iloc[:max_samples]
    transformed = pipeline.named_steps["scale"].transform(sample)
    booster = pipeline.named_steps["model"].get_booster()
    try:
        explainer = shap.TreeExplainer(booster)
    except (ValueError, TypeError) as error:
        config = json.loads(booster.save_config())
        score = config["learner"]["learner_model_param"]["base_score"]
        if not isinstance(score, str) or not score.startswith("["):
            raise error
        booster = booster.copy()
        booster.set_param({"base_score": float(json.loads(score)[0])})
        explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != transformed.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {transformed.shape}")
    result = pd.DataFrame({
        "feature": X_test.columns, "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    result.to_csv(output_path, index=False)
    return result


def _read_results(path):
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    result = pd.read_csv(path)
    if list(result.columns) != FIELDS:
        raise ValueError(f"Incompatible results schema: {path}")
    if result.duplicated(KEYS).any() or result[list(METRICS)].isna().any().any():
        raise ValueError(f"Duplicate/incomplete results in {path}")
    return result


def _run_partition(X, y, train, test, dataset, phase, split, seed, models,
                   approaches, balances, output_path, completed, n_jobs,
                   shap_dir=None):
    for balanced in balances:
        for approach in approaches:
            for model in models:
                record = dict(dataset=dataset, phase=phase, balanced=balanced,
                              approach=approach, model=model, split=split, seed=seed)
                key = tuple(str(record[name]) for name in KEYS)
                if key in completed:
                    continue
                estimator = make_estimator(dataset, model, seed, balanced, approach,
                                           n_jobs=n_jobs)
                estimator.fit(X.iloc[train], y.iloc[train])
                record.update(evaluate(y.iloc[test], estimator.predict_proba(X.iloc[test])[:, 1]))
                record.update(train_n=len(train), test_n=len(test),
                              best_params=json.dumps(getattr(estimator, "best_params_", {}),
                                                     sort_keys=True))
                # Save explanations before marking a model complete, so resume
                # cannot silently skip a failed SHAP calculation.
                if shap_dir is not None and model == "XGB":
                    shap_features(estimator, X.iloc[test],
                                  Path(shap_dir) / f"shap_{dataset}_{approach}.csv")
                with output_path.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(record)
                completed.add(key)
                print(f"{dataset} {phase} split={split} {approach} "
                      f"balanced={balanced} {model} AUROC={record['auroc']:.4f}", flush=True)


def run_analysis(datasets, output_dir, *, models=MODELS,
                 approaches=("paper", "tuned"), seeds=range(500),
                 balances=(True, False), include_cv=True, cv_folds=10,
                 cv_repeats=5, cv_seed=42, n_jobs=1, reproduction=False,
                 explain=True):
    """Append one completed fit at a time; resume only identical run settings.

    Each outer holdout/CV training fold has its own five-fold grid search.
    Repeated CV distributions are descriptive, not independent-sample CIs.
    """
    from .data import split_xy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    if not datasets or not models or not approaches or not balances or not seeds:
        raise ValueError("At least one dataset, model, approach, balance and seed is required.")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be unique nonnegative integers.")
    if reproduction and (len(seeds) != 1 or tuple(balances) != (True,) or include_cv):
        raise ValueError("Reproduction requires one seed, balancing, and no outer CV.")
    prefix = "reproduction" if reproduction else "stability"
    output_path = output_dir / f"{prefix}_raw.csv"
    # Data fingerprints prevent mixing runs after edits or changed row order.
    fingerprints = {
        name: __import__("hashlib").sha256(
            pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        ).hexdigest() for name, frame in datasets.items()
    }
    settings = dict(datasets=sorted(datasets), fingerprints=fingerprints,
                    models=list(models), approaches=list(approaches), seeds=seeds,
                    balances=list(balances), include_cv=include_cv, cv_folds=cv_folds,
                    cv_repeats=cv_repeats, cv_seed=cv_seed, explain=explain,
                    protocol="fold-local-v1")
    manifest = output_dir / f"{prefix}_settings.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            raise ValueError("Run settings changed. Use a separate output directory.")
    elif output_path.exists():
        raise ValueError("Raw results exist without their settings manifest.")
    else:
        manifest.write_text(json.dumps(settings, indent=2) + "\n")
    existing = _read_results(output_path)
    completed = set(tuple(str(value) for value in row)
                    for row in existing[KEYS].itertuples(index=False, name=None))
    for dataset, frame in datasets.items():
        X, y = split_xy(dataset, frame)
        for seed in seeds:
            train, test = train_test_split(np.arange(len(y)), test_size=0.2,
                                          stratify=y, random_state=seed)
            _run_partition(X, y, train, test, dataset, "holdout", seed, seed,
                           models, approaches, balances, output_path, completed,
                           n_jobs, output_dir if reproduction and explain else None)
        if include_cv:
            cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats,
                                        random_state=cv_seed)
            for split, (train, test) in enumerate(cv.split(X, y)):
                _run_partition(X, y, train, test, dataset, "repeated_cv", split,
                               cv_seed, models, approaches, balances, output_path,
                               completed, n_jobs)
    results = _read_results(output_path)
    write_summaries(results, output_dir, prefix)
    return results


def parse_seeds(text):
    """Accept comma-separated integers or a half-open start:stop range."""
    try:
        if ":" in text:
            start, stop = map(int, text.split(":"))
            values = list(range(start, stop))
        else:
            values = [int(value) for value in text.split(",")]
        if not values or min(values) < 0 or len(set(values)) != len(values):
            raise ValueError
        return values
    except ValueError as error:
        raise ValueError("Use unique nonnegative seeds: 0:500 or 0,1,2") from error
