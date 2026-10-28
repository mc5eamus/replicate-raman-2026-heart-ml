"""Fold-local models and a small, explicitly specified hyperparameter search."""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBClassifier

MODELS = ("LR", "RF", "KNN", "XGB", "NN")
PAPER_PARAMS = {
    "framingham": {
        "LR": dict(C=0.01, penalty="l2", solver="liblinear"),
        "RF": dict(criterion="gini", max_depth=8, max_features="sqrt", n_estimators=75),
        "KNN": dict(metric="manhattan", n_neighbors=15, weights="uniform"),
        "XGB": dict(colsample_bytree=0.5, learning_rate=0.01, max_depth=3,
                    min_child_weight=1, n_estimators=50, subsample=0.5),
    },
    "cleveland": {
        "LR": dict(C=0.001, penalty="l2", solver="liblinear"),
        "RF": dict(criterion="entropy", max_depth=12, max_features="sqrt", n_estimators=200),
        "KNN": dict(metric="manhattan", n_neighbors=15, weights="distance"),
        "XGB": dict(colsample_bytree=0.5, learning_rate=0.1, max_depth=3,
                    min_child_weight=3, n_estimators=50, subsample=0.8),
    },
}


class TorchClassifier(ClassifierMixin, BaseEstimator):
    """Binary CPU MLP: 32/16 ReLU units, Adam, BCE, 100 full-batch epochs.

    The paper does not specify the architecture. This deterministic surrogate
    uses a fixed 0.5 decision threshold and does not use held-out early stopping.
    """

    def __init__(self, hidden_sizes=(32, 16), learning_rate=0.001,
                 epochs=100, weight_decay=0.0, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from torch import nn

        X, y = check_X_y(X, y, dtype=np.float32)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("TorchClassifier requires both binary labels 0 and 1.")
        self.n_features_in_ = X.shape[1]
        torch.set_num_threads(1)
        # fork_rng avoids changing the caller's PyTorch random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            layers = []
            width = self.n_features_in_
            for size in self.hidden_sizes:
                layers.extend([nn.Linear(width, size), nn.ReLU()])
                width = size
            layers.append(nn.Linear(width, 1))
            self.network_ = nn.Sequential(*layers)
            optimizer = torch.optim.Adam(
                self.network_.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            inputs = torch.from_numpy(X)
            target = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            loss_fn = nn.BCEWithLogitsLoss()
            self.network_.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = loss_fn(self.network_(inputs), target)
                loss.backward()
                optimizer.step()
        self.network_.eval()
        return self

    def predict_proba(self, X):
        import torch

        check_is_fitted(self, "network_")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training.")
        with torch.no_grad():
            positive = torch.sigmoid(self.network_(torch.from_numpy(X))).numpy().ravel()
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def compact_grid(dataset, model):
    """Two-to-four candidates, including the paper point; not its unknown grid.

    RF varies tree count, KNN varies weighting, XGB varies learning rate and
    child weight, and NN varies learning rate. Other paper settings stay fixed.
    """
    params = PAPER_PARAMS[dataset].get(model, {})
    grids = {
        "LR": {"C": sorted({0.001, 0.01, 0.1, params.get("C", 0.01)})},
        "RF": {"n_estimators": sorted({75, 200})},
        "KNN": {"weights": ["uniform", "distance"]},
        "XGB": {"learning_rate": [0.01, 0.1], "min_child_weight": [1, 3]},
        "NN": {"learning_rate": [0.001, 0.01]},
    }
    return {f"model__{key}": values for key, values in grids[model].items()}


def make_estimator(dataset, model, seed=42, balanced=True, approach="paper",
                   n_jobs=1, inner_folds=5):
    """Construct an unfitted pipeline; resampling is never outside CV folds."""
    if dataset not in PAPER_PARAMS or model not in MODELS:
        raise ValueError(f"Unknown dataset/model: {dataset}/{model}")
    if approach not in ("paper", "tuned"):
        raise ValueError("approach must be paper or tuned")
    constructors = {
        "LR": lambda: LogisticRegression(random_state=seed, max_iter=2000),
        "RF": lambda: RandomForestClassifier(random_state=seed, n_jobs=1),
        "KNN": KNeighborsClassifier,
        "XGB": lambda: XGBClassifier(random_state=seed, n_jobs=1,
                                     objective="binary:logistic", eval_metric="logloss"),
        "NN": lambda: TorchClassifier(random_state=seed),
    }
    estimator = constructors[model]()
    estimator.set_params(**PAPER_PARAMS[dataset].get(model, {}))
    steps = []
    if balanced:
        # Default SMOTE reaches 1:1; the requested undersampler then preserves
        # that ratio (a no-op), because no intermediate ratios were specified.
        steps.extend([
            ("smote", SMOTE(random_state=seed)),
            ("undersample", RandomUnderSampler(random_state=seed)),
        ])
    steps.extend([("scale", StandardScaler()), ("model", estimator)])
    pipeline = Pipeline(steps)
    if approach == "paper":
        return pipeline
    return GridSearchCV(
        pipeline, compact_grid(dataset, model), scoring="roc_auc",
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed),
        n_jobs=n_jobs, refit=True, error_score="raise",
    )
