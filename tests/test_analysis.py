"""Model protocol tests; no downloads and no reliance on unpublished results."""

import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from raman.analysis import (METRICS, PAPER_TARGETS, compare_paper, evaluate, parse_seeds,
                            rank_wins, run_analysis, shap_features, summarize)
from raman.models import MODELS, PAPER_PARAMS, TorchClassifier, make_estimator

class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(n_samples=180, n_features=6, n_informative=4,
                                   weights=[0.65, 0.35], random_state=10)
        cls.X = pd.DataFrame(X, columns=list("abcdef"))
        cls.y = pd.Series(y)

    def setUp(self):
        self.output = ROOT / "tests" / f".analysis-output-{uuid4().hex}"
        self.output.mkdir()

    def tearDown(self):
        shutil.rmtree(self.output)

    def test_metrics_and_zero_predictions(self):
        scores = evaluate([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(set(scores), set(METRICS))
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["auroc"], 0.75)
        self.assertAlmostEqual(scores["brier"], 0.158125)
        self.assertEqual(evaluate([0, 1], [0.1, 0.1])["precision"], 0)

    def test_complete_paper_comparison_targets(self):
        for dataset in PAPER_PARAMS:
            for model in MODELS:
                self.assertEqual(set(PAPER_TARGETS[dataset][model]), set(METRICS))
        self.assertEqual(PAPER_TARGETS["cleveland"]["LR"]["brier"], 0.2096)
        self.assertEqual(PAPER_TARGETS["framingham"]["XGB"]["precision"], 0.2456)

    def test_exact_paper_parameters_and_pipeline_order(self):
        for dataset, parameters in PAPER_PARAMS.items():
            for model, expected in parameters.items():
                pipeline = make_estimator(dataset, model)
                self.assertEqual(list(pipeline.named_steps),
                                 ["smote", "undersample", "scale", "model"])
                for key, value in expected.items():
                    self.assertEqual(pipeline.named_steps["model"].get_params()[key], value)
                self.assertEqual(list(make_estimator(dataset, model, balanced=False).named_steps),
                                 ["scale", "model"])
        self.assertEqual(PAPER_PARAMS["cleveland"]["LR"]["C"], 0.001)
        self.assertEqual(PAPER_PARAMS["framingham"]["RF"]["n_estimators"], 75)
        self.assertEqual(PAPER_PARAMS["cleveland"]["KNN"]["weights"], "distance")

    def test_all_paper_models_predict_probabilities(self):
        for dataset in PAPER_PARAMS:
            for model in MODELS:
                with self.subTest(dataset=dataset, model=model):
                    estimator = make_estimator(dataset, model)
                    if model == "NN":
                        estimator.set_params(model__epochs=3)
                    estimator.fit(self.X.iloc[:140], self.y.iloc[:140])
                    probabilities = estimator.predict_proba(self.X.iloc[140:])
                    self.assertEqual(probabilities.shape, (40, 2))
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    self.assertTrue(np.isfinite(list(evaluate(self.y.iloc[140:],
                                                              probabilities[:, 1]).values())).all())

    def test_grid_search_resamples_only_inner_training(self):
        estimator = make_estimator("cleveland", "LR", approach="tuned")
        self.assertIsInstance(estimator, GridSearchCV)
        self.assertEqual(estimator.cv.n_splits, 5)
        sizes = []
        original = SMOTE.fit_resample

        def spy(sampler, X, y, **kwargs):
            sizes.append(len(y))
            return original(sampler, X, y, **kwargs)

        with patch.object(SMOTE, "fit_resample", spy):
            estimator.fit(self.X.iloc[:150], self.y.iloc[:150])
        # 3 candidates x 5 folds, then one refit on the outer training set.
        self.assertEqual(sizes.count(120), 15)
        self.assertEqual(sizes.count(150), 1)
        self.assertEqual(len(sizes), 16)

    def test_torch_sklearn_clone_and_determinism(self):
        estimator = TorchClassifier(epochs=3, random_state=7)
        self.assertTrue(is_classifier(estimator))
        other = clone(estimator)
        estimator.fit(self.X, self.y)
        other.fit(self.X, self.y)
        np.testing.assert_array_equal(estimator.predict_proba(self.X),
                                      other.predict_proba(self.X))
        with self.assertRaises(ValueError):
            estimator.predict_proba(np.ones((4, 2)))

    def test_tuned_torch_estimator_scoring(self):
        estimator = make_estimator("cleveland", "NN", approach="tuned")
        estimator.estimator.set_params(model__epochs=2)
        estimator.fit(self.X, self.y)
        self.assertEqual(estimator.predict_proba(self.X.iloc[:5]).shape, (5, 2))

    def test_fractional_paired_rankings_and_intervals(self):
        records = []
        for split, scores in enumerate([(0.9, 0.9), (0.8, 0.7)]):
            for model, score in zip(["LR", "RF"], scores):
                row = dict(dataset="cleveland", phase="holdout", balanced=True,
                           approach="paper", model=model, split=split, seed=split)
                row.update({metric: score for metric in METRICS})
                records.append(row)
        frame = pd.DataFrame(records)
        wins = rank_wins(frame).set_index("model")
        self.assertEqual(wins.loc["LR", "win_fraction"], 0.75)
        self.assertEqual(wins.loc["RF", "win_fraction"], 0.25)
        self.assertTrue(wins.loc["LR", "clear_majority"])
        self.assertEqual(wins.fractional_wins.sum(), 2)
        summary = summarize(frame)
        lr_auc = summary[(summary.model == "LR") & (summary.metric == "auroc")].iloc[0]
        self.assertAlmostEqual(lr_auc["mean"], 0.85)
        self.assertAlmostEqual(lr_auc["sd"], np.std([0.9, 0.8], ddof=1))
        comparison = compare_paper(summary)
        self.assertFalse(comparison.loc[(comparison.model == "LR") &
                                       (comparison.metric == "auroc"), "reproduced"].iloc[0])

    def test_parse_seed_options(self):
        self.assertEqual(parse_seeds("0:3"), [0, 1, 2])
        self.assertEqual(parse_seeds("42,7"), [42, 7])
        for text in ["0:0", "-1", "1,1", "abc", "1:2:3"]:
            with self.assertRaises(ValueError):
                parse_seeds(text)

    def test_resumption_and_outer_cv(self):
        frame = self.X.assign(target=self.y)
        kwargs = dict(models=["LR"], approaches=["paper", "tuned"], seeds=[0],
                      balances=(True, False), include_cv=True, cv_folds=2,
                      cv_repeats=1, explain=False)
        result = run_analysis({"cleveland": frame}, self.output, **kwargs)
        self.assertEqual(len(result), 12)
        self.assertEqual(set(result.phase), {"holdout", "repeated_cv"})
        self.assertEqual(set(result[result.phase == "holdout"].test_n), {36})
        with patch("raman.analysis.make_estimator", side_effect=AssertionError("Must resume")):
            resumed = run_analysis({"cleveland": frame}, self.output, **kwargs)
        pd.testing.assert_frame_equal(result, resumed)
        with self.assertRaisesRegex(ValueError, "settings changed"):
            run_analysis({"cleveland": frame}, self.output, **{**kwargs, "seeds": [1]})
        self.assertTrue((self.output / "stability_summary.csv").exists())
        self.assertTrue(any(self.output.glob("*.png")))

    def test_xgboost_shap_features(self):
        estimator = make_estimator("cleveland", "XGB")
        estimator.fit(self.X.iloc[:140], self.y.iloc[:140])
        result = shap_features(estimator, self.X.iloc[140:],
                               self.output / "shap.csv")
        self.assertEqual(set(result.feature), set(self.X.columns))
        self.assertTrue((result.mean_abs_shap >= 0).all())
        self.assertTrue(result.mean_abs_shap.is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
