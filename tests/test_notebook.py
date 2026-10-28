"""Contract tests for the standalone notebook; no repository model imports."""

import ast
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "raman2026_reproduction.ipynb"


def read_notebook(path=NOTEBOOK):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def source(cell):
    return "".join(cell["source"])


class NotebookProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = read_notebook()
        cls.namespace = {"__name__": "notebook_contract"}
        tags = {
            "configuration", "imports", "parameters", "torch-model",
            "pipeline", "metrics", "paper-targets", "rank-wins",
        }
        # Execute definitions, not downloads, training loops, or plotting.
        for cell in cls.notebook["cells"]:
            if tags.intersection(cell["metadata"].get("tags", [])):
                tree = ast.parse(source(cell))
                definitions = [
                    node for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                         ast.ClassDef, ast.Assign, ast.AnnAssign))
                ]
                # Cells mix definitions with analysis; only selected constants
                # can be evaluated without their preceding dataset cells.
                allowed_assignments = {
                    "START", "SEED", "N_SPLITS", "MODELS", "COLORS",
                    "PAPER_PARAMS", "METRICS", "PAPER_VALUES",
                }
                definitions = [
                    node for node in definitions
                    if not isinstance(node, ast.Assign)
                    or all(isinstance(t, ast.Name) and t.id in allowed_assignments
                           for t in node.targets)
                ]
                module = ast.Module(body=definitions, type_ignores=[])
                exec(compile(module, "<notebook definitions>", "exec"), cls.namespace)
        cls.namespace["torch"].set_num_threads(1)

    def test_only_allowed_imports_and_focused_cells(self):
        allowed = {
            "os", "random", "time", "pathlib", "hashlib", "urllib",
            "numpy", "pandas", "matplotlib", "seaborn", "torch",
            "IPython", "imblearn", "sklearn", "xgboost", "shap",
        }
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = source(cell)
            self.assertLessEqual(len(code.splitlines()), 40)
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], allowed)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    self.assertIn((node.module or "").split(".")[0], allowed)
            self.assertNotRegex(code, r"sys\.path|importlib|subprocess")
        self.assertEqual(self.namespace["SEED"], 42)
        self.assertEqual(self.namespace["N_SPLITS"], 50)

    def test_reported_parameters_and_pipeline_order(self):
        make = self.namespace["make_pipeline"]
        for dataset in ["Cleveland", "Framingham"]:
            for model in self.namespace["MODELS"]:
                pipeline = make(dataset, model)
                self.assertEqual(list(pipeline.named_steps),
                                 ["smote", "undersample", "scale", "model"])
                self.assertEqual(list(make(dataset, model, balanced=False).named_steps),
                                 ["scale", "model"])
                for key, value in self.namespace["PAPER_PARAMS"][dataset].get(model, {}).items():
                    self.assertEqual(pipeline.named_steps["model"].get_params()[key], value)
        self.assertEqual(make("Cleveland", "LR").get_params()["model__C"], 0.001)
        self.assertEqual(make("Framingham", "RF").get_params()["model__n_estimators"], 75)
        self.assertEqual(make("Cleveland", "KNN").get_params()["model__weights"], "distance")
        self.assertEqual(make("Framingham", "NN").get_params()["model__epochs"], 100)

    def test_inner_cv_resampling_is_fold_local(self):
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=150, n_features=6, n_informative=4,
                                   weights=[0.7, 0.3], random_state=42)
        sampler = self.namespace["SMOTE"]
        original = sampler.fit_resample
        sizes = []

        def spy(instance, features, labels, **kwargs):
            sizes.append(len(labels))
            return original(instance, features, labels, **kwargs)

        search = self.namespace["GridSearchCV"](
            self.namespace["make_pipeline"]("Cleveland", "LR"),
            {"model__C": [0.001, 0.01]}, scoring="roc_auc",
            cv=self.namespace["StratifiedKFold"](5, shuffle=True, random_state=42),
            n_jobs=1, error_score="raise",
        )
        with patch.object(sampler, "fit_resample", spy):
            search.fit(X[:120], y[:120])
        self.assertEqual(sizes.count(96), 10)
        self.assertEqual(sizes.count(120), 1)
        self.assertEqual(len(sizes), 11)
        self.assertEqual(search.predict_proba(X[120:]).shape, (30, 2))
        self.assertEqual(len(sizes), 11)  # Prediction must not resample.

    def test_nn_is_cloneable_and_deterministic(self):
        from sklearn.base import clone

        np = self.namespace["np"]
        X = np.random.default_rng(42).normal(size=(30, 6))
        y = np.arange(30) % 2
        model = self.namespace["TorchClassifier"](epochs=3)
        other = clone(model)
        model.fit(X, y)
        other.fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), other.predict_proba(X))
        np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1, atol=1e-6)
        self.assertEqual(len(model.losses_), 3)

    def test_metrics_and_paper_targets(self):
        scores = self.namespace["evaluate"]([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(scores["Accuracy"], 0.75)
        self.assertEqual(scores["AUROC"], 0.75)
        self.assertAlmostEqual(scores["Brier"], 0.158125)
        self.assertEqual(self.namespace["evaluate"]([0, 1], [0.1, 0.1])["Precision"], 0)
        for models in self.namespace["PAPER_VALUES"].values():
            self.assertEqual(set(models), set(self.namespace["MODELS"]))
            self.assertTrue(all(len(values) == 7 for values in models.values()))
        self.assertEqual(self.namespace["PAPER_VALUES"]["Cleveland"]["LR"][-1], 0.2096)
        self.assertEqual(self.namespace["PAPER_VALUES"]["Framingham"]["XGB"][1], 0.2456)

    def test_tied_rankings_share_one_win(self):
        pd = self.namespace["pd"]
        frame = pd.DataFrame({
            "Dataset": ["example"] * 4, "Split": [0, 0, 1, 1],
            "Balanced": [True] * 4, "Model": ["LR", "RF"] * 2,
            "AUROC": [0.9, 0.9, 0.8, 0.7],
        })
        ranked = self.namespace["fractional_wins"](frame)
        self.assertEqual(ranked.groupby("Split")["Win share"].sum().tolist(), [1, 1])
        self.assertEqual(ranked.groupby("Model")["Win share"].mean()["LR"], 0.75)


class NotebookOutputTests(unittest.TestCase):
    def test_executed_outputs_and_sections(self):
        notebook = read_notebook(os.environ.get("RAMAN_EXECUTED_NOTEBOOK", NOTEBOOK))
        markdown = "\n".join(source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")
        for section in range(11):
            self.assertRegex(markdown, rf"(?m)^#{{1,3}} {section}\. ")
        images = 0
        executed = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            executed += 1
            self.assertEqual(cell["execution_count"], executed)
            for output in cell["outputs"]:
                self.assertNotEqual(output["output_type"], "error")
                images += "image/png" in output.get("data", {})
                text = "".join(output.get("text", []))
                text += "".join(output.get("data", {}).get("text/plain", []))
                text += "".join(output.get("data", {}).get("text/html", []))
                self.assertNotRegex(text, r"/(?:home|tmp|Users|opt|usr)/|[A-Z]:\\")
            if set(cell["metadata"].get("tags", [])) & {
                "paper-targets", "numeric-claims", "conclusions",
            }:
                self.assertTrue(any("text/html" in output.get("data", {})
                                    for output in cell["outputs"]))
        self.assertGreaterEqual(images, 32)
        final = next(c for c in notebook["cells"] if "conclusions" in c["metadata"].get("tags", []))
        output_text = "".join("".join(o.get("text", [])) for o in final["outputs"])
        match = re.search(r"End-to-end notebook runtime: ([0-9.]+) minutes", output_text)
        self.assertIsNotNone(match)
        self.assertLess(float(match.group(1)), 15)
        self.assertIn("CPU budget (<15 minutes): PASS", output_text)


if __name__ == "__main__":
    unittest.main()
