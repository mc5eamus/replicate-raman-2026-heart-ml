"""Download, fingerprint, and validate the two complete-case datasets."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


SOURCES = {
    "cleveland": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
        "processed.cleveland.data",
        "processed.cleveland.data",
        "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    ),
    "framingham": (
        "https://raw.githubusercontent.com/sta210-sp20/datasets/master/framingham.csv",
        "framingham.csv",
        "2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018",
    ),
}
CLEVELAND_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
FRAMINGHAM_COLUMNS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
    "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD",
]
EXPECTED = {"cleveland": (303, 297, 137), "framingham": (4240, 3658, 557)}


class DataValidationError(ValueError):
    """Input differs from the pre-specified analytic sample; do not model."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(data_dir):
    """Cache source bytes without overwriting existing files; verify every hash."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, filename, expected_hash) in SOURCES.items():
        path = data_dir / filename
        if not path.exists():
            with urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise DataValidationError(f"{name}: downloaded SHA-256 differs; stop")
            path.write_bytes(content)
        if sha256(path) != expected_hash:
            raise DataValidationError(f"{name}: cached SHA-256 differs; stop")


def clean_dataset(name, frame):
    """Apply only the issue's cleaning rules, then enforce the data gate."""
    if name not in EXPECTED:
        raise ValueError(f"Unknown dataset: {name}")
    expected_columns = CLEVELAND_COLUMNS if name == "cleveland" else FRAMINGHAM_COLUMNS
    if list(frame.columns) != expected_columns:
        raise DataValidationError(f"{name}: unexpected columns; stop")
    raw_count, expected_count, expected_positive = EXPECTED[name]
    if len(frame) != raw_count:
        raise DataValidationError(f"{name}: expected {raw_count} raw rows, got {len(frame)}; stop")
    frame = frame.mask(frame.eq("?")).apply(pd.to_numeric, errors="raise")
    frame = frame.dropna().copy()
    if name == "cleveland":
        if not frame["num"].isin([0, 1, 2, 3, 4]).all():
            raise DataValidationError("cleveland: invalid disease code; stop")
        frame["target"] = (frame.pop("num") > 0).astype(int)
    else:
        frame = frame.drop_duplicates()
    target = "target" if name == "cleveland" else "TenYearCHD"
    if not frame[target].isin([0, 1]).all():
        raise DataValidationError(f"{name}: target is not binary; stop")
    positives = int(frame[target].sum())
    if len(frame) != expected_count or positives != expected_positive:
        raise DataValidationError(
            f"{name}: expected {expected_count} rows/{expected_positive} positives, "
            f"got {len(frame)}/{positives}; stop before modelling"
        )
    return frame.reset_index(drop=True)


def load_datasets(data_dir):
    """Validate both datasets before returning either for modelling."""
    download(data_dir)
    frames = {}
    for name, (_, filename, _) in SOURCES.items():
        kwargs = {"header": None, "names": CLEVELAND_COLUMNS} if name == "cleveland" else {}
        raw = pd.read_csv(Path(data_dir) / filename, na_values=["?"], **kwargs)
        frames[name] = clean_dataset(name, raw)
    return frames


def split_xy(name, frame):
    target = "target" if name == "cleveland" else "TenYearCHD"
    return frame.drop(columns=target), frame[target].astype(int)


def data_summary(frames, data_dir):
    return pd.DataFrame([
        {
            "dataset": name,
            "raw_rows": EXPECTED[name][0],
            "clean_rows": len(frame),
            "features": frame.shape[1] - 1,
            "positive": int(split_xy(name, frame)[1].sum()),
            "prevalence": float(split_xy(name, frame)[1].mean()),
            "sha256": sha256(Path(data_dir) / SOURCES[name][1]),
            "source": SOURCES[name][0],
        }
        for name, frame in frames.items()
    ])
