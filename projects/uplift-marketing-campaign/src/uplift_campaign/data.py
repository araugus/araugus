"""Download, load and prepare the Hillstrom MineThatData e-mail experiment."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "hillstrom.csv"

# Original location published by Kevin Hillstrom (2008). Plain HTTP only, so the
# file is verified against a known SHA-256 after download.
SOURCE_URL = (
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

CONTROL = "No E-Mail"
ARMS = ("Womens E-Mail", "Mens E-Mail")
OUTCOMES = ("visit", "conversion", "spend")
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]


def download(path: Path = RAW_FILE, url: str = SOURCE_URL, force: bool = False) -> Path:
    """Download the raw CSV (about 4 MB) and verify its checksum."""
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed URL)
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Checksum mismatch for {url}: got {digest}, expected {SHA256}")
    path.write_bytes(content)
    return path


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(path)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment covariates as a numeric matrix (one-hot for categoricals).

    `history_segment` is dropped because it is a binned copy of `history`.
    """
    X = df[NUMERIC_FEATURES].astype(float).copy()
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    X.columns = [c.lower().replace(" ", "_") for c in X.columns]
    return X


@dataclass(frozen=True)
class UpliftData:
    """Binary-treatment view of the experiment used for modelling."""

    X: pd.DataFrame
    treatment: np.ndarray  # 1 = received the e-mail, 0 = control
    y: np.ndarray  # outcome
    raw: pd.DataFrame


def binary_setup(df: pd.DataFrame, treatment_arm: str = "Womens E-Mail", outcome: str = "visit") -> UpliftData:
    """Restrict the 3-arm experiment to `treatment_arm` vs control."""
    if treatment_arm not in ARMS:
        raise ValueError(f"treatment_arm must be one of {ARMS}")
    sub = df[df["segment"].isin([treatment_arm, CONTROL])].reset_index(drop=True)
    return UpliftData(
        X=make_features(sub),
        treatment=(sub["segment"] == treatment_arm).to_numpy(dtype=int),
        y=sub[outcome].to_numpy(),
        raw=sub,
    )
