#!/usr/bin/env python3
import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
_CACHE_PATH = os.path.join(_DATA_DIR, "creditcard.csv")


def load_creditcard_fraud(
    max_samples: Optional[int] = None,
    subsample_seed: int = 42,
    cache: bool = True,
) -> pd.DataFrame:
    df = _load_raw(cache)
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "y"})
    if "transaction_id" not in df.columns:
        df.insert(0, "transaction_id", [f"TXN{i:06d}" for i in range(1, len(df) + 1)])
    if max_samples is not None and len(df) > max_samples:
        rng = np.random.default_rng(subsample_seed)
        fraud = df[df["y"] == 1]
        normal = df[df["y"] == 0]
        n_fraud = min(len(fraud), max(1, max_samples // 10))
        n_normal = max_samples - n_fraud
        if len(normal) >= n_normal and len(fraud) >= n_fraud:
            df = pd.concat([
                normal.sample(n=n_normal, random_state=rng),
                fraud.sample(n=n_fraud, random_state=rng),
            ], ignore_index=True)
            df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
        else:
            df = df.sample(n=max_samples, random_state=rng)
        df["transaction_id"] = [f"TXN{i:06d}" for i in range(1, len(df) + 1)]
    return df


def _load_raw(cache: bool) -> pd.DataFrame:
    if cache and os.path.isfile(_CACHE_PATH):
        logger.info("Loading cached credit card fraud data: %s", _CACHE_PATH)
        return pd.read_csv(_CACHE_PATH)
    try:
        from sklearn.datasets import fetch_openml
        logger.info("Fetching Credit Card Fraud dataset from OpenML (id=42397)...")
        data = fetch_openml(data_id=42397, as_frame=True, parser="auto")
        df = data.frame
        if "class" in df.columns and "Class" not in df.columns:
            df = df.rename(columns={"class": "Class"})
        if cache:
            os.makedirs(_DATA_DIR, exist_ok=True)
            df.to_csv(_CACHE_PATH, index=False)
            logger.info("Cached to %s", _CACHE_PATH)
        return df
    except Exception as e:
        logger.warning("fetch_openml failed: %s", e)
        if cache and os.path.isfile(_CACHE_PATH):
            return pd.read_csv(_CACHE_PATH)
        raise FileNotFoundError(
            "Could not download Credit Card Fraud dataset. Install: pip install scikit-learn, then retry. "
            "Or download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and save as octopus/data/creditcard.csv (columns: Time, V1–V28, Amount, Class)."
        ) from e


def get_fraud_data_info() -> str:
    return (
        "Credit Card Fraud Detection (OpenML 42397 / Kaggle mlg-ulb/creditcardfraud): "
        "European cardholders, Sept 2013; 284,807 transactions, 492 frauds; "
        "features V1–V28 (PCA), Time, Amount; target Class (0/1)."
    )
