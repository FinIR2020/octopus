#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _create_demo_payment_data(n_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "transaction_id": [f"TXN{i:06d}" for i in range(1, n_samples + 1)],
        "amount": np.round(np.abs(rng.lognormal(4, 1.2, n_samples)), 2),
        "merchant_category": rng.choice(
            ["retail", "food", "travel", "online", "utility"], n_samples
        ),
        "hour_of_day": rng.integers(0, 24, n_samples),
        "day_of_week": rng.integers(0, 7, n_samples),
        "country_code": rng.choice(["US", "UK", "DE", "FR", "OTHER"], n_samples),
        "device_type": rng.choice(["mobile", "desktop", "tablet"], n_samples),
        "y": rng.choice([0, 1], n_samples, p=[0.92, 0.08]),
    }
    return pd.DataFrame(data)


def _generate_synthetic_statistical(
    X_train: pd.DataFrame, n_synthetic: int, y_col: str = "y", seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_synthetic):
        row = {}
        for col in X_train.columns:
            if col == y_col:
                vals = X_train[col].unique()
                probs = X_train[col].value_counts(normalize=True).reindex(vals).fillna(0).values
                probs = probs / probs.sum()
                row[col] = rng.choice(vals, p=probs)
            elif pd.api.types.is_numeric_dtype(X_train[col]):
                mean, std = X_train[col].mean(), X_train[col].std()
                row[col] = max(0, rng.normal(mean, max(std, 1e-6)))
            else:
                vals = X_train[col].dropna().unique()
                if len(vals) == 0:
                    row[col] = ""
                else:
                    probs = X_train[col].value_counts(normalize=True).reindex(vals).fillna(0).values
                    probs = probs / probs.sum()
                    row[col] = rng.choice(vals, p=probs)
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_xy(
    df: pd.DataFrame,
    target_col: str = "y",
    drop_ids: bool = True,
    train_columns: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Index]]:
    id_like = [c for c in df.columns if "id" in c.lower() or c == "transaction_id"]
    if drop_ids and id_like:
        df = df.drop(columns=[c for c in id_like if c in df.columns], errors="ignore")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    if train_columns is not None:
        for c in train_columns:
            if c not in X.columns:
                X[c] = 0
        X = X.reindex(columns=train_columns, fill_value=0)
    return X, y, X.columns if train_columns is None else None


def _train_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    except ImportError:
        logger.warning("sklearn not found; install with: pip install scikit-learn")
        return {"accuracy": 0, "f1": 0, "auc": 0}

    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0
    return {"accuracy": acc, "f1": f1, "auc": auc}


def run_verification(
    *,
    n_real: int = 400,
    test_ratio: float = 0.3,
    train_subset_ratio: Optional[float] = None,
    n_synthetic: int = 200,
    use_api: bool = True,
) -> dict:
    logger.info("Preparing data: real payment/fraud demo...")
    full = _create_demo_payment_data(n_samples=n_real)
    train_df = full.sample(frac=1 - test_ratio, random_state=42)
    test_df = full.drop(train_df.index)
    if train_subset_ratio is not None:
        n_subset = max(20, int(len(train_df) * train_subset_ratio))
        train_small_df = train_df.sample(n=n_subset, random_state=42)
        source_for_synthetic = train_small_df
    else:
        train_small_df = train_df
        source_for_synthetic = train_df

    _, _, cols = _prepare_xy(train_df, train_columns=None)
    X_train_small, y_train_small, _ = _prepare_xy(train_small_df, train_columns=cols)
    X_test, y_test, _ = _prepare_xy(test_df, train_columns=cols)

    logger.info("(A) Training on real data only...")
    metrics_a = _train_eval(X_train_small, y_train_small, X_test, y_test)

    if use_api:
        try:
            from octopus.payment_guard_system import PaymentGuardSystem
            api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
            if not api_key:
                api_key = "b7245f997cdb46e155f0ff8dd1cdd89fff5707109e6c2d84b4b5fb1c01f60f76"
            system = PaymentGuardSystem(api_config={"api_key": api_key, "api_base": "https://api.together.xyz/v1"}, privacy_level="high")
            res = system.generate_payment_synthetic_data(
                dataset="verify", X_train=source_for_synthetic, domain="financial",
                n_synthetic=n_synthetic, privacy_strategy="generalization", use_case="fraud_detection",
            )
            synthetic_df = res.get("synthetic_data")
        except Exception as e:
            logger.warning("PaymentGuard API failed (%s), falling back to statistical synthetic.", e)
            synthetic_df = _generate_synthetic_statistical(source_for_synthetic, n_synthetic)
    else:
        logger.info("Generating synthetic data (statistical, no API)...")
        synthetic_df = _generate_synthetic_statistical(source_for_synthetic, n_synthetic)

    if synthetic_df is None or synthetic_df.empty:
        logger.warning("No synthetic data; (B) will equal (A).")
        X_train_b = X_train_small
        y_train_b = y_train_small
    else:
        X_syn, y_syn, _ = _prepare_xy(synthetic_df, train_columns=cols)
        X_train_b = pd.concat([X_train_small, X_syn], ignore_index=True)
        y_train_b = pd.concat([y_train_small, y_syn], ignore_index=True)

    logger.info("(B) Training on real + synthetic...")
    metrics_b = _train_eval(X_train_b, y_train_b, X_test, y_test)

    report = {
        "metrics_real_only": metrics_a,
        "metrics_real_plus_synthetic": metrics_b,
        "n_train_real": len(X_train_small),
        "n_synthetic": len(synthetic_df) if synthetic_df is not None else 0,
        "n_test": len(X_test),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Verify synthetic data improves financial prediction")
    parser.add_argument("--no-api", action="store_true", help="Use statistical synthetic only (no LLM API)")
    parser.add_argument("--train-subset", type=float, default=0.5, help="Use only this fraction of train to simulate data scarcity (default 0.5)")
    parser.add_argument("--n-synthetic", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--n-real", type=int, default=400, help="Total real samples")
    args = parser.parse_args()

    report = run_verification(
        n_real=args.n_real,
        test_ratio=0.3,
        train_subset_ratio=args.train_subset if args.train_subset > 0 else None,
        n_synthetic=args.n_synthetic,
        use_api=not args.no_api,
    )

    print()
    print("=" * 60)
    print("Verification: Does synthetic data improve prediction?")
    print("=" * 60)
    print(f"  Train (real): {report['n_train_real']}  |  Synthetic: {report['n_synthetic']}  |  Test: {report['n_test']}")
    print()
    a = report["metrics_real_only"]
    b = report["metrics_real_plus_synthetic"]
    print("  (A) Real only        -> Accuracy: {:.4f}  F1: {:.4f}  AUC: {:.4f}".format(a["accuracy"], a["f1"], a["auc"]))
    print("  (B) Real + Synthetic -> Accuracy: {:.4f}  F1: {:.4f}  AUC: {:.4f}".format(b["accuracy"], b["f1"], b["auc"]))
    print()
    if b["f1"] >= a["f1"] and (b["auc"] >= a["auc"] or b["accuracy"] >= a["accuracy"]):
        print("  => Synthetic data improves or matches prediction on the same test set.")
    else:
        print("  => Run with more synthetic (--n-synthetic) or with --train-subset 0.5 to see gain under data scarcity.")
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()
