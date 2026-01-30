#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _create_demo_payment_data(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
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


def _features_for_pca(
    df: pd.DataFrame,
    target_col: str = "y",
    train_columns: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    id_like = [c for c in df.columns if "id" in c.lower() or c == "transaction_id"]
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
    return X, y, X.columns


def _plot_2d(
    X_orig: np.ndarray,
    X_syn: np.ndarray,
    out_path: str,
    title: str = "Original vs Synthetic (PCA 2D)",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not found. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_orig[:, 0], X_orig[:, 1],
        c="#1f77b4", alpha=0.5, s=20, label="Original", edgecolors="none"
    )
    ax.scatter(
        X_syn[:, 0], X_syn[:, 1],
        c="#ff7f0e", alpha=0.5, s=20, label="Synthetic", edgecolors="none"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def run_visualization(
    *,
    n_original: int = 300,
    n_synthetic: int = 200,
    use_api: bool = False,
    out_path: Optional[str] = None,
) -> str:
    if out_path is None:
        out_path = os.path.join(_SCRIPT_DIR, "distribution_plot.png")

    logger.info("Loading / generating original data...")
    orig_df = _create_demo_payment_data(n_samples=n_original)

    if use_api:
        try:
            from octopus.payment_guard_system import PaymentGuardSystem
            api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
            if not api_key:
                api_key = "b7245f997cdb46e155f0ff8dd1cdd89fff5707109e6c2d84b4b5fb1c01f60f76"
            system = PaymentGuardSystem(
                api_config={"api_key": api_key, "api_base": "https://api.together.xyz/v1"},
                privacy_level="high",
            )
            res = system.generate_payment_synthetic_data(
                dataset="visualize",
                X_train=orig_df,
                domain="financial",
                n_synthetic=n_synthetic,
                privacy_strategy="generalization",
                use_case="fraud_detection",
            )
            syn_df = res.get("synthetic_data")
        except Exception as e:
            logger.warning("PaymentGuard failed (%s), using statistical synthetic.", e)
            syn_df = _generate_synthetic_statistical(orig_df, n_synthetic)
    else:
        logger.info("Generating synthetic data (statistical, no API)...")
        syn_df = _generate_synthetic_statistical(orig_df, n_synthetic)

    if syn_df is None or syn_df.empty:
        logger.warning("No synthetic data; plot will show original only.")
        syn_df = orig_df.head(0)

    X_orig, _, cols = _features_for_pca(orig_df, train_columns=None)
    X_syn, _, _ = _features_for_pca(syn_df, train_columns=cols)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    combined = np.vstack([X_orig.values, X_syn.values])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(combined_scaled)

    orig_scaled = scaler.transform(X_orig.values)
    syn_scaled = scaler.transform(X_syn.values)
    orig_2d = pca.transform(orig_scaled)
    syn_2d = pca.transform(syn_scaled)

    _plot_2d(
        orig_2d, syn_2d, out_path,
        title="Original vs Synthetic — PCA 2D (distribution overlap = similar data)",
    )
    return out_path


def plot_2d_from_dfs(
    orig_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    out_path: str,
    title: str = "Original vs Synthetic — PCA 2D (distribution overlap = similar data)",
) -> Optional[str]:
    if syn_df is None or syn_df.empty:
        return None
    try:
        X_orig, _, cols = _features_for_pca(orig_df, train_columns=None)
        X_syn, _, _ = _features_for_pca(syn_df, train_columns=cols)
    except Exception as e:
        logger.warning("Feature prep for plot failed: %s", e)
        return None
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("sklearn not found; skip distribution plot.")
        return None
    combined = np.vstack([X_orig.values, X_syn.values])
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(combined_scaled)
    orig_2d = pca.transform(scaler.transform(X_orig.values))
    syn_2d = pca.transform(scaler.transform(X_syn.values))
    _plot_2d(orig_2d, syn_2d, out_path, title=title)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot original and synthetic data in 2D (PCA) to show distribution similarity"
    )
    parser.add_argument("--no-api", action="store_true", help="Use statistical synthetic only (no LLM)")
    parser.add_argument("--n-original", type=int, default=300, help="Number of original samples")
    parser.add_argument("--n-synthetic", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("-o", "--output", default=None, help="Output image path (default: octopus/distribution_plot.png)")
    args = parser.parse_args()

    path = run_visualization(
        n_original=args.n_original,
        n_synthetic=args.n_synthetic,
        use_api=not args.no_api,
        out_path=args.output,
    )
    print()
    print("Done. Open the image to see overlap: original (blue) vs synthetic (orange).")
    print("  " + path)
    return path


if __name__ == "__main__":
    main()
