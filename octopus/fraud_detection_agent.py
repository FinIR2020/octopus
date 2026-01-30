#!/usr/bin/env python3
import os
import logging
from typing import Optional, List, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "data", "fraud_detector.joblib")


def _prepare_features(
    df: pd.DataFrame,
    target_col: str = "y",
    drop_ids: bool = True,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    id_like = [c for c in df.columns if "id" in str(c).lower() or c == "transaction_id"]
    if drop_ids and id_like:
        df = df.drop(columns=[c for c in id_like if c in df.columns], errors="ignore")
    y = df[target_col].copy() if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore") if target_col in df.columns else df.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    cols = list(X.columns)
    if feature_columns is not None:
        for c in feature_columns:
            if c not in X.columns:
                X[c] = 0
        X = X.reindex(columns=feature_columns, fill_value=0)
        cols = feature_columns
    return X, y, cols


class FraudDetectionAgent:

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self._model = None
        self._feature_columns: Optional[List[str]] = None
        self._target_col = "y"

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str = "y",
    ) -> "FraudDetectionAgent":
        X, y, cols = _prepare_features(df, target_col=y_col, feature_columns=None)
        if y is None or y.isna().all():
            raise ValueError("fit requires a target column '%s' with labels" % y_col)
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("fraud_detection_agent requires scikit-learn")
        self._scaler = StandardScaler()
        self._model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._feature_columns = cols
        self._target_col = y_col
        logger.info("FraudDetectionAgent fitted on %d samples, %d features", len(X), len(cols))
        return self

    def retrain(
        self,
        df_original: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        y_col: str = "y",
    ) -> "FraudDetectionAgent":
        if df_synthetic is None or df_synthetic.empty:
            return self.fit(df_original, y_col=y_col)
        combined = pd.concat([df_original, df_synthetic], ignore_index=True)
        return self.fit(combined, y_col=y_col)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = _prepare_features(
            df, target_col=self._target_col, feature_columns=self._feature_columns
        )
        if self._model is None:
            raise RuntimeError("FraudDetectionAgent not fitted; run fit() or load() first.")
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = _prepare_features(
            df, target_col=self._target_col, feature_columns=self._feature_columns
        )
        if self._model is None:
            raise RuntimeError("FraudDetectionAgent not fitted; run fit() or load() first.")
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.model_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            import joblib
        except ImportError:
            raise ImportError("save/load requires joblib; pip install joblib")
        payload = {
            "model": self._model,
            "scaler": getattr(self, "_scaler", None),
            "feature_columns": self._feature_columns,
            "target_col": self._target_col,
        }
        joblib.dump(payload, path)
        logger.info("FraudDetectionAgent saved to %s", path)
        return path

    def load(self, path: Optional[str] = None) -> "FraudDetectionAgent":
        path = path or self.model_path
        if not os.path.isfile(path):
            raise FileNotFoundError("No saved model at %s; run fit() and save() first." % path)
        try:
            import joblib
        except ImportError:
            raise ImportError("load requires joblib; pip install joblib")
        payload = joblib.load(path)
        self._model = payload.get("model")
        self._scaler = payload.get("scaler")
        self._feature_columns = payload.get("feature_columns")
        self._target_col = payload.get("target_col", "y")
        logger.info("FraudDetectionAgent loaded from %s", path)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and self._feature_columns is not None
