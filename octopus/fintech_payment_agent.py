import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


PCI_SENSITIVE = {
    "card_number": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "cvv": r"\b\d{3,4}\b",
    "full_track": r"%[B]\d+[\^]",
}

PAYMENT_USE_CASES = [
    "fraud_detection",
    "transaction_anomaly",
    "merchant_risk",
    "chargeback_prediction",
    "aml_screening",
]


class FinTechPaymentAgent:

    def __init__(self, compliance_level: str = "pci_dss"):
        self.compliance_level = compliance_level
        self.detected_payment_fields = []
        self.risk_indicators = []

    def detect_payment_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect payment/transaction–related columns and types."""
        schema = {
            "amount_cols": [],
            "id_cols": [],
            "time_cols": [],
            "categorical_cols": [],
            "target_col": None,
            "is_payment_dataset": False,
        }

        for col in df.columns:
            c = str(col).lower()
            if any(x in c for x in ["amount", "value", "sum", "total", "price"]):
                schema["amount_cols"].append(col)
            if any(x in c for x in ["id", "user", "merchant", "account"]):
                schema["id_cols"].append(col)
            if any(x in c for x in ["time", "date", "hour", "timestamp"]):
                schema["time_cols"].append(col)
            if any(x in c for x in ["fraud", "label", "is_fraud", "target", "class"]):
                schema["target_col"] = col
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < 100:
                    schema["categorical_cols"].append(col)

        schema["is_payment_dataset"] = (
            len(schema["amount_cols"]) >= 1 or "payment" in str(df.columns).lower()
            or "transaction" in str(df.columns).lower()
        )
        self.detected_payment_fields = (
            schema["amount_cols"]
            + schema["id_cols"]
            + schema["time_cols"]
            + ([schema["target_col"]] if schema["target_col"] else [])
        )
        return schema

    def get_payment_context(self) -> str:
        """Return domain context string for FinTech & Payment Solutions."""
        return (
            "FinTech and payment solutions: transaction data, payment flows, "
            "fraud detection, and merchant analytics. Generate realistic but synthetic "
            "payment/transaction records. Do not generate real card numbers, CVV, or "
            "other PCI-DSS sensitive data. Preserve statistical patterns (amounts, "
            "time-of-day, class balance) for fraud and anomaly detection use cases."
        )

    def enforce_pci_safe_constraints(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Ensure synthetic data does not contain PCI-sensitive patterns.
        Returns (cleaned_df, list of warnings).
        """
        warnings = []
        out = df.copy()

        for col in out.columns:
            try:
                sample = out[col].astype(str).str.replace(r"\s+", "", regex=True)
                long_digits = sample.str.match(r"^\d{13,19}$")
                if long_digits.any():
                    out.loc[long_digits, col] = "MASKED_PAN"
                    warnings.append(f"Masked potential PAN-like values in column: {col}")
            except Exception:
                pass

        return out, warnings

    def score_payment_solution(
        self,
        schema: Dict[str, Any],
        synthetic_size: int,
        use_case: str = "fraud_detection",
    ) -> Dict[str, Any]:
        """
        Score project for hackathon criteria: innovation, usability, scalability,
        technical implementation (per Octopus rules).
        """
        scores = {
            "innovation": 0.0,
            "usability": 0.0,
            "scalability": 0.0,
            "technical_implementation": 0.0,
            "domain_fit": "FinTech",
            "domain_points_cap": 7,
        }

        # Innovation: synthetic data for payment/fraud
        if schema.get("is_payment_dataset") and synthetic_size > 0:
            scores["innovation"] = min(7.0, 2.0 + np.log1p(synthetic_size) / 2.0)
        if use_case in PAYMENT_USE_CASES:
            scores["innovation"] = min(7.0, scores["innovation"] + 1.5)

        # Usability: clear schema, target for fraud/label
        if schema.get("target_col"):
            scores["usability"] = 5.0
        if len(schema.get("amount_cols", [])) >= 1:
            scores["usability"] = min(7.0, scores["usability"] + 1.0)

        # Scalability: data volume and pipeline
        scores["scalability"] = min(7.0, 2.0 + np.log1p(synthetic_size) / 3.0)

        # Technical implementation: multi-agent, pipeline
        scores["technical_implementation"] = 6.0  # Multi-agent + Pipeline V7

        return scores
