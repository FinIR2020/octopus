import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class PrivacyGuardAgent:
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
    }
    PHI_PATTERNS = {
        'patient_id': r'\bPAT\d+\b',
        'medical_record': r'\bMRN\d+\b',
        'date_of_birth': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
    }
    
    def __init__(self, privacy_level: str = "high"):
        self.privacy_level = privacy_level
        self.detected_sensitive_columns = []
        self.privacy_risks = {}
    
    def detect_sensitive_columns(self, df: pd.DataFrame) -> List[str]:
        sensitive_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            col_data = df[col].astype(str)
            
            sensitive_keywords = [
                'id', 'name', 'email', 'phone', 'address', 'ssn',
                'patient', 'medical', 'record', 'diagnosis', 'treatment',
                'age', 'birth', 'gender', 'race', 'ethnicity'
            ]
            
            if any(keyword in col_lower for keyword in sensitive_keywords):
                sensitive_cols.append(col)
                continue
            sample_values = col_data.head(100).tolist()
            for pattern_name, pattern in {**self.PII_PATTERNS, **self.PHI_PATTERNS}.items():
                matches = sum(1 for val in sample_values if pd.notna(val) and 
                             any(char.isdigit() or char.isalpha() for char in str(val)))
                if matches > len(sample_values) * 0.1:
                    sensitive_cols.append(col)
                    break
        
        self.detected_sensitive_columns = list(set(sensitive_cols))
        logger.info(f"Detected {len(self.detected_sensitive_columns)} sensitive columns: {self.detected_sensitive_columns}")
        
        return self.detected_sensitive_columns

    def harden_synthetic_for_privacy(
        self,
        df_synthetic: pd.DataFrame,
        df_real: pd.DataFrame,
        *,
        noise_scale: float = 0.03,
        overlap_reduce_ratio: float = 0.4,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        if df_synthetic is None or df_synthetic.empty:
            return df_synthetic
        rng = np.random.default_rng(random_state)
        out = df_synthetic.copy()

        for col in out.columns:
            if col not in df_real.columns:
                continue
            if not pd.api.types.is_numeric_dtype(out[col]):
                continue
            real_std = df_real[col].std()
            if pd.isna(real_std) or real_std <= 0:
                real_std = out[col].std() or 1.0
            noise = rng.normal(0, noise_scale * real_std, size=len(out))
            out[col] = out[col].astype(float) + noise
            if (df_real[col] >= 0).all():
                out[col] = out[col].clip(lower=0)
            if pd.api.types.is_integer_dtype(df_real[col]):
                out[col] = np.round(out[col]).astype(df_real[col].dtype)
            elif df_real[col].dtype.kind == "i":
                out[col] = np.round(out[col]).astype(np.int64)

        for col in self.detected_sensitive_columns:
            if col not in out.columns or col not in df_real.columns:
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                continue
            real_unique = set(df_real[col].dropna().astype(str).unique())
            syn_unique = set(out[col].dropna().astype(str).unique())
            overlap = real_unique & syn_unique
            overlap_ratio = len(overlap) / len(real_unique) if real_unique else 0
            if overlap_ratio <= 0.1:
                continue
            col_safe = str(col).replace(" ", "_").lower()[:30]
            out[col] = [f"SYN_{col_safe}_{i + 1:06d}" for i in range(len(out))]

        logger.info("Privacy hardening applied to synthetic data (noise + overlap reduction).")
        return out
    
    def assess_privacy_risk(self, df_real: pd.DataFrame, df_synthetic: pd.DataFrame) -> Dict[str, Any]:
        risks = {
            "overall_risk": "low",
            "risks": [],
            "recommendations": []
        }
        
        for col in self.detected_sensitive_columns:
            if col in df_real.columns and col in df_synthetic.columns:
                real_unique = set(df_real[col].dropna().astype(str).unique())
                synthetic_unique = set(df_synthetic[col].dropna().astype(str).unique())
                overlap = real_unique & synthetic_unique
                overlap_count = len(overlap)
                overlap_ratio = overlap_count / len(real_unique) if len(real_unique) > 0 else 0
                if overlap_count > 0 and overlap_ratio > 0.1:
                    risks["risks"].append({
                        "type": "unique_value_leakage",
                        "column": col,
                        "overlap_ratio": overlap_ratio,
                        "severity": "high" if overlap_ratio > 0.3 else "medium"
                    })
        
        for col in df_real.columns:
            if col in df_synthetic.columns and pd.api.types.is_numeric_dtype(df_real[col]):
                real_mean = df_real[col].mean()
                synthetic_mean = df_synthetic[col].mean()
                denom = abs(real_mean) + 1e-10
                mean_diff = abs(real_mean - synthetic_mean) / denom
                if mean_diff < 0.005:
                    risks["risks"].append({
                        "type": "statistical_leakage",
                        "column": col,
                        "mean_difference": mean_diff,
                        "severity": "medium"
                    })
        
        if len(self.detected_sensitive_columns) >= 2:
            risks["risks"].append({
                "type": "combination_attack_risk",
                "columns": self.detected_sensitive_columns[:3],
                "severity": "medium"
            })
        
        high_risks = sum(1 for r in risks["risks"] if r.get("severity") == "high")
        medium_risks = sum(1 for r in risks["risks"] if r.get("severity") == "medium")
        
        if high_risks > 0:
            risks["overall_risk"] = "high"
        elif medium_risks > 2:
            risks["overall_risk"] = "medium"
        else:
            risks["overall_risk"] = "low"
        
        if risks["overall_risk"] != "low":
            risks["recommendations"].append("Apply differential privacy noise")
            risks["recommendations"].append("Remove or generalize sensitive columns")
            risks["recommendations"].append("Increase synthetic data diversity")
        
        self.privacy_risks = risks
        return risks
    
    def apply_privacy_protection(
        self,
        df: pd.DataFrame,
        strategy: str = "generalization"
    ) -> pd.DataFrame:
        df_protected = df.copy()
        if strategy == "removal":
            df_protected = df_protected.drop(columns=self.detected_sensitive_columns)
            logger.info(f"Removed {len(self.detected_sensitive_columns)} sensitive columns")
        
        elif strategy == "generalization":
            for col in self.detected_sensitive_columns:
                if col in df_protected.columns:
                    if pd.api.types.is_numeric_dtype(df_protected[col]):
                        df_protected[col] = pd.cut(
                            df_protected[col],
                            bins=5,
                            labels=False,
                            duplicates='drop'
                        )
                    else:
                        top_n = df_protected[col].value_counts().head(5).index.tolist()
                        df_protected[col] = df_protected[col].apply(
                            lambda x: x if x in top_n else "Other"
                        )
            logger.info(f"Generalized {len(self.detected_sensitive_columns)} sensitive columns")
        
        elif strategy == "noise":
            for col in self.detected_sensitive_columns:
                if col in df_protected.columns and pd.api.types.is_numeric_dtype(df_protected[col]):
                    noise = np.random.normal(0, df_protected[col].std() * 0.1, len(df_protected))
                    df_protected[col] = df_protected[col] + noise
            logger.info(f"Added noise to {len(self.detected_sensitive_columns)} sensitive columns")
        
        return df_protected
    
    def validate_privacy(self, df_synthetic: pd.DataFrame) -> Tuple[bool, str]:
        for col in df_synthetic.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['id', 'name', 'email', 'phone', 'ssn']):
                sample = df_synthetic[col].head(10).astype(str)
                if any('@' in str(val) for val in sample):
                    return False, f"Potential PII detected in column: {col}"
        
        for col in df_synthetic.columns:
            unique_ratio = df_synthetic[col].nunique() / len(df_synthetic)
            if unique_ratio > 0.9:
                return False, f"High uniqueness in {col} may indicate data leakage"
        
        return True, "Privacy validation passed"
    
    def generate_privacy_report(self) -> str:
        report = f"""
# Privacy Protection Report

## Privacy Level: {self.privacy_level.upper()}

## Detected Sensitive Columns: {len(self.detected_sensitive_columns)}
{', '.join(self.detected_sensitive_columns[:5])}

## Privacy Risk Assessment
Overall Risk: {self.privacy_risks.get('overall_risk', 'unknown').upper()}

Identified Risks: {len(self.privacy_risks.get('risks', []))}
"""
        
        for risk in self.privacy_risks.get('risks', [])[:5]:
            report += f"\n- {risk.get('type', 'unknown')}: {risk.get('severity', 'unknown')} severity"
        
        if self.privacy_risks.get('recommendations'):
            report += "\n\n## Recommendations:\n"
            for rec in self.privacy_risks['recommendations']:
                report += f"- {rec}\n"
        
        return report
