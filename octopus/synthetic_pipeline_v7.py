import re
import time
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:
    try:
        from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
    except ImportError:
        class ResponseSchema:
            def __init__(self, name: str, description: str):
                self.name = name
                self.description = description
        class StructuredOutputParser:
            def __init__(self, response_schemas):
                self.response_schemas = response_schemas
            @classmethod
            def from_response_schemas(cls, response_schemas):
                return cls(response_schemas)
            def get_format_instructions(self) -> str:
                schema_strs = [f'"{s.name}": {s.description}' for s in self.response_schemas]
                return "Output should be a JSON object with the following fields:\n{" + ", ".join(schema_strs) + "}"

logger = logging.getLogger(__name__)


def _has_required_placeholders(text: str) -> bool:
    if text is None:
        return False
    t = str(text)
    has_data = ("{data}" in t) or ("{{data}}" in t)
    has_fmt = ("{format_instructions}" in t) or ("{{format_instructions}}" in t)
    return bool(has_data and has_fmt)


class MetaOptimizerV7:
    def __init__(self, api_config: Dict[str, str], llm_serving: str = "together"):
        self.api_config = api_config
        self.llm_serving = llm_serving
        self.client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            api_key = self.api_config.get("api_key")
            base_url = self.api_config.get("api_base", "https://api.together.xyz/v1")
            return OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return None

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        model = self.api_config.get("model", "meta-llama/Llama-3-70b-chat-hf")
        if not self.client:
            return ""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=2048,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Meta-Optimizer LLM call failed (attempt {attempt+1}/3): {e}")
                time.sleep(2)
        return ""

    def analyze_distribution(self, df: pd.DataFrame, *, y_col: str = "y") -> str:
        stats_report: List[str] = []
        stats_report.append(f"Dataset Size: {len(df)} samples, {len(df.columns)} columns.")
        if y_col in df.columns:
            try:
                vc = df[y_col].value_counts(normalize=True)
                bal = ", ".join([f"{k}: {v:.1%}" for k, v in vc.items()])
                stats_report.append(f"Class Balance ({y_col}): {bal}")
            except Exception:
                pass
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats_report.append("\nFeature Distributions:")
        for col in df.columns:
            if col == y_col:
                continue
            if col in numeric_cols:
                s = df[col]
                try:
                    q01, q05, q50, q95, q99 = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).tolist()
                    stats_report.append(
                        f"- {col} (Numeric): mean={s.mean():.3f}, std={s.std():.3f}, "
                        f"q01={q01:.3f}, q05={q05:.3f}, q50={q50:.3f}, q95={q95:.3f}, q99={q99:.3f}"
                    )
                except Exception:
                    stats_report.append(f"- {col} (Numeric): mean={s.mean():.3f}, std={s.std():.3f}, min={s.min():.3f}, max={s.max():.3f}")
            else:
                try:
                    vc = df[col].value_counts(normalize=True).head(5)
                    top_cats = ", ".join([f"{k} ({v:.1%})" for k, v in vc.items()])
                    stats_report.append(f"- {col} (Categorical): Top values: {top_cats}")
                except Exception:
                    stats_report.append(f"- {col} (Categorical): unable to summarize")
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr()
                pairs: List[Tuple[str, str, float]] = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        c_val = float(corr.iloc[i, j])
                        if np.isfinite(c_val) and abs(c_val) > 0.3:
                            pairs.append((numeric_cols[i], numeric_cols[j], c_val))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs:
                    stats_report.append("\nKey Correlations (abs > 0.3):")
                    for a, b, v in pairs[:5]:
                        stats_report.append(f"- {a} ~ {b}: corr={v:.2f}")
            except Exception:
                pass
        return "\n".join(stats_report)

    def optimize_prompt(self, base_template: str, distribution_report: str, domain: str) -> str:
        meta_prompt = f"""
You are an expert Prompt Engineer specializing in Synthetic Tabular Data Generation.
Optimize the baseline prompt using the statistical audit below.

### GOAL
Return an improved "Final Generator Prompt" that produces realistic, diverse samples while preserving distributions and correlations.

### BASELINE PROMPT TEMPLATE
```
{base_template}
```

### DATA DISTRIBUTION AUDIT (x_train DNA)
{distribution_report}

### DOMAIN CONTEXT
{domain}

### RULES
- Keep the placeholders `{{data}}` and `{{format_instructions}}` EXACTLY (do not rename them).
- Return ONLY the optimized prompt text (no explanations).

### OPTIMIZED PROMPT:
"""
        messages = [
            {"role": "system", "content": "You are a master of prompt engineering for data science."},
            {"role": "user", "content": meta_prompt},
        ]
        optimized = self._call_llm(messages, temperature=0.3).strip()
        if "```" in optimized:
            matches = re.findall(r"```(?:\w+)?\n?(.*?)\n?```", optimized, re.DOTALL)
            if matches:
                optimized = matches[0].strip()
        return optimized


def build_class_conditional_template(
    generator_template: str, *, y_col: str = "y", target_label: int,
) -> str:
    t = "" if generator_template is None else str(generator_template)
    constraint = f"""

### CLASS-CONDITIONAL CONSTRAINT (STRICT)
Generate ONLY samples where `{y_col}` == {int(target_label)}.
Every object in the output JSON array MUST set `{y_col}` to {int(target_label)}.
If you are unsure about a row's label, discard it and generate a different row.
"""
    return t + constraint


def _is_integer_like_series(s: pd.Series) -> bool:
    try:
        x = pd.to_numeric(s.dropna(), errors="coerce")
        if x.empty:
            return False
        frac = np.abs(x - np.round(x))
        return bool((frac < 1e-6).mean() >= 0.98)
    except Exception:
        return False


def infer_constraints(
    df_real: pd.DataFrame, *, y_col: str = "y", max_unique_for_enum: int = 30,
) -> Dict[str, Dict[str, Any]]:
    constraints: Dict[str, Dict[str, Any]] = {}
    for col in df_real.columns:
        if col == y_col:
            continue
        s = df_real[col]
        col_info: Dict[str, Any] = {}
        is_num = pd.api.types.is_numeric_dtype(s)
        nunique = int(s.dropna().nunique())
        col_info["is_numeric"] = bool(is_num)
        col_info["n_unique"] = nunique
        if nunique > 0 and nunique <= max_unique_for_enum:
            try:
                col_info["allowed_values"] = sorted(s.dropna().unique().tolist(), key=lambda x: str(x))
            except Exception:
                col_info["allowed_values"] = None
        else:
            col_info["allowed_values"] = None
        if is_num:
            try:
                q01, q99 = s.quantile([0.01, 0.99]).tolist()
                col_info["q01"], col_info["q99"] = float(q01), float(q99)
            except Exception:
                col_info["q01"] = col_info["q99"] = None
            col_info["integer_like"] = _is_integer_like_series(s)
        else:
            col_info["q01"] = col_info["q99"] = col_info["integer_like"] = False
        constraints[col] = col_info
    return constraints


def _encode_for_proxy(
    X_train: pd.DataFrame, X_cand: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_e = X_train.copy()
    X_cand_e = X_cand.copy()
    cols = list(X_train_e.columns)
    X_cand_e = X_cand_e.reindex(columns=cols)
    for col in cols:
        if pd.api.types.is_numeric_dtype(X_train_e[col]):
            X_train_e[col] = pd.to_numeric(X_train_e[col], errors="coerce")
            X_cand_e[col] = pd.to_numeric(X_cand_e[col], errors="coerce")
        else:
            combo = pd.concat([X_train_e[col].astype(str), X_cand_e[col].astype(str)], axis=0)
            codes, _ = pd.factorize(combo, sort=True)
            X_train_e[col] = codes[: len(X_train_e)]
            X_cand_e[col] = codes[len(X_train_e) :]
    return X_train_e.fillna(0), X_cand_e.fillna(0)


def filter_synthetic_candidates(
    df_candidates: pd.DataFrame,
    *,
    df_real_train: pd.DataFrame,
    y_col: str = "y",
    n_keep: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    if df_candidates is None or df_candidates.empty:
        return pd.DataFrame()
    df_cand = df_candidates.copy()
    if y_col not in df_cand.columns:
        return df_cand.head(n_keep)
    real_cols = list(df_real_train.columns)
    if not [c for c in real_cols if c not in df_cand.columns]:
        df_cand = df_cand[real_cols]
    try:
        rows_with_lists = df_cand.map(lambda x: isinstance(x, list)).any(axis=1)
        df_cand = df_cand[~rows_with_lists]
    except Exception:
        pass
    constraints = infer_constraints(df_real_train, y_col=y_col)
    for col, info in constraints.items():
        if info.get("is_numeric"):
            df_cand[col] = pd.to_numeric(df_cand[col], errors="coerce")
            if info.get("integer_like"):
                df_cand[col] = np.round(df_cand[col])
    mask_valid = pd.Series(True, index=df_cand.index)
    for col, info in constraints.items():
        allowed = info.get("allowed_values")
        if allowed is not None and len(allowed) > 0:
            mask_valid &= df_cand[col].isin(allowed) | df_cand[col].isna()
        else:
            q01, q99 = info.get("q01"), info.get("q99")
            if info.get("is_numeric") and q01 is not None and q99 is not None:
                mask_valid &= (df_cand[col] >= q01) & (df_cand[col] <= q99)
    df_cand = df_cand[mask_valid]
    df_cand = df_cand.dropna(axis=0, how="any")
    if df_cand.empty:
        return df_candidates.head(n_keep)
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return df_cand.head(n_keep)
    df_real = df_real_train.dropna(axis=0, how="any").copy()
    if y_col not in df_real.columns:
        return df_cand.head(n_keep)
    y_real = df_real[y_col]
    X_real = df_real.drop(columns=[y_col])
    y_cand = df_cand[y_col]
    X_cand = df_cand.drop(columns=[y_col])
    X_real_e, X_cand_e = _encode_for_proxy(X_real, X_cand)
    proxy = LogisticRegression(max_iter=2000, random_state=random_state)
    proxy.fit(X_real_e, y_real)
    proba = proxy.predict_proba(X_cand_e)
    classes = list(proxy.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = y_cand.map(class_to_idx).fillna(0).astype(int).to_numpy()
    scores = proba[np.arange(len(proba)), idx]
    df_scored = df_cand.copy()
    df_scored["_v7_score"] = scores
    df_scored = df_scored.sort_values("_v7_score", ascending=False)
    df_scored = df_scored.drop(columns=["_v7_score"])
    return df_scored.head(n_keep)


def integrate_with_cllm(
    dataset: str,
    X_train: pd.DataFrame,
    api_config: Dict[str, str],
    base_template: Optional[str] = None,
    llm_serving: str = "together",
    domain: Optional[str] = None,
) -> Tuple[str, str, str]:
    domain_mapping = {
        "compas": "criminal recidivism", "adult": "census data", "seer": "cancer survival",
        "cutract": "prostate cancer", "support": "survival prediction", "maggic": "heart failure",
        "covid": "covid-19 patient data", "drug": "drug consumption", "fraud": "credit card fraud",
        "higgs": "higgs boson", "mimic": "ICU patient data",
    }
    if not domain:
        domain = domain_mapping.get(dataset.lower(), "general data patterns")
    response_schemas = []
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            desc = f"numeric value for {col}"
        else:
            try:
                unique_vals = X_train[col].dropna().unique()
                if len(unique_vals) <= 20:
                    unique_str = ", ".join([str(x) for x in sorted(unique_vals, key=lambda x: str(x))])
                    desc = f"categorical value for {col} (allowed values: [{unique_str}])"
                else:
                    desc = f"categorical value for {col}"
            except Exception:
                desc = f"categorical value for {col}"
        response_schemas.append(ResponseSchema(name=col, description=desc))
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    if not base_template:
        base_template = f"""You are a synthetic data generator.
Your goal is to produce data which mirrors the given examples in feature and label distributions, while producing diverse, realistic samples.
I will give you real examples first.
Leverage your knowledge about {domain} to generate realistic but diverse samples.
IMPORTANT: Sometimes you will be asked to generate samples ONLY for a specific target class.
If the prompt says to generate only class 0 or only class 1, you MUST set the label accordingly for every row.
example data: {{data}}
Output MUST be a valid JSON array (e.g. [{{"col": value, ...}}, ...]) and NOTHING else.
Do not include any explanation, markdown, or code fences.
{{format_instructions}}
DO NOT COPY THE EXAMPLES; generate new, realistic samples with correct labels."""
    optimizer = MetaOptimizerV7(api_config=api_config, llm_serving=llm_serving)
    dist_report = optimizer.analyze_distribution(X_train, y_col="y" if "y" in X_train.columns else "y")
    optimized_prompt = optimizer.optimize_prompt(base_template, dist_report, domain)
    if not _has_required_placeholders(optimized_prompt):
        logger.warning("v7 optimized prompt missing required placeholders; reverting to baseline.")
        optimized_prompt = base_template
    final_prompt = optimized_prompt.replace("{{data}}", "{data}").replace("{{format_instructions}}", "{format_instructions}")
    return final_prompt, final_prompt, format_instructions
