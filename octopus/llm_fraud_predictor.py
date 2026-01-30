#!/usr/bin/env python3
import os
import re
import json
import logging
import urllib.request
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FRAUD_BACKGROUND_KNOWLEDGE = """
Background — why transactions are flagged fraud or not:
- Fraud indicators: unusually high amount vs user history; transaction at unusual hour or location; first-time merchant; velocity (many transactions in short time); mismatch between device/location and cardholder; PCA features (V1–V28) far from normal distribution.
- Legitimate (unfraud) indicators: moderate amount; common merchant category (e.g. retail, food); consistent hour/country; recurring pattern; PCA features close to typical non-fraud cluster.
- Industry practice: no single feature proves fraud; combination of several risk factors increases probability. Small recurring payments at known merchants are often legitimate; large one-off at odd hour in new country often flagged.
"""


def _get_fraud_background_for_prompt(use_web: bool = False) -> str:
    out = FRAUD_BACKGROUND_KNOWLEDGE.strip()
    if not use_web:
        return out
    try:
        req = urllib.request.Request(
            "https://docs.stripe.com/disputes/prevention/identifying-fraud",
            headers={"User-Agent": "PaymentGuard/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        excerpt = re.sub(r"<[^>]+>", " ", raw)[:600].strip()
        excerpt = re.sub(r"\s+", " ", excerpt)
        if len(excerpt) > 100:
            out += "\n\nSimilar industry guidance (reference): " + excerpt[:400] + "..."
    except Exception:
        pass
    return out


def _row_to_text(row: pd.Series, cols: List[str], y_col: Optional[str] = None) -> str:
    parts = []
    for c in cols:
        if c == y_col:
            continue
        v = row.get(c)
        if pd.isna(v):
            v = ""
        parts.append("%s=%s" % (c, v))
    if y_col and y_col in row.index:
        parts.append("fraud=%s" % int(row[y_col]))
    return ", ".join(parts)


def _sample_few_shot(
    df_original: pd.DataFrame,
    df_synthetic: Optional[pd.DataFrame],
    n: int,
    y_col: str = "y",
    seed: int = 42,
) -> pd.DataFrame:
    if df_synthetic is not None and not df_synthetic.empty:
        combined = pd.concat([df_original, df_synthetic], ignore_index=True)
    else:
        combined = df_original
    if len(combined) <= n:
        return combined.sample(n=min(n, len(combined)), random_state=seed)
    rng = np.random.default_rng(seed)
    fraud = combined[combined[y_col] == 1]
    normal = combined[combined[y_col] == 0]
    n1 = min(len(fraud), max(1, n // 2))
    n0 = min(len(normal), n - n1)
    if n0 <= 0:
        n0 = n - n1
    out = pd.concat([
        fraud.sample(n=n1, random_state=rng) if len(fraud) >= n1 else fraud,
        normal.sample(n=n0, random_state=rng) if len(normal) >= n0 else normal,
    ], ignore_index=True)
    return out.sample(frac=1, random_state=rng).reset_index(drop=True)


def _call_llm(prompt: str, api_config: Dict[str, str], max_tokens: int = 150) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("llm_fraud_predictor requires openai; pip install openai")
    client = OpenAI(
        api_key=api_config.get("api_key", ""),
        base_url=api_config.get("api_base", "https://api.together.xyz/v1"),
    )
    model = api_config.get("model", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text


def _parse_01_lines(text: str, n_expected: int) -> List[int]:
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"\b(0|1)\b", line)
        if m:
            out.append(int(m.group(1)))
        if len(out) >= n_expected:
            break
    return out


def _parse_prediction_reason_lines(text: str, n_expected: int) -> Tuple[List[int], List[str]]:
    preds, reasons = [], []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line or len(preds) >= n_expected:
            continue
        obj = None
        i = line.find("{")
        if i >= 0:
            depth = 0
            for j in range(i, len(line)):
                if line[j] == "{":
                    depth += 1
                elif line[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(line[i : j + 1])
                        except json.JSONDecodeError:
                            pass
                        break
        if obj is not None and isinstance(obj, dict):
            p = obj.get("prediction", 0)
            if p not in (0, 1):
                p = 1 if p else 0
            preds.append(int(p))
            reasons.append(str(obj.get("reason", "")).strip() or "")
        else:
            m = re.search(r"\b(0|1)\b", line)
            preds.append(int(m.group(1)) if m else 0)
            reasons.append("")
        if len(preds) >= n_expected:
            break
    return preds, reasons


def predict_fraud_llm(
    query_df: pd.DataFrame,
    df_original: pd.DataFrame,
    df_synthetic: Optional[pd.DataFrame],
    api_config: Dict[str, str],
    *,
    n_few_shot: int = 12,
    y_col: str = "y",
    batch_size: int = 15,
    use_web_context: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    if query_df.empty:
        return np.array([]), []
    cols = [c for c in query_df.columns if c != y_col]
    if not cols:
        cols = list(query_df.columns)
    if y_col not in df_original.columns:
        raise ValueError("df_original must have label column '%s'" % y_col)
    example_df = _sample_few_shot(df_original, df_synthetic, n_few_shot, y_col=y_col)
    example_lines = [
        _row_to_text(example_df.iloc[i], list(example_df.columns), y_col=y_col)
        for i in range(len(example_df))
    ]
    examples_blob = "\n".join(example_lines)
    background = _get_fraud_background_for_prompt(use_web=use_web_context)

    all_preds = []
    all_reasons = []
    for start in range(0, len(query_df), batch_size):
        batch = query_df.iloc[start : start + batch_size]
        query_lines = [
            _row_to_text(batch.iloc[i], cols, y_col=None)
            for i in range(len(batch))
        ]
        query_blob = "\n".join(query_lines)
        prompt = f"""You are a fraud detection expert. Use the background below and the labeled examples to predict fraud (1) or not fraud (0), and give a concrete reason for each transaction.

{background}

Labeled examples (learn from these):
{examples_blob}

For each transaction output exactly one line: {{"prediction": 0 or 1, "reason": "detailed judgment logic"}}.
In "reason" give a **detailed judgment logic** in 2–3 short sentences: (1) What patterns or values stand out (e.g. amount vs time, merchant type, PCA or other features). (2) How they relate to known fraud or legitimate behaviour (e.g. "high amount at 3am with online merchant fits card-not-present fraud; low amount in retail at noon fits normal use"). (3) Clear conclusion (fraud / not fraud). Do NOT only list column names; explain the reasoning chain. One line per transaction, in order. Example:
{{"prediction": 1, "reason": "Amount is far above typical for 3am and online merchant; this hour and channel combination is commonly abused in card-not-present fraud. PCA-style or amount/time mismatch supports that. Conclusion: fraud."}}
{{"prediction": 0, "reason": "Moderate amount, retail category, daytime hour; all within normal range for in-store use. No mismatch between amount, time and merchant. Conclusion: not fraud."}}

Transactions:
{query_blob}"""
        try:
            raw = _call_llm(prompt, api_config, max_tokens=min(700, batch_size * 90))
            preds, batch_reasons = _parse_prediction_reason_lines(raw, len(batch))
            while len(preds) < len(batch):
                preds.append(0)
                batch_reasons.append("")
            all_preds.extend(preds[: len(batch)])
            all_reasons.extend(batch_reasons[: len(batch)])
        except Exception as e:
            logger.warning("LLM call failed for batch: %s", e)
            all_preds.extend([0] * len(batch))
            all_reasons.extend([""] * len(batch))
    return np.array(all_preds[: len(query_df)]), all_reasons[: len(query_df)]


def predict_fraud_llm_with_fallback(
    query_df: pd.DataFrame,
    df_original: pd.DataFrame,
    df_synthetic: Optional[pd.DataFrame],
    api_config: Dict[str, str],
    *,
    n_few_shot: int = 12,
    y_col: str = "y",
    batch_size: int = 15,
    use_web_context: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    try:
        return predict_fraud_llm(
            query_df, df_original, df_synthetic, api_config,
            n_few_shot=n_few_shot, y_col=y_col, batch_size=batch_size,
            use_web_context=use_web_context,
        )
    except Exception as e:
        logger.warning("LLM predictor failed, returning zeros: %s", e)
        n = len(query_df)
        return np.zeros(n, dtype=int), [""] * n
