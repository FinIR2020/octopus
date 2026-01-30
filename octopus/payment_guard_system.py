import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from octopus.synthetic_pipeline_v7 import (
    MetaOptimizerV7,
    integrate_with_cllm,
    build_class_conditional_template,
    filter_synthetic_candidates,
)
from octopus.cllm.src.llm_gen import llm_gen

from octopus.domain_adapter_agent import DomainAdapterAgent
from octopus.task_coordinator_agent import TaskCoordinatorAgent
from octopus.privacy_guard_agent import PrivacyGuardAgent
from octopus.impact_evaluator_agent import ImpactEvaluatorAgent
from octopus.fintech_payment_agent import FinTechPaymentAgent

logger = logging.getLogger(__name__)


class PaymentGuardSystem:

    def __init__(
        self,
        api_config: Dict[str, str],
        privacy_level: str = "high",
        llm_serving: str = "together",
        max_parallel_tasks: int = 3,
    ):
        self.api_config = api_config
        self.llm_serving = llm_serving
        self.domain_adapter = DomainAdapterAgent()
        self.task_coordinator = TaskCoordinatorAgent(max_workers=max_parallel_tasks)
        self.meta_optimizer = MetaOptimizerV7(api_config=api_config, llm_serving=llm_serving)
        self.privacy_guard = PrivacyGuardAgent(privacy_level=privacy_level)
        self.impact_evaluator = ImpactEvaluatorAgent(domain="financial")
        self.fintech_agent = FinTechPaymentAgent(compliance_level="pci_dss")
        self._full_pipeline_available = True

    def generate_payment_synthetic_data(
        self,
        dataset: str,
        X_train: pd.DataFrame,
        domain: Optional[str] = None,
        n_synthetic: int = 500,
        privacy_strategy: str = "generalization",
        use_case: str = "fraud_detection",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate PCI-DSS safe synthetic payment/transaction data for FinTech use cases.
        """
        logger.info("=" * 60)
        logger.info("PaymentGuard: FinTech & Payment Solutions")
        logger.info("=" * 60)

        results = {
            "synthetic_data": None,
            "privacy_report": None,
            "impact_report": None,
            "privacy_risk": None,
            "social_value": None,
            "payment_scores": None,
            "pci_safe": True,
        }

        # Step 1: FinTech payment schema
        logger.info("Step 1: Detecting payment/transaction schema...")
        payment_schema = self.fintech_agent.detect_payment_schema(X_train)
        logger.info(f"  Payment dataset: {payment_schema.get('is_payment_dataset')}")
        logger.info(f"  Amount columns: {payment_schema.get('amount_cols')}")
        logger.info(f"  Target column: {payment_schema.get('target_col')}")

        # Step 2: Sensitive info (PII/PCI)
        logger.info("Step 2: Detecting sensitive information...")
        sensitive_cols = self.privacy_guard.detect_sensitive_columns(X_train)

        logger.info(f"Step 3: Applying privacy protection ({privacy_strategy})...")
        X_train_protected = self.privacy_guard.apply_privacy_protection(
            X_train, strategy=privacy_strategy
        )

        if domain is None:
            domain = self.domain_adapter.detect_domain(
                X_train_protected, dataset_name=dataset
            )
        # Force FinTech context
        domain_context = self.fintech_agent.get_payment_context()
        logger.info("Step 4: Domain: FinTech & Payment Solutions")

        logger.info("Step 5: Generating synthetic payment data...")
        y_col = "y" if "y" in X_train_protected.columns else "is_fraud"
        dist_report = self.meta_optimizer.analyze_distribution(
            X_train_protected,
            y_col=y_col,
        )

        base_template = kwargs.get("base_template", None)
        if base_template is None:
            base_template = f"""You are a synthetic data generator for FinTech and payment solutions.
Your goal is to produce realistic transaction/payment data that preserves statistical properties for fraud detection and analytics.
Do NOT generate real card numbers, CVV, or any PCI-DSS sensitive data. Use masked or synthetic identifiers only.

example data: {{data}}

Output MUST be a valid JSON array and NOTHING else.
{{format_instructions}}

DO NOT COPY THE EXAMPLES; generate new, realistic payment/transaction samples with correct labels."""

        api_key = (self.api_config or {}).get("api_key") or ""
        use_full_pipeline = (
            api_key
            and str(api_key).strip()
            and str(api_key).strip().lower() != "placeholder"
        )

        if use_full_pipeline:
            synthetic_data = self._generate_synthetic_data_full_pipeline(
                X_train_protected=X_train_protected,
                dataset=dataset,
                base_template=base_template,
                n_synthetic=n_synthetic,
                y_col=y_col,
                **kwargs,
            )
        else:
            optimized_prompt = self.meta_optimizer.optimize_prompt(
                base_template, dist_report, domain_context
            )
            synthetic_data = self._generate_synthetic_data(
                X_train_protected, optimized_prompt, n_synthetic, **kwargs
            )

        # PCI-DSS safety
        logger.info("Step 6: Enforcing PCI-DSS safe constraints...")
        synthetic_data, pci_warnings = self.fintech_agent.enforce_pci_safe_constraints(
            synthetic_data
        )
        results["pci_safe"] = len(pci_warnings) == 0
        for w in pci_warnings:
            logger.warning(f"  {w}")

        # Privacy hardening: reduce statistical/unique-value overlap with real data
        logger.info("Step 6b: Privacy hardening (noise + overlap reduction)...")
        synthetic_data = self.privacy_guard.harden_synthetic_for_privacy(
            synthetic_data, X_train, noise_scale=0.03, overlap_reduce_ratio=0.4
        )

        results["synthetic_data"] = synthetic_data

        logger.info("Step 7: Assessing privacy risks...")
        privacy_risk = self.privacy_guard.assess_privacy_risk(X_train, synthetic_data)
        results["privacy_risk"] = privacy_risk

        logger.info("Step 8: Evaluating impact (FinTech & Payment)...")
        impact_metrics = self.impact_evaluator.evaluate_research_impact(
            original_size=len(X_train),
            synthetic_size=len(synthetic_data),
            use_case="fraud_detection",
        )
        social_value = self.impact_evaluator.calculate_social_value(
            impact_metrics, domain="financial"
        )
        results["social_value"] = social_value

        # Payment solution scoring (innovation, usability, scalability, technical)
        payment_scores = self.fintech_agent.score_payment_solution(
            payment_schema, len(synthetic_data), use_case=use_case
        )
        results["payment_scores"] = payment_scores

        logger.info("Step 9: Generating reports...")
        results["privacy_report"] = self.privacy_guard.generate_privacy_report()
        results["impact_report"] = self.impact_evaluator.generate_impact_report(
            impact_metrics, social_value
        )

        logger.info("=" * 60)
        logger.info("PaymentGuard completed")
        logger.info(f"  Synthetic data: {len(synthetic_data)} samples")
        logger.info(f"  PCI-DSS safe: {results['pci_safe']}")
        logger.info("=" * 60)

        return results

    def _generate_synthetic_data_full_pipeline(
        self,
        X_train_protected: pd.DataFrame,
        dataset: str,
        base_template: str,
        n_synthetic: int,
        y_col: str = "y",
        **kwargs,
    ) -> pd.DataFrame:
        """Full pipeline (synthetic_pipeline_v7 + cllm under octopus): integrate_with_cllm -> llm_gen -> filter_synthetic_candidates."""
        # Ensure training data has column "y" for CLLM
        if y_col != "y" and y_col in X_train_protected.columns:
            X_train_evo = X_train_protected.copy()
            X_train_evo["y"] = X_train_evo[y_col]
        else:
            X_train_evo = X_train_protected.copy()
            if "y" not in X_train_evo.columns and y_col in X_train_evo.columns:
                X_train_evo["y"] = X_train_evo[y_col]

        api_config = dict(self.api_config or {})
        api_config.setdefault("api_base", "https://api.together.xyz/v1")
        api_config.setdefault("api_type", "together")
        model = api_config.get("model", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        oversample_factor = float(kwargs.get("oversample_factor", 2.0))
        temperature = float(kwargs.get("temperature", 0.9))
        max_tokens = int(kwargs.get("max_tokens", 2048))
        ic_samples = int(kwargs.get("ic_samples", 10))
        seed = int(kwargs.get("seed", 42))

        try:
            _, generator_template, format_instructions = integrate_with_cllm(
                dataset=dataset,
                X_train=X_train_evo,
                api_config=api_config,
                base_template=base_template,
                llm_serving=self.llm_serving,
                domain="financial",
            )
        except Exception as e:
            logger.warning("integrate_with_cllm failed, falling back to simplified: %s", e)
            dist_report = self.meta_optimizer.analyze_distribution(
                X_train_protected, y_col=y_col
            )
            optimized_prompt = self.meta_optimizer.optimize_prompt(
                base_template, dist_report, "financial"
            )
            return self._generate_synthetic_data(
                X_train_protected, optimized_prompt, n_synthetic, **kwargs
            )

        p1 = (X_train_evo["y"] == 1).mean() if "y" in X_train_evo.columns else 0.5
        n1_keep = int(round(n_synthetic * p1))
        n0_keep = n_synthetic - n1_keep
        n0_cand = max(n0_keep, int(np.ceil(n0_keep * oversample_factor)))
        n1_cand = max(n1_keep, int(np.ceil(n1_keep * oversample_factor)))

        df0_examples = X_train_evo[X_train_evo["y"] == 0] if "y" in X_train_evo.columns else X_train_evo
        df1_examples = X_train_evo[X_train_evo["y"] == 1] if "y" in X_train_evo.columns else X_train_evo
        if df0_examples.empty:
            df0_examples = X_train_evo
        if df1_examples.empty:
            df1_examples = X_train_evo
        ic0 = min(ic_samples, len(df0_examples))
        ic1 = min(ic_samples, len(df1_examples))

        def _gen(label: int, n_cand: int, ex_df: pd.DataFrame, ic: int) -> pd.DataFrame:
            tmpl = build_class_conditional_template(
                generator_template, target_label=label, y_col="y"
            )
            return llm_gen(
                prompt=None,
                generator_template=tmpl,
                format_instructions=format_instructions,
                example_df=ex_df,
                llm_serving=self.llm_serving,
                api_details=api_config,
                n_samples=n_cand,
                temperature=temperature,
                model=model,
                n_processes=min(5, max(1, n_cand // 20)),
                ic_samples=ic,
                max_tokens=max_tokens,
            )

        try:
            df0_cand = _gen(0, n0_cand, df0_examples, ic0)
            df1_cand = _gen(1, n1_cand, df1_examples, ic1)
        except Exception as e:
            logger.warning("llm_gen failed, falling back to simplified: %s", e)
            optimized_prompt = self.meta_optimizer.optimize_prompt(
                base_template,
                self.meta_optimizer.analyze_distribution(X_train_protected, y_col=y_col),
                "financial",
            )
            return self._generate_synthetic_data(
                X_train_protected, optimized_prompt, n_synthetic, **kwargs
            )

        df_candidates = pd.concat([df0_cand, df1_cand], ignore_index=True)
        if df_candidates.empty:
            return self._generate_synthetic_data(
                X_train_protected, base_template, n_synthetic, **kwargs
            )

        if "y" in df_candidates.columns:
            df0_syn = filter_synthetic_candidates(
                df_candidates[df_candidates["y"] == 0],
                df_real_train=X_train_evo,
                y_col="y",
                n_keep=n0_keep,
                random_state=seed,
            )
            df1_syn = filter_synthetic_candidates(
                df_candidates[df_candidates["y"] == 1],
                df_real_train=X_train_evo,
                y_col="y",
                n_keep=n1_keep,
                random_state=seed,
            )
            df_syn = pd.concat([df0_syn, df1_syn], ignore_index=True)
        else:
            df_syn = filter_synthetic_candidates(
                df_candidates,
                df_real_train=X_train_evo,
                y_col="y",
                n_keep=n_synthetic,
                random_state=seed,
            )
        df_syn = df_syn.head(n_synthetic)
        # When using real data (e.g. Credit Card with V1–V28), LLM candidates may all be filtered out
        # (schema/range mismatch). Fall back to statistical generation so samples > 0.
        if df_syn.empty:
            logger.warning(
                "All LLM candidates filtered out (e.g. real-data schema/range mismatch). "
                "Falling back to statistical synthetic generation."
            )
            optimized_prompt = self.meta_optimizer.optimize_prompt(
                base_template,
                self.meta_optimizer.analyze_distribution(X_train_protected, y_col=y_col),
                "financial",
            )
            return self._generate_synthetic_data(
                X_train_protected, optimized_prompt, n_synthetic, **kwargs
            )
        return df_syn

    def _generate_synthetic_data(
        self,
        X_train: pd.DataFrame,
        optimized_prompt: str,
        n_synthetic: int,
        **kwargs
    ) -> pd.DataFrame:
        """Generate synthetic rows preserving distributions (simplified)."""
        synthetic_rows = []
        y_col = "y" if "y" in X_train.columns else "is_fraud"
        # ID-like columns: do not resample from real to avoid 100% overlap → high privacy risk
        id_like = [
            c for c in X_train.columns
            if c != y_col
            and (
                ("transaction" in str(c).lower() and "id" in str(c).lower())
                or (str(c).lower().endswith("_id") and X_train[c].nunique() >= max(20, len(X_train) // 2))
            )
        ]
        for i in range(n_synthetic):
            row = {}
            for col in X_train.columns:
                if col == y_col:
                    vals = X_train[col].unique()
                    probs = X_train[col].value_counts(normalize=True).reindex(vals).fillna(0).values
                    probs = probs / probs.sum()
                    row[col] = np.random.choice(vals, p=probs)
                elif col in id_like:
                    # Generate new synthetic IDs so they do not overlap with real (keeps privacy risk low)
                    if "transaction" in str(col).lower():
                        row[col] = f"TXN_SYN{i + 1:06d}"
                    else:
                        row[col] = f"SYN_{col}_{i + 1:06d}"
                elif pd.api.types.is_numeric_dtype(X_train[col]):
                    mean, std = X_train[col].mean(), X_train[col].std()
                    row[col] = max(0, np.random.normal(mean, max(std, 1e-6)))
                else:
                    vals = X_train[col].dropna().unique()
                    if len(vals) == 0:
                        row[col] = ""
                    else:
                        probs = X_train[col].value_counts(normalize=True).reindex(vals).fillna(0).values
                        probs = probs / probs.sum()
                        row[col] = np.random.choice(vals, p=probs)
            synthetic_rows.append(row)
        return pd.DataFrame(synthetic_rows)

    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Report: synthetic data, PCI safety, privacy risk."""
        syn = results.get("synthetic_data")
        n_syn = len(syn) if syn is not None else 0
        report = f"""
# PaymentGuard Report

## Synthetic data
- Samples generated: {n_syn}
- PCI-DSS safe: {results.get('pci_safe', True)}

## Privacy
- Overall risk: {results.get('privacy_risk', {}).get('overall_risk', 'unknown')}

---
Generated by PaymentGuard (synthetic pipeline v7 multi-agent framework).
"""
        return report
