# PaymentGuard: FinTech & Payment Solutions

Octopus Hackathon 2026

Multi-agent system for PCI-DSS safe synthetic payment data and fraud detection.

---

## Core features

1. **Few-shot synthetic fraud data generation**  
   Uses prompt-based context-learning: the LLM sees a small set of real (and optionally synthetic) transaction rows and generates new rows that match the same distribution. The pipeline enforces schema alignment, value ranges, and PCI-DSS safety (no real PAN/CVV). Class-conditional generation and filtering keep fraud/non-fraud balance controllable.

2. **Explainable fraud detection**  
   Few-shot LLM prediction outputs Fraud (0/1), a score, and a **Reason** (2–3 sentences of judgment logic) per transaction. Every decision is interpretable: what stands out, how it relates to fraud/legitimate behavior, and a clear conclusion. The web app displays Fraud, Fraud_Score, and Reason in the results table.

3. **Multi-agent pipeline**  
   FinTech Payment Agent (schema + PCI-DSS), Privacy Guard (PII/PCI detection and protection), Synthetic Pipeline V7 (meta-optimization, class-conditional generation, filter and rank), and Impact Evaluator (business impact and reports). One run produces PCI-safe synthetic data and an HTML report; the web app then uses original + synthetic data as few-shot sources for the fraud predictor.

---

## Quick run

From project root:

- **Windows:** `python octopus\run_payment_guard.py --fast`
- **macOS / Linux:** `python octopus/run_payment_guard.py --fast`

Opens `octopus/paymentguard_report.html` after run. See **RUN_DEMO.md** for details. please make sure that you have the togetherai key in environment.

---

## Features

- **FinTech Payment Agent** — Schema detection, PCI-DSS constraints, payment scoring
- **Privacy Guard Agent** — PII/PCI detection and protection
- **Synthetic Pipeline V7** — Class-conditional generation and filtering
- **Impact Evaluator** — Business impact and social value

Demo splits data into train/test (80/20). Train set drives synthesis and few-shot; test set is saved as `octopus/data/test_transactions.csv` (10 rows) for the web app.

---

## Web app: LLM fraud prediction

1. Run the demo once to generate `last_original.csv` and `last_synthetic.csv`.
2. Start: `python -m octopus.app_fraud_detection`
3. Open http://127.0.0.1:5000 and upload `octopus/data/test_transactions.csv`.

Predictions show Fraud (0/1), Fraud_Score, and Reason (judgment logic) on separate rows.

---

## Optional commands

**Verify synthetic utility (A vs B):**
```bash
python -m octopus.verify_synthetic_utility --no-api
python -m octopus.verify_synthetic_utility
```

**Distribution plot (PCA 2D):**
```bash
python -m octopus.visualize_distribution --no-api
```
Output: `octopus/distribution_plot.png`.

---

## Dependencies

```bash
pip install -r octopus/requirements.txt
```
Python 3.8+, pandas, numpy, scikit-learn, openai. Set `TOGETHER_API_KEY` to override bundled key.

---

## Project layout (submission)

```
octopus/
├── README.md
├── RUN_DEMO.md
├── requirements.txt
├── demo_payment_guard.py
├── run_payment_guard.py
├── run_payment_guard.sh
├── payment_guard_system.py
├── synthetic_pipeline_v7.py
├── cllm/src/llm_gen.py
├── fintech_payment_agent.py
├── privacy_guard_agent.py
├── impact_evaluator_agent.py
├── domain_adapter_agent.py
├── task_coordinator_agent.py
├── report_ui.py
├── fraud_detection_agent.py
├── llm_fraud_predictor.py
├── app_fraud_detection.py
├── run_fraud_web.py
├── verify_synthetic_utility.py
├── visualize_distribution.py
├── load_fraud_data.py
├── THE_STORY_FINTECH.md
├── PITCH_SCRIPT_FINTECH.md
└── SUBMISSION_FINTECH_7PTS.md
