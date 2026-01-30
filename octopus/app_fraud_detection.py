#!/usr/bin/env python3
import os
import sys
import io
import logging
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

for _name in ("httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_agent = None
_DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "data", "fraud_detector.joblib")


def _get_agent():
    global _agent
    if _agent is not None:
        return _agent
    from octopus.fraud_detection_agent import FraudDetectionAgent
    agent = FraudDetectionAgent(model_path=_DEFAULT_MODEL_PATH)
    if os.path.isfile(agent.model_path):
        try:
            agent.load()
            _agent = agent
            return agent
        except Exception as e:
            logger.warning("Failed to load saved model: %s", e)
    logger.info("No saved model; training FraudDetectionAgent on demo data...")
    from octopus.demo_payment_guard import create_demo_payment_data
    df = create_demo_payment_data(n_samples=200)
    agent.fit(df, y_col="y")
    agent.save()
    _agent = agent
    return agent


def create_app():
    try:
        from flask import Flask, request, render_template_string, redirect, url_for
    except ImportError:
        raise ImportError("Web app requires Flask: pip install flask")

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    HTML_INDEX = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PaymentGuard · Fraud Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0c10;
      --bg-elevated: #111318;
      --surface: rgba(22, 26, 34, 0.85);
      --surface-border: rgba(255, 255, 255, 0.06);
      --text: #f0f2f5;
      --text-muted: #8b92a0;
      --accent: #6366f1;
      --accent-hover: #818cf8;
      --safe: #22c55e;
      --safe-bg: rgba(34, 197, 94, 0.12);
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.12);
      --radius: 14px;
      --radius-sm: 10px;
      --shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
      --transition: 0.2s ease;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      background-image: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(99, 102, 241, 0.15), transparent),
                        radial-gradient(ellipse 80% 50% at 100% 50%, rgba(99, 102, 241, 0.06), transparent);
    }
    .container { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.5rem; }
    .header {
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--surface-border);
    }
    .logo {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
      margin-bottom: 0.5rem;
    }
    .logo::before {
      content: '';
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--accent), #a78bfa);
      box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    }
    .sub {
      color: var(--text-muted);
      font-size: 0.95rem;
      max-width: 560px;
      line-height: 1.65;
    }
    .sub code {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85em;
      padding: 0.15em 0.45em;
      background: var(--bg-elevated);
      border-radius: 6px;
      color: var(--accent);
      border: 1px solid var(--surface-border);
    }
    .card {
      background: var(--surface);
      backdrop-filter: blur(12px);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius);
      padding: 1.75rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--shadow);
    }
    .card-title {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 1rem;
    }
    .predictor-options {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }
    .predictor-option {
      flex: 1;
      min-width: 200px;
      position: relative;
    }
    .predictor-option input { position: absolute; opacity: 0; width: 0; height: 0; }
    .predictor-option label {
      display: block;
      padding: 1rem 1.2rem;
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: var(--transition);
      background: var(--bg-elevated);
      font-weight: 500;
    }
    .predictor-option input:checked + label {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.08);
      color: var(--accent);
      box-shadow: 0 0 0 1px var(--accent);
    }
    .predictor-option label:hover { border-color: rgba(99, 102, 241, 0.4); }
    .file-zone {
      border: 2px dashed var(--surface-border);
      border-radius: var(--radius-sm);
      padding: 1.5rem;
      text-align: center;
      margin-bottom: 1.25rem;
      transition: var(--transition);
      background: var(--bg-elevated);
    }
    .file-zone:hover, .file-zone:focus-within { border-color: var(--accent); background: rgba(99, 102, 241, 0.04); }
    .file-zone input[type="file"] {
      position: absolute;
      opacity: 0;
      width: 0;
      height: 0;
    }
    .file-zone .file-label {
      display: inline-block;
      padding: 0.6rem 1.2rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
      color: var(--accent);
      background: var(--bg-elevated);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: var(--transition);
    }
    .file-zone .file-label:hover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.06);
    }
    .file-zone .file-name {
      display: block;
      margin-top: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
      word-break: break-all;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: linear-gradient(135deg, var(--accent), #7c3aed);
      color: #fff;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: var(--transition);
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .btn:hover { background: linear-gradient(135deg, var(--accent-hover), #8b5cf6); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4); }
    .btn:active { transform: translateY(0); }
    .error-box {
      background: var(--danger-bg);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: var(--danger);
      padding: 1rem 1.25rem;
      border-radius: var(--radius-sm);
      margin-bottom: 1.5rem;
      font-weight: 500;
    }
    .results-section { margin-top: 2rem; }
    .results-section h2 {
      font-size: 1.35rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .results-section h2::before {
      content: '';
      width: 4px;
      height: 1.2em;
      border-radius: 2px;
      background: linear-gradient(180deg, var(--accent), #a78bfa);
    }
    .result-count { font-weight: 400; color: var(--text-muted); font-size: 0.85em; }
    .table-wrap {
      overflow-x: auto;
      border-radius: var(--radius);
      border: 1px solid var(--surface-border);
      background: var(--bg-elevated);
      box-shadow: var(--shadow);
    }
    .table-wrap table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }
    .table-wrap thead {
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(17, 19, 24, 0.95);
      backdrop-filter: blur(8px);
    }
    .table-wrap th {
      padding: 0.85rem 1rem;
      text-align: left;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--surface-border);
      white-space: nowrap;
    }
    .table-wrap td {
      padding: 0.7rem 1rem;
      border-bottom: 1px solid var(--surface-border);
    }
    .table-wrap tbody tr { transition: var(--transition); }
    .table-wrap tbody tr:hover { background: rgba(99, 102, 241, 0.04); }
    .fraud-1 {
      color: var(--danger);
      font-weight: 600;
      padding: 0.25em 0.6em;
      border-radius: 6px;
      background: var(--danger-bg);
      display: inline-block;
    }
    .fraud-0 {
      color: var(--safe);
      font-weight: 600;
      padding: 0.25em 0.6em;
      border-radius: 6px;
      background: var(--safe-bg);
      display: inline-block;
    }
    .score { font-family: 'JetBrains Mono', monospace; font-size: 0.9em; color: var(--text-muted); }
    .reason {
      max-width: 320px;
      word-break: break-word;
      color: var(--text-muted);
      font-size: 0.85em;
      line-height: 1.5;
    }
    .reason-row { background: var(--bg-elevated); }
    .reason-cell {
      padding: 0.6rem 1rem 0.9rem;
      color: var(--text-muted);
      font-size: 0.9em;
      line-height: 1.6;
      word-break: break-word;
      border-bottom: 1px solid var(--surface-border);
      vertical-align: top;
    }
    .reason-cell:first-letter { text-transform: none; }
    .back-link {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      margin-top: 1.5rem;
      font-weight: 500;
      transition: var(--transition);
    }
    .back-link a { color: var(--accent); text-decoration: none; }
    .back-link a:hover { color: var(--accent-hover); }
    .row-count { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.75rem; }
    @media (max-width: 640px) {
      .container { padding: 1.5rem 1rem; }
      .logo { font-size: 1.35rem; }
      .predictor-options { flex-direction: column; }
      .predictor-option { min-width: 100%; }
      .file-zone { padding: 1rem; }
      .table-wrap { border-radius: var(--radius-sm); }
      .table-wrap th, .table-wrap td { padding: 0.6rem 0.75rem; font-size: 0.8rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="header">
      <h1 class="logo">PaymentGuard</h1>
      <p class="sub">Upload a CSV of transactions; LLM few-shot predicts fraud (0/1) and gives <strong>Reason</strong>. Use <code>octopus/data/test_transactions.csv</code> (10 rows) from the demo for quick runs.</p>
    </header>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <div class="card">
        <div class="card-title">CSV file</div>
        <div class="file-zone">
          <input type="file" name="file" id="file" accept=".csv" required>
          <label for="file" class="file-label">Choose file</label>
          <span class="file-name" id="file-name"></span>
        </div>
        <script>
          document.getElementById('file').addEventListener('change', function() {
            var el = document.getElementById('file-name');
            el.textContent = this.files.length ? this.files[0].name : '';
          });
        </script>
        <button type="submit" class="btn">Predict fraud (LLM)</button>
      </div>
    </form>
    {% if error %}
    <div class="error-box">{{ error }}</div>
    {% endif %}
    {% if table_html %}
    <section class="results-section">
      <h2>Results {% if result_count is not none %}<span class="result-count">({{ result_count }} transactions)</span>{% endif %}</h2>
      <div class="table-wrap">{{ table_html | safe }}</div>
      <p class="back-link"><a href="/">← Upload another file</a></p>
    </section>
    {% endif %}
  </div>
</body>
</html>
"""

    @app.route("/")
    def index():
        return render_template_string(HTML_INDEX, error=None, table_html=None, result_count=None)

    @app.route("/predict", methods=["POST"])
    def predict():
        error = None
        table_html = None
        try:
            if "file" not in request.files:
                return render_template_string(HTML_INDEX, error="No file uploaded.", table_html=None, result_count=None)
            f = request.files["file"]
            if f.filename == "" or not f.filename.lower().endswith(".csv"):
                return render_template_string(HTML_INDEX, error="Please upload a CSV file.", table_html=None, result_count=None)
            try:
                df = pd.read_csv(io.BytesIO(f.read()))
            except Exception as e:
                return render_template_string(HTML_INDEX, error="Failed to read CSV: " + str(e), table_html=None, result_count=None)
            if df.empty:
                return render_template_string(HTML_INDEX, error="CSV is empty.", table_html=None, result_count=None)
            _LAST_ORIG = os.path.join(_SCRIPT_DIR, "data", "last_original.csv")
            _LAST_SYN = os.path.join(_SCRIPT_DIR, "data", "last_synthetic.csv")
            if not os.path.isfile(_LAST_ORIG):
                return render_template_string(
                    HTML_INDEX,
                    error="LLM predictor needs original + synthetic data. Run the demo first (python octopus/run_payment_guard.py --fast), then try again.",
                    table_html=None,
                    result_count=None,
                )
            try:
                df_orig = pd.read_csv(_LAST_ORIG)
                df_syn = pd.read_csv(_LAST_SYN) if os.path.isfile(_LAST_SYN) else None
                api_key = os.environ.get("TOGETHER_API_KEY", "").strip() or "b7245f997cdb46e155f0ff8dd1cdd89fff5707109e6c2d84b4b5fb1c01f60f76"
                api_config = {"api_key": api_key, "api_base": "https://api.together.xyz/v1", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"}
                from octopus.llm_fraud_predictor import predict_fraud_llm_with_fallback
                pred, reasons = predict_fraud_llm_with_fallback(df, df_orig, df_syn, api_config, n_few_shot=6, batch_size=50)
                proba = pred.astype(float)
            except Exception as e:
                logger.exception("LLM predict failed: %s", e)
                return render_template_string(HTML_INDEX, error="LLM prediction failed: " + str(e), table_html=None, result_count=None)
            n = len(df)
            if pred is None or len(pred) != n:
                return render_template_string(HTML_INDEX, error="Prediction length mismatch.", table_html=None, result_count=None)
            df = df.copy()
            df["Fraud"] = pred
            proba_arr = proba.round(4) if hasattr(proba, "round") else proba
            if hasattr(proba_arr, "flatten"):
                proba_arr = proba_arr.flatten()
            df["Fraud_Score"] = proba_arr[:n] if len(proba_arr) >= n else list(proba_arr) + [0.0] * (n - len(proba_arr))
            reasons = list(reasons)[:n] if len(reasons) >= n else list(reasons) + [""] * (n - len(reasons))
            df["Reason"] = reasons
            def esc(s):
                if s is None or (isinstance(s, float) and pd.isna(s)):
                    return ""
                return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            def safe_int(v):
                try:
                    return int(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else 0
                except (ValueError, TypeError):
                    return 0
            data_cols = [c for c in df.columns if c != "Reason"]
            ncols = len(data_cols)
            rows = ["<table><thead><tr>" + "".join("<th>" + esc(c) + "</th>" for c in data_cols) + "</tr></thead><tbody>"]
            for _, row in df.head(500).iterrows():
                cells = []
                for c in data_cols:
                    v = row[c]
                    cls = ""
                    if c == "Fraud":
                        cls = ' class="fraud-' + str(safe_int(v)) + '"'
                    elif c == "Fraud_Score":
                        cls = ' class="score"'
                    cells.append("<td" + cls + ">" + esc(v) + "</td>")
                rows.append("<tr>" + "".join(cells) + "</tr>")
                reason_text = esc(row["Reason"]) if "Reason" in row.index else ""
                rows.append('<tr class="reason-row"><td colspan="' + str(ncols) + '" class="reason-cell">Reason: ' + reason_text + "</td></tr>")
            rows.append("</tbody></table>")
            if len(df) > 500:
                rows.append("<p class=\"row-count\">Showing first 500 rows of %d.</p>" % len(df))
            table_html = "\n".join(rows)
            return render_template_string(HTML_INDEX, error=None, table_html=table_html, result_count=n)
        except Exception as e:
            logger.exception("Unhandled error in /predict: %s", e)
            return render_template_string(
                HTML_INDEX,
                error="Internal error: " + str(e) + ". Check server logs for details.",
                table_html=None,
                result_count=None,
            )

    @app.errorhandler(500)
    def handle_500(err):
        logger.exception("Server error: %s", err)
        return render_template_string(
            HTML_INDEX,
            error="A server error occurred. Check the terminal/logs for details.",
            table_html=None,
            result_count=None,
        ), 500

    return app


def main():
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print("Fraud Detection web app: http://%s:%d" % (host, port))
    print("Upload a transaction CSV to get fraud predictions.")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
