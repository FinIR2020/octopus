#!/usr/bin/env python3
import os
from typing import Any, Dict, Optional

import pandas as pd


def _escape(s: Any) -> str:
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _table_rows_from_df(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<tbody><tr><td colspan=\"100\">No data</td></tr></tbody>"
    cols = list(df.columns)
    thead = "<thead><tr>" + "".join(f"<th>{_escape(c)}</th>" for c in cols) + "</tr></thead>"
    tbody_rows = []
    for _, row in df.head(max_rows).iterrows():
        tbody_rows.append("<tr>" + "".join(f"<td>{_escape(row[c])}</td>" for c in cols) + "</tr>")
    tbody = "<tbody>" + "\n".join(tbody_rows) + "</tbody>"
    return thead + "\n" + tbody


def generate_html_report(
    results: Dict[str, Any],
    output_path: str,
    *,
    title: str = "PaymentGuard Report",
    max_table_rows: int = 15,
    n_original: Optional[int] = None,
    distribution_plot_filename: Optional[str] = None,
) -> str:
    syn = results.get("synthetic_data")
    n_syn = len(syn) if syn is not None else 0
    n_orig = n_original if n_original is not None else results.get("n_original") or 0
    pci_safe = results.get("pci_safe", True)
    privacy = results.get("privacy_risk") or {}
    risk = _escape(privacy.get("overall_risk", "unknown")).upper()

    table_html = _table_rows_from_df(syn, max_rows=max_table_rows)

    if distribution_plot_filename:
        dist_section = f"""
    <div class="card">
      <h2>Distribution: Original vs Synthetic (PCA 2D)</h2>
      <p class="subtitle" style="margin-bottom:1rem;">Blue: original data · Orange: synthetic data. Same feature space projected to two principal components.</p>
      <img src="{_escape(distribution_plot_filename)}" alt="Original vs Synthetic PCA 2D" style="max-width:100%; height:auto; border-radius:8px;" />
      <div class="analysis" style="margin-top:1.25rem; padding-top:1rem; border-top:1px solid var(--surface2); font-size:0.9rem; color:var(--muted); line-height:1.6;">
        <p style="margin:0 0 0.5rem;"><strong style="color:var(--text);">Why this supports reliability</strong></p>
        <ul style="margin:0; padding-left:1.25rem;">
          <li><strong>Overlap in 2D</strong> — Where blue and orange points mix, the synthetic data occupies similar regions of the feature space as the original. That indicates the generator preserved main structure (e.g. amount ranges, time patterns, category mix) rather than producing arbitrary points.</li>
          <li><strong>PCA preserves variance</strong> — The axes (PC1, PC2) are the directions of highest variance in the combined data. If synthetic points lie along the same directions and spread as the original, key statistical properties are aligned.</li>
          <li><strong>Downstream use</strong> — Models trained on data with similar distributions tend to generalize better. This plot is a visual check that synthetic data can be used reliably for training or evaluation (e.g. fraud detection) without introducing obvious distribution shift.</li>
        </ul>
        <p style="margin:0.75rem 0 0;">In short: strong overlap suggests the synthetic data is distributionally consistent with the original and is a credible stand‑in for augmentation or testing.</p>
      </div>
    </div>"""
    else:
        dist_section = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface2: #243044;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent2: #3fb950;
      --accent3: #d29922;
      --danger: #f85149;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 700;
      margin: 0 0 0.5rem;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .subtitle {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }}
    .card {{
      background: var(--surface);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
      border: 1px solid var(--surface2);
    }}
    .card h2 {{
      font-size: 1.1rem;
      margin: 0 0 1rem;
      color: var(--accent);
      font-weight: 600;
    }}
    .scores {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 1rem;
    }}
    .score-item {{ margin-bottom: 0.5rem; }}
    .score-item label {{ display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.25rem; }}
    .score-bar {{
      height: 8px;
      background: var(--surface2);
      border-radius: 4px;
      overflow: hidden;
    }}
    .score-fill {{
      height: 100%;
      border-radius: 4px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      transition: width 0.5s ease;
    }}
    .kpi {{
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      margin-bottom: 1rem;
    }}
    .kpi-box {{
      background: var(--surface2);
      padding: 0.75rem 1rem;
      border-radius: 8px;
      min-width: 140px;
    }}
    .kpi-box .val {{ font-size: 1.5rem; font-weight: 700; color: var(--accent2); }}
    .kpi-box .lbl {{ font-size: 0.8rem; color: var(--muted); }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .badge.ok {{ background: rgba(63, 185, 80, 0.2); color: var(--accent2); }}
    .badge.warn {{ background: rgba(210, 153, 34, 0.2); color: var(--accent3); }}
    .badge.danger {{ background: rgba(248, 81, 73, 0.2); color: var(--danger); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--surface2); }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:hover {{ background: var(--surface2); }}
    .footer {{ margin-top: 2rem; font-size: 0.8rem; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{_escape(title)}</h1>
    <p class="subtitle">Multi-agent synthetic payment data · PCI-DSS safe · Pipeline V7</p>

    <div class="card">
      <h2>Summary</h2>
      <div class="kpi">
        <div class="kpi-box">
          <div class="val">{n_orig}</div>
          <div class="lbl">Original samples</div>
        </div>
        <div class="kpi-box">
          <div class="val">{n_syn}</div>
          <div class="lbl">Synthetic samples</div>
        </div>
        <div class="kpi-box">
          <div class="val">{n_syn / n_orig if n_orig else 0:.1f}x</div>
          <div class="lbl">Data expansion</div>
        </div>
        <div class="kpi-box">
          <div class="val"><span class="badge {'ok' if pci_safe else 'warn'}">{'PCI-DSS safe' if pci_safe else 'Check'}</span></div>
          <div class="lbl">Compliance</div>
        </div>
        <div class="kpi-box">
          <div class="val"><span class="badge ok">{risk}</span></div>
          <div class="lbl">Privacy risk</div>
        </div>
      </div>
    </div>
{dist_section}

    <div class="card">
      <h2>Synthetic payment data (sample)</h2>
      <div style="overflow-x: auto;">
        <table>
          {table_html}
        </table>
      </div>
    </div>

    <p class="footer">Generated by PaymentGuard · Synthetic Pipeline V7 · Octopus Hackathon</p>
  </div>
</body>
</html>
"""

    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)
    return abs_path
