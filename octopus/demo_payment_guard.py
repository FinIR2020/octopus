#!/usr/bin/env python3
"""
PaymentGuard Demo: FinTech & Payment Solutions

Aligned with Octopus Hackathon rules: https://octopushack.devpost.com/rules
- Evaluation: innovation, usability, scalability, technical implementation

"""

import os
import sys
import logging
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT and _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

DEFAULT_TOGETHER_API_KEY = ""

from octopus.payment_guard_system import PaymentGuardSystem
from octopus import report_ui
from octopus.visualize_distribution import plot_2d_from_dfs
from octopus.load_fraud_data import load_creditcard_fraud, get_fraud_data_info


def _console():
    return Console() if _RICH else None


def print_header(title: str, width: int = 70):
    c = _console()
    if c:
        c.print()
        c.print(Panel(f"[bold cyan]{title}[/bold cyan]", box=box.DOUBLE, border_style="cyan"))
        c.print()
    else:
        print("\n" + "=" * width)
        print(f" {title}")
        print("=" * width + "\n")


def create_demo_payment_data(n_samples: int = 80) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "transaction_id": [f"TXN{i:06d}" for i in range(1, n_samples + 1)],
        "amount": np.round(np.abs(np.random.lognormal(4, 1.2, n_samples)), 2),
        "merchant_category": np.random.choice(
            ["retail", "food", "travel", "online", "utility"], n_samples
        ),
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "country_code": np.random.choice(["US", "UK", "DE", "FR", "OTHER"], n_samples),
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
        "y": np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),  # is_fraud
    }
    return pd.DataFrame(data)


def _train_test_split_stratified(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    y_col: str = "y",
    seed: int = 42,
) -> tuple:
    """Stratified train/test split. Returns (train_df, test_df)."""
    if y_col not in df.columns or test_ratio <= 0 or test_ratio >= 1:
        # fallback: simple shuffle split
        n = len(df)
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        k = int(n * (1 - test_ratio))
        return df.iloc[idx[:k]].reset_index(drop=True), df.iloc[idx[k:]].reset_index(drop=True)
    try:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, stratify=df[y_col], random_state=seed
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    except Exception:
        fraud = df[df[y_col] == 1]
        normal = df[df[y_col] == 0]
        rng = np.random.default_rng(seed)
        n_fraud_test = max(0, int(len(fraud) * test_ratio))
        n_normal_test = max(0, int(len(normal) * test_ratio))
        fraud_test = fraud.sample(n=min(n_fraud_test, len(fraud)), random_state=rng) if len(fraud) else pd.DataFrame()
        normal_test = normal.sample(n=min(n_normal_test, len(normal)), random_state=rng) if len(normal) else pd.DataFrame()
        test_df = pd.concat([fraud_test, normal_test], ignore_index=True).sample(frac=1, random_state=rng).reset_index(drop=True)
        train_df = df[~df.index.isin(test_df.index)].reset_index(drop=True)
        return train_df, test_df


def demo_payment_guard(use_real_data: bool = False, real_data_max_samples: int = 5000):
    print_header("PaymentGuard: FinTech & Payment Solutions")

    c = _console()
    if c:
        c.print("[dim]PaymentGuard · multi-agent synthetic payment data[/dim]")
        c.print()
    else:
        print("PaymentGuard · multi-agent synthetic payment data")
        print()

    api_key = os.environ.get("TOGETHER_API_KEY", "").strip() or DEFAULT_TOGETHER_API_KEY
    if c:
        c.print("[green]✓[/green] Together AI API key configured — full pipeline enabled.")
    else:
        print("Using configured Together AI API key (full agent pipeline enabled).")
    print()

    api_config = {
        "api_key": api_key,
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "api_base": "https://api.together.xyz/v1",
    }

    if c:
        c.print("[bold]Step 1[/bold] Initializing PaymentGuard...")
    else:
        print("[Step 1] Initializing PaymentGuard...")
    system = PaymentGuardSystem(api_config=api_config, privacy_level="high")
    if getattr(system, "_full_pipeline_available", False):
        mode_msg = "Full CLLM pipeline (integrate_with_cllm → llm_gen → filter_synthetic_candidates)."
    else:
        mode_msg = "Simplified (statistical generation)."
    if c:
        c.print(f"  [dim]{mode_msg}[/dim]")
        c.print("  [green]Done.[/green]\n")
    else:
        print(f"  Mode: {mode_msg}")
        print("  Done.\n")
    time.sleep(0.5)

    if c:
        c.print("[bold]Step 2[/bold] Loading payment/transaction data...")
    else:
        print("[Step 2] Loading payment/transaction data...")
    if use_real_data:
        try:
            original_data = load_creditcard_fraud(max_samples=real_data_max_samples, cache=True)
            if c:
                c.print(f"  [green]Real fraud data[/green]: {len(original_data)} transactions · {get_fraud_data_info()[:60]}...")
            else:
                print(f"  Real fraud data: {len(original_data)} transactions")
                print(f"  Source: " + get_fraud_data_info())
        except FileNotFoundError as e:
            logger.error("%s", e)
            if c:
                c.print("[yellow]Falling back to demo data.[/yellow]")
            original_data = create_demo_payment_data(n_samples=80)
    else:
        original_data = create_demo_payment_data(n_samples=80)

    # Split into train (for pipeline + few-shot) and test (for web upload and prediction)
    train_data, test_data = _train_test_split_stratified(original_data, test_ratio=0.2, y_col="y", seed=42)
    if c:
        c.print(f"  [green]{len(train_data)}[/green] train, [green]{len(test_data)}[/green] test · columns: {', '.join(original_data.columns[:5])}...")
    else:
        print(f"  Train: {len(train_data)}, Test: {len(test_data)} transactions")
        print(f"  Columns: {', '.join(original_data.columns[:5])}...")
    print()
    time.sleep(0.5)

    if c:
        tab = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        for col in train_data.head(3).columns:
            tab.add_column(col)
        for _, row in train_data.head(3).iterrows():
            tab.add_row(*[str(row[col]) for col in train_data.columns])
        c.print(Panel(tab, title="Sample train transactions (first 3)", border_style="blue"))
    else:
        print("Sample train transactions (first 3):")
        print(train_data.head(3).to_string(index=False))
    print()
    time.sleep(1)

    if c:
        c.print("[bold]Step 3[/bold] Running PaymentGuard (multi-agent pipeline)...")
        c.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=c,
        ) as progress:
            task = progress.add_task("Generating synthetic data…", total=None)
            results = system.generate_payment_synthetic_data(
                dataset="demo_payment",
                X_train=train_data,
                domain="financial",
                n_synthetic=300,
                privacy_strategy="generalization",
                use_case="fraud_detection",
            )
            progress.update(task_id=task, completed=True)
    else:
        print("[Step 3] Running PaymentGuard...")
        print()
        results = system.generate_payment_synthetic_data(
            dataset="demo_payment",
            X_train=train_data,
            domain="financial",
            n_synthetic=300,
            privacy_strategy="generalization",
            use_case="fraud_detection",
        )

    try:
        print_header("Results Summary")

        synthetic_data = results.get("synthetic_data")
        if synthetic_data is not None and not synthetic_data.empty:
            if c:
                kpi = Table.grid(expand=True)
                kpi.add_column(style="cyan")
                kpi.add_column(style="green")
                kpi.add_row("Train (original) samples", str(len(train_data)))
                kpi.add_row("Synthetic samples", str(len(synthetic_data)))
                kpi.add_row("Data expansion", f"{len(synthetic_data) / len(train_data):.1f}x")
                c.print(Panel(kpi, title="Synthetic data", border_style="green"))
                tab = Table(show_header=True, header_style="bold", box=box.SIMPLE)
                for col in synthetic_data.columns:
                    tab.add_column(col)
                for _, row in synthetic_data.head(5).iterrows():
                    tab.add_row(*[str(row[col]) for col in synthetic_data.columns])
                c.print(Panel(tab, title="Sample synthetic transactions", border_style="blue"))
            else:
                print("Synthetic payment data generated")
                print(f"  Train (original): {len(train_data)} samples")
                print(f"  Synthetic: {len(synthetic_data)} samples")
                print(f"  Increase: {len(synthetic_data) / len(train_data):.1f}x")
                print()
                print("Sample synthetic transactions:")
                print(synthetic_data.head(3).to_string(index=False))
            print()
        time.sleep(1)

        privacy_risk = results.get("privacy_risk")
        if c:
            risk = privacy_risk.get("overall_risk", "unknown").upper() if privacy_risk else "UNKNOWN"
            pci = "PCI-DSS safe: [green]✓[/green]" if results.get("pci_safe", True) else "PCI-DSS safe: [red]—[/red]"
            c.print(Panel(f"Privacy risk: [bold]{risk}[/bold]\n{pci}", title="Privacy & compliance", border_style="blue"))
        else:
            if privacy_risk:
                print("Privacy risk:", privacy_risk.get("overall_risk", "unknown").upper())
            print("PCI-DSS safe:", results.get("pci_safe", True))
        print()

        report = system.generate_comprehensive_report(results)
        report_file = os.path.join(_SCRIPT_DIR, "paymentguard_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        report_rel = os.path.relpath(report_file, _PROJECT_ROOT)
        if c:
            c.print(f"[green]Report saved:[/green] {report_rel}")

        # Distribution plot (PCA 2D) for HTML report
        dist_plot_path = os.path.join(_SCRIPT_DIR, "distribution_plot.png")
        plot_path = plot_2d_from_dfs(
            train_data,
            results.get("synthetic_data"),
            dist_plot_path,
        )
        dist_filename = "distribution_plot.png" if plot_path else None

        # HTML report for browser showcase (includes distribution plot if generated)
        html_file = os.path.join(_SCRIPT_DIR, "paymentguard_report.html")
        report_ui.generate_html_report(
            results,
            html_file,
            n_original=len(train_data),
            title="PaymentGuard Report",
            distribution_plot_filename=dist_filename,
        )
        html_rel = os.path.relpath(html_file, _PROJECT_ROOT)
        if c:
            c.print(Panel(
                f"[bold green]Open in browser for full dashboard:[/bold green]\n\n  [bold]{html_rel}[/bold]\n\n"
                "[dim]macOS: open octopus/paymentguard_report.html[/dim]\n"
                "[dim]Windows: start octopus\\paymentguard_report.html[/dim]",
                title="✨ Showcase report",
                border_style="green",
                padding=(1, 2),
            ))
        else:
            print(f"Report saved: {report_rel}")
            print(f"HTML report: {html_rel}")
            print("  → Open in browser for full dashboard (open/start octopus/paymentguard_report.html).")
        print()

        if c:
            c.print("[dim]Report preview (first 20 lines):[/dim]")
            c.print(Panel(Markdown("\n".join(report.strip().split("\n")[:20]) + "\n…"), border_style="dim"))
        else:
            print("Report preview:")
            print("-" * 70)
            for line in report.strip().split("\n")[:20]:
                print(line)
            print("...")
        print()

        # Fraud Detection Agent: retrain on original + synthetic so web app uses latest model
        try:
            from octopus.fraud_detection_agent import FraudDetectionAgent
            agent = FraudDetectionAgent()
            agent.retrain(train_data, results.get("synthetic_data"), y_col="y")
            agent.save()
            if c:
                c.print("[green]FraudDetectionAgent[/green] retrained on original + synthetic and saved (use web app to predict).")
            else:
                print("FraudDetectionAgent retrained on original + synthetic and saved (use web app to predict).")
        except Exception as ea:
            logger.warning("FraudDetectionAgent retrain/save skipped: %s", ea)

        # Save train + synthetic for LLM predictor; save test set (10 rows only) for web upload
        try:
            data_dir = os.path.join(_SCRIPT_DIR, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_data.to_csv(os.path.join(data_dir, "last_original.csv"), index=False)
            if results.get("synthetic_data") is not None and not results["synthetic_data"].empty:
                results["synthetic_data"].to_csv(os.path.join(data_dir, "last_synthetic.csv"), index=False)
            test_export = test_data.head(10)
            test_export.to_csv(os.path.join(data_dir, "test_transactions.csv"), index=False)
            if c:
                c.print(f"[green]Test set saved:[/green] octopus/data/test_transactions.csv ({len(test_export)} rows) — upload to web for prediction + reason.")
            else:
                print("Test set saved: octopus/data/test_transactions.csv (%d rows) — upload to web for prediction + reason." % len(test_export))
        except Exception as es:
            logger.warning("Save last_original/last_synthetic/test_transactions skipped: %s", es)

    except Exception as e:
        logger.exception("Error during generation: %s", e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PaymentGuard Demo")
    parser.add_argument("--fast", action="store_true", help="Skip delays")
    parser.add_argument("--real-data", action="store_true", help="Use real Credit Card Fraud dataset (OpenML 42397) instead of demo data")
    parser.add_argument("--real-data-samples", type=int, default=5000, help="Max samples when using --real-data (default 5000)")
    args = parser.parse_args()
    if args.fast:
        time.sleep = lambda x: None

    c = _console()
    if c:
        c.print()
        c.print(Panel("[bold]PaymentGuard[/bold] · FinTech & Payment Solutions", box=box.DOUBLE, border_style="cyan"))
        c.print()
    else:
        print()
        print("=" * 70)
        print(" PaymentGuard: FinTech & Payment Solutions")
        print("=" * 70)
        print()

    demo_payment_guard(use_real_data=args.real_data, real_data_max_samples=args.real_data_samples)

    if c:
        c.print(Panel("[dim]End of demo[/dim]", border_style="dim"))
    else:
        print("=" * 70)
        print(" End of demo")
        print("=" * 70)
    print()


if __name__ == "__main__":
    main()
