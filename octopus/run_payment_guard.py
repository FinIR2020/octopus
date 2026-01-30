#!/usr/bin/env python3
import os
import sys

def _project_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(this_dir)

def _check_deps():
    missing = []
    try:
        import pandas
        import numpy
    except ImportError:
        missing.append("pandas numpy scikit-learn")
    try:
        import openai
    except ImportError:
        missing.append("openai")
    if not missing:
        return
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    print("Missing dependencies:", ", ".join(missing))
    print()
    print("On macOS/Linux with Homebrew Python (PEP 668), use a virtual environment:")
    print("  python3 -m venv paymentguard")
    print("  source paymentguard/bin/activate   # Windows: paymentguard\\Scripts\\activate")
    print("  pip install -r octopus/requirements.txt")
    if os.path.isfile(req):
        print("  # or: pip install -r", req)
    print()
    print("Then run again: python octopus/run_payment_guard.py --fast")
    sys.exit(1)

def main():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    _check_deps()

    from octopus.demo_payment_guard import main as demo_main
    demo_main()

    report_abs = os.path.normpath(os.path.join(root, "octopus", "paymentguard_report.md"))
    html_abs = os.path.normpath(os.path.join(root, "octopus", "paymentguard_report.html"))
    if os.path.isfile(report_abs):
        report_rel = os.path.relpath(report_abs, root)
        print()
        print("Report:", report_rel)
    if os.path.isfile(html_abs):
        html_rel = os.path.relpath(html_abs, root)
        print("Showcase (open in browser):", html_rel)

if __name__ == "__main__":
    main()
