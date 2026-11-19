#!/usr/bin/env python
"""Run the complete Customer Voice Intelligence pipeline.

Usage
-----
    python run_pipeline.py                  # all stages (ingest -> ... -> validate)
    python run_pipeline.py --from eda       # resume from a stage
    python run_pipeline.py --only decision figures
    python run_pipeline.py --list           # list stage names

Outputs are written to data/ (bronze, silver, gold warehouse) and outputs/ (tables, figures, models, metrics,
data-quality and validation evidence).  The run is deterministic (seed 42).
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))      # works without `pip install -e .`

from cvi.utils import PIPELINE_PACKAGES, environment_issues  # noqa: E402


def check_environment() -> None:
    """Fail fast, in plain English, when this is not the project's Python (instead of a traceback mid-import)."""
    issues = environment_issues(only=PIPELINE_PACKAGES)
    missing = [i for i in issues if "not installed" in i]
    if missing:
        sys.exit(f"This Python ({sys.executable}) is missing packages the pipeline needs:\n  - " + "\n  - ".join(missing)
                 + "\nUse the project's environment:   ./.venv/bin/python run_pipeline.py"
                 + "\nor install the pinned packages:  python -m pip install -r requirements.txt")
    if issues:
        print("NOTE: package versions differ from requirements.txt; results are only guaranteed identical in the pinned environment:\n  - "
              + "\n  - ".join(issues) + "\n")


check_environment()
from cvi.pipeline import STAGES, run_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="ingest", choices=STAGES, help="first stage to run")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    a = ap.parse_args()
    if a.list:
        print("\n".join(f"{i + 1:>2}. {s}" for i, s in enumerate(STAGES)))
        return
    run_all(start=a.start, only=a.only)


if __name__ == "__main__":
    main()
