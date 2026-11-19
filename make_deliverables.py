#!/usr/bin/env python
"""Build the submission deliverables that depend on executed results.

Run AFTER `python run_pipeline.py` and `pytest`:

    python make_deliverables.py

Writes deliverables/Completed_Data_Analysis_Template.xlsx (the supplied template, populated from the saved outputs
and the executed test results).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))      # works without `pip install -e .`

from cvi import template_xlsx  # noqa: E402

if __name__ == "__main__":
    print("wrote", template_xlsx.build())
