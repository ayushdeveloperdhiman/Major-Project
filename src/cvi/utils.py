"""Small shared helpers: logging, hashing, JSON I/O, stats utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config as C


def get_logger(name: str = "cvi") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Order-sensitive, dtype-insensitive content hash of a DataFrame (for reproducibility tests)."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes())
    return h.hexdigest()


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _sig12(o: Any) -> Any:
    """Round floats to 12 significant digits (removes last-bit noise from threaded reductions; keeps 1e-72 p-values)."""
    if isinstance(o, (float, np.floating)):
        return float(f"{float(o):.12g}") if np.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _sig12(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sig12(v) for v in o]
    return o


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sig12(obj), fh, indent=2, default=_jsonable, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_metrics(stage: str, metrics: dict) -> None:
    """Persist a stage's headline numbers; the workbook/dashboard read these (no hand-typed numbers)."""
    write_json(metrics, C.METRIC_DIR / f"{stage}.json")


def load_metrics(stage: str) -> dict:
    return read_json(C.METRIC_DIR / f"{stage}.json")


def environment_info() -> dict:
    import importlib
    pk = {}
    for m in ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "nltk", "vaderSentiment",
              "mlxtend", "matplotlib", "seaborn", "plotly", "streamlit", "pyarrow", "openpyxl", "docx"]:
        try:
            mod = importlib.import_module(m)
            pk[m] = getattr(mod, "__version__", "n/a")
        except Exception:  # pragma: no cover
            pk[m] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pk}


# Packages whose absence or version changes behaviour: the pipeline needs all of the first set; the dashboard needs the
# second.  (pytest, notebook and screenshot tooling are deliberately not in either: they never run in the app.)
PIPELINE_PACKAGES = {"pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "patsy", "nltk", "vaderSentiment", "mlxtend",
                     "matplotlib", "pillow", "plotly", "pyarrow", "openpyxl", "joblib"}
DASHBOARD_PACKAGES = {"pandas", "numpy", "scikit-learn", "vaderSentiment", "plotly", "streamlit", "pyarrow", "joblib"}


def environment_issues(requirements: Path | None = None, only: set[str] | None = None) -> list[str]:
    """Differences between the packages installed for THIS Python and the pins in requirements.txt (empty list = match).

    This exists to explain "wrong interpreter" problems, e.g. a bare ``streamlit`` command that belongs to Anaconda
    (no vaderSentiment, a different scikit-learn) instead of the project's venv.  Without it they surface as unrelated
    tracebacks: ``No module named 'vaderSentiment'``, ``'LogisticRegression' object has no attribute 'multi_class'``.
    """
    from importlib import metadata
    path = requirements or (C.ROOT / "requirements.txt")
    if not path.exists():
        return []
    issues = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, want = (s.strip() for s in line.split("==", 1))
        if only is not None and name not in only:
            continue
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            issues.append(f"{name} is not installed (pinned {want})")
            continue
        if have != want:
            issues.append(f"{name} {have} installed (pinned {want})")
    return issues


# ----------------------------- statistics helpers -------------------------- #
def wilson_ci(k: int | float, n: int | float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (robust for small n / extreme p)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, stat: Callable = np.mean, n_boot: int = C.N_BOOT,
                 level: float = C.CI_LEVEL, seed: int = C.SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = np.apply_along_axis(stat, 1, values[idx]) if stat is not np.mean else values[idx].mean(axis=1)
    a = (1 - level) / 2
    return (float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)))


def cramers_v(table: pd.DataFrame) -> float:
    from scipy import stats
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def fmt_inr(x: float) -> str:
    """Indian-style compact currency: ₹3.70M / ₹15.12M (millions; lakh/crore shown in docs where useful)."""
    if abs(x) >= 1e6:
        return f"₹{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"₹{x / 1e3:,.1f}K"
    return f"₹{x:,.0f}"
