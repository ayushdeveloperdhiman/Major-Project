"""Dashboard data layer (no Streamlit import, so the static HTML export can reuse it).

The interactive frame is read from the WAREHOUSE (``vw_feedback_360``), i.e. the dashboard consumes the analytical
layer rather than the raw files.  Statistical outputs that are computed once on the full dataset (driver model,
bootstrap rank stability, model validation, DQ) are read from ``outputs/tables``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvi import config as C  # noqa: E402
from cvi import warehouse as W  # noqa: E402
from cvi.utils import load_metrics  # noqa: E402

FILTERS = [("Region", "region_name"), ("Channel", "channel_name"), ("Product", "product_category_name"),
           ("Issue / theme", "issue_category"), ("Feedback source", "source_name")]


def load_frame() -> pd.DataFrame:
    df = W.query("SELECT * FROM vw_feedback_360")
    seg = pd.read_parquet(C.SILVER_DIR / "mining_segments.parquet")[["feedback_id", "pc1", "pc2"]]
    df = df.merge(seg, on="feedback_id", how="left")
    df["sent"] = pd.Categorical(df.sentiment_pred, categories=C.SENTIMENT_ORDER, ordered=True)
    df["resolution_band"] = pd.Categorical(df.resolution_band, categories=C.RES_BAND_LABELS, ordered=True)
    df["revenue_tier"] = pd.Categorical(df.revenue_tier, categories=C.REVENUE_TIER_LABELS, ordered=True)
    df["repeat_purchase_flag"] = df.repeat_purchase_flag.astype(float)
    return df


def apply_filters(df: pd.DataFrame, sel: dict[str, list[str]]) -> pd.DataFrame:
    out = df
    for label, col in FILTERS:
        v = sel.get(label)
        if v:
            out = out[out[col].isin(v)]
    return out


def options(df: pd.DataFrame) -> dict[str, list[str]]:
    return {label: sorted(df[col].dropna().unique().tolist()) for label, col in FILTERS}


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(C.TABLE_DIR / f"{name}.csv")


def metrics() -> dict:
    return {k: load_metrics(k) for k in ["ingest", "transform", "eda", "sentiment", "topics", "mining", "integrate", "decision", "validate"]}


def test_summary() -> pd.DataFrame:
    p = C.VALIDATION_DIR / "test_results.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["Test ID", "Component", "Pass / Fail"])
