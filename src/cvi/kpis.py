"""KPI definitions - the single source of truth (pandas), reconciled against the SQL views in tests.

Definitions follow the supplied *Metric Guidance* sheet:

    Negative Sentiment Rate = negative feedback / total feedback
    Repeat Purchase Rate    = records with repeat flag / total records
    Average Resolution Time = mean hours to resolution
    Revenue at Risk         = revenue associated with negative-sentiment records
    Topic Share             = feedback assigned to a topic / total feedback

Extras used by the decision layer: Net Sentiment Score (positive% - negative%), average CSAT/rating,
SLA-breach rate.  All arithmetic is done in float64 (narrow integer flag columns can overflow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .utils import wilson_ci


def _f(s: pd.Series) -> pd.Series:
    return s.astype("float64")


def overall_kpis(df: pd.DataFrame, sent_col: str = "sent_pred") -> dict:
    n = len(df)
    neg = df[sent_col] == "Negative"
    pos = df[sent_col] == "Positive"
    rev = _f(df["customer_revenue_inr"])
    lo, hi = wilson_ci(int(neg.sum()), n)
    rlo, rhi = wilson_ci(int(_f(df["repeat_purchase_flag"]).sum()), n)
    return {
        "feedback_count": int(n),
        "negative_count": int(neg.sum()),
        "negative_sentiment_rate": float(neg.mean()),
        "negative_rate_ci": [float(lo), float(hi)],
        "positive_sentiment_rate": float(pos.mean()),
        "neutral_sentiment_rate": float((df[sent_col] == "Neutral").mean()),
        "net_sentiment_score": float(100 * (pos.mean() - neg.mean())),
        "repeat_purchase_rate": float(_f(df["repeat_purchase_flag"]).mean()),
        "repeat_rate_ci": [float(rlo), float(rhi)],
        "avg_resolution_hours": float(_f(df["resolution_time_hours"]).mean()),
        "avg_csat": float(_f(df["csat_score"]).mean()),
        "avg_rating": float(_f(df["rating"]).mean()),
        "total_revenue_inr": float(rev.sum()),
        "revenue_at_risk_inr": float(rev[neg].sum()),
        "revenue_at_risk_share": float(rev[neg].sum() / rev.sum()),
        "sla_breach_rate": float(_f(df["sla_breach"]).mean()),
        "critical_delay_rate": float(_f(df["critical_delay"]).mean()),
        "negative_avg_resolution_hours": float(_f(df.loc[neg, "resolution_time_hours"]).mean()),
        "non_negative_avg_resolution_hours": float(_f(df.loc[~neg, "resolution_time_hours"]).mean()),
        "negative_repeat_rate": float(_f(df.loc[neg, "repeat_purchase_flag"]).mean()),
        "positive_repeat_rate": float(_f(df.loc[pos, "repeat_purchase_flag"]).mean()),
        "negative_sla_breach_rate": float(_f(df.loc[neg, "sla_breach"]).mean()),
    }


def kpi_table(df: pd.DataFrame, by: str | list[str], sent_col: str = "sent_pred", min_n: int = 1) -> pd.DataFrame:
    """KPI table for any grouping, with Wilson CIs on proportions and normal-approx CIs on means."""
    d = df.copy()
    d["_neg"] = (d[sent_col] == "Negative").astype("float64")
    d["_pos"] = (d[sent_col] == "Positive").astype("float64")
    d["_rep"] = _f(d["repeat_purchase_flag"])
    d["_rev"] = _f(d["customer_revenue_inr"])
    d["_rar"] = d["_rev"] * d["_neg"]
    d["_res"] = _f(d["resolution_time_hours"])
    d["_sla"] = _f(d["sla_breach"])
    total_rev, total_rar, total_n = d["_rev"].sum(), d["_rar"].sum(), len(d)
    g = d.groupby(by, observed=True)
    out = g.agg(feedback_count=("_neg", "size"), negative_count=("_neg", "sum"), positive_count=("_pos", "sum"),
                repeat_count=("_rep", "sum"), avg_resolution_hours=("_res", "mean"), sd_res=("_res", "std"),
                avg_csat=("csat_score", "mean"), avg_rating=("rating", "mean"), total_revenue_inr=("_rev", "sum"),
                avg_revenue_inr=("_rev", "mean"), revenue_at_risk_inr=("_rar", "sum"), sla_breach_rate=("_sla", "mean")).reset_index()
    out = out[out.feedback_count >= min_n].copy()
    n = out.feedback_count.astype(float)
    out["topic_share"] = n / total_n
    out["negative_rate"] = out.negative_count / n
    out["positive_rate"] = out.positive_count / n
    out["net_sentiment_score"] = 100 * (out.positive_rate - out.negative_rate)
    out["repeat_rate"] = out.repeat_count / n
    ci = np.array([wilson_ci(a, b) for a, b in zip(out.negative_count, n)])
    out["negative_rate_lo"], out["negative_rate_hi"] = ci[:, 0], ci[:, 1]
    ci = np.array([wilson_ci(a, b) for a, b in zip(out.repeat_count, n)])
    out["repeat_rate_lo"], out["repeat_rate_hi"] = ci[:, 0], ci[:, 1]
    se = out.sd_res / np.sqrt(n)
    out["avg_resolution_lo"], out["avg_resolution_hi"] = out.avg_resolution_hours - 1.96 * se, out.avg_resolution_hours + 1.96 * se
    out["rar_share_of_total"] = out.revenue_at_risk_inr / total_rar if total_rar else np.nan
    out["rar_pct_of_segment_revenue"] = out.revenue_at_risk_inr / out.total_revenue_inr
    out["revenue_share"] = out.total_revenue_inr / total_rev
    return out.drop(columns=["sd_res"]).reset_index(drop=True)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"
