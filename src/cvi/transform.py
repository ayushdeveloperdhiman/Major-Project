"""Stage 3 - Cleaning, standardisation and feature preparation (Silver layer).

Transformation rules (every one is logged with rows in/out and cells changed):

TR-01  Unicode NFKC normalisation, trim, collapse internal whitespace on all strings
TR-02  Map categorical values to canonical casing (case-insensitive lookup); blank -> 'Unknown'
TR-03  Coerce numerics (unparseable -> NaN).  Outcome measures are NEVER imputed
TR-04  Quarantine rows failing any *error* rule, plus duplicates (keep first occurrence)
TR-05  Fix the repeated-word template glitch ("experience experience") in ``text_clean``
TR-06  Derive analysis-ready structured features (bands, tiers, flags, ratios)
TR-07  Derive basic text features (length, word count)
TR-08  Cast to strict types; persist with lineage columns

The function is deterministic and idempotent: running it twice on the same
Bronze batch yields byte-identical Silver data (tested).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config as C
from . import quality as Q
from .utils import get_logger, save_metrics, write_json

log = get_logger("cvi.transform")

KEEP_REPEATS = {"very", "really", "so", "no", "not", "never", "too", "much", "more", "most", "had", "that"}
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)
QUARANTINE_ON_WARN = {"DQ-U02"}          # exact duplicates are removed (keep first), never silently

LINEAGE = ["_source_row", "_batch_id", "_row_hash"]


# ------------------------------------------------------------------ steps #
def norm_string(s: pd.Series) -> pd.Series:
    def f(v):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        v = unicodedata.normalize("NFKC", str(v))
        v = re.sub(r"\s+", " ", v).strip()
        return v if v != "" else pd.NA
    return s.map(f).astype("string")


def canonicalise(s: pd.Series, allowed: list[str]) -> pd.Series:
    lut = {a.casefold(): a for a in allowed}
    lut[Q.UNKNOWN.casefold()] = Q.UNKNOWN
    out = s.map(lambda v: lut.get(str(v).casefold(), v) if v is not pd.NA and pd.notna(v) else Q.UNKNOWN)
    return out.astype("string")


def collapse_repeated_words(text: str) -> str:
    """'The app experience experience was good' -> 'The app experience was good' (intensifiers/negations preserved)."""
    def sub(m: re.Match) -> str:
        return m.group(0) if m.group(1).casefold() in KEEP_REPEATS else m.group(1)
    return _REPEAT.sub(sub, text)


def fit_derivation_params(df: pd.DataFrame) -> dict:
    """Parameters learned from the reference batch (persisted so later batches use identical thresholds)."""
    q = df["customer_revenue_inr"].quantile([0.25, 0.5, 0.75]).tolist()
    return {"revenue_quartile_edges": q, "sla_target_hours": C.SLA_TARGET_HOURS,
            "critical_delay_hours": C.CRITICAL_DELAY_HOURS, "res_band_edges": [float(x) if np.isfinite(x) else "inf" for x in C.RES_BAND_EDGES]}


def derive_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = df.copy()
    d["sentiment_ref"] = pd.Categorical(d["reference_sentiment"], categories=C.SENTIMENT_ORDER + [Q.UNKNOWN], ordered=False)
    d["is_negative_ref"] = (d["reference_sentiment"] == "Negative").astype("int64")
    d["sentiment_ref_score"] = d["reference_sentiment"].map(C.SENTIMENT_SCORE).astype("float")
    d["resolution_band"] = pd.cut(d["resolution_time_hours"], bins=C.RES_BAND_EDGES, labels=C.RES_BAND_LABELS,
                                  include_lowest=True, right=True)
    d["sla_breach"] = (d["resolution_time_hours"] > params["sla_target_hours"]).astype("int64")
    d["critical_delay"] = (d["resolution_time_hours"] > params["critical_delay_hours"]).astype("int64")
    q1, q2, q3 = params["revenue_quartile_edges"]
    d["revenue_tier"] = pd.cut(d["customer_revenue_inr"], bins=[-np.inf, q1, q2, q3, np.inf],
                               labels=C.REVENUE_TIER_LABELS, right=True)
    d["order_band"] = pd.cut(d["order_count"].astype("float"), bins=C.ORDER_BAND_EDGES, labels=C.ORDER_BAND_LABELS, right=True)
    d["csat_band"] = pd.cut(d["csat_score"], bins=C.CSAT_BAND_EDGES, labels=C.CSAT_BAND_LABELS, right=False)
    d["avg_order_value_inr"] = (d["customer_revenue_inr"] / d["order_count"].astype("float")).round(2)
    d["issue_theme_group"] = d["issue_category"].map(C.ISSUE_THEME_GROUP).fillna(Q.UNKNOWN)
    d["channel_type"] = d["channel"].map(C.CHANNEL_TYPE).fillna(Q.UNKNOWN)
    d["source_class"] = d["feedback_source"].map(C.SOURCE_CLASS).fillna(Q.UNKNOWN)
    d["flag_repeat_without_orders"] = ((d["repeat_purchase_flag"] == 1) & (d["order_count"] == 1)).astype("int64")
    d["customer_record_count"] = d.groupby("customer_id")["feedback_id"].transform("size").astype("int64")
    return d


def to_contract_view(silver: pd.DataFrame) -> pd.DataFrame:
    """Silver (snake_case) -> original Title_Case contract names, so the same DQ rules can re-run on clean data."""
    inv = {v: k for k, v in C.SNAKE.items()}
    v = silver[[c for c in inv if c in silver.columns]].rename(columns=inv)
    return v


# ------------------------------------------------------------------ driver #
def clean(bronze: pd.DataFrame, params: dict | None = None) -> dict:
    """Bronze -> Silver.  Returns dict with silver, quarantine, log, params and both DQ results."""
    steps: list[dict] = []

    def record(tr: str, desc: str, rows_in: int, rows_out: int, changed: int = 0):
        steps.append({"step": tr, "description": desc, "rows_in": int(rows_in), "rows_out": int(rows_out), "cells_changed": int(changed)})

    contract = [c for c in C.EXPECTED_COLUMNS if c in bronze.columns]
    lineage = [c for c in LINEAGE if c in bronze.columns]
    work = bronze[contract + lineage].copy()
    n0 = len(work)

    # DQ on the RAW data (before any fix) -> quantifies the problems found
    dq_raw = Q.run_rules(work[contract])

    # TR-01 string normalisation
    changed = 0
    for c in Q.STRING_COLS:
        before = work[c].astype("string")
        work[c] = norm_string(work[c])
        changed += int((before.fillna("") != work[c].fillna("")).sum())
    record("TR-01", "NFKC-normalise, trim, collapse whitespace on all string fields", n0, len(work), changed)

    # TR-02 canonical categorical values
    changed = 0
    for c, allowed in C.ALLOWED_VALUES.items():
        before = work[c].copy()
        work[c] = canonicalise(work[c], allowed)
        changed += int((before.fillna("<NA>") != work[c].fillna("<NA>")).sum())
    record("TR-02", "Map categoricals to canonical casing; blank -> 'Unknown'", n0, len(work), changed)

    # TR-03 numeric coercion
    changed = 0
    for c in Q.NUMERIC_COLS:
        before = work[c].astype("string")
        work[c] = pd.to_numeric(work[c], errors="coerce")
        changed += int((before.notna() & work[c].isna()).sum())
    record("TR-03", "Coerce numeric fields (unparseable -> NaN, no imputation of outcome measures)", n0, len(work), changed)

    # TR-04 quarantine: error rules + duplicates (re-evaluated on the normalised data)
    post = Q.run_rules(work[contract])
    qmask = post["error_mask"].copy()
    reason = post["error_reason"].copy()
    for rid in QUARANTINE_ON_WARN:
        m = post["masks"][rid]
        qmask |= m
        reason = reason.where(~m, reason.where(reason != "", rid) + np.where(reason != "", ";" + rid, ""))
    quarantine = work.loc[qmask].copy()
    quarantine["dq_error_reason"] = reason[qmask]
    silver = work.loc[~qmask].copy()
    # Row-level flags = issues seen on the LANDED data (before auto-fixes) UNION issues seen after normalisation,
    # so a blank -> 'Unknown' or a trimmed / re-cased value stays traceable even though it was fixed.
    union = [";".join(sorted(set(filter(None, (a + ";" + b).split(";"))))) for a, b in zip(dq_raw["flags"], post["flags"])]
    flags = pd.Series(union, index=work.index)[~qmask]
    record("TR-04", "Quarantine rows failing error rules / duplicates (keep first occurrence)", n0, len(silver), int(qmask.sum()))

    # snake_case + strict types
    silver = silver.rename(columns=C.SNAKE)
    silver["dq_flags"] = flags.values
    for c in ["rating", "order_count", "repeat_purchase_flag"]:
        silver[c] = silver[c].astype("Int64")
    for c in ["customer_revenue_inr", "resolution_time_hours", "csat_score"]:
        silver[c] = silver[c].astype("float64")
    for c in ["region", "channel", "product_category", "issue_category", "reference_sentiment", "feedback_source",
              "feedback_id", "customer_id", "feedback_text"]:
        silver[c] = silver[c].astype("string")
    if (silver[["rating", "order_count", "repeat_purchase_flag"]].notna().all().all()):
        for c in ["rating", "order_count", "repeat_purchase_flag"]:
            silver[c] = silver[c].astype("int64")

    # TR-05 text clean (glitch fix) + TR-07 text features
    before = silver["feedback_text"]
    silver["text_clean"] = before.map(lambda t: collapse_repeated_words(t) if isinstance(t, str) else t).astype("string")
    record("TR-05", "Collapse repeated adjacent words in text_clean (template glitch 'experience experience')", len(silver), len(silver),
           int((before != silver["text_clean"]).sum()))
    silver["text_len_chars"] = silver["text_clean"].str.len().astype("int64")
    silver["text_len_words"] = silver["text_clean"].str.split().map(len).astype("int64")
    record("TR-07", "Derive text features: character and word counts", len(silver), len(silver), 0)

    # TR-06 structured features
    params = params or fit_derivation_params(silver)
    silver = derive_features(silver, params)
    record("TR-06", "Derive resolution/SLA bands, revenue tiers, order bands, AOV, theme/channel/source classes, flags", len(silver), len(silver), 0)

    silver = silver.sort_values("feedback_id").reset_index(drop=True)
    record("TR-08", "Strict typing, deterministic ordering by feedback_id, lineage retained", len(silver), len(silver), 0)

    # DQ on the CLEAN data -> proves the fixes
    clean_view = to_contract_view(silver)
    clean_view["Feedback_Text"] = silver["text_clean"].values       # rules see the fixed text
    dq_clean = Q.run_rules(clean_view)

    return {"silver": silver, "quarantine": quarantine, "log": pd.DataFrame(steps), "params": params,
            "dq_raw": dq_raw, "dq_clean": dq_clean}


def run(save: bool = True) -> dict:
    from .ingest import load_bronze
    C.ensure_dirs()
    b = load_bronze()
    out = clean(b)
    silver, quarantine = out["silver"], out["quarantine"]
    if save:
        silver.to_parquet(C.SILVER_FILE, index=False)
        quarantine.to_parquet(C.QUARANTINE_FILE, index=False)
        write_json(out["params"], C.DERIVATION_PARAMS_FILE)
        out["log"].to_csv(C.DQ_DIR / "transformation_log.csv", index=False)
        Q.save_report(out["dq_raw"], "raw")
        Q.save_report(out["dq_clean"], "clean")
    rr, rc = out["dq_raw"]["report"].set_index("rule_id"), out["dq_clean"]["report"].set_index("rule_id")
    metrics = {
        "rows_bronze": int(len(b)), "rows_silver": int(len(silver)), "rows_quarantined": int(len(quarantine)),
        "silver_columns": int(silver.shape[1]),
        "dq_raw": out["dq_raw"]["score"], "dq_clean": out["dq_clean"]["score"],
        "findings": {
            "repeat_without_orders_rows": int(rr.loc["DQ-K04", "rows_failed"]),
            "repeated_word_glitch_rows_raw": int(rr.loc["DQ-K05", "rows_failed"]),
            "repeated_word_glitch_rows_clean": int(rc.loc["DQ-K05", "rows_failed"]),
            "resolution_floor_censored_rows": int(rr.loc["DQ-P03", "rows_failed"]),
            "resolution_mild_outliers": int(rr.loc["DQ-P02", "rows_failed"]),
            "region_unstable_pct_of_multi_record_customers": float(rr.loc["DQ-E01", "pct_failed"]),
            "revenue_unstable_pct_of_multi_record_customers": float(rr.loc["DQ-E02", "pct_failed"]),
            "text_template_pct": float(rr.loc["DQ-X01", "pct_failed"]),
            "same_customer_same_text_rows": int(rr.loc["DQ-U03", "rows_failed"]),
        },
        "derivation_params": out["params"],
    }
    if save:
        save_metrics("transform", metrics)
    log.info("Silver: %d rows kept, %d quarantined | DQ error-rule pass %.2f%% | clean-row rate raw %.1f%% -> clean %.1f%%",
             len(silver), len(quarantine), out["dq_raw"]["score"]["pct_rows_passing_all_error_rules"],
             out["dq_raw"]["score"]["pct_rows_fully_clean"], out["dq_clean"]["score"]["pct_rows_fully_clean"])
    return metrics


def load_silver() -> pd.DataFrame:
    return pd.read_parquet(C.SILVER_FILE)


if __name__ == "__main__":  # pragma: no cover
    run()
