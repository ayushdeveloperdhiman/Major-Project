"""Stage 4 - Analytical warehouse (Gold layer): Kimball star schema in SQLite.

``build_core``      creates the schema, loads dimensions + ``fact_feedback`` + ``feedback_text``
                    from Silver, creates the KPI views and runs integrity / reconciliation checks.
``load_analytics``  later loads model outputs (``fact_feedback_nlp``, ``dim_segment``,
                    ``fact_feedback_segment``) once the NLP and mining stages have run.

Surrogate keys are assigned deterministically (sorted / declared order) so a rebuild
produces an identical database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .quality import UNKNOWN
from .utils import get_logger, read_json, save_metrics

log = get_logger("cvi.warehouse")
SCHEMA_SQL = C.SQL_DIR / "schema.sql"
VIEWS_SQL = C.SQL_DIR / "views.sql"
MODEL_VERSION = "tfidf-lr-v1"


def connect(path: Path = C.WAREHOUSE_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def query(sql: str, params: tuple = (), path: Path = C.WAREHOUSE_DB) -> pd.DataFrame:
    with connect(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _with_keys(values: list[str], extra_unknown: bool = True) -> dict[str, int]:
    vals = sorted(set(values) - {UNKNOWN})
    if extra_unknown:
        vals = vals + [UNKNOWN]
    return {v: i + 1 for i, v in enumerate(vals)}


def _insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df = df.astype(object).where(df.notna(), None)
    cols = list(df.columns)
    conn.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [tuple(r) for r in df.itertuples(index=False, name=None)])


# ------------------------------------------------------------------ build #
def build_core(silver: pd.DataFrame, params: dict, path: Path = C.WAREHOUSE_DB, write_evidence: bool = True) -> dict:
    C.ensure_dirs()
    if path.exists():
        path.unlink()
    conn = connect(path)
    conn.executescript(SCHEMA_SQL.read_text())

    # ---- dimensions
    keys: dict[str, dict] = {}
    k = keys["region"] = _with_keys(C.ALLOWED_VALUES["Region"])
    _insert(conn, "dim_region", pd.DataFrame({"region_key": list(k.values()), "region_name": list(k.keys())}))

    k = keys["channel"] = _with_keys(C.ALLOWED_VALUES["Channel"])
    _insert(conn, "dim_channel", pd.DataFrame({"channel_key": list(k.values()), "channel_name": list(k.keys()),
                                               "channel_type": [C.CHANNEL_TYPE.get(x, "Unknown") for x in k]}))

    k = keys["product"] = _with_keys(C.ALLOWED_VALUES["Product_Category"])
    _insert(conn, "dim_product_category", pd.DataFrame({"product_category_key": list(k.values()), "product_category_name": list(k.keys())}))

    k = keys["issue"] = _with_keys(C.ALLOWED_VALUES["Issue_Category"])
    _insert(conn, "dim_issue", pd.DataFrame({"issue_key": list(k.values()), "issue_category": list(k.keys()),
                                             "issue_theme_group": [C.ISSUE_THEME_GROUP.get(x, UNKNOWN) for x in k]}))

    k = keys["source"] = _with_keys(C.ALLOWED_VALUES["Feedback_Source"])
    _insert(conn, "dim_feedback_source", pd.DataFrame({"source_key": list(k.values()), "source_name": list(k.keys()),
                                                       "source_class": [C.SOURCE_CLASS.get(x, UNKNOWN) for x in k]}))

    labels = C.SENTIMENT_ORDER + [UNKNOWN]
    keys["sentiment"] = {v: i + 1 for i, v in enumerate(labels)}
    _insert(conn, "dim_sentiment", pd.DataFrame({"sentiment_key": list(keys["sentiment"].values()), "sentiment_label": labels,
                                                 "sentiment_score": [C.SENTIMENT_SCORE.get(x, 0) for x in labels],
                                                 "sort_order": list(range(1, len(labels) + 1))}))

    rating_labels = {1: "1 - Very poor", 2: "2 - Poor", 3: "3 - Average", 4: "4 - Good", 5: "5 - Excellent"}
    _insert(conn, "dim_rating", pd.DataFrame({"rating_key": list(rating_labels), "rating_label": list(rating_labels.values()),
                                              "implied_sentiment": [C.RATING_TO_SENTIMENT[r] for r in rating_labels]}))

    edges = C.RES_BAND_EDGES
    status = ["Within SLA", "Within SLA", "SLA breach", "Critical delay", "Critical delay"]
    keys["res_band"] = {lab: i + 1 for i, lab in enumerate(C.RES_BAND_LABELS)}
    _insert(conn, "dim_resolution_band", pd.DataFrame({
        "band_key": list(keys["res_band"].values()), "band_label": C.RES_BAND_LABELS,
        "lower_hours": [float(edges[i]) for i in range(len(C.RES_BAND_LABELS))],
        "upper_hours": [float(edges[i + 1]) if np.isfinite(edges[i + 1]) else None for i in range(len(C.RES_BAND_LABELS))],
        "sla_status": status}))

    q = params["revenue_quartile_edges"]
    lo = [None] + q
    hi = q + [None]
    keys["rev_tier"] = {lab: i + 1 for i, lab in enumerate(C.REVENUE_TIER_LABELS)}
    _insert(conn, "dim_revenue_tier", pd.DataFrame({"tier_key": list(keys["rev_tier"].values()), "tier_label": C.REVENUE_TIER_LABELS,
                                                    "lower_inr": lo, "upper_inr": hi}))

    cust = silver.groupby("customer_id").agg(feedback_record_count=("feedback_id", "size"),
                                              distinct_region_count=("region", "nunique")).reset_index().sort_values("customer_id")
    cust.insert(0, "customer_key", np.arange(1, len(cust) + 1))
    keys["customer"] = dict(zip(cust.customer_id, cust.customer_key))
    _insert(conn, "dim_customer", cust)

    # ---- fact + text
    s = silver
    fact = pd.DataFrame({
        "feedback_id": s.feedback_id,
        "customer_key": s.customer_id.map(keys["customer"]),
        "region_key": s.region.map(keys["region"]),
        "channel_key": s.channel.map(keys["channel"]),
        "product_category_key": s.product_category.map(keys["product"]),
        "issue_key": s.issue_category.map(keys["issue"]),
        "source_key": s.feedback_source.map(keys["source"]),
        "rating_key": s.rating,
        "ref_sentiment_key": s.reference_sentiment.map(keys["sentiment"]),
        "resolution_band_key": s.resolution_band.astype("string").map(keys["res_band"]),
        "revenue_tier_key": s.revenue_tier.astype("string").map(keys["rev_tier"]),
        "customer_revenue_inr": s.customer_revenue_inr, "order_count": s.order_count,
        "resolution_time_hours": s.resolution_time_hours, "rating": s.rating, "csat_score": s.csat_score,
        "avg_order_value_inr": s.avg_order_value_inr, "repeat_purchase_flag": s.repeat_purchase_flag,
        "is_negative_ref": s.is_negative_ref, "sla_breach": s.sla_breach, "critical_delay": s.critical_delay,
        "flag_repeat_without_orders": s.flag_repeat_without_orders, "dq_flags": s.dq_flags,
        "batch_id": s._batch_id, "row_hash": s._row_hash, "source_row": s._source_row,
    })
    _insert(conn, "fact_feedback", fact)
    _insert(conn, "feedback_text", s[["feedback_id", "feedback_text", "text_clean", "text_len_chars", "text_len_words"]])
    conn.commit()

    conn.executescript(VIEWS_SQL.read_text())
    conn.commit()
    checks = integrity_checks(conn, silver)
    conn.close()
    if write_evidence:      # tests that build a throw-away database must not overwrite the pipeline's evidence
        pd.DataFrame(checks).to_csv(C.VALIDATION_DIR / "warehouse_reconciliation.csv", index=False)
    ok = all(c["status"] == "PASS" for c in checks)
    log.info("Warehouse core built: fact=%d rows | integrity checks %s", len(fact), "ALL PASS" if ok else "FAILURES")
    keys_flat = {name: v for name, v in keys.items() if name != "customer"}
    return {"keys": keys_flat, "checks": checks}


def integrity_checks(conn: sqlite3.Connection, silver: pd.DataFrame) -> list[dict]:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    out = []

    def chk(cid: str, desc: str, expected, actual, tol: float = 0.0):
        passed = (abs(float(expected) - float(actual)) <= tol)
        out.append({"check_id": cid, "description": desc, "expected": expected, "actual": actual, "status": "PASS" if passed else "FAIL"})

    chk("WH-01", "fact_feedback rows == Silver rows", len(silver), q("SELECT COUNT(*) FROM fact_feedback"))
    chk("WH-02", "feedback_text rows == Silver rows", len(silver), q("SELECT COUNT(*) FROM feedback_text"))
    chk("WH-03", "distinct feedback_id in fact == Silver", silver.feedback_id.nunique(), q("SELECT COUNT(DISTINCT feedback_id) FROM fact_feedback"))
    chk("WH-04", "SUM(customer_revenue_inr) fact == Silver (INR, tol 0.01)", round(float(silver.customer_revenue_inr.sum()), 2),
        round(q("SELECT SUM(customer_revenue_inr) FROM fact_feedback"), 2), 0.01)
    chk("WH-05", "dim_customer rows == distinct Customer_ID", silver.customer_id.nunique(), q("SELECT COUNT(*) FROM dim_customer"))
    chk("WH-06", "Foreign-key violations (PRAGMA foreign_key_check)", 0, len(conn.execute("PRAGMA foreign_key_check").fetchall()))
    chk("WH-07", "SQLite integrity_check returns 'ok'", 1, 1 if q("PRAGMA integrity_check") == "ok" else 0)
    chk("WH-08", "Fact rows with NULL dimension keys", 0, q("""SELECT COUNT(*) FROM fact_feedback WHERE customer_key IS NULL OR region_key IS NULL
        OR channel_key IS NULL OR product_category_key IS NULL OR issue_key IS NULL OR source_key IS NULL OR ref_sentiment_key IS NULL"""))
    chk("WH-09", "SUM(order_count) fact == Silver", int(silver.order_count.sum()), q("SELECT SUM(order_count) FROM fact_feedback"))
    chk("WH-10", "vw_feedback_360 rows == fact rows (no join fan-out)", len(silver), q("SELECT COUNT(*) FROM vw_feedback_360"))
    return out


def load_analytics(nlp: pd.DataFrame, seg: pd.DataFrame, segments: pd.DataFrame, path: Path = C.WAREHOUSE_DB) -> list[dict]:
    """Load model outputs into the warehouse.

    nlp      : feedback_id, sent_pred, p_negative, p_neutral, p_positive, vader_compound, vader_label,
               topic_issue, nmf_topic_id, descriptor, negative_descriptor_theme
    seg      : feedback_id, segment_name, risk_tier
    segments : segment_name, segment_description
    """
    conn = connect(path)
    sk = dict(conn.execute("SELECT sentiment_label, sentiment_key FROM dim_sentiment").fetchall())
    ik = dict(conn.execute("SELECT issue_category, issue_key FROM dim_issue").fetchall())
    conn.execute("DELETE FROM fact_feedback_nlp")
    conn.execute("DELETE FROM fact_feedback_segment")
    conn.execute("DELETE FROM dim_segment")
    n = pd.DataFrame({
        "feedback_id": nlp.feedback_id, "pred_sentiment_key": nlp.sent_pred.map(sk),
        "p_negative": nlp.p_negative, "p_neutral": nlp.p_neutral, "p_positive": nlp.p_positive,
        "vader_compound": nlp.vader_compound, "vader_sentiment_key": nlp.vader_label.map(sk),
        "topic_issue_key": nlp.topic_issue.map(ik), "nmf_topic_id": nlp.nmf_topic_id,
        "descriptor": nlp.descriptor, "negative_descriptor_theme": nlp.negative_descriptor_theme,
        "model_version": MODEL_VERSION})
    _insert(conn, "fact_feedback_nlp", n)
    seg_keys = {name: i + 1 for i, name in enumerate(segments.segment_name)}
    _insert(conn, "dim_segment", pd.DataFrame({"segment_key": list(seg_keys.values()), "segment_name": list(seg_keys),
                                               "segment_description": segments.segment_description.values}))
    _insert(conn, "fact_feedback_segment", pd.DataFrame({"feedback_id": seg.feedback_id, "segment_key": seg.segment_name.map(seg_keys),
                                                         "risk_tier": seg.risk_tier}))
    conn.commit()
    fk = len(conn.execute("PRAGMA foreign_key_check").fetchall())
    n_nlp = conn.execute("SELECT COUNT(*) FROM fact_feedback_nlp").fetchone()[0]
    n_seg = conn.execute("SELECT COUNT(*) FROM fact_feedback_segment").fetchone()[0]
    n_360 = conn.execute("SELECT COUNT(*) FROM vw_feedback_360").fetchone()[0]
    n_fact = conn.execute("SELECT COUNT(*) FROM fact_feedback").fetchone()[0]
    conn.close()
    res = [
        {"check_id": "WH-11", "description": "fact_feedback_nlp rows == fact rows", "expected": n_fact, "actual": n_nlp, "status": "PASS" if n_nlp == n_fact else "FAIL"},
        {"check_id": "WH-12", "description": "fact_feedback_segment rows == fact rows", "expected": n_fact, "actual": n_seg, "status": "PASS" if n_seg == n_fact else "FAIL"},
        {"check_id": "WH-13", "description": "FK violations after analytics load", "expected": 0, "actual": fk, "status": "PASS" if fk == 0 else "FAIL"},
        {"check_id": "WH-14", "description": "vw_feedback_360 still one row per feedback (no fan-out)", "expected": n_fact, "actual": n_360, "status": "PASS" if n_360 == n_fact else "FAIL"},
    ]
    old = pd.read_csv(C.VALIDATION_DIR / "warehouse_reconciliation.csv")
    old = old[~old.check_id.isin([r["check_id"] for r in res])]
    pd.concat([old, pd.DataFrame(res)], ignore_index=True).to_csv(C.VALIDATION_DIR / "warehouse_reconciliation.csv", index=False)
    log.info("Analytics loaded into warehouse: nlp=%d seg=%d", n_nlp, n_seg)
    return res


def run(save: bool = True) -> dict:
    from .transform import load_silver
    silver = load_silver()
    params = read_json(C.DERIVATION_PARAMS_FILE)
    out = build_core(silver, params)
    metrics = {"fact_rows": int(len(silver)), "checks_passed": int(sum(c["status"] == "PASS" for c in out["checks"])),
               "checks_total": len(out["checks"])}
    if save:
        save_metrics("warehouse", metrics)
    return metrics


if __name__ == "__main__":  # pragma: no cover
    run()
