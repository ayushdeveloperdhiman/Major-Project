"""End-to-end orchestration + validation + reproducibility manifest.

    python run_pipeline.py                # everything
    python run_pipeline.py --from eda     # resume from a stage

Stage order mirrors the project workflow:
ingest -> transform(+DQ) -> warehouse -> eda -> sentiment -> topics -> mining -> analytics_load -> integrate
-> decision -> figures -> validate (reconciliation + manifest)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from . import kpis as K
from .utils import environment_info, get_logger, read_json, sha256_file, sha256_frame, write_json

log = get_logger("cvi.pipeline")
STAGES = ["ingest", "transform", "warehouse", "eda", "sentiment", "topics", "mining", "analytics_load", "integrate", "decision", "figures", "validate"]


def analytics_load() -> None:
    """Load model outputs (NLP + segments) into the warehouse."""
    from . import warehouse as W
    s = pd.read_parquet(C.SILVER_FILE)
    sc = pd.read_parquet(C.SILVER_DIR / "nlp_sentiment.parquet")
    tp = pd.read_parquet(C.SILVER_DIR / "nlp_topics.parquet")
    sg = pd.read_parquet(C.SILVER_DIR / "mining_segments.parquet")
    nlp = sc.merge(tp[["feedback_id", "topic_issue", "nmf_topic_id"]], on="feedback_id")
    from .mining import SEGMENT_DESCRIPTION
    segments = pd.DataFrame({"segment_name": list(SEGMENT_DESCRIPTION), "segment_description": list(SEGMENT_DESCRIPTION.values())})
    W.load_analytics(nlp, sg[["feedback_id", "segment_name", "risk_tier"]], segments)


def validate_kpi_reconciliation() -> pd.DataFrame:
    """SQL semantic layer (views) must equal the pandas KPI functions - two independent implementations."""
    from . import warehouse as W
    df = pd.read_parquet(C.ANALYTICS_FILE)
    sql = W.query("SELECT * FROM vw_kpi_overall").iloc[0]
    py = K.overall_kpis(df, "sent_pred")
    pairs = [("feedback_count", "feedback_count", 0), ("negative_sentiment_rate", "negative_sentiment_rate", 1e-4),
             ("positive_sentiment_rate", "positive_sentiment_rate", 1e-4), ("net_sentiment_score", "net_sentiment_score", 1e-2),
             ("repeat_purchase_rate", "repeat_purchase_rate", 1e-4), ("avg_resolution_hours", "avg_resolution_hours", 1e-3),
             ("avg_csat", "avg_csat", 1e-3), ("total_revenue_inr", "total_revenue_inr", 0.01), ("revenue_at_risk_inr", "revenue_at_risk_inr", 0.01)]
    rows = []
    for name, key, tol in pairs:
        a, b = float(sql[name]), float(py[key])
        rows.append({"check": f"overall: {name}", "sql": a, "pandas": b, "abs_diff": abs(a - b), "tolerance": tol, "status": "PASS" if abs(a - b) <= max(tol, 1e-12) else "FAIL"})
    by_sql = W.query("SELECT issue_category AS level, feedback_count, negative_sentiment_rate, revenue_at_risk_inr FROM vw_kpi_by_issue").set_index("level")
    by_py = K.kpi_table(df, "issue_category", "sent_pred").set_index("issue_category")
    for lvl in by_py.index:
        for sql_c, py_c, tol in [("negative_sentiment_rate", "negative_rate", 1e-4), ("revenue_at_risk_inr", "revenue_at_risk_inr", 0.01)]:
            a, b = float(by_sql.loc[lvl, sql_c]), float(by_py.loc[lvl, py_c])
            rows.append({"check": f"by issue [{lvl}]: {sql_c}", "sql": a, "pandas": b, "abs_diff": abs(a - b), "tolerance": tol, "status": "PASS" if abs(a - b) <= tol else "FAIL"})
    seg_sql = W.query("SELECT segment_name, feedback_count FROM vw_kpi_by_segment").set_index("segment_name").feedback_count.to_dict()
    seg_py = df.segment_name.value_counts().to_dict()
    for k_, v in seg_py.items():
        rows.append({"check": f"by segment [{k_}]: feedback_count", "sql": float(seg_sql.get(k_, -1)), "pandas": float(v), "abs_diff": abs(seg_sql.get(k_, -1) - v), "tolerance": 0, "status": "PASS" if seg_sql.get(k_) == v else "FAIL"})
    out = pd.DataFrame(rows)
    out.to_csv(C.VALIDATION_DIR / "kpi_reconciliation_sql_vs_pandas.csv", index=False)
    return out


def write_manifest(timings: dict[str, float]) -> dict:
    files = {}
    for p in sorted(list(C.SILVER_DIR.glob("*.parquet")) + [C.BRONZE_FILE, C.WAREHOUSE_DB]):
        if p.exists():
            files[str(p.relative_to(C.ROOT))] = {"bytes": p.stat().st_size}
    silver = pd.read_parquet(C.SILVER_FILE)
    ing = read_json(C.METRIC_DIR / "ingest.json")
    manifest_path = C.OUT_DIR / "run_manifest.json"
    prev = read_json(manifest_path).get("stage_seconds", {}) if manifest_path.exists() else {}
    merged = {**prev, **{k: round(v, 2) for k, v in timings.items()}}      # a partial re-run must not erase earlier stage timings
    man = {"project": "MSDS-MAJOR-CVI-001", "client": "InsightRetail India (anonymised)",
           "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "last_run_stages": list(timings),
           "input": {"file": C.RAW_XLSX.name, "sha256": sha256_file(C.RAW_XLSX), "rows": ing["rows"], "columns": ing["columns"]},
           "seed": C.SEED, "environment": environment_info(), "stage_seconds": merged,
           "deterministic_hashes": {"silver_content_sha256": sha256_frame(silver.drop(columns=["_batch_id"])),
                                    "analytics_kpi_table_sha256": sha256_frame(pd.read_csv(C.TABLE_DIR / "integrated_kpis_by_dimension.csv").round(8))},
           "artifacts": files}
    write_json(man, C.OUT_DIR / "run_manifest.json")
    return man


def validate() -> dict:
    rec = validate_kpi_reconciliation()
    wh = pd.read_csv(C.VALIDATION_DIR / "warehouse_reconciliation.csv")
    res = {"kpi_reconciliation_checks": int(len(rec)), "kpi_reconciliation_pass": int((rec.status == "PASS").sum()),
           "warehouse_checks": int(len(wh)), "warehouse_pass": int((wh.status == "PASS").sum())}
    from .utils import save_metrics
    save_metrics("validate", res)
    if res["kpi_reconciliation_pass"] != res["kpi_reconciliation_checks"] or res["warehouse_pass"] != res["warehouse_checks"]:
        raise RuntimeError(f"Validation failed: {res}")
    log.info("Validation | warehouse %d/%d PASS | SQL-vs-pandas KPI reconciliation %d/%d PASS", res["warehouse_pass"], res["warehouse_checks"],
             res["kpi_reconciliation_pass"], res["kpi_reconciliation_checks"])
    return res


def run_all(start: str = "ingest", only: list[str] | None = None) -> dict:
    C.ensure_dirs()
    from . import (decision, eda, ingest, integrate, mining, sentiment, topics, transform, warehouse)
    runners = {
        "ingest": ingest.ingest, "transform": transform.run, "warehouse": warehouse.run, "eda": eda.run,
        "sentiment": sentiment.run, "topics": topics.run, "mining": mining.run, "analytics_load": analytics_load,
        "integrate": integrate.run, "decision": decision.run, "validate": validate,
    }

    def figures():
        from . import figures as F
        return F.make_all()
    runners["figures"] = figures
    order = STAGES[STAGES.index(start):] if only is None else only
    timings: dict[str, float] = {}
    for st in order:
        t0 = time.time()
        log.info("=== STAGE: %s ===", st)
        runners[st]()
        timings[st] = time.time() - t0
    if only is None or "validate" in only:
        write_manifest(timings)
    log.info("Pipeline finished in %.1fs", sum(timings.values()))
    return timings
