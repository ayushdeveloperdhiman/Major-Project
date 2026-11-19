"""Warehouse integrity, KPI correctness (independent recomputation), integration statistics and decision-layer tests."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from cvi import config as C
from cvi import kpis as K
from cvi import warehouse as W
from cvi.utils import load_metrics, sha256_frame, wilson_ci


@pytest.fixture(scope="module")
def conn():
    c = W.connect()
    yield c
    c.close()


# ------------------------------------------------------------------ warehouse
@pytest.mark.case("T-WH-01", "Warehouse", "Run all integrity / reconciliation checks", "All 14 checks PASS (row counts, revenue sum, FK, PK uniqueness, no fan-out)")
def test_wh_checks(actual):
    t = pd.read_csv(C.VALIDATION_DIR / "warehouse_reconciliation.csv")
    actual(f"{(t.status == 'PASS').sum()}/{len(t)} PASS")
    assert len(t) == 14 and (t.status == "PASS").all()


@pytest.mark.case("T-WH-02", "Warehouse", "Insert a fact row whose customer_key does not exist", "sqlite3.IntegrityError (foreign key enforced)")
def test_fk_enforced(conn, actual):
    cols = "feedback_id, customer_key, region_key, channel_key, product_category_key, issue_key, source_key, rating_key, ref_sentiment_key, rating, is_negative_ref, sla_breach, critical_delay, flag_repeat_without_orders, batch_id, row_hash, source_row"
    conn.execute("BEGIN")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(f"INSERT INTO fact_feedback ({cols}) VALUES ('FB-X', 999999, 1, 1, 1, 1, 1, 3, 1, 3, 0, 0, 0, 0, 'b', 'h', 1)")
    conn.execute("ROLLBACK")
    actual("IntegrityError raised for customer_key=999999")


@pytest.mark.case("T-WH-03", "Warehouse", "Insert a fact row with Rating = 9 and repeat flag = 3", "sqlite3.IntegrityError (CHECK constraints)")
def test_check_constraints(conn, actual):
    cols = "feedback_id, customer_key, region_key, channel_key, product_category_key, issue_key, source_key, rating_key, ref_sentiment_key, rating, repeat_purchase_flag, is_negative_ref, sla_breach, critical_delay, flag_repeat_without_orders, batch_id, row_hash, source_row"
    conn.execute("BEGIN")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(f"INSERT INTO fact_feedback ({cols}) VALUES ('FB-Y', 1, 1, 1, 1, 1, 1, 3, 1, 9, 3, 0, 0, 0, 0, 'b', 'h', 1)")
    conn.execute("ROLLBACK")
    actual("IntegrityError raised for rating=9 / repeat=3")


@pytest.mark.case("T-WH-04", "Warehouse", "Grain check: fact_feedback and the 360 view", "2,400 rows, feedback_id unique, no join fan-out in vw_feedback_360")
def test_grain(conn, actual):
    n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT feedback_id) FROM fact_feedback").fetchone()
    v = conn.execute("SELECT COUNT(*), COUNT(DISTINCT feedback_id) FROM vw_feedback_360").fetchone()
    actual(f"fact={n}; view={v}")
    assert n == (2400, 2400) and v == (2400, 2400)


@pytest.mark.case("T-WH-05", "Warehouse", "Compare the SQL semantic layer with the pandas KPI functions", "28 of 28 reconciliations PASS (overall, by issue, by segment)")
def test_sql_vs_pandas(actual):
    t = pd.read_csv(C.VALIDATION_DIR / "kpi_reconciliation_sql_vs_pandas.csv")
    actual(f"{(t.status == 'PASS').sum()}/{len(t)} PASS; max abs diff={t.abs_diff.max():.6f}")
    assert len(t) == 28 and (t.status == "PASS").all()


@pytest.mark.case("T-WH-06", "Warehouse", "Rebuild the warehouse core from Silver", "Identical dimension tables (deterministic surrogate keys)")
def test_wh_deterministic(silver, tmp_path, actual):
    from cvi.utils import read_json
    params = read_json(C.DERIVATION_PARAMS_FILE)
    W.build_core(silver, params, path=tmp_path / "again.db", write_evidence=False)
    dims = ["dim_region", "dim_channel", "dim_issue", "dim_feedback_source", "dim_sentiment", "dim_customer", "dim_resolution_band", "dim_revenue_tier"]
    ok = []
    for d in dims:
        a = W.query(f"SELECT * FROM {d} ORDER BY 1")
        b = W.query(f"SELECT * FROM {d} ORDER BY 1", path=tmp_path / "again.db")
        ok.append(sha256_frame(a) == sha256_frame(b))
    actual(f"{sum(ok)}/{len(dims)} dimension tables identical")
    assert all(ok)


# ------------------------------------------------------------------------ KPIs
@pytest.mark.case("T-KPI-01", "KPI", "Recompute the five Metric Guidance KPIs from the raw Excel with plain pandas (independent path) and compare with the warehouse views", "Negative rate 24.04%, repeat 57.46%, avg resolution 11.256h, Revenue at Risk INR 3,703,115.01 - all equal")
def test_kpis_independent(raw_excel, actual):
    r = raw_excel
    neg = r.Reference_Sentiment == "Negative"
    exp = {"negative_sentiment_rate": neg.mean(), "repeat_purchase_rate": r.Repeat_Purchase_Flag.mean(), "avg_resolution_hours": r.Resolution_Time_Hours.mean(),
           "revenue_at_risk_inr": r.Customer_Revenue_INR[neg].sum()}
    sql = W.query("SELECT * FROM vw_kpi_overall").iloc[0]
    diffs = {k: abs(float(sql[k]) - float(v)) for k, v in exp.items()}
    actual(f"neg={float(sql.negative_sentiment_rate):.4f} repeat={float(sql.repeat_purchase_rate):.4f} res={float(sql.avg_resolution_hours):.3f} RaR={float(sql.revenue_at_risk_inr):,.2f}")
    assert diffs["negative_sentiment_rate"] < 1e-4 and diffs["repeat_purchase_rate"] < 1e-4 and diffs["avg_resolution_hours"] < 1e-3 and diffs["revenue_at_risk_inr"] < 0.01
    assert round(exp["revenue_at_risk_inr"], 2) == 3703115.01


@pytest.mark.case("T-KPI-02", "KPI", "Wilson 95% CI for 50 successes out of 100 and for 0 out of 50", "(0.404, 0.596) and a lower bound of exactly 0")
def test_wilson(actual):
    lo, hi = wilson_ci(50, 100)
    lo0, hi0 = wilson_ci(0, 50)
    actual(f"({lo:.3f}, {hi:.3f}); 0/50 -> ({lo0:.3f}, {hi0:.3f})")
    assert abs(lo - 0.404) < 0.001 and abs(hi - 0.596) < 0.001 and lo0 == 0.0 and 0 < hi0 < 0.1


@pytest.mark.case("T-KPI-03", "KPI", "Topic Share per theme", "Shares sum to 1.0 and match the SQL view")
def test_topic_share(analytics, actual):
    t = K.kpi_table(analytics, "topic_issue", "sent_pred")
    v = W.query("SELECT issue_category, topic_share FROM vw_kpi_by_issue").set_index("issue_category").topic_share
    diff = (t.set_index("topic_issue").topic_share - v).abs().max()
    actual(f"sum={t.topic_share.sum():.6f}; max diff vs SQL={diff:.5f}")
    assert abs(t.topic_share.sum() - 1) < 1e-9 and diff < 1e-4


@pytest.mark.case("T-KPI-04", "KPI", "Net Sentiment Score identity", "NSS = 100 x (positive share - negative share) = +33.04")
def test_nss(analytics, actual):
    k = K.overall_kpis(analytics)
    actual(f"NSS={k['net_sentiment_score']:.2f}")
    assert abs(k["net_sentiment_score"] - 33.04) < 0.01


# ---------------------------------------------------------------- integration
@pytest.mark.case("T-INT-01", "Integration", "Bootstrap rank-stability probabilities", "Probability of ranking #1 sums to 1 across levels; every p in [0,1]")
def test_rank_probs(actual):
    r = pd.read_csv(C.TABLE_DIR / "integrated_rank_stability_issue.csv")
    actual(f"sum p_rank1={r.p_rank1_rar.sum():.4f}; sum p_rank1(rate)={r.p_rank1_rate.sum():.4f}")
    assert abs(r.p_rank1_rar.sum() - 1) < 1e-9 and abs(r.p_rank1_rate.sum() - 1) < 1e-9 and r.p_rank1_rar.between(0, 1).all()


@pytest.mark.case("T-INT-02", "Integration", "Revenue-at-Risk scenarios", "conservative < base < optimistic; exposure = Revenue at Risk x repeat gap; decomposition sums to total revenue")
def test_scenarios(actual):
    m = load_metrics("integrate")
    sc = pd.DataFrame(m["scenarios"]).set_index("scenario")
    rar = m["revenue_at_risk"]
    dec = pd.DataFrame(m["decomposition"])
    actual(f"scenarios INR {sc.recoverable_revenue_inr.round(0).astype(int).tolist()}; exposure {rar['exposure_inr']:,.0f} = RaR {rar['revenue_at_risk_inr']:,.0f} x gap {rar['repeat_gap_pp']:.2f}pp; decomposition sums to {dec.revenue_inr.sum():,.2f}")
    assert sc.loc["conservative", "recoverable_revenue_inr"] < sc.loc["base", "recoverable_revenue_inr"] < sc.loc["optimistic", "recoverable_revenue_inr"]
    assert abs(rar["exposure_inr"] - rar["revenue_at_risk_inr"] * rar["repeat_gap_pp"] / 100) < 1.0
    assert abs(dec.revenue_inr.sum() - rar["total_revenue_inr"]) < 0.01


@pytest.mark.case("T-INT-03", "Integration", "Repeat-purchase driver model", "Adjusted OR(Negative vs Positive) < 0.25 with p < 0.001; resolution time NOT significant once sentiment is controlled (p > 0.05)")
def test_driver_model(actual):
    d = {r["label"]: r for r in load_metrics("integrate")["driver"]["key_terms"]}
    neg, res = d["Negative vs Positive sentiment"], d["Resolution time (per +1 SD)"]
    actual(f"OR_neg={neg['odds_ratio']:.3f} (p={neg['p_value']:.1e}); OR_resolution={res['odds_ratio']:.3f} (p={res['p_value']:.3f})")
    assert neg["odds_ratio"] < 0.25 and neg["p_value"] < 0.001 and res["p_value"] > 0.05


@pytest.mark.case("T-INT-04", "Integration", "FDR-controlled hotspot scan on the supplied data", "314 cells scanned; 0 significant after Benjamini-Hochberg; raw p<0.05 count near chance expectation")
def test_hotspot(actual):
    h = load_metrics("eda")["hotspot"]
    actual(f"tests={h['tests']}; raw p<0.05={h['raw_p_lt_0.05']} (chance {h['expected_by_chance']:.1f}); FDR-significant={h['significant_after_fdr']}")
    assert h["tests"] == 314 and h["significant_after_fdr"] == 0 and abs(h["raw_p_lt_0.05"] - h["expected_by_chance"]) < 10


# ------------------------------------------------------------------- decision
@pytest.mark.case("T-DEC-01", "Decision layer", "Recommendation table", ">= 5 recommendations; unique ids R1..Rn follow the priority rank; priority = impact x confidence / effort; all fields non-empty; the dashboard ranking chart lists them in that same order (ties included)")
def test_recs(actual):
    import sys
    r = pd.read_csv(C.TABLE_DIR / "decision_recommendations.csv")
    need = ["finding", "evidence", "action", "owner", "timeline", "kpi", "baseline", "target", "expected_impact", "confidence_basis", "caveat"]
    score_ok = ((r.impact_score * r.confidence / r.effort).round(2) == r.priority_score).all()
    sys.path.insert(0, str(C.ROOT / "dashboard"))
    import charts as CH                                   # the dashboard's own chart builder, not a re-implementation
    fig, twin = CH.priority(r, "light")
    plotted = [s.split()[0] for s in fig.data[0].y][::-1]    # plotly draws the first category at the bottom, so reverse for top-down
    ties = int(r.priority_score.duplicated(keep=False).sum())
    actual(f"{len(r)} recommendations; ids={r.id.tolist()}; score formula ok={bool(score_ok)}; dashboard chart top-down={plotted} ({ties} tied rows keep table order)")
    assert len(r) >= 5 and r.id.is_unique and r.id.tolist() == [f"R{i}" for i in range(1, len(r) + 1)] and score_ok
    assert r[need].notna().all().all() and (r[need].astype(str).apply(lambda s: s.str.strip().str.len()).min() >= 2).all()
    assert r.priority_score.is_monotonic_decreasing
    assert plotted == r.sort_values("priority_rank").id.tolist() and twin.id.tolist() == plotted


@pytest.mark.case("T-DEC-02", "Decision layer", "Cross-references between recommendations", "No unresolved '[[...]]' placeholders; every 'R<n>' mention points to an existing id")
def test_xrefs(actual):
    import re
    r = pd.read_csv(C.TABLE_DIR / "decision_recommendations.csv")
    txt = " ".join(r[c].astype(str).str.cat(sep=" ") for c in ["finding", "evidence", "action", "caveat", "expected_impact"])
    refs = set(re.findall(r"\bR\d+\b", txt))
    actual(f"references found={sorted(refs)}")
    assert "[[" not in txt and refs <= set(r.id)


@pytest.mark.case("T-DEC-03", "Decision layer", "KPI scorecard", "8 KPIs; statuses follow the documented rule; every target flagged as an assumption")
def test_scorecard(actual):
    s = pd.read_csv(C.TABLE_DIR / "decision_kpi_scorecard.csv")
    actual(f"{len(s)} KPIs; statuses={s.status.tolist()}")
    assert len(s) == 8 and s.target_is_assumption.all() and set(s.status) <= {"good", "warning", "serious", "critical"}
    nsr = s.set_index("kpi").loc["Negative Sentiment Rate"]
    assert nsr.current > nsr.proposed_target and nsr.status in ("serious", "critical")
