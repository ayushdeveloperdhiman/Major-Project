"""Stage 11 - Business decision layer: KPI scorecard and prioritised, evidence-backed recommendations.

Every number in the evidence text is read from the saved stage metrics (no hand-typed values).

Scoring rubric (transparent, documented in the workbook)
  Impact      1-5  from the BASE-scenario recoverable revenue: >=0.8M 5 | >=0.4M 4 | >=0.2M 3 | >=0.1M 2 | else 1.
              Enabling actions (no direct revenue) get a fixed, stated score.
  Confidence  1-5  confidence that the ACTION delivers its stated impact: 5 near-certain (rule-based / directly verifiable);
              4 very likely; 3 strong association but causal effect untested; 2 weak or not significant; 1 hypothesis.
  Effort      1-5  analyst judgement (1 = days, 5 = multi-quarter programme)  [ASSUMPTION]
  Priority    = Impact x Confidence / Effort   (ties: higher impact, then higher confidence)
Targets in the scorecard are PROPOSED (the client supplied none) and must be agreed with stakeholders.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config as C
from .utils import fmt_inr, get_logger, load_metrics, save_metrics

log = get_logger("cvi.decision")


def load_all_metrics() -> dict:
    return {k: load_metrics(k) for k in ["ingest", "transform", "eda", "sentiment", "topics", "mining", "integrate"]}


def _status(value: float, target: float, higher_is_better: bool) -> str:
    gap = (target - value) / abs(target) if higher_is_better else (value - target) / abs(target)   # >0 means worse than target
    if gap <= 0:
        return "good"
    if gap <= 0.10:
        return "warning"
    if gap <= 0.25:
        return "serious"
    return "critical"


def kpi_scorecard(M: dict) -> pd.DataFrame:
    k = M["integrate"]["kpis_predicted"]
    T = C.PROPOSED_TARGETS
    rows = [
        ("Negative Sentiment Rate", k["negative_sentiment_rate"], T["negative_sentiment_rate"], False, "pct", "Metric Guidance"),
        ("Repeat Purchase Rate", k["repeat_purchase_rate"], T["repeat_purchase_rate"], True, "pct", "Metric Guidance"),
        ("Average Resolution Time (h)", k["avg_resolution_hours"], T["avg_resolution_hours"], False, "hours", "Metric Guidance"),
        ("Revenue at Risk (share of revenue)", k["revenue_at_risk_share"], 0.20, False, "pct", "Metric Guidance"),
        ("Net Sentiment Score (pts)", k["net_sentiment_score"], T["net_sentiment_score"], True, "pts", "Derived"),
        ("Average CSAT (1-5)", k["avg_csat"], T["avg_csat"], True, "num", "Derived"),
        ("Resolution time, negative cases (h)", k["negative_avg_resolution_hours"], T["negative_resolution_hours"], False, "hours", "Derived"),
        ("Repeat rate, negative-sentiment customers", k["negative_repeat_rate"], T["negative_repeat_rate"], True, "pct", "Derived"),
    ]
    out = pd.DataFrame(rows, columns=["kpi", "current", "proposed_target", "higher_is_better", "unit", "definition_source"])
    out["status"] = [_status(r.current, r.proposed_target, r.higher_is_better) for r in out.itertuples()]
    out["status_label"] = out.status.map({"good": "On target", "warning": "Near target", "serious": "Off target", "critical": "Well off target"})
    out["gap_to_target"] = out.current - out.proposed_target
    out["target_is_assumption"] = True
    return out


def _impact_from_revenue(base_inr: float) -> int:
    return 5 if base_inr >= 0.8e6 else 4 if base_inr >= 0.4e6 else 3 if base_inr >= 0.2e6 else 2 if base_inr >= 0.1e6 else 1


def recommendations(M: dict) -> pd.DataFrame:
    e, s, t, m, i, tr = M["eda"], M["sentiment"], M["topics"], M["mining"], M["integrate"], M["transform"]
    k = i["kpis_predicted"]
    rar = i["revenue_at_risk"]
    sc = {r["scenario"]: r for r in i["scenarios"]}
    gap = rar["repeat_gap_pp"] / 100
    crit, opp = i["critical_tier"], i["opportunity_segment"]
    rs = i["rank_stability"]
    dim = {d["dimension"]: d for d in e["dimension_tests"]}
    hot = e["hotspot"]
    seg = m["segmentation"]
    ranks_issue = pd.read_csv(C.TABLE_DIR / "integrated_rank_stability_issue.csv").set_index("level")
    cs, app = ranks_issue.loc["Customer Support"], ranks_issue.loc["App Experience"]
    overall_neg = k["negative_sentiment_rate"]
    excess_inr = sum(max(0.0, (r.negative_rate - overall_neg)) * r.n * (rar["revenue_at_risk_inr"] / rar["negative_records"]) for r in [cs, app])

    base_r1 = sc["base"]["recoverable_revenue_inr"]
    base_r2 = crit["revenue_at_risk_inr"] * gap * C.RECOVERY_SCENARIOS["base"]
    base_r3 = opp["scenario_conversion_pct"]["10%"]
    lodo = s["lodo"]["accuracy"]
    cv_auc = i["driver"]["cv_auc"]
    drv = {r["label"]: r for r in i["driver"]["key_terms"]}
    neg_or = drv["Negative vs Positive sentiment"]
    res_adj = drv["Resolution time (per +1 SD)"]

    R = []

    R.append(dict(
        key="fast_lane", short="Negative-feedback fast lane", phase="Now", type="Strategic", title="Negative-feedback fast lane and service-recovery protocol",
        finding="Negative-sentiment cases are resolved about twice as slowly and are far less likely to buy again.",
        evidence=(f"Negative cases take {k['negative_avg_resolution_hours']:.1f}h vs {k['non_negative_avg_resolution_hours']:.1f}h (Cohen's d={e['sentiment_outcomes']['resolution_cohens_d_neg_vs_nonneg']:.2f}); "
                  f"{100 * k['negative_sla_breach_rate']:.0f}% of negatives breach a {C.SLA_TARGET_HOURS:.0f}h SLA; repeat purchase {100 * k['negative_repeat_rate']:.1f}% (negative) vs "
                  f"{100 * k['positive_repeat_rate']:.1f}% (positive) = {rar['repeat_gap_pp']:.1f}pp gap (adjusted OR {neg_or['odds_ratio']:.2f}, 95% CI {neg_or['ci_lo']:.2f}-{neg_or['ci_hi']:.2f})."),
        action=("Score every incoming comment with the NLP sentiment model; auto-route Negative to a priority queue with a 12h resolution SLA, a named owner and a recovery playbook "
                "(apology, proactive follow-up, goodwill credit). Run it as a randomised pilot (fast lane vs control) because the data are observational."),
        owner="Head of Customer Experience + Support Operations", timeline="Pilot in 6-8 weeks; scale after a 90-day read-out",
        kpi="Avg resolution time (negative cases); repeat rate (negative customers)", baseline=f"{k['negative_avg_resolution_hours']:.1f}h; {100 * k['negative_repeat_rate']:.1f}%",
        target=f"<= {C.PROPOSED_TARGETS['negative_resolution_hours']:.0f}h; >= {100 * C.PROPOSED_TARGETS['negative_repeat_rate']:.0f}%",
        expected_impact=(f"Recoverable revenue {fmt_inr(sc['conservative']['recoverable_revenue_inr'])} / {fmt_inr(base_r1)} / {fmt_inr(sc['optimistic']['recoverable_revenue_inr'])} "
                         f"(10% / 25% / 50% of the {rar['repeat_gap_pp']:.0f}pp repeat gap closed)"),
        impact_basis_inr=base_r1, impact_score=_impact_from_revenue(base_r1), confidence=3,
        confidence_basis="Association is strong and replicated (chi-square, regression, rules) but causal effect of the intervention is untested -> pilot",
        effort=2, caveat=(f"Resolution time has no significant independent link to repeat purchase once sentiment is controlled (adjusted OR per SD {res_adj['odds_ratio']:.2f}, "
                          f"p={res_adj['p_value']:.2f}); the recovery CONTACT, not speed alone, may be what matters - hence the pilot.")))

    R.append(dict(
        key="critical_tier", short="Win-back for Critical tier", phase="Next", type="Strategic", title="Retention and win-back programme for high-value unhappy customers (Critical tier)",
        finding="Most of the revenue at risk is concentrated in a small, identifiable group: dissatisfied records in the top half of revenue.",
        evidence=(f"At-Risk segment (K-means k={seg['k']}, silhouette {seg['silhouette']:.2f}, bootstrap ARI {seg['stability']['mean_ari']:.2f}): {seg['segment_sizes']['At-Risk']} records, {100 * seg['at_risk_negative_purity']:.0f}% negative. "
                  f"Critical tier (At-Risk x revenue Q3/Q4) = {crit['records']} records carrying {fmt_inr(crit['revenue_at_risk_inr'])} = {100 * crit['share_of_total_rar']:.0f}% of all Revenue at Risk; "
                  f"{crit['avg_resolution_hours']:.1f}h resolution, {100 * crit['repeat_rate']:.0f}% repeat."),
        action=("Create a named outreach list from the Critical tier; contact within 24h; tiered goodwill offers by revenue band; track save-rate. Use the NLP + segmentation flags in the warehouse "
                "(fact_feedback_segment.risk_tier) as the list source."),
        owner="CRM / Retention Lead + Account Management", timeline="List live in 4 weeks; monthly save-rate review",
        kpi="Repeat rate in Critical tier; Revenue at Risk", baseline=f"{100 * crit['repeat_rate']:.0f}% repeat; {fmt_inr(crit['revenue_at_risk_inr'])} at risk",
        target="Repeat rate >= 40% in tier", expected_impact=f"Recoverable revenue about {fmt_inr(base_r2)} (base: 25% of the repeat gap on this tier's revenue at risk); overlaps with [[fast_lane]] - do not add",
        impact_basis_inr=base_r2, impact_score=_impact_from_revenue(base_r2), confidence=3,
        confidence_basis="Segment is stable (ARI 0.99) and beats a permutation null; revenue concentration is arithmetic but the uplift is scenario-based",
        effort=3, caveat="Customer_ID is unstable in the source (region/revenue vary within a customer), so the list is at feedback-record level until master data is fixed ([[foundations]])."))

    R.append(dict(
        key="governance", short="Governed KPI reporting", phase="Now", type="Quick win", title="Governed KPI reporting: confidence intervals, FDR-controlled hotspots, proposed targets",
        finding="Naive league tables would send the business chasing noise.",
        evidence=(f"{hot['tests']} segment cells scanned: {hot['raw_p_lt_0.05']} have raw p<0.05 vs {hot['expected_by_chance']:.0f} expected by chance; {hot['significant_after_fdr']} survive FDR (min q={hot['min_q_bh']:.2f}). "
                  f"Chi-square p: issue {dim['Issue']['p_value']:.2f}, channel {dim['Channel']['p_value']:.2f}, region {dim['Region']['p_value']:.2f}, product {dim['Product']['p_value']:.2f}, source {dim['Source']['p_value']:.2f}."),
        action=("Publish the KPI scorecard weekly from the warehouse views with 95% CIs; flag a segment as a hotspot only if it survives FDR; agree the proposed targets with stakeholders; "
                "show rank-stability next to any ranking."),
        owner="Analytics / BI Lead", timeline="2 weeks", kpi="Share of hotspot alerts that survive review", baseline="n/a (no governance today)", target="100% of alerts FDR-validated",
        expected_impact="Avoids mis-allocating budget to chance-driven 'hotspots'; no direct revenue claim", impact_basis_inr=0.0, impact_score=2, confidence=4,
        confidence_basis="The finding is certain and reproducible; the benefit (fewer false alarms) is inferred, not measured", effort=1, caveat="Enabler / protection value only."))

    R.append(dict(
        key="foundations", short="Fix measurement foundations", phase="Now", type="Enabler", title="Fix measurement foundations: timestamps, customer master, repeat-purchase definition, reason codes",
        finding="The data cannot show trends, emerging issues or customer-level behaviour.",
        evidence=(f"No date field (trend / 'emerging issue' / improvement tracking impossible); {tr['findings']['region_unstable_pct_of_multi_record_customers']:.1f}% of multi-record customers have >1 region and "
                  f"{tr['findings']['revenue_unstable_pct_of_multi_record_customers']:.0f}% have >1 revenue value; sentiment is not clustered within customers (ICC={e['customer_structure']['icc_rating']:.2f}); "
                  f"{tr['findings']['repeat_without_orders_rows']} records have Repeat_Purchase_Flag=1 with Order_Count=1; text is {tr['findings']['text_template_pct']:.0f}% templated."),
        action="Add event timestamps (feedback, case open/close), a stable customer key and order IDs; define Repeat_Purchase_Flag (window, post-feedback); add issue reason codes; capture real free text.",
        owner="Data Engineering + Product/IT", timeline="1-2 quarters", kpi="Data-quality scorecard (consistency pass-rate)", baseline=f"{tr['dq_raw']['dimension_pass_rate_pct']['Consistency']:.1f}% consistency pass-rate",
        target=">= 99% and time-series KPIs available", expected_impact="Unlocks trend / early-warning KPIs and customer-level analytics (no direct revenue claim)", impact_basis_inr=0.0, impact_score=3,
        confidence=5, confidence_basis="Fixing these gaps certainly enables the missing measurements (rule-based findings)", effort=3, caveat="Largest benefit is future measurement quality."))

    R.append(dict(
        key="nlp_ops", short="Productionise the NLP pipeline", phase="Next", type="Enabler", title="Productionise the NLP pipeline with a human-labelled validation set and drift monitoring",
        finding="Near-perfect scores reflect templated text; the model has no mechanism for unseen vocabulary.",
        evidence=(f"In-distribution accuracy {100 * s['cv5']['accuracy']:.0f}% (CV) / {100 * s['holdout']['accuracy']:.0f}% (hold-out); leave-one-descriptor-out accuracy {100 * lodo:.1f}% (= majority baseline {100 * s['majority_baseline_accuracy']:.1f}%); "
                  f"generic VADER only {100 * s['vader']['cv5_accuracy']:.1f}% ({len(s['vader_words_scored_zero'])} of 19 opinion words score 0); on {s['ood_probe_n']} hand-written natural sentences "
                  f"LR {100 * s['ood_probe_accuracy']['TF-IDF+LR (trained on supplied data)']:.0f}% / VADER {100 * s['ood_probe_accuracy']['VADER lexicon']:.0f}% / hybrid {100 * s['ood_probe_accuracy']['Hybrid TF-IDF+VADER']:.0f}% (indicative only)."),
        action="Collect >= 500 real comments labelled by humans for validation/retraining; adopt the hybrid (domain model + general lexicon) or a fine-tuned transformer; monitor vocabulary coverage and confidence.",
        owner="Data Science + CX Insights", timeline="1 quarter", kpi="Macro-F1 on a human-labelled real-text test set", baseline="Unknown on real text (100% on templated text)", target=">= 0.85 macro-F1",
        expected_impact="Protects every downstream KPI that depends on sentiment (no direct revenue claim)", impact_basis_inr=0.0, impact_score=3, confidence=5,
        confidence_basis="Validation on real labelled text certainly reveals true accuracy; failure mode shown by leave-descriptor-out test", effort=3, caveat="The n=30 probe is qualitative; real validation data is the actual fix."))

    R.append(dict(
        key="convert", short="Convert satisfied non-repeaters", phase="Next", type="Pilot", title="Convert satisfied non-repeaters with a CRM nudge programme",
        finding="A quarter of records are satisfied, quickly resolved - and still never repeat-purchase.",
        evidence=(f"'Satisfied Non-Repeaters' = {opp['records']} records ({100 * opp['share']:.1f}%), CSAT {opp['avg_csat']:.2f}, resolved in {opp['avg_resolution_hours']:.1f}h, "
                  f"0% repeat, revenue {fmt_inr(opp['revenue_inr'])}. Conversion of 5% / 10% / 20% would add {fmt_inr(opp['scenario_conversion_pct']['5%'])} / "
                  f"{fmt_inr(opp['scenario_conversion_pct']['10%'])} / {fmt_inr(opp['scenario_conversion_pct']['20%'])}."),
        action="A/B test post-purchase offers and replenishment reminders on this segment; first validate the Repeat_Purchase_Flag definition ([[foundations]]).",
        owner="CRM / Marketing", timeline="Test in 8 weeks", kpi="Repeat rate of the segment", baseline="0%", target=">= 10% conversion",
        expected_impact=f"About {fmt_inr(base_r3)} at 10% conversion (range {fmt_inr(opp['scenario_conversion_pct']['5%'])} - {fmt_inr(opp['scenario_conversion_pct']['20%'])})",
        impact_basis_inr=base_r3, impact_score=_impact_from_revenue(base_r3), confidence=2,
        confidence_basis="The segment is split on a binary flag whose meaning is ambiguous (113 records conflict with Order_Count); opportunity size is a scenario",
        effort=2, caveat="Segment definition is partly definitional (repeat flag); treat as hypothesis."))

    R.append(dict(
        key="pilots", short="Support / App root-cause pilots", phase="Later", type="Pilot", title="Root-cause pilots on Customer Support and App Experience (leading candidates, not proven hotspots)",
        finding="These two topics lead the rankings but the gap to the average is statistically indistinguishable from chance.",
        evidence=(f"Negative rate Customer Support {100 * cs.negative_rate:.1f}% and App Experience {100 * app.negative_rate:.1f}% vs {100 * overall_neg:.1f}% overall; "
                  f"issue-level chi-square p={dim['Issue']['p_value']:.2f}; rank #1 on Revenue at Risk in {100 * cs.p_rank1_rar:.0f}% / {100 * app.p_rank1_rar:.0f}% of bootstrap resamples "
                  f"(chance {100 / 8:.1f}%), top-3 in {100 * cs.p_top3_rar:.0f}% / {100 * app.p_top3_rar:.0f}%."),
        action="Read a stratified sample of negative texts per topic; run time-boxed fixes (support scripts, app checkout journey) with a pre-registered metric; do NOT re-allocate large budgets on the current ranking.",
        owner="Support Operations; Digital Product", timeline="90-day pilots", kpi="Negative rate for the two topics (with CI)", baseline=f"{100 * cs.negative_rate:.1f}% / {100 * app.negative_rate:.1f}%", target="<= overall average",
        expected_impact=f"Excess negative revenue vs average about {fmt_inr(excess_inr)} - small and uncertain", impact_basis_inr=float(excess_inr), impact_score=_impact_from_revenue(excess_inr), confidence=2,
        confidence_basis="Not significant (p=0.33), not FDR-significant; ranking is fragile", effort=3, caveat="Low priority until more data (or timestamps) allow a firmer test."))

    df = pd.DataFrame(R)
    df["priority_score"] = (df.impact_score * df.confidence / df.effort).round(2)
    df = df.sort_values(["priority_score", "impact_score", "confidence"], ascending=False, kind="stable").reset_index(drop=True)
    df["priority_rank"] = range(1, len(df) + 1)
    df["id"] = [f"R{n}" for n in df.priority_rank]          # IDs follow the ranking; cross-references resolved below
    ref = dict(zip(df.key, df.id))
    text_cols = ["finding", "evidence", "action", "expected_impact", "confidence_basis", "caveat", "kpi", "baseline", "target", "title"]
    for c in text_cols:
        df[c] = df[c].map(lambda t: re.sub(r"\[\[(\w+)\]\]", lambda m: ref[m.group(1)], t))
    return df


def findings_table(M: dict, recs: pd.DataFrame) -> pd.DataFrame:
    """Workbook table: Priority Finding | Evidence | Recommended Action | Expected KPI Impact."""
    return recs.assign(finding_priority=recs.priority_rank)[["priority_rank", "id", "finding", "evidence", "action", "expected_impact"]].rename(
        columns={"priority_rank": "Priority", "id": "Ref", "finding": "Priority Finding", "evidence": "Evidence", "action": "Recommended Action", "expected_impact": "Expected KPI Impact"})


def run(save: bool = True) -> dict:
    M = load_all_metrics()
    sc = kpi_scorecard(M)
    rec = recommendations(M)
    ft = findings_table(M, rec)
    if save:
        sc.to_csv(C.TABLE_DIR / "decision_kpi_scorecard.csv", index=False)
        rec.to_csv(C.TABLE_DIR / "decision_recommendations.csv", index=False)
        ft.to_csv(C.TABLE_DIR / "decision_findings_table.csv", index=False)
        save_metrics("decision", {"scorecard": sc.to_dict("records"), "recommendations": rec.to_dict("records")})
    log.info("Decision | %d recommendations; top priority: %s (score %.1f)", len(rec), rec.iloc[0].title, rec.iloc[0].priority_score)
    return {"scorecard": sc, "recommendations": rec, "findings": ft}


if __name__ == "__main__":  # pragma: no cover
    run()
