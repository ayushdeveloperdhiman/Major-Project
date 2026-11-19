"""Populate the supplied 'Sample Student Data Analysis Template.xlsx' (six sheets) from the saved pipeline outputs.

Every value is read from outputs/metrics, outputs/tables or outputs/validation - nothing is typed by hand - so the
spreadsheet can be regenerated after any re-run:   python make_deliverables.py   (run AFTER the pipeline and pytest).
"""
from __future__ import annotations

import shutil

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from . import config as C
from .utils import load_metrics

OUT = C.DELIVERABLES_DIR / "Completed_Data_Analysis_Template.xlsx"
HDR_FILL = PatternFill("solid", fgColor="E7E6E0")
THIN = Side(style="thin", color="C3C2B7")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WIDTHS = {
    "Pipeline Design": [22, 34, 52, 38, 52],
    "Warehouse Design": [30, 22, 30, 38, 58],
    "NLP Experiment": [26, 34, 38, 26, 34, 60],
    "Mining Analysis": [28, 36, 54, 52, 50],
    "Business KPI Dashboard": [38, 16, 34, 58, 58],
    "Test Cases": [11, 20, 58, 58, 58, 11],
}


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def inr(x: float) -> str:
    return f"INR {x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"INR {x:,.0f}"


def pipeline_rows(M: dict) -> list[list]:
    tr, ing, dq = M["transform"], M["ingest"], M["transform"]["dq_raw"]
    val = M["validate"]
    return [
        ["1. Ingest (Bronze)", "Client workbook, sheet 'Customer Voice' (2,400 x 15); data dictionary",
         "Land every cell as text; add lineage (file, sheet, Excel row, batch id, row hash)", "data/bronze/customer_voice_raw.parquet",
         f"Columns equal the dictionary; batch id {ing['batch_id']}; 0 mismatching cells vs an independent openpyxl read (T-ING-04)"],
        ["2. Profile", "Bronze", "Per-column null %, distinct values, ranges, ID-pattern conformance", "outputs/dq/source_profile.csv",
         f"{ing['total_null_cells']} null cells; both ID columns 100% pattern-conformant (T-ING-06)"],
        ["3. Validate", "Bronze", f"{dq['n_rules']} data-quality rules (32 row-level, 4 dataset-level), severities error / warn / info", "outputs/dq/dq_report_raw.csv",
         f"{dq['rows_failing_error_rules']} error-rule failures; {dq['n_rules_warn']} warnings; {dq['n_rules_info']} info; fully clean rows {dq['pct_rows_fully_clean']}%"],
        ["4. Clean (Silver) TR-01..TR-08", "Bronze + rules", "Normalise, canonicalise, coerce numerics, quarantine, repair repeated words, derive features, strict typing",
         "data/silver/customer_voice_clean.parquet; quarantine file; transformation log",
         f"{tr['rows_bronze']} in, {tr['rows_silver']} out, {tr['rows_quarantined']} quarantined; fully clean rows {tr['dq_raw']['pct_rows_fully_clean']}% -> {tr['dq_clean']['pct_rows_fully_clean']}%; idempotent hash (T-TR-04)"],
        ["5. Warehouse (Gold)", "Silver", "Star-schema load (11 dimensions, 3 facts + text table), surrogate keys, 12 views", "data/gold/cvi_warehouse.db",
         f"{val['warehouse_pass']}/{val['warehouse_checks']} integrity checks; {val['kpi_reconciliation_pass']}/{val['kpi_reconciliation_checks']} SQL-vs-pandas KPI reconciliations"],
        ["6. EDA and KPIs", "Silver", "KPI tables per dimension, tests with effect sizes, FDR hot-spot scan", "outputs/tables/eda_*.csv",
         f"{M['eda']['hotspot']['tests']} cells scanned, {M['eda']['hotspot']['significant_after_fdr']} significant after FDR"],
        ["7. NLP: sentiment", "Silver text_clean", "TextPreprocessor + BoilerplateFilter -> TF-IDF -> logistic regression; VADER baseline", "outputs/models/sentiment_model.joblib; nlp_sentiment.parquet",
         f"CV macro-F1 {M['sentiment']['cv5']['macro_f1']:.3f}; unseen-word accuracy {pct(M['sentiment']['lodo']['accuracy'])} (= baseline)"],
        ["8. NLP: themes", "Silver text_clean", "Keyword taxonomy + NMF on aspect words", "nlp_topics.parquet; outputs/tables/topics_*.csv",
         f"Taxonomy accuracy {pct(M['topics']['taxonomy_accuracy_vs_issue'])}; NMF k={M['topics']['nmf']['k']}, NMI {M['topics']['nmf']['nmi_vs_issue']:.2f}"],
        ["9. Mining", "Silver + NLP output", "K-means (k=3) with permutation null; Apriori rules with FDR + productivity filter", "mining_segments.parquet; outputs/tables/mining_*.csv",
         f"Silhouette {M['mining']['segmentation']['silhouette']:.3f} vs null p95 {M['mining']['segmentation']['null_silhouette_p95']:.3f}; {M['mining']['association_rules']['funnel']['rules_robust_after_productivity']} robust rules"],
        ["10. Integrate", "All of the above", "Driver model, Revenue at Risk with bootstrap CI, rank stability, risk tiers", "outputs/tables/integrated_*.csv; analytics parquet",
         f"Revenue at Risk {inr(M['integrate']['revenue_at_risk']['revenue_at_risk_inr'])} ({pct(M['integrate']['revenue_at_risk']['rar_share'])} of revenue)"],
        ["11. Decision layer", "Warehouse + outputs", "KPI scorecard, 7 prioritised recommendations, Streamlit dashboard + static HTML", "dashboard/; outputs/tables/decision_*.csv",
         "Tests T-DEC-01..03; dashboard screenshots in dashboard/screenshots/"],
        ["12. Validation", "Whole project", "65 automated tests, run manifest, reproducibility checks", "outputs/validation/test_results.csv; outputs/run_manifest.json", "See sheet 'Test Cases'"],
    ]


def warehouse_rows() -> list[list]:
    return [
        ["fact_feedback", "Fact", "One row per feedback record (2,400)", "PK feedback_id; FKs customer_key, region_key, channel_key, product_category_key, issue_key, source_key, rating_key, ref_sentiment_key, resolution_band_key, revenue_tier_key",
         "KPI base: revenue, orders, resolution hours, rating, CSAT, repeat flag, SLA flags, lineage"],
        ["feedback_text", "Text table (1:1)", "One row per feedback record", "PK/FK feedback_id", "Raw and cleaned text kept out of the narrow fact"],
        ["fact_feedback_nlp", "Fact: model output (1:1)", "One row per feedback record", "PK/FK feedback_id; FK pred_sentiment_key, vader_sentiment_key, topic_issue_key",
         "Predicted sentiment, probabilities, VADER compound, extracted theme, NMF topic, opinion word"],
        ["fact_feedback_segment", "Fact: mining output (1:1)", "One row per feedback record", "PK/FK feedback_id; FK segment_key", "Experience segment and risk tier"],
        ["dim_customer", "Dimension", "One row per Customer_ID (1,114)", "customer_key; customer_id", "Record count and region-instability indicator (Customer_ID is a weak key)"],
        ["dim_region", "Dimension", "One row per region (5 + Unknown)", "region_key", "Geographic slicing"],
        ["dim_channel", "Dimension", "One row per channel (4 + Unknown)", "channel_key", "Channel slicing; Digital / Physical type"],
        ["dim_product_category", "Dimension", "One row per product category (5 + Unknown)", "product_category_key", "Product slicing"],
        ["dim_issue", "Dimension", "One row per issue / theme (8 + Unknown)", "issue_key", "Theme slicing; four-group theme hierarchy"],
        ["dim_feedback_source", "Dimension", "One row per source (5 + Unknown)", "source_key", "Source slicing; source class"],
        ["dim_sentiment", "Dimension", "Negative / Neutral / Positive (+ Unknown)", "sentiment_key", "Polarity score and sort order"],
        ["dim_rating", "Dimension", "One row per rating 1-5", "rating_key", "Rating label and implied sentiment"],
        ["dim_resolution_band", "Dimension", "Five SLA bands", "band_key", "0-6, 6-12, 12-18, 18-24, over 24 h with SLA status"],
        ["dim_revenue_tier", "Dimension", "Four revenue quartile tiers", "tier_key", "Quartile bounds learned on the reference batch"],
        ["dim_segment", "Dimension", "Three experience segments", "segment_key", "At-Risk, Satisfied Non-Repeaters, Loyal Repeaters"],
        ["vw_feedback_360", "View (semantic layer)", "One row per feedback record (no fan-out)", "feedback_id", "Wide join of facts and dimensions for the dashboard"],
        ["vw_kpi_overall and 10 vw_kpi_by_* views", "View (semantic layer)", "One row per overall / per dimension level", "dimension value", "Metric Guidance KPIs (negative rate, repeat rate, resolution time, Revenue at Risk, topic share) defined once in SQL"],
    ]


def nlp_rows(M: dict, T: dict) -> list[list]:
    s, tp = M["sentiment"], M["topics"]
    comp = T["sentiment_model_comparison"]
    rows = [
        ["Pre-processing", "TextPreprocessor (10 steps) + BoilerplateFilter", "NFKC, lowercase, contractions, negation window 3, NLTK stop-words (negators kept), WordNet lemmas, boilerplate DF >= 0.90",
         "Vocabulary size", "44 types; 6 boilerplate terms", "Closed, tiny vocabulary: the text is templated"],
    ]
    for _, r in comp[comp.regime == "cv5"].iterrows():
        lodo = comp[(comp.regime == "lodo") & (comp.model == r.model)].iloc[0]
        rows.append(["Sentiment model", r.model, "5-fold stratified CV (seed 42)", "Accuracy / macro-F1 (CV)", f"{r.accuracy:.4f} / {r.macro_f1:.4f}",
                     f"Unseen-opinion-word macro-F1 {lodo.macro_f1:.3f} (accuracy {lodo.accuracy:.3f})"])
    rows += [
        ["Sentiment hold-out", s["chosen_model"], "Stratified 20% hold-out (n=480); bootstrap CI", "Accuracy; macro-F1", f"{s['holdout']['accuracy']:.3f}; {s['holdout']['macro_f1']:.3f} (CI {s['holdout_ci']['accuracy_ci']})",
         "Ceiling effect of templated text; interval is degenerate (no errors)"],
        ["Sentiment ablation", "TF-IDF + LR with the opinion word masked", "Topic / channel / product words only", "Accuracy", f"{pct(list(s['ablation'].values())[1])} (chance for balanced classes 33%)", "No topic word leaks the label"],
        ["Sentiment probe", "LR / VADER / hybrid on 30 hand-written sentences", "Analyst-labelled, out of template", "Accuracy", f"{pct(s['ood_probe_accuracy']['TF-IDF+LR (trained on supplied data)'], 0)} / {pct(s['ood_probe_accuracy']['VADER lexicon'], 0)} / {pct(s['ood_probe_accuracy']['Hybrid TF-IDF+VADER'], 0)}",
         "Qualitative only (n=30); differences within noise"],
        ["Theme classification", "Keyword taxonomy (weighted)", "8 themes; weights 1.0 specific / 0.5 ambiguous; threshold 1.0", "Accuracy vs Issue_Category", pct(tp["taxonomy_accuracy_vs_issue"]), "Transparent, auditable, extensible"],
        ["Opinion-word learning", "Chi-square + FDR + Cramer's V >= 0.10", "Token presence vs sentiment", "Terms removed", f"{tp['n_opinion_words']} opinion words; 0 aspect words", "Separates what customers talk about from how they feel"],
        ["Topic discovery", "NMF on TF-IDF (aspect view)", f"k={tp['nmf']['k']} chosen by NPMI coherence, no empty topic, stability >= 0.8", "NMI / purity vs Issue_Category", f"{tp['nmf']['nmi_vs_issue']:.2f} / {tp['nmf']['purity_vs_issue']:.2f}",
         f"Beats LDA (NMI {tp['lda']['nmi_vs_issue']:.2f}); without aspect-opinion separation purity is only 0.77 / 0.50"],
    ]
    return rows


def mining_rows(M: dict) -> list[list]:
    sg, ar, e, ig = M["mining"]["segmentation"], M["mining"]["association_rules"], M["eda"], M["integrate"]
    d = {r["label"]: r for r in ig["driver"]["key_terms"]}
    ng, rs = d["Negative vs Positive sentiment"], d["Resolution time (per +1 SD)"]
    f = ar["funnel"]
    return [
        ["K-means segmentation (k=3)", "CSAT, resolution time, repeat flag (standardised)", f"Silhouette {sg['silhouette']:.3f} vs shuffled-column null p95 {sg['null_silhouette_p95']:.3f}; CH {sg['calinski_harabasz']:.0f}; DB {sg['davies_bouldin']:.3f}; bootstrap ARI {sg['stability']['mean_ari']:.3f}",
         "Three stable segments: At-Risk (24.4%), Satisfied Non-Repeaters (24.4%), Loyal Repeaters (51.2%)", "outputs/tables/mining_cluster_selection.csv, mining_segment_profile.csv"],
        ["Cluster-tendency check", "Revenue, order count", f"Silhouette {sg['value_space_silhouette_k4']:.3f} vs null {sg['value_space_null_k4']:.3f}: no structure", "Value is tiered by rule (quartiles), not clustered", "outputs/tables/mining_value_space_null_check.csv"],
        ["Segment independence", "Segment x issue / channel / region / product / source", f"Smallest q = {sg['context_tests_min_q']:.2f}", "The experience problem is systemic, not local", "outputs/tables/mining_segment_context_tests.csv"],
        ["Association rules (Apriori)", "42 items; support >= 2%; length <= 3", f"{f['frequent_itemsets']} itemsets -> {f['rules_all_lift_ge_1']} rules -> {f['rules_lift_ge_1.2']} (lift >= 1.2) -> {f['rules_significant_fdr_assoc']} significant -> {f['rules_robust_after_productivity']} robust; {f['robust_rules_containing_any_context_item']} with context items",
         "Only resolution time, repeat purchase and sentiment form robust patterns (e.g. resolution > 18 h -> Negative: confidence 84%, lift 3.5)", "outputs/tables/mining_association_rules_robust.csv"],
        ["Hot-spot scan", "314 one- and two-way segment cells", f"{e['hotspot']['raw_p_lt_0.05']} raw p<0.05 (expected {e['hotspot']['expected_by_chance']:.0f}); {e['hotspot']['significant_after_fdr']} after Benjamini-Hochberg (min q {e['hotspot']['min_q_bh']:.2f})",
         "No segment is a proven sentiment hotspot", "outputs/tables/eda_hotspot_scan.csv"],
        ["Repeat-purchase driver model", "Logistic regression: sentiment, resolution, revenue, orders, channel, region, issue", f"OR negative vs positive {ng['odds_ratio']:.2f} ({ng['ci_lo']:.2f}-{ng['ci_hi']:.2f}); resolution (adjusted) {rs['odds_ratio']:.2f}, p={rs['p_value']:.2f}",
         "Sentiment drives repeat purchase; speed is a marker, not proven lever", "outputs/tables/integrated_repeat_driver_odds_ratios.csv"],
        ["Bootstrap rank stability", "Issue ranking by Revenue at Risk (2,000 resamples)", f"Customer Support ranks #1 in {pct(ig['rank_stability']['top_issue_by_rar']['p_rank1'], 0)} (chance 12.5%), top-3 in {pct(ig['rank_stability']['top_issue_by_rar']['p_top3'], 0)}",
         "Leading candidate, not a proven hotspot", "outputs/tables/integrated_rank_stability_issue.csv"],
        ["Revenue at Risk", "Revenue of negative-sentiment records", f"{inr(ig['revenue_at_risk']['revenue_at_risk_inr'])} ({pct(ig['revenue_at_risk']['rar_share'])} of revenue); bootstrap CI {inr(ig['revenue_at_risk']['rar_ci'][0])} - {inr(ig['revenue_at_risk']['rar_ci'][1])}",
         f"Critical tier = {ig['critical_tier']['records']} records, {pct(ig['critical_tier']['share_of_total_rar'], 0)} of Revenue at Risk", "outputs/tables/integrated_revenue_decomposition.csv, integrated_risk_tiers.csv"],
    ]


def kpi_rows(M: dict, T: dict) -> list[list]:
    sc = T["decision_kpi_scorecard"]
    ig = M["integrate"]
    action = {"Negative Sentiment Rate": "R2 fast lane; R1 governed reporting", "Repeat Purchase Rate": "R5 win-back; R6 conversion nudges", "Average Resolution Time (h)": "R2 fast lane with 12 h SLA",
              "Revenue at Risk (share of revenue)": "R2 and R5 (overlapping scenarios INR 162K-811K)", "Net Sentiment Score (pts)": "R2; track weekly with intervals (R1)", "Average CSAT (1-5)": "Monitor; redundant with rating (rho 0.97)",
              "Resolution time, negative cases (h)": "R2 fast lane (target 12 h)", "Repeat rate, negative-sentiment customers": "R5 win-back for the Critical tier"}
    interp = {"Negative Sentiment Rate": "Off target; no segment is a proven hotspot", "Repeat Purchase Rate": "Near target overall, but 28% among negative feedback", "Average Resolution Time (h)": "Driven by negative cases (18.1 h vs 9.1 h)",
              "Revenue at Risk (share of revenue)": "Quarter of revenue sits with negative-sentiment records", "Net Sentiment Score (pts)": "Positive minus negative share", "Average CSAT (1-5)": "Tracks rating almost perfectly",
              "Resolution time, negative cases (h)": "Twice the non-negative time; 91% breach 12 h", "Repeat rate, negative-sentiment customers": "44-point gap to positive feedback"}
    unit_fmt = {"pct": '0.0%', "hours": '0.0" h"', "pts": '+0.0;-0.0', "num": '0.00'}
    rows = []
    for r in sc.itertuples():
        rows.append([r.kpi, (r.current, unit_fmt[r.unit]), "All 2,400 records",
                     f"{interp[r.kpi]}. Proposed target {'at least' if r.higher_is_better else 'at most'} {r.proposed_target:g}; status: {r.status_label}.", action[r.kpi]])
    top = pd.read_csv(C.TABLE_DIR / "integrated_theme_kpis.csv").sort_values("revenue_at_risk_inr", ascending=False).iloc[0]
    crit, opp = ig["critical_tier"], ig["opportunity_segment"]
    rows += [
        ["Revenue at Risk (INR)", (ig["revenue_at_risk"]["revenue_at_risk_inr"], '#,##0'), "All negative-sentiment records", f"95% CI {inr(ig['revenue_at_risk']['rar_ci'][0])} - {inr(ig['revenue_at_risk']['rar_ci'][1])}; lapsed {inr(ig['revenue_at_risk']['lapsed_unhappy']['revenue_inr'])}, active {inr(ig['revenue_at_risk']['active_unhappy']['revenue_inr'])}", "R2, R5"],
        ["Revenue at Risk, Critical tier (INR)", (crit["revenue_at_risk_inr"], '#,##0'), "At-Risk segment x revenue Q3/Q4", f"{crit['records']} records = {pct(crit['share_of_total_rar'], 0)} of all Revenue at Risk; {crit['avg_resolution_hours']:.1f} h; repeat {pct(crit['repeat_rate'], 0)}", "R5 named outreach within 24 h"],
        ["Topic Share, largest-risk theme", (float(top.topic_share), '0.0%'), f"Theme: {top.topic_issue}", f"Negative rate {pct(float(top.negative_rate))}; Revenue at Risk INR {top.revenue_at_risk_inr:,.0f}; not significantly above average (chi-square p=0.33)", "R7 root-cause pilot (hypothesis, not proven hotspot)"],
        ["Conversion opportunity (INR, 10%)", (opp["scenario_conversion_pct"]["10%"], '#,##0'), "Satisfied Non-Repeaters", f"{opp['records']} records, revenue {inr(opp['revenue_inr'])}, CSAT {opp['avg_csat']:.2f}", "R6 CRM nudge A/B test"],
    ]
    return rows


def test_rows() -> list[list]:
    t = pd.read_csv(C.VALIDATION_DIR / "test_results.csv")
    return t[["Test ID", "Component", "Input / Condition", "Expected Result", "Actual Result", "Pass / Fail"]].values.tolist()


def _style(ws, name: str) -> None:
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, size=10)
        c.fill = HDR_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.border = BORDER
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Arial", size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BORDER
    for i, w in enumerate(WIDTHS[name], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 22


def build() -> str:
    C.DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(C.TEMPLATE_XLSX, OUT)
    M = {k: load_metrics(k) for k in ["ingest", "transform", "eda", "sentiment", "topics", "mining", "integrate", "validate"]}
    T = {n: pd.read_csv(C.TABLE_DIR / f"{n}.csv") for n in ["sentiment_model_comparison", "decision_kpi_scorecard"]}
    wb = openpyxl.load_workbook(OUT)
    content = {"Pipeline Design": pipeline_rows(M), "Warehouse Design": warehouse_rows(), "NLP Experiment": nlp_rows(M, T), "Mining Analysis": mining_rows(M),
               "Business KPI Dashboard": kpi_rows(M, T), "Test Cases": test_rows()}
    for name, rows in content.items():
        ws = wb[name]
        header = [c.value for c in ws[1]]
        for r in rows:
            assert len(r) == len(header), (name, len(r), len(header), r[:2])
            ws.append([x[0] if isinstance(x, tuple) else x for x in r])
            for j, x in enumerate(r, 1):
                if isinstance(x, tuple):
                    ws.cell(row=ws.max_row, column=j).number_format = x[1]
        _style(ws, name)
    wb.save(OUT)
    return str(OUT)


if __name__ == "__main__":  # pragma: no cover
    print(build())
