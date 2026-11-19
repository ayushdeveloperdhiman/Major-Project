#!/usr/bin/env python
"""Generate and EXECUTE the end-to-end analysis notebook (outputs are embedded, so it can be reviewed without running).

    python make_notebook.py        ->  notebooks/Customer_Voice_Intelligence_End_to_End.ipynb

Prerequisite: `python run_pipeline.py` has been run (the notebook reads the saved outputs and recomputes the quick parts live).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))      # works without `pip install -e .`

import nbformat as nbf  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from cvi import config as C  # noqa: E402

OUT = C.ROOT / "notebooks" / "Customer_Voice_Intelligence_End_to_End.ipynb"
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))  # noqa: E731
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))  # noqa: E731

md("""# AI-Powered Customer Voice Intelligence and Business Decision Platform
**Project MSDS-MAJOR-CVI-001 | Client: InsightRetail India (anonymised) | MSDS Major Project**

This notebook follows the 12 workbook activities. It **recomputes the quick parts live** (data-quality rules, fault injection, SQL KPIs, hypothesis tests, pre-processing, model scoring) and **loads the saved outputs** of the full pipeline (`python run_pipeline.py`) for the heavier steps (cross-validation, mining, bootstrap). Figures are the project's validated visual system (`outputs/figures`).

**Read first - three data facts that shape every result:** (1) the feedback text is 100% templated, so model scores of 100% are a ceiling effect; (2) `Reference_Sentiment` is a deterministic function of `Rating`; (3) `Customer_ID` is not a stable key, so every customer metric is at feedback-record level.""")
code("""
import warnings; warnings.filterwarnings("ignore")
import sys, json, sqlite3
from pathlib import Path
for _p in [Path.cwd(), *Path.cwd().parents]:          # find the project's src/ so the notebook runs without `pip install -e .`
    if (_p / "src" / "cvi").is_dir():
        sys.path.insert(0, str(_p / "src")); break
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
from scipy import stats
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 90)
from cvi import config as C, ingest, quality, transform, kpis as K, warehouse as W, nlp_prep, sentiment, topics
from cvi.utils import load_metrics
print("python", sys.version.split()[0], "| seed", C.SEED, "| project root:", C.ROOT.name)
def fig(name, width=860): display(Image(filename=str(C.FIG_DIR / f"{name}.png"), width=width))
def table(name): return pd.read_csv(C.TABLE_DIR / f"{name}.csv")
M = {k: load_metrics(k) for k in ["ingest", "transform", "eda", "sentiment", "topics", "mining", "integrate", "decision", "validate"]}
""")

md("## Activity 1 - Problem framing and success metrics\nFive analytical questions: (1) how much feedback is negative and does it vary by segment; (2) which themes dominate and carry risk; (3) how sentiment links to repeat purchase, CSAT, resolution time and revenue; (4) which segments are high-risk or high-opportunity; (5) how much revenue is at risk and which actions come first. KPI definitions follow the supplied *Metric Guidance*; targets are **proposed assumptions** (none were supplied).")
code("""
sc = table("decision_kpi_scorecard")
display(sc[["kpi", "current", "proposed_target", "status_label", "definition_source"]].round(4))
fig("fig27_kpi_scorecard", 820)
""")

md("## Activity 2 - Data engineering and ingestion\nBronze / Silver / Gold pipeline; one command (`python run_pipeline.py`) runs all 12 stages in about two minutes.")
code("""
fig("fig01_pipeline_architecture", 820)
raw = ingest.read_source()
print("source:", raw.shape, "| schema drift:", ingest.schema_drift(raw))
prof = ingest.profile_source(raw)
display(prof[["column", "dictionary_type", "non_null", "null_count", "distinct", "min", "max", "pattern_conformance_pct"]])
""")

md("## Activity 3 - Data quality and feature preparation\nThe 36 rules are executed live on the raw data, then **fault injection** proves the pipeline catches problems that the (clean) supplied data does not contain.")
code("""
bronze = ingest.load_bronze()
rep = quality.run_rules(bronze[C.EXPECTED_COLUMNS])
print("score:", {k: v for k, v in rep["score"].items() if k != "dimension_pass_rate_pct"})
display(rep["report"][rep["report"].rows_failed > 0][["rule_id", "dimension", "severity", "description", "rows_failed", "pct_failed", "status"]])
""")
code("""
# Fault injection on a 60-row sample: each fault must be caught and handled as specified
s = bronze.head(60).copy().reset_index(drop=True)
s.loc[1, "Region"] = "Mars"                      # invalid category      -> quarantine (DQ-V03)
s.loc[2, "Region"] = "north"                     # case variant          -> canonicalised to 'North'
s.loc[3, "Rating"] = "7"                         # impossible value      -> quarantine (DQ-V04)
s.loc[4, "Customer_Revenue_INR"] = pd.NA         # missing outcome       -> kept as NaN, never imputed (DQ-C03)
s.loc[5, "Channel"] = ""                         # blank categorical     -> 'Unknown' + flagged (DQ-C02)
s.loc[6, "Feedback_ID"] = s.loc[0, "Feedback_ID"]# duplicate ID          -> quarantine (DQ-U01)
out = transform.clean(s)
print("kept", len(out["silver"]), "| quarantined", len(out["quarantine"]))
display(out["quarantine"][["Feedback_ID", "dq_error_reason"]])
display(out["silver"].set_index("feedback_id").loc[[s.loc[2, "Feedback_ID"], s.loc[4, "Feedback_ID"], s.loc[5, "Feedback_ID"]], ["region", "customer_revenue_inr", "channel", "dq_flags"]])
display(pd.read_csv(C.DQ_DIR / "transformation_log.csv"))
fig("fig04_dq_findings", 820)
""")

md("## Activity 4 - Warehouse and analytical model\nStar schema at feedback grain (Customer_ID is a weak key). The KPI views in SQL are the semantic layer; they reconcile with the pandas KPI functions (28 of 28).")
code("""
fig("fig02_star_schema_erd", 820)
print(open(C.SQL_DIR / "schema.sql").read()[:1500], "...")
display(pd.read_csv(C.VALIDATION_DIR / "warehouse_reconciliation.csv"))
display(W.query("SELECT * FROM vw_kpi_overall").T.rename(columns={0: "value"}))
display(W.query("SELECT issue_category, feedback_count, topic_share, negative_sentiment_rate, repeat_purchase_rate, avg_resolution_hours, revenue_at_risk_inr FROM vw_kpi_by_issue ORDER BY negative_sentiment_rate DESC"))
""")

md("## Activity 5 - Exploratory business analytics\nNo pattern is reported without an effect size, an uncertainty statement and (when many cells are scanned) a multiple-testing correction.")
code("""
df = transform.load_silver()
# Pattern 1: sentiment vs repeat purchase (chi-square + Cramer's V)
ct = pd.crosstab(df.reference_sentiment, df.repeat_purchase_flag)
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"repeat purchase by sentiment: chi2={chi2:.1f}, p={p:.1e}, Cramer's V={np.sqrt(chi2 / (ct.values.sum())):.2f}")
print((df.groupby("reference_sentiment").repeat_purchase_flag.mean() * 100).round(1).to_dict())
# Pattern 2: resolution time
neg, non = df[df.is_negative_ref == 1].resolution_time_hours, df[df.is_negative_ref == 0].resolution_time_hours
print(f"resolution: negative {neg.mean():.1f} h vs other {non.mean():.1f} h | Mann-Whitney p={stats.mannwhitneyu(neg, non).pvalue:.1e}")
# Null results: segment dimensions vs sentiment
for dim in ["issue_category", "channel", "region", "product_category", "feedback_source"]:
    c2 = pd.crosstab(df[dim], df.reference_sentiment); print(f"  {dim:18s} chi-square p = {stats.chi2_contingency(c2)[1]:.2f}")
display(pd.DataFrame(M["eda"]["material_patterns"])[["id", "type", "pattern", "evidence"]])
""")
code("""
for f in ["fig03_data_profile", "fig07_sentiment_vs_outcomes", "fig06_negative_rate_by_dimension", "fig09_sentiment_by_resolution_band", "fig11_fdr_hotspot_scan"]:
    fig(f)
""")

md("## Activity 6 - Data mining\nK-means is compared with a **column-permutation null** (a good-looking silhouette is not evidence of clusters); association rules go through a Fisher + FDR + productivity funnel.")
code("""
sel = table("mining_cluster_selection"); display(sel.round(3))
display(table("mining_segment_profile")[["segment_name", "feedback_count", "negative_rate", "avg_csat", "avg_resolution_hours", "repeat_rate", "revenue_at_risk_inr", "rar_share_of_total"]].round(3))
print("rule funnel:", M["mining"]["association_rules"]["funnel"])
display(table("mining_association_rules_robust").round(3))
for f in ["fig19_cluster_selection", "fig20_segments", "fig21_association_rules"]:
    fig(f)
""")

md("## Activity 7 - NLP pre-processing\nA 10-step workflow (negation marking, lemmatisation, data-driven boilerplate removal), one object used in training, scoring and the dashboard.")
code("""
display(nlp_prep.describe_steps())
pre = nlp_prep.TextPreprocessor()
samples = ["The app experience experience was helpful. I used the web channel for my fashion order and this affected my overall experience.",
           "The delivery wasn't good, and the courier didn't call! Visit https://x.co or mail a@b.com <b>NOW</b> @brand #fail",
           "Not happy. Never again; refund delayed by 12 days -- very very disappointed"]
display(nlp_prep.before_after_examples(samples, pre))
st = nlp_prep.corpus_stats(nlp_prep.TextPreprocessor(return_tokens=True).transform(df.text_clean))
print({k: v for k, v in st.items() if k != "top_terms"})
""")

md("## Activity 8 - Sentiment analysis\nSix approaches, three evaluation regimes. **Read the numbers carefully:** the 100% scores are a ceiling effect of templated text; leave-one-opinion-word-out shows what happens on unseen vocabulary.")
code("""
comp = table("sentiment_model_comparison")
display(comp.pivot_table(index="model", columns="regime", values=["accuracy", "macro_f1"]).round(3))
s = M["sentiment"]
print("ablation (accuracy):", {k[:48]: round(v, 3) for k, v in s["ablation"].items()})
print("30-sentence natural-language probe:", {k: round(v, 2) for k, v in s["ood_probe_accuracy"].items()}, "(qualitative, n=30)")
for f in ["fig12_nlp_model_comparison", "fig13_confusion_matrices", "fig14_vader_by_descriptor", "fig15_learned_lexicon"]:
    fig(f)
""")
code("""
# Score new text with the saved production model, next to the general-purpose lexicon
new = ["The delivery delay experience was slow. I used the app channel for my home order and this affected my overall experience.",
       "Love the new app, checkout is easy and fast.",
       "Worst service ever, never ordering again.",
       "The order arrived on the expected date."]
out = sentiment.predict_sentiment(new)
out["vader"] = sentiment.vader_label(sentiment.vader_scores(new).vader_compound)
display(out.round(2))
""")

md("## Activity 9 - Topic / theme intelligence\nKeyword taxonomy (classify) + NMF on aspect words (discover). Topics describe what customers talk about, so the 19 opinion words are learned from sentiment association and removed first.")
code("""
t = M["topics"]
print("taxonomy accuracy vs Issue_Category:", t["taxonomy_accuracy_vs_issue"], "| opinion words removed:", t["n_opinion_words"])
print("NMF:", {k: round(v, 3) if isinstance(v, float) else v for k, v in t["nmf"].items()}, "| LDA:", {k: round(v, 3) for k, v in t["lda"].items()})
display(table("topics_nmf_topics"))
display(table("topics_prevalence_relevance")[["topic_issue", "feedback_count", "topic_share", "negative_rate", "repeat_rate", "revenue_at_risk_inr", "rar_share_of_total"]].round(3))
print("taxonomy on free text:", [topics.classify_theme(x, clause_only=False) for x in ["The refund is still pending", "The courier delivered it very late", "Checkout keeps crashing in the app"]])
for f in ["fig16_topic_selection", "fig17_theme_overview", "fig18_theme_by_channel"]:
    fig(f)
""")

md("## Activity 10 - Integrated customer intelligence\nText intelligence (predicted sentiment, theme) joined to revenue, repeat purchase, CSAT and resolution time in the warehouse view `vw_feedback_360`.")
code("""
display(table("integrated_kpis_by_sentiment")[["sent_pred", "feedback_count", "repeat_rate", "avg_resolution_hours", "avg_csat", "avg_revenue_inr"]].round(3))
display(table("integrated_repeat_driver_odds_ratios").round(4))
ig = M["integrate"]; rar = ig["revenue_at_risk"]
print(f"Revenue at Risk INR {rar['revenue_at_risk_inr']:,.0f} ({rar['rar_share']:.1%} of revenue), bootstrap 95% CI {rar['rar_ci'][0]:,.0f} - {rar['rar_ci'][1]:,.0f}")
display(pd.DataFrame(ig["scenarios"]).round(0)); display(pd.DataFrame(ig["decomposition"]).round(3))
print("critical tier:", ig["critical_tier"]); print("rank stability (top issue):", ig["rank_stability"]["top_issue_by_rar"])
for f in ["fig22_repeat_driver_forest", "fig23_revenue_at_risk", "fig24_rank_stability", "fig25_risk_matrix"]:
    fig(f)
""")

md("## Activity 11 - Business decision layer\nKPI scorecard, seven prioritised actions (impact x confidence / effort), and the dashboard. Run it with `python -m streamlit run dashboard/app.py` (use the project's Python: `./.venv/bin/python -m streamlit ...`); a single-file offline export is in `dashboard/static/cvi_dashboard.html`; screenshots are in `dashboard/screenshots/`.")
code("""
rec = table("decision_recommendations")
display(rec[["id", "phase", "type", "title", "impact_score", "confidence", "effort", "priority_score", "expected_impact"]])
fig("fig26_priority_ranking", 820)
display(Image(filename=str(C.ROOT / "dashboard" / "screenshots" / "streamlit_1_overview.png"), width=820))
""")

md("## Activity 12 - Validation, documentation and final review\n65 automated tests (written by the test run itself into the template's *Test Cases* layout), a run manifest and the limitations.")
code("""
tests = pd.read_csv(C.VALIDATION_DIR / "test_results.csv")
print(len(tests), "tests |", tests["Pass / Fail"].value_counts().to_dict())
display(tests.groupby("Component")["Pass / Fail"].agg(tests="size", passed=lambda x: int((x == "Pass").sum())).reset_index())
man = json.load(open(C.OUT_DIR / "run_manifest.json"))
print("input sha256:", man["input"]["sha256"][:16], "... | seed", man["seed"], "| python", man["environment"]["python"], "| stages:", list(man["stage_seconds"]))
print("packages:", {k: v for k, v in man["environment"]["packages"].items() if k in ("pandas", "numpy", "scikit-learn", "sklearn", "scipy", "statsmodels", "plotly", "streamlit")})
""")
md("""**Limitations (consolidated):** templated text and rating-implied labels; no timestamps and no stable customer key; cross-sectional data, so impacts are scenarios not forecasts; no competitor data was supplied (the 'Competitor Alpha' workbook holds only the data dictionary and metric guidance); targets and effort scores are proposed assumptions; English-only NLP; single-node tooling.""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(C.ROOT / "notebooks")}})
client.execute()
nbf.write(nb, OUT)
errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"wrote {OUT.name} | {len(cells)} cells | error outputs: {len(errs)} | size {OUT.stat().st_size / 1e6:.1f} MB")
