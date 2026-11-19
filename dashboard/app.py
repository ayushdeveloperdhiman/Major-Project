"""Customer Voice Intelligence - management dashboard (Streamlit).

Run:   python -m streamlit run dashboard/app.py      (with the project's Python: ./.venv/bin/python -m streamlit run dashboard/app.py)
       A bare `streamlit ...` command runs whichever Python owns the launcher on your PATH (often Anaconda), not the venv.
Data:  warehouse view vw_feedback_360 (filtered interactively) + statistical outputs computed on the full dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import charts as CH  # noqa: E402
import data as DT  # noqa: E402
from cvi import config as C  # noqa: E402
from cvi import kpis as K  # noqa: E402
from cvi.utils import DASHBOARD_PACKAGES, environment_issues  # noqa: E402

st.set_page_config(page_title="Customer Voice Intelligence", page_icon="\U0001F4CA", layout="wide", initial_sidebar_state="collapsed")


def theme_name() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


TH = theme_name()
st.markdown(CH.css(TH), unsafe_allow_html=True)
st.markdown("<style>.block-container{padding-top:1.4rem;max-width:1280px}</style>", unsafe_allow_html=True)

# A bare `streamlit` command often belongs to another Python (e.g. Anaconda) that lacks vaderSentiment and has a different
# scikit-learn, so the saved model cannot be used.  Say so, instead of leaving the viewer with an unrelated traceback.
_env_issues = environment_issues(only=DASHBOARD_PACKAGES)
if _env_issues:
    st.warning(f"This dashboard is running on `{sys.executable}`, which differs from the project's pinned environment: " + "; ".join(_env_issues)
               + ". The charts are unaffected, but the 'Try the sentiment model' panel may be limited. For the full dashboard start it with "
               "`./.venv/bin/python -m streamlit run dashboard/app.py` (or install the pins: `python -m pip install -r requirements.txt`).")


@st.cache_data(show_spinner=False)
def frame() -> pd.DataFrame:
    return DT.load_frame()


@st.cache_data(show_spinner=False)
def stats() -> dict:
    return DT.metrics()


@st.cache_data(show_spinner=False)
def tables() -> dict:
    names = ["integrated_repeat_driver_odds_ratios", "integrated_rank_stability_issue", "decision_recommendations", "decision_kpi_scorecard", "sentiment_model_comparison",
             "eda_dimension_tests", "mining_segment_profile", "topics_negative_descriptor_themes"]
    return {n: DT.table(n) for n in names}


def block(title: str, result, caption: str | None = None, key: str = "", note: str | None = None) -> None:
    fig, tbl = result
    st.markdown(f"**{title}**")
    if caption:
        st.caption(caption)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False}, key=f"ch_{key}")
    if note:
        st.caption(note)
    if tbl is not None and len(tbl):
        with st.expander("Table view"):
            st.dataframe(tbl, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------- header --
df_all, M, TB = frame(), stats(), tables()
st.markdown("### AI-Powered Customer Voice Intelligence and Business Decision Platform")
st.caption("InsightRetail India (anonymised) | MSDS-MAJOR-CVI-001 | 2,400 feedback records | sentiment = NLP prediction | KPI definitions follow the supplied Metric Guidance")

# one filter row above everything it scopes
opts = DT.options(df_all)
cols = st.columns(len(DT.FILTERS) + 1)
sel = {}
for c, (label, _) in zip(cols, DT.FILTERS):
    sel[label] = c.multiselect(label, opts[label], default=[], placeholder="All")
df = DT.apply_filters(df_all, sel)
cols[-1].metric(f"Records in view (of {len(df_all):,})", f"{len(df):,}")

st.markdown(CH.kpi_tiles_html(df), unsafe_allow_html=True)
st.caption("Targets are PROPOSED assumptions (none were supplied); status = icon + label. Statistical tests, bootstrap rankings, driver model and model validation are computed once on the full dataset and do not change with the filters.")

tab_ov, tab_voc, tab_drv, tab_seg, tab_act, tab_health = st.tabs(["Overview", "Voice of customer", "Drivers", "Segments & risk", "Actions", "Data & model health"])
e = M["eda"]; ig = M["integrate"]

# ----------------------------------------------------------------- overview --
with tab_ov:
    c1, c2 = st.columns([1, 1.6])
    k = K.overall_kpis(df, "sentiment_pred") if len(df) else None
    with c1:
        if k:
            st.markdown(f"<div class='cvi-hero'><div class='n'>{k['net_sentiment_score']:+.1f}</div><div class='l'>Net Sentiment Score (positive % minus negative %)</div></div>", unsafe_allow_html=True)
            st.caption(f"Negative rate {100 * k['negative_sentiment_rate']:.1f}%  (95% CI {100 * k['negative_rate_ci'][0]:.1f}-{100 * k['negative_rate_ci'][1]:.1f}%)")
    with c2:
        block("Sentiment mix", CH.sentiment_mix(df, TH), key="mix")
        block("Revenue by sentiment", CH.revenue_split(df, TH), key="revsplit", caption="Negative-sentiment revenue = Revenue at Risk (Metric Guidance)")
    st.markdown("#### Headline findings")
    f1, f2, f3 = st.columns(3)
    so = e["sentiment_outcomes"]
    f1.markdown(f"<div class='cvi-card'><h4>Sentiment drives repeat purchase</h4><p>Repeat rate {100 * ig['kpis_predicted']['negative_repeat_rate']:.1f}% (negative) vs {100 * ig['kpis_predicted']['positive_repeat_rate']:.1f}% (positive): a {ig['revenue_at_risk']['repeat_gap_pp']:.1f}pp gap, chi-square p&lt;0.001.</p></div>", unsafe_allow_html=True)
    f2.markdown(f"<div class='cvi-card'><h4>Negative cases are resolved twice as slowly</h4><p>{ig['kpis_predicted']['negative_avg_resolution_hours']:.1f}h vs {ig['kpis_predicted']['non_negative_avg_resolution_hours']:.1f}h (Cohen's d {so['resolution_cohens_d_neg_vs_nonneg']:.1f}); {100 * ig['kpis_predicted']['negative_sla_breach_rate']:.0f}% of negatives breach a 12h SLA.</p></div>", unsafe_allow_html=True)
    hs = e["hotspot"]
    f3.markdown(f"<div class='cvi-card'><h4>No segment is a proven hotspot</h4><p>{hs['tests']} segment cells scanned: {hs['raw_p_lt_0.05']} raw p&lt;0.05 (chance: ~{hs['expected_by_chance']:.0f}); {hs['significant_after_fdr']} survive FDR control. Treat issue rankings as hypotheses.</p></div>", unsafe_allow_html=True)
    dim = st.selectbox("Negative rate by", [d for d, _ in [("Issue", "issue_category"), ("Channel", "channel_name"), ("Region", "region_name"), ("Product", "product_category_name"), ("Source", "source_name")]], index=0)
    colmap = {"Issue": "issue_category", "Channel": "channel_name", "Region": "region_name", "Product": "product_category_name", "Source": "source_name"}
    p = TB["eda_dimension_tests"].set_index("dimension").loc[dim, "p_value"]
    block(f"Negative-feedback rate by {dim.lower()} (95% CI)", CH.neg_rate_by(df, colmap[dim], dim, TH), key="negdim",
          caption=f"Whiskers: Wilson 95% CI. Full-data chi-square test p = {p:.2f}: differences are not statistically significant." if p > 0.05 else f"Full-data chi-square p = {p:.3f}.")

# ------------------------------------------------------------------- voice ---
with tab_voc:
    a, b = st.columns(2)
    with a:
        block("Themes: share of feedback vs share of Revenue at Risk", CH.theme_overview(df, TH), key="theme", caption="Topic Share = feedback assigned to a theme / total feedback (Metric Guidance).")
    with b:
        block("Theme mix by channel", CH.theme_heatmap(df, TH), key="themeheat", caption="Evenly spread: chi-square p = 0.62 (no theme depends on channel).")
    st.markdown("#### Try the sentiment model")
    txt = st.text_area("Type a customer comment", "The delivery delay experience was slow. I used the app channel for my home order and this affected my overall experience.", height=80)
    if txt.strip():
        from cvi import sentiment as S
        from cvi import topics as TP
        r1, r2, r3 = st.columns(3)
        try:    # each card is independent: an unusable saved model or a missing package must never stop the rest of the page rendering
            out = S.predict_sentiment([txt]).iloc[0]
            r1.metric("Domain model (TF-IDF + LR)", out.sentiment); r1.caption(f"P(neg) {out.p_negative:.2f} | P(neu) {out.p_neutral:.2f} | P(pos) {out.p_positive:.2f}")
        except Exception as err:    # typically: the model was pickled by a different scikit-learn from the one running
            r1.metric("Domain model (TF-IDF + LR)", "not available")
            r1.caption(f"Saved model cannot be used by this Python ({type(err).__name__}). Start with ./.venv/bin/python -m streamlit run dashboard/app.py")
        try:
            v = S.vader_scores([txt]).iloc[0]
            r2.metric("General lexicon (VADER)", str(S.vader_label([v.vader_compound])[0])); r2.caption(f"compound score {v.vader_compound:+.2f}")
        except ModuleNotFoundError:
            r2.metric("General lexicon (VADER)", "not available")
            r2.caption("vaderSentiment is not installed for the Python running this app. Start with: ./.venv/bin/python -m streamlit run dashboard/app.py")
        r3.metric("Detected theme", TP.classify_theme(txt, clause_only=False))
        st.caption("Caution: the domain model was trained on templated text with 19 opinion words; for words it has never seen it falls back to the prior (see 'Data & model health'). Compare with the general lexicon.")

# ------------------------------------------------------------------ drivers --
with tab_drv:
    block("What sentiment changes: repeat purchase, resolution time - and revenue (no effect)", CH.outcomes(df, TH), key="outcomes", caption="Mean by sentiment class with 95% CI whiskers.")
    a, b = st.columns(2)
    with a:
        block("Resolution-time distribution", CH.resolution_hist(df, TH), key="reshist", caption=f"Resolution time alone predicts negative sentiment with AUC {e['resolution_threshold']['auc_resolution_predicts_negative']:.2f}.")
    with b:
        block("Sentiment mix by resolution-time band", CH.sentiment_by_band(df, TH), key="band", caption="Negative share rises from ~0% (<=6h) to 100% (>24h).")
    block("Repeat-purchase drivers: adjusted odds ratios (full dataset)", CH.forest(TB["integrated_repeat_driver_odds_ratios"], TH), key="forest",
          caption="Sentiment is the driver. Resolution time loses its effect once sentiment is accounted for (adjusted OR 0.92, p = 0.17): faster handling is a marker of negative experience, not proven to lift repeat purchase.")

# ----------------------------------------------------------------- segments --
with tab_seg:
    a, b = st.columns([1.1, 1])
    with a:
        block("Experience segments (K-means, k = 3)", CH.segment_profile(df, TH), key="segprof", caption="Standardised CSAT, resolution time and repeat flag; stable under bootstrap (ARI 0.99).")
    with b:
        block("Segment map (PCA)", CH.segment_scatter(df, TH), key="segmap")
    a, b = st.columns([1.1, 1])
    with a:
        block("Where the Revenue at Risk sits: segment x revenue tier", CH.risk_matrix(df, TH), key="riskm", caption="Outlined cells = Critical tier (At-Risk in the top half of revenue).")
    with b:
        rar = ig["revenue_at_risk"]
        st.markdown("**Revenue at Risk (full dataset)**")
        st.markdown(f"<div class='cvi-card'><h4>INR {rar['revenue_at_risk_inr'] / 1e6:.2f}M &nbsp;({100 * rar['rar_share']:.1f}% of revenue)</h4><p>Bootstrap 95% CI INR {rar['rar_ci'][0] / 1e6:.2f}M - {rar['rar_ci'][1] / 1e6:.2f}M</p>"
                    f"<p>Lapsed (no repeat): INR {rar['lapsed_unhappy']['revenue_inr'] / 1e6:.2f}M / {rar['lapsed_unhappy']['records']} records</p><p>Still active: INR {rar['active_unhappy']['revenue_inr'] / 1e6:.2f}M / {rar['active_unhappy']['records']} records</p>"
                    f"<p>Critical tier: {ig['critical_tier']['records']} records = {100 * ig['critical_tier']['share_of_total_rar']:.0f}% of all Revenue at Risk</p></div>", unsafe_allow_html=True)
        block("Recoverable revenue scenarios", CH.scenarios(ig, TH), key="scen", caption="Scenarios, not forecasts: share of the repeat-purchase gap assumed recoverable (data are cross-sectional).")
    block("Is the top issue statistically firm? Bootstrap rank stability (full dataset)", CH.rank_stability(TB["integrated_rank_stability_issue"], TH), key="rank",
          caption="Customer Support ranks #1 in 43% of 2,000 resamples (chance 12.5%): a leading candidate, not a proven hotspot.")

# ------------------------------------------------------------------ actions --
with tab_act:
    rec = TB["decision_recommendations"]
    block("Prioritised actions", CH.priority(rec, TH), key="prio", caption="Priority = Impact x Confidence / Effort. Impact = base-scenario recoverable revenue; effort is analyst judgement (assumption).")
    for r in rec.itertuples():
        with st.expander(f"{r.id}  |  {r.title}   [{r.phase}  |  {r.type}  |  score {r.priority_score:.1f}]"):
            st.markdown(f"**Finding.** {r.finding}")
            st.markdown(f"**Evidence.** {r.evidence}")
            st.markdown(f"**Action.** {r.action}")
            st.markdown(f"**Owner / timeline.** {r.owner} | {r.timeline}")
            st.markdown(f"**KPI.** {r.kpi}: baseline {r.baseline} -> target {r.target}")
            st.markdown(f"**Expected impact.** {r.expected_impact}")
            st.markdown(f"**Confidence ({r.confidence}/5).** {r.confidence_basis}")
            st.markdown(f"**Caveat.** {r.caveat}")
    st.markdown("#### KPI scorecard")
    raw = TB["decision_kpi_scorecard"]
    fmt = {"pct": lambda v: f"{100 * v:.1f}%", "hours": lambda v: f"{v:.1f} h", "pts": lambda v: f"{v:+.1f}", "num": lambda v: f"{v:.2f}"}
    sc = pd.DataFrame({"KPI": raw.kpi, "Current": [fmt[u](v) for u, v in zip(raw.unit, raw.current)],
                       "Proposed target (assumption)": [("<= " if not h else ">= ") + fmt[u](v).replace("+", "") for u, v, h in zip(raw.unit, raw.proposed_target, raw.higher_is_better)],
                       "Status": raw.status_label, "Definition": raw.definition_source})
    st.dataframe(sc, hide_index=True, use_container_width=True, height=36 * (len(sc) + 1) + 4)

# ------------------------------------------------------------------- health --
with tab_health:
    tr = M["transform"]; s = M["sentiment"]; v = M["validate"]
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Rows ingested to kept", f"{tr['rows_bronze']:,} to {tr['rows_silver']:,}"); h1.caption(f"{tr['rows_quarantined']} quarantined")
    h2.metric("DQ error-rule pass rate", f"{tr['dq_raw']['pct_rows_passing_all_error_rules']:.1f}%"); h2.caption(f"fully clean rows {tr['dq_raw']['pct_rows_fully_clean']:.1f}% raw, {tr['dq_clean']['pct_rows_fully_clean']:.1f}% after cleaning")
    h3.metric("Warehouse checks", f"{v['warehouse_pass']}/{v['warehouse_checks']}"); h3.caption(f"SQL = pandas KPI reconciliation: {v['kpi_reconciliation_pass']}/{v['kpi_reconciliation_checks']}")
    ts = DT.test_summary()
    h4.metric("Automated tests", f"{(ts['Pass / Fail'] == 'Pass').sum()}/{len(ts)} pass" if len(ts) else "run pytest")
    block("Sentiment model: perfect in-distribution, no mechanism for new vocabulary", CH.model_comparison(TB["sentiment_model_comparison"], TH), key="models",
          caption=f"Leave-one-descriptor-out accuracy {100 * s['lodo']['accuracy']:.1f}% equals the majority baseline; general VADER only {100 * s['vader']['cv5_accuracy']:.1f}%. The supplied text is templated (19 opinion words) - treat 100% as an upper bound.")
    st.markdown("**Data-quality findings**")
    rep = pd.read_csv(C.DQ_DIR / "dq_report_raw.csv")
    dq = rep[rep.rows_failed > 0][["rule_id", "dimension", "severity", "description", "rows_failed", "pct_failed", "status"]]
    st.dataframe(dq, hide_index=True, use_container_width=True, height=36 * (len(dq) + 1) + 4)
    if len(ts):
        st.markdown("**Test results by component**")
        comp = ts.groupby("Component")["Pass / Fail"].agg(tests="size", passed=lambda x: int((x == "Pass").sum())).reset_index()
        st.dataframe(comp, hide_index=True, use_container_width=True, height=36 * (len(comp) + 1) + 4)
    st.markdown("**Known limitations**")
    st.markdown("- Text is 100% templated; sentiment labels are rating-implied (not independent human labels).  \n- `Customer_ID` is unstable (revenue/region vary within a customer): all customer metrics are at feedback-record level.  \n"
                "- No timestamps: no trend or emerging-issue analysis.  \n- Cross-sectional data: impact figures are scenarios, not causal forecasts.  \n- No competitor data was supplied (the 'Competitor Alpha' file only holds the dictionary and metric guidance).")
