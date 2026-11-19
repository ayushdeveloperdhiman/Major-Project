"""Static, single-file, offline HTML export of the dashboard (full dataset, light/dark theme toggle).

    python dashboard/build_static.py      ->  dashboard/static/cvi_dashboard.html

Charts are rendered client-side by an inlined Plotly build; every chart has a 'Table view' twin.  The interactive
filters live in the Streamlit app (dashboard/app.py); this export is the portable evidence copy.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.offline as po

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import charts as CH  # noqa: E402
import data as DT  # noqa: E402
from cvi import config as C  # noqa: E402
from cvi import kpis as K  # noqa: E402
from cvi import viz  # noqa: E402

OUT = Path(__file__).resolve().parent / "static" / "cvi_dashboard.html"


def tokens_css() -> str:
    def block(sel: str, th: str) -> str:
        t = viz.THEMES[th]
        return (f"{sel}{{--surface:{t['surface']};--page:{t['page']};--ink:{t['ink']};--ink2:{t['ink2']};--muted:{t['muted']};--grid:{t['grid']};--axis:{t['axis']};}}")
    return (block(":root", "light") + "@media (prefers-color-scheme: dark){" + block(":root:not([data-theme='light'])", "dark") + "}" + block(":root[data-theme='dark']", "dark") + block(":root[data-theme='light']", "light"))


PAGE_CSS = """
*{box-sizing:border-box} body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;line-height:1.45}
.wrap{max-width:1240px;margin:0 auto;padding:20px 16px 60px} h1{font-size:22px;margin:0 0 4px;font-weight:600} h2{font-size:18px;margin:34px 0 10px;font-weight:600;scroll-margin-top:64px}
h3{font-size:14px;margin:0 0 2px;font-weight:600} .sub{color:var(--ink2);font-size:13px;margin:0 0 14px} .cap{color:var(--ink2);font-size:12.5px;margin:0 0 6px}
nav{position:sticky;top:0;background:var(--page);border-bottom:1px solid var(--axis);padding:8px 0;z-index:5;display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:13px}
nav a{color:var(--ink2);text-decoration:none} nav a:hover{color:var(--ink)} nav .sp{flex:1} button{background:var(--surface);color:var(--ink);border:1px solid var(--axis);border-radius:8px;padding:5px 12px;font-size:12px;cursor:pointer}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px} .card{background:var(--surface);border:1px solid var(--axis);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.card p{margin:2px 0;font-size:12.5px;color:var(--ink2)} .plot{width:100%} details{margin-top:6px;font-size:12.5px} summary{cursor:pointer;color:var(--ink2)}
table.tbl{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px} .tbl th,.tbl td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right;color:var(--ink2)} .tbl th:first-child,.tbl td:first-child{text-align:left} .tbl th{color:var(--ink);font-weight:600}
.cvi-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:8px 0 6px} .cvi-tile{background:var(--surface);border:1px solid var(--axis);border-radius:10px;padding:12px 14px 10px}
.cvi-tl{font-size:12px;color:var(--ink2)} .cvi-tv{font-size:26px;font-weight:600;color:var(--ink);margin:6px 0 2px;line-height:1.1} .cvi-tt{font-size:11px;color:var(--muted)} .cvi-ts{display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:var(--ink)}
.hero .n{font-size:56px;font-weight:600;line-height:1} .hero .l{font-size:13px;color:var(--ink2);margin-top:4px} .hero{margin:10px 0}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px} .kv .card .n{font-size:24px;font-weight:600;color:var(--ink)} .kv .card p{margin-top:2px}
.rec summary{color:var(--ink);font-weight:600;font-size:13.5px} .rec b{color:var(--ink)} ul{padding-left:20px;color:var(--ink2);font-size:13px}
@media print{nav{display:none}}
"""


def plot_card(cid: str, title: str, builder, caption: str | None, figs: dict, extra_cls: str = "") -> str:
    fl, tbl = builder("light")
    fd, _ = builder("dark")
    figs[cid] = {"light": fl.to_json(), "dark": fd.to_json()}
    tbl_html = tbl.round(4).to_html(index=False, border=0, classes="tbl", na_rep="") if tbl is not None and len(tbl) else "<p>No table.</p>"
    cap = f"<p class='cap'>{html.escape(caption)}</p>" if caption else ""
    return f"<div class='card {extra_cls}'><h3>{html.escape(title)}</h3>{cap}<div class='plot' id='{cid}'></div><details><summary>Table view</summary>{tbl_html}</details></div>"


def build() -> Path:
    df = DT.load_frame()
    M = DT.metrics()
    TB = {n: DT.table(n) for n in ["integrated_repeat_driver_odds_ratios", "integrated_rank_stability_issue", "decision_recommendations", "decision_kpi_scorecard", "sentiment_model_comparison", "eda_dimension_tests"]}
    e, ig, s, tr, v = M["eda"], M["integrate"], M["sentiment"], M["transform"], M["validate"]
    k = K.overall_kpis(df, "sentiment_pred")
    figs: dict = {}
    h: list[str] = []

    # ---------------------------------------------------------------- overview
    h.append("<h2 id='overview'>Overview</h2>")
    h.append(f"<div class='grid2'><div class='card hero'><div class='n'>{k['net_sentiment_score']:+.1f}</div><div class='l'>Net Sentiment Score (positive % minus negative %)</div>"
             f"<p>Negative rate {100 * k['negative_sentiment_rate']:.1f}% (95% CI {100 * k['negative_rate_ci'][0]:.1f}-{100 * k['negative_rate_ci'][1]:.1f}%)</p></div>"
             + plot_card("c_mix", "Sentiment mix", lambda t: CH.sentiment_mix(df, t), None, figs) + "</div>")
    h.append(plot_card("c_rev", "Revenue by sentiment", lambda t: CH.revenue_split(df, t), "Negative-sentiment revenue = Revenue at Risk (Metric Guidance).", figs))
    so, hs = e["sentiment_outcomes"], e["hotspot"]
    h.append("<div class='kv'>"
             f"<div class='card'><h3>Sentiment drives repeat purchase</h3><p>Repeat rate {100 * k['negative_repeat_rate']:.1f}% (negative) vs {100 * k['positive_repeat_rate']:.1f}% (positive): a {ig['revenue_at_risk']['repeat_gap_pp']:.1f}pp gap, chi-square p&lt;0.001.</p></div>"
             f"<div class='card'><h3>Negative cases resolve twice as slowly</h3><p>{k['negative_avg_resolution_hours']:.1f}h vs {k['non_negative_avg_resolution_hours']:.1f}h (Cohen's d {so['resolution_cohens_d_neg_vs_nonneg']:.1f}); {100 * k['negative_sla_breach_rate']:.0f}% of negatives breach a 12h SLA.</p></div>"
             f"<div class='card'><h3>No segment is a proven hotspot</h3><p>{hs['tests']} cells scanned: {hs['raw_p_lt_0.05']} raw p&lt;0.05 (chance ~{hs['expected_by_chance']:.0f}); {hs['significant_after_fdr']} survive FDR control.</p></div></div>")
    dt = TB["eda_dimension_tests"].set_index("dimension")
    cols = {"Issue": "issue_category", "Channel": "channel_name", "Region": "region_name", "Product": "product_category_name", "Source": "source_name"}
    h.append("<div class='grid2'>" + "".join(plot_card(f"c_neg_{d.lower()}", f"Negative rate by {d.lower()} (95% CI)", (lambda c, d: (lambda t: CH.neg_rate_by(df, c, d, t)))(c, d),
                                                      f"Full-data chi-square p = {dt.loc[d, 'p_value']:.2f}: no significant difference.", figs) for d, c in cols.items()) + "</div>")

    # ------------------------------------------------------------------- voice
    h.append("<h2 id='voice'>Voice of customer</h2><div class='grid2'>"
             + plot_card("c_theme", "Themes: share of feedback vs share of Revenue at Risk", lambda t: CH.theme_overview(df, t), "Topic Share = feedback assigned to a theme / total feedback.", figs)
             + plot_card("c_heat", "Theme mix by channel", lambda t: CH.theme_heatmap(df, t), "Evenly spread: chi-square p = 0.62.", figs) + "</div>")

    # ----------------------------------------------------------------- drivers
    h.append("<h2 id='drivers'>Drivers</h2>")
    h.append(plot_card("c_out", "What sentiment changes: repeat purchase, resolution time, and revenue (no effect)", lambda t: CH.outcomes(df, t), "Mean by sentiment class with 95% CI whiskers.", figs))
    h.append("<div class='grid2'>" + plot_card("c_hist", "Resolution-time distribution", lambda t: CH.resolution_hist(df, t), f"Resolution time alone predicts negative sentiment with AUC {e['resolution_threshold']['auc_resolution_predicts_negative']:.2f}.", figs)
             + plot_card("c_band", "Sentiment mix by resolution-time band", lambda t: CH.sentiment_by_band(df, t), "Negative share rises from ~0% (<=6h) to 100% (>24h).", figs) + "</div>")
    h.append(plot_card("c_forest", "Repeat-purchase drivers: adjusted odds ratios", lambda t: CH.forest(TB["integrated_repeat_driver_odds_ratios"], t),
                       "Sentiment is the driver. Resolution time loses its effect once sentiment is accounted for (adjusted OR 0.92, p = 0.17).", figs))

    # ---------------------------------------------------------------- segments
    rar = ig["revenue_at_risk"]
    h.append("<h2 id='segments'>Segments &amp; risk</h2><div class='grid2'>"
             + plot_card("c_segprof", "Experience segments (K-means, k = 3)", lambda t: CH.segment_profile(df, t), "Standardised CSAT, resolution time and repeat flag; bootstrap ARI 0.99.", figs)
             + plot_card("c_segmap", "Segment map (PCA)", lambda t: CH.segment_scatter(df, t), None, figs) + "</div>")
    h.append("<div class='grid2'>" + plot_card("c_risk", "Where the Revenue at Risk sits: segment x revenue tier", lambda t: CH.risk_matrix(df, t), "Outlined cells = Critical tier.", figs)
             + f"<div><div class='card'><h3>INR {rar['revenue_at_risk_inr'] / 1e6:.2f}M Revenue at Risk ({100 * rar['rar_share']:.1f}% of revenue)</h3><p>Bootstrap 95% CI INR {rar['rar_ci'][0] / 1e6:.2f}M - {rar['rar_ci'][1] / 1e6:.2f}M</p>"
               f"<p>Lapsed (no repeat): INR {rar['lapsed_unhappy']['revenue_inr'] / 1e6:.2f}M / {rar['lapsed_unhappy']['records']} records; still active: INR {rar['active_unhappy']['revenue_inr'] / 1e6:.2f}M / {rar['active_unhappy']['records']} records</p>"
               f"<p>Critical tier: {ig['critical_tier']['records']} records = {100 * ig['critical_tier']['share_of_total_rar']:.0f}% of all Revenue at Risk</p></div>"
             + plot_card("c_scen", "Recoverable revenue scenarios", lambda t: CH.scenarios(ig, t), "Scenarios, not forecasts: the repeat-purchase gap is observed in cross-sectional data.", figs) + "</div></div>")
    h.append(plot_card("c_rank", "Is the top issue statistically firm? Bootstrap rank stability", lambda t: CH.rank_stability(TB["integrated_rank_stability_issue"], t),
                       "Customer Support ranks #1 in 43% of 2,000 resamples (chance 12.5%): a leading candidate, not a proven hotspot.", figs))

    # ----------------------------------------------------------------- actions
    rec = TB["decision_recommendations"]
    h.append("<h2 id='actions'>Actions</h2>")
    h.append(plot_card("c_prio", "Prioritised actions", lambda t: CH.priority(rec, t), "Priority = Impact x Confidence / Effort; effort is analyst judgement (assumption).", figs))
    for r in rec.itertuples():
        h.append(f"<details class='card rec'><summary>{html.escape(r.id)} | {html.escape(r.title)} <span style='font-weight:400;color:var(--ink2)'>[{html.escape(r.phase)} | {html.escape(r.type)} | score {r.priority_score:.1f}]</span></summary>"
                 + "".join(f"<p><b>{lab}.</b> {html.escape(str(val))}</p>" for lab, val in [("Finding", r.finding), ("Evidence", r.evidence), ("Action", r.action), ("Owner / timeline", f"{r.owner} | {r.timeline}"),
                                                                                         ("KPI", f"{r.kpi}: baseline {r.baseline} -> target {r.target}"), ("Expected impact", r.expected_impact), (f"Confidence ({r.confidence}/5)", r.confidence_basis), ("Caveat", r.caveat)]) + "</details>")

    # ------------------------------------------------------------------ health
    ts = DT.test_summary()
    h.append("<h2 id='health'>Data &amp; model health</h2><div class='kv'>"
             f"<div class='card'><p>Rows ingested to kept</p><div class='n'>{tr['rows_bronze']:,} to {tr['rows_silver']:,}</div><p>{tr['rows_quarantined']} quarantined</p></div>"
             f"<div class='card'><p>DQ error-rule pass rate</p><div class='n'>{tr['dq_raw']['pct_rows_passing_all_error_rules']:.1f}%</div><p>fully clean rows {tr['dq_raw']['pct_rows_fully_clean']:.1f}% raw, {tr['dq_clean']['pct_rows_fully_clean']:.1f}% cleaned</p></div>"
             f"<div class='card'><p>Warehouse checks</p><div class='n'>{v['warehouse_pass']}/{v['warehouse_checks']}</div><p>SQL = pandas KPI reconciliation {v['kpi_reconciliation_pass']}/{v['kpi_reconciliation_checks']}</p></div>"
             f"<div class='card'><p>Automated tests</p><div class='n'>{(ts['Pass / Fail'] == 'Pass').sum()}/{len(ts)}</div><p>pass</p></div></div>")
    h.append(plot_card("c_models", "Sentiment model: perfect in-distribution, no mechanism for new vocabulary", lambda t: CH.model_comparison(TB["sentiment_model_comparison"], t),
                       f"Leave-one-descriptor-out accuracy {100 * s['lodo']['accuracy']:.1f}% equals the majority baseline; general VADER only {100 * s['vader']['cv5_accuracy']:.1f}%. Templated text: treat 100% as an upper bound.", figs))
    rep = pd.read_csv(C.DQ_DIR / "dq_report_raw.csv")
    dq = rep[rep.rows_failed > 0][["rule_id", "dimension", "severity", "description", "rows_failed", "pct_failed", "status"]]
    h.append("<div class='card'><h3>Data-quality findings (raw data)</h3>" + dq.to_html(index=False, border=0, classes="tbl") + "</div>")
    if len(ts):
        comp = ts.groupby("Component")["Pass / Fail"].agg(tests="size", passed=lambda x: int((x == "Pass").sum())).reset_index()
        h.append("<div class='card'><h3>Test results by component</h3>" + comp.to_html(index=False, border=0, classes="tbl") + "</div>")
    h.append("<div class='card'><h3>Known limitations</h3><ul><li>Text is 100% templated; sentiment labels are rating-implied, not independent human labels.</li><li>Customer_ID is unstable (revenue / region vary within a customer): all customer metrics are at feedback-record level.</li>"
             "<li>No timestamps: no trend or emerging-issue analysis.</li><li>Cross-sectional data: impact figures are scenarios, not causal forecasts.</li><li>No competitor data was supplied (the 'Competitor Alpha' file only holds the dictionary and metric guidance).</li></ul></div>")

    body = "".join(h)
    js = (
        "const FIGS=" + "{" + ",".join(json.dumps(k) + ':{"light":' + v["light"] + ',"dark":' + v["dark"] + "}" for k, v in figs.items()) + "};\n"
        "function curTheme(){return document.documentElement.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}\n"
        "function renderAll(){const t=curTheme();for(const id in FIGS){const f=FIGS[id][t];Plotly.react(id,f.data,f.layout,{displayModeBar:false,responsive:true})}}\n"
        "(function(){let t=new URLSearchParams(location.search).get('theme');try{if(!t)t=localStorage.getItem('cvi-theme')}catch(e){}if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)})();\n"
        "document.getElementById('themeBtn').addEventListener('click',()=>{const n=curTheme()==='dark'?'light':'dark';document.documentElement.setAttribute('data-theme',n);try{localStorage.setItem('cvi-theme',n)}catch(e){}renderAll()});\n"
        "window.addEventListener('load',renderAll);\n"
    )
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Customer Voice Intelligence</title>"
            f"<style>{tokens_css()}{PAGE_CSS}</style></head><body><div class='wrap'><h1>AI-Powered Customer Voice Intelligence and Business Decision Platform</h1>"
            "<p class='sub'>InsightRetail India (anonymised) | MSDS-MAJOR-CVI-001 | 2,400 feedback records | sentiment = NLP prediction | KPI definitions follow the supplied Metric Guidance | static export (full dataset)</p>"
            "<nav><a href='#overview'>Overview</a><a href='#voice'>Voice of customer</a><a href='#drivers'>Drivers</a><a href='#segments'>Segments &amp; risk</a><a href='#actions'>Actions</a><a href='#health'>Data &amp; model health</a><span class='sp'></span><button id='themeBtn' type='button'>Light / dark</button></nav>"
            f"{CH.kpi_tiles_html(df)}"
            "<p class='cap'>Targets are PROPOSED assumptions (none were supplied); status = icon + label.</p>"
            f"{body}</div><script>{po.get_plotlyjs()}</script><script>{js}</script></body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
