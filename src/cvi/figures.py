"""Static figures (PNG, light theme) for the notebook, the workbook and the README - part 1:
architecture diagrams, data profile, data quality, exploratory analytics.  NLP / mining / integration figures
live in :mod:`cvi.figures2`.  Every figure reads the saved tables/metrics (no recomputation) and has a CSV twin
in ``outputs/tables`` (the 'table view').
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from . import config as C
from . import diagrams as D
from . import viz
from .utils import load_metrics, wilson_ci

T = viz.THEMES["light"]
SENT = viz.SENTIMENT["light"]
ACC = viz.ACCENT["light"]


def tbl(name: str) -> pd.DataFrame:
    return pd.read_csv(C.TABLE_DIR / f"{name}.csv")


def silver() -> pd.DataFrame:
    return pd.read_parquet(C.ANALYTICS_FILE if C.ANALYTICS_FILE.exists() else C.SILVER_FILE)


def _sent_col(df: pd.DataFrame) -> str:
    return "sent_pred" if "sent_pred" in df.columns else "reference_sentiment"


# ----------------------------------------------------------------- diagrams #
def fig01_pipeline_architecture() -> str:
    W, H = 7.0, 5.15
    fig, ax = D.canvas(W, H)
    ramp = {"bronze": viz.BLUE_RAMP[3], "silver": viz.BLUE_RAMP[6], "gold": viz.BLUE_RAMP[9]}      # ordinal layers
    stages = [
        ("SOURCES", T["muted"], ["3 Excel workbooks", "Customer Voice 2,400 x 15", "data dictionary, metrics", "analysis template"]),
        ("BRONZE  raw landing", ramp["bronze"], ["schema-on-read, all text", "lineage: file / row / batch", "SHA-256 file + row hash", "column-level source profile"]),
        ("SILVER  clean + typed", ramp["silver"], ["36 DQ rules, quarantine", "8 logged transformations", "bands, tiers, flags, text_clean", "re-validated after cleaning"]),
        ("GOLD  warehouse", ramp["gold"], ["star schema in SQLite", "11 dimensions, 3 facts + text", "FK + CHECK constraints", "12 KPI / 360-degree views"]),
        ("ANALYTICS", T["ink2"], ["sentiment + themes (NLP)", "K-means, association rules", "drivers, Revenue at Risk", "bootstrap rank stability"]),
        ("DECISION LAYER", ACC, ["KPI scorecard + targets", "Streamlit + HTML dashboard", "7 prioritised actions", "workbook + test evidence"]),
    ]
    boxes = [D.Box(t, ln, accent=c) for t, c, ln in stages]
    bw = max(b.size(fig, ax)[0] for b in boxes)
    bh = max(b.size(fig, ax)[1] for b in boxes)
    gap = (W - 0.5 - 3 * bw) / 2
    xs = [0.25 + i * (bw + gap) for i in range(3)]
    ytop1, ytop2 = H - 0.78, H - 0.78 - bh - 0.55
    pos = [(xs[0], ytop1), (xs[1], ytop1), (xs[2], ytop1), (xs[0], ytop2), (xs[1], ytop2), (xs[2], ytop2)]
    pos = pos[:3] + pos[3:]
    ed = [b.draw(fig, ax, x, y, bw, bh) for b, (x, y) in zip(boxes, pos)]
    # arrows with validation checkpoints
    D.arrow(ax, (ed[0]["r"] + 0.02, ed[0]["cy"]), (ed[1]["l"] - 0.02, ed[1]["cy"])); D.badge(ax, (ed[0]["r"] + ed[1]["l"]) / 2, ed[0]["cy"] + 0.2, "1")
    D.arrow(ax, (ed[1]["r"] + 0.02, ed[1]["cy"]), (ed[2]["l"] - 0.02, ed[2]["cy"])); D.badge(ax, (ed[1]["r"] + ed[2]["l"]) / 2, ed[1]["cy"] + 0.2, "2")
    ymid = (ed[2]["b"] + ed[5]["t"]) / 2
    D.polyline(ax, [(ed[2]["cx"], ed[2]["b"] - 0.02), (ed[2]["cx"], ymid), (ed[3]["cx"], ymid), (ed[3]["cx"], ed[3]["t"] + 0.03)])
    D.badge(ax, (ed[2]["cx"] + ed[3]["cx"]) / 2, ymid, "3")
    D.arrow(ax, (ed[3]["r"] + 0.02, ed[3]["cy"]), (ed[4]["l"] - 0.02, ed[4]["cy"])); D.badge(ax, (ed[3]["r"] + ed[4]["l"]) / 2, ed[3]["cy"] + 0.2, "4")
    D.arrow(ax, (ed[4]["r"] + 0.02, ed[4]["cy"]), (ed[5]["l"] - 0.02, ed[5]["cy"])); D.badge(ax, (ed[4]["r"] + ed[5]["l"]) / 2, ed[4]["cy"] + 0.2, "5")
    y = ed[3]["b"] - 0.30
    ax.text(0.25, y, "Validation checkpoints", fontsize=8.2, fontweight=500, color=T["ink"], va="top")
    cps = ["Schema contract vs data dictionary; source profile; row counts",
           "DQ gate: 32 row rules + 4 dataset checks -> quarantine / flag / fix, re-run after cleaning",
           "Warehouse reconciliation: row counts, revenue sums, FK check, no join fan-out (14 checks)",
           "Model validation: CV, hold-out + CI, leave-descriptor-out, ablation, permutation nulls, FDR",
           "SQL semantic layer == pandas KPIs (28 reconciliations); automated tests; run manifest"]
    for i, t in enumerate(cps):
        yy = y - 0.27 - i * 0.235
        D.badge(ax, 0.36, yy, str(i + 1), r=0.085)
        ax.text(0.58, yy, t, ha="left", va="center", fontsize=7.4, color=T["ink2"])
    viz.heading(fig, "Source-to-decision pipeline (medallion architecture)", "Deterministic (seed 42); every stage writes evidence to data/ and outputs/")
    return viz.save(fig, "fig01_pipeline_architecture")


def fig02_star_schema_erd() -> str:
    W, H = 7.0, 5.95
    fig, ax = D.canvas(W, H)
    fs = 6.2
    left = [("dim_customer", ["PK customer_key", "customer_id, record_count"]), ("dim_region", ["PK region_key", "region_name"]),
            ("dim_channel", ["PK channel_key", "channel_name, channel_type"]), ("dim_product_category", ["PK product_category_key", "name"]),
            ("dim_issue", ["PK issue_key", "issue_category, theme_group"])]
    right = [("dim_feedback_source", ["PK source_key", "source_name, source_class"]), ("dim_rating", ["PK rating_key (1-5)", "label, implied_sentiment"]),
             ("dim_sentiment", ["PK sentiment_key", "label, score, sort_order"]), ("dim_resolution_band", ["PK band_key", "label, hours, sla_status"]),
             ("dim_revenue_tier", ["PK tier_key", "label, lower_inr, upper_inr"])]
    dims_l = [D.Box(n, ln, fs=fs, title_fs=7.4) for n, ln in left]
    dims_r = [D.Box(n, ln, fs=fs, title_fs=7.4) for n, ln in right]
    dw = max(b.size(fig, ax)[0] for b in dims_l + dims_r)
    dh = max(b.size(fig, ax)[1] for b in dims_l + dims_r)
    vgap = 0.11
    stack_h = 5 * dh + 4 * vgap
    top = H - 0.78
    fact = D.Box("fact_feedback", ["PK  feedback_id   (grain: one row per feedback record)", "FK  customer, region, channel, product, issue, source,",
                                   "      rating, ref_sentiment, resolution_band, revenue_tier", "", "Measures  revenue_inr, order_count, resolution_hours,",
                                   "     rating, csat_score, repeat_flag, avg_order_value", "", "Flags  is_negative_ref, sla_breach, critical_delay,",
                                   "     repeat_without_orders", "", "Lineage  batch_id, row_hash, source_row, dq_flags"],
                 fs=fs, title_fs=8.4, fill=T["page"], edge=viz.BLUE_RAMP[9], bold_edge=True)
    fw = fact.size(fig, ax)[0]
    gap = (W - 0.4 - 2 * dw - fw) / 2
    xl, xf, xr = 0.2, 0.2 + dw + gap, 0.2 + dw + gap + fw + gap
    fe = fact.draw(fig, ax, xf, top, fw, stack_h)
    for i, b in enumerate(dims_l):
        e = b.draw(fig, ax, xl, top - i * (dh + vgap), dw, dh)
        D.line(ax, (e["r"], e["cy"]), (fe["l"], e["cy"]))
        ax.text(e["r"] + 0.05, e["cy"] + 0.05, "1", fontsize=6.4, color=T["muted"], va="bottom"); ax.text(fe["l"] - 0.05, e["cy"] + 0.05, "*", fontsize=7.4, color=T["muted"], va="bottom", ha="right")
    for i, b in enumerate(dims_r):
        e = b.draw(fig, ax, xr, top - i * (dh + vgap), dw, dh)
        D.line(ax, (e["l"], e["cy"]), (fe["r"], e["cy"]))
        ax.text(e["l"] - 0.05, e["cy"] + 0.05, "1", fontsize=6.4, color=T["muted"], va="bottom", ha="right"); ax.text(fe["r"] + 0.05, e["cy"] + 0.05, "*", fontsize=7.4, color=T["muted"], va="bottom")
    bottom = [D.Box("feedback_text", ["PK/FK feedback_id", "text, text_clean, lengths"], fs=fs, title_fs=7.4),
              D.Box("fact_feedback_nlp", ["PK/FK feedback_id", "pred_sentiment_key, p_neg/neu/pos", "vader_compound, topic_issue_key"], fs=fs, title_fs=7.4),
              D.Box("fact_feedback_segment", ["PK/FK feedback_id", "segment_key, risk_tier"], fs=fs, title_fs=7.4),
              D.Box("dim_segment", ["PK segment_key", "segment_name, description"], fs=fs, title_fs=7.4)]
    bw = [b.size(fig, ax)[0] for b in bottom]
    bgap = (W - 0.4 - sum(bw)) / 3
    ytop_b = fe["b"] - 0.62
    xs, x = [], 0.2
    for w_ in bw:
        xs.append(x)
        x += w_ + bgap
    bes = [b.draw(fig, ax, xx, ytop_b) for b, xx in zip(bottom[:3], xs[:3])]
    de = bottom[3].draw(fig, ax, xs[3], ytop_b)
    ybus = fe["b"] - 0.30
    D.line(ax, (fe["cx"], fe["b"]), (fe["cx"], ybus))
    D.line(ax, (bes[0]["cx"], ybus), (bes[2]["cx"], ybus))
    for e in bes:
        D.line(ax, (e["cx"], ybus), (e["cx"], e["t"]))
    D.line(ax, (bes[2]["r"], bes[2]["cy"]), (de["l"], bes[2]["cy"]))
    ax.text(0.2, fe["b"] - 0.12, "1:1 tables keyed by feedback_id (text, model output, segment)", fontsize=7, color=T["muted"], va="top")
    viz.heading(fig, "Analytical warehouse: star schema at feedback grain",
                "Customer_ID is a weak key (revenue, orders, repeat flag and region vary within a customer), so those measures live in the fact")
    return viz.save(fig, "fig02_star_schema_erd")


# ------------------------------------------------------------ data profile --
def fig03_data_profile() -> str:
    d = silver()
    sc = viz.SENTIMENT["light"]
    fig, axes, _ = viz.new_grid(2, 2, w=6.8, h=5.2, theme="light", gridspec_kw={"hspace": 0.62, "wspace": 0.28})
    ax = axes[0, 0]
    cnt = d.rating.value_counts().sort_index()
    cols = [sc["Negative"] if r <= 2 else sc["Neutral"] if r == 3 else sc["Positive"] for r in cnt.index]
    viz.vbars(ax, [str(r) for r in cnt.index], cnt.values, cols, fmt="{:.0f}", thick_px=24)
    ax.set_title("Rating (1-5) by implied sentiment")
    ax.set_ylim(0, cnt.max() * 1.55)
    viz.legend_keys(ax, [("Negative (1-2)", sc["Negative"]), ("Neutral (3)", sc["Neutral"]), ("Positive (4-5)", sc["Positive"])], loc="upper left")
    ax = axes[0, 1]
    h, e = np.histogram(d.csat_score, bins=np.arange(1, 5.51, 0.5))
    viz.vbars(ax, [f"{a:.1f}" for a in e[:-1]], h, ACC, thick_px=17, label=False)
    ax.set_title("CSAT score (0.5-point bins)")
    ax.text(0.02, 0.93, "tracks Rating: Spearman 0.97", transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color=T["ink2"])
    ax = axes[1, 0]
    edges = np.linspace(0.5, 12.0, 12)
    h, e = np.histogram(d.customer_revenue_inr / 1000, bins=edges)
    viz.vbars(ax, [f"{(a + b) / 2:.1f}" for a, b in zip(e[:-1], e[1:])], h, ACC, thick_px=14, label=False)
    ax.set_title("Revenue per record (INR thousand)")
    ax.text(0.02, 0.93, "statistically uniform (KS p = 0.50)", transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color=T["ink2"])
    ax.set_ylim(0, h.max() * 1.35)
    ax = axes[1, 1]
    h, e = np.histogram(d.resolution_time_hours, bins=np.arange(0, 33, 2))
    viz.vbars(ax, [f"{int(a)}" for a in e[:-1]], h, ACC, thick_px=14, label=False)
    ax.set_title("Resolution time (hours, 2h bins)")
    ax.axvline(12 / 2 - 0.5, color=T["ink2"], lw=1.0, zorder=5)
    ax.text(12 / 2 - 0.35, ax.get_ylim()[1] * 0.97, "12h SLA", fontsize=7.6, color=T["ink2"], va="top")
    for a in axes.ravel():
        a.tick_params(labelsize=7.4)
    fig.subplots_adjust(left=0.09, right=0.98, top=viz.top(fig, 0.95), bottom=0.07)
    viz.heading(fig, "Data profile: what the supplied data look like", "2,400 feedback records, no missing values; revenue is uniform noise, Rating and CSAT move together")
    return viz.save(fig, "fig03_data_profile")


def fig04_dq_findings() -> str:
    raw = pd.read_csv(C.DQ_DIR / "dq_report_raw.csv").set_index("rule_id")
    cln = pd.read_csv(C.DQ_DIR / "dq_report_clean.csv").set_index("rule_id")
    fig, axes, _ = viz.new_grid(1, 2, w=7.0, h=3.45, theme="light")
    axes[0].set_position([0.335, 0.13, 0.27, 0.60])
    axes[1].set_position([0.805, 0.13, 0.17, 0.60])
    rows = [("Repeated word 'experience experience'\n(auto-fixed in text_clean)", "DQ-K05"), ("Repeat flag = 1 but Order_Count = 1\n(flagged, retained)", "DQ-K04"),
            ("Resolution censored at 1.0h floor\n(monitored)", "DQ-P03"), ("Resolution mild outliers (1.5 x IQR)\n(monitored)", "DQ-P02"), ("Same customer + identical text\n(legitimate, kept)", "DQ-U03")]
    ax = axes[0]
    b = [int(raw.loc[r[1], "rows_failed"]) for r in rows]
    a = [int(cln.loc[r[1], "rows_failed"]) for r in rows]
    ypos = np.arange(len(rows))[::-1]
    ax.set_xlim(0, 360); ax.set_ylim(-0.7, len(rows) - 0.3)
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=7.2, color=T["ink2"])
    py = viz._px_per_unit(ax, "y"); px = viz._px_per_unit(ax, "x")
    for y, vb, va in zip(ypos, b, a):
        viz.rounded_bar(ax, 0, y + 0.19, vb, 10 / py, T["deemph"], "h")
        viz.rounded_bar(ax, 0, y - 0.19, max(va, 0.0001), 10 / py, ACC, "h")
        ax.text(vb + 5 / px, y + 0.19, f"{vb}", va="center", fontsize=7.6, color=T["ink"])
        ax.text(max(va, 0) + 5 / px, y - 0.19, f"{va}", va="center", fontsize=7.6, color=T["ink"])
    ax.set_title("Row-level findings (records)")
    viz.legend_keys(ax, [("Raw (Bronze)", T["deemph"]), ("After cleaning (Silver)", ACC)], loc="lower right")
    ax = axes[1]
    tr = load_metrics("transform")["findings"]
    rows2 = [("Customers with >1\nrevenue value", tr["revenue_unstable_pct_of_multi_record_customers"]), ("Customers with >1\nregion", tr["region_unstable_pct_of_multi_record_customers"]),
             ("Texts following ONE\nsentence template", tr["text_template_pct"])]
    viz.hbars(ax, [r[0] for r in rows2], [r[1] for r in rows2], ACC, fmt="{:.1f}%", xmax=140, thick_px=14)
    ax.set_yticklabels([r[0] for r in rows2], fontsize=7.2, color=T["ink2"])
    ax.set_title("Dataset-level (%)")
    viz.heading(fig, "Data quality: 0 blocking errors, 5 behaviours worth knowing", "36 rules run on raw and cleaned data; fully-clean rows 83.5% -> 95.3%")
    return viz.save(fig, "fig04_dq_findings")


# --------------------------------------------------------------------- EDA --
def fig05_sentiment_mix() -> str:
    d = silver(); sc = viz.SENTIMENT["light"]; col = _sent_col(d)
    shares = d[col].value_counts(normalize=True).reindex(C.SENTIMENT_ORDER) * 100
    n_neg = int((d[col] == "Negative").sum()); lo, hi = wilson_ci(n_neg, len(d))
    W, H = 6.8, 2.05
    fig, ax = D.canvas(W, H)
    x0, x1, yb, hb = 0.2, W - 0.2, 0.62, 0.30
    x = x0
    span = x1 - x0
    for name in C.SENTIMENT_ORDER:
        w = span * shares[name] / 100
        ax.add_patch(FancyBboxPatch((x, yb), max(w - 0.025, 0), hb, boxstyle="round,pad=0,rounding_size=0.04", fc=sc[name], ec="none", zorder=3))
        ax.text(x + 0.1, yb + hb / 2, f"{shares[name]:.1f}%", ha="left", va="center", fontsize=9.5, color="#0b0b0b" if name == "Neutral" else "#ffffff", fontweight=500, zorder=5)
        x += w
    xs = x0
    for name in C.SENTIMENT_ORDER:
        ax.add_patch(FancyBboxPatch((xs, 0.30), 0.13, 0.13, boxstyle="square,pad=0", fc=sc[name], ec="none"))
        ax.text(xs + 0.2, 0.365, f"{name}  ({int((d[col] == name).sum()):,} records)", fontsize=8, color=T["ink2"], va="center")
        xs += 2.0
    nss = shares["Positive"] - shares["Negative"]
    ax.text(x0, H - 0.78, f"Net Sentiment Score  {nss:+.1f} pts", fontsize=14, fontweight=500, color=T["ink"], va="top")
    ax.text(x1, H - 0.84, f"Negative rate {shares['Negative']:.1f}%   (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)", fontsize=8.4, color=T["ink2"], va="top", ha="right")
    viz.heading(fig, "Voice of customer at a glance", "Share of 2,400 feedback records by sentiment (NLP prediction)")
    return viz.save(fig, "fig05_sentiment_mix")


def fig06_negative_rate_by_dimension() -> str:
    d = silver(); col = _sent_col(d); sc = viz.SENTIMENT["light"]
    dt = tbl("eda_dimension_tests").set_index("dimension")
    dims = [("Issue", "issue_category"), ("Channel", "channel"), ("Region", "region"), ("Product", "product_category"), ("Source", "feedback_source")]
    overall = 100 * (d[col] == "Negative").mean()
    fig, axes, _ = viz.new_grid(2, 3, w=7.0, h=5.3, theme="light", gridspec_kw={"hspace": 0.55, "wspace": 0.95})
    for ax, (name, c) in zip(axes.ravel(), dims):
        g = d.groupby(c)[col].agg(neg=lambda s: (s == "Negative").sum(), n="size")
        g["rate"] = 100 * g.neg / g.n
        ci = np.array([wilson_ci(a, b) for a, b in zip(g.neg, g.n)]) * 100
        g = g.assign(lo=ci[:, 0], hi=ci[:, 1]).sort_values("rate", ascending=False)
        viz.hbars(ax, list(g.index), g.rate.values, sc["Negative"], err=np.array([g.lo.values, g.hi.values]), ref=overall, xmax=47, thick_px=10, fmt="{:.1f}")
        ax.set_title(name, fontsize=9)
        ax.text(1.0, 1.02, f"p = {dt.loc[name, 'p_value']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4, color=T["muted"])
        ax.tick_params(labelsize=7)
        ax.set_yticklabels([t.get_text() for t in ax.get_yticklabels()], fontsize=7)
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(0, 0.98, "How to read", fontsize=8.6, fontweight=500, color=T["ink"], va="top")
    ax.text(0, 0.86, f"Bars: negative rate (%)\nWhiskers: 95% Wilson CI\nGray line: overall {overall:.1f}%\np: chi-square per dimension\n\nEvery interval crosses the\nline, so no level is\nsignificantly worse.\n0 of 314 cells survive FDR.", fontsize=7.2, color=T["ink2"], va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.13, right=0.985, top=viz.top(fig, 0.98), bottom=0.05)
    viz.heading(fig, "No segment is a statistically proven sentiment hotspot", "Negative-feedback rate by business dimension, with 95% confidence intervals")
    return viz.save(fig, "fig06_negative_rate_by_dimension")


def fig07_sentiment_vs_outcomes() -> str:
    o = tbl("eda_sentiment_outcomes"); t = load_metrics("eda")["sentiment_outcomes"]; sc = viz.SENTIMENT["light"]
    fig, axes, _ = viz.new_grid(1, 3, w=7.0, h=3.35, theme="light", gridspec_kw={"wspace": 0.5})
    cols = [sc[s] for s in C.SENTIMENT_ORDER]
    labs = C.SENTIMENT_ORDER
    viz.vbars(axes[0], labs, 100 * o.repeat_rate.values, cols, err=np.array([100 * o.repeat_lo.values, 100 * o.repeat_hi.values]), fmt="{:.0f}%", thick_px=26, ymax=100)
    axes[0].set_title("Repeat purchase rate (%)")
    axes[0].text(0.5, -0.30, f"strong effect: chi-square p = {t['repeat_vs_sentiment_p']:.0e}", transform=axes[0].transAxes, ha="center", fontsize=7.6, color=T["ink2"])
    viz.vbars(axes[1], labs, o.avg_resolution_hours.values, cols, err=np.array([o.res_lo.values, o.res_hi.values]), fmt="{:.1f}h", thick_px=26, ymax=24)
    axes[1].set_title("Avg resolution time (hours)")
    axes[1].text(0.5, -0.30, f"strong effect: Cohen's d = {t['resolution_cohens_d_neg_vs_nonneg']:.1f}", transform=axes[1].transAxes, ha="center", fontsize=7.6, color=T["ink2"])
    viz.vbars(axes[2], labs, o.avg_revenue_inr.values / 1000, cols, err=np.array([o.rev_lo.values / 1000, o.rev_hi.values / 1000]), fmt="{:.1f}", thick_px=26, ymax=9)
    axes[2].set_title("Avg revenue (INR thousand)")
    axes[2].text(0.5, -0.30, f"no effect: Kruskal p = {t['revenue_kruskal_p']:.2f}", transform=axes[2].transAxes, ha="center", fontsize=7.6, color=T["ink2"])
    for a in axes:
        a.tick_params(labelsize=7.6)
    fig.subplots_adjust(left=0.06, right=0.985, top=viz.top(fig, 0.85), bottom=0.2)
    viz.heading(fig, "Sentiment moves repeat purchase and resolution time, but not revenue", "Mean by sentiment class with 95% CI (whiskers); reference sentiment")
    return viz.save(fig, "fig07_sentiment_vs_outcomes")


def fig08_resolution_distribution() -> str:
    d = silver(); col = _sent_col(d); sc = viz.SENTIMENT["light"]
    neg = d[d[col] == "Negative"].resolution_time_hours; non = d[d[col] != "Negative"].resolution_time_hours
    fig, ax, _ = viz.new_fig(6.6, 3.55)
    bins = np.arange(0, 33, 1.5)
    for s_, c_ in [(non, sc["Positive"]), (neg, sc["Negative"])]:
        h, e = np.histogram(s_, bins=bins, density=True)
        xs = np.repeat(e, 2)[1:-1]; ys = np.repeat(h, 2)
        ax.fill_between(xs, ys, color=c_, alpha=0.10, lw=0)
        ax.plot(xs, ys, color=c_, lw=1.6, solid_joinstyle="round")
        ax.plot([s_.mean()], [0.0], marker="o", ms=7, color=c_, mec=T["surface"], mew=1.5, zorder=6, clip_on=False)
    viz.style_axes(ax, "light", grid="y")
    ax.set_xlim(0, 32); ax.set_ylim(0, 0.115)
    ax.set_xlabel("Resolution time (hours)   -   dots mark the group means", fontsize=8, color=T["ink2"]); ax.set_ylabel("Density", fontsize=8, color=T["ink2"])
    for x_, lab in [(12, "12h SLA target"), (18, "18h critical delay")]:
        ax.axvline(x_, color=T["ink2"], lw=1.0, zorder=4)
        ax.text(x_ + 0.3, 0.112, lab, fontsize=7.6, color=T["ink2"], va="top")
    viz.legend_keys(ax, [(f"Negative  (mean {neg.mean():.1f}h)", sc["Negative"]), (f"Neutral / Positive  (mean {non.mean():.1f}h)", sc["Positive"])], loc="upper right", kind="line", bbox=(1.0, 0.86))
    fig.subplots_adjust(left=0.1, right=0.97, top=viz.top(fig, 0.78), bottom=0.15)
    viz.heading(fig, "Negative cases take twice as long to resolve", "Two near-separate distributions: resolution time alone predicts negative sentiment with AUC 0.91")
    return viz.save(fig, "fig08_resolution_distribution")


def fig09_sentiment_by_resolution_band() -> str:
    d = silver(); col = _sent_col(d); sc = viz.SENTIMENT["light"]
    ct = pd.crosstab(d.resolution_band, d[col], normalize="index").reindex(C.RES_BAND_LABELS) * 100
    n = d.resolution_band.value_counts().reindex(C.RES_BAND_LABELS)
    fig, ax, _ = viz.new_fig(6.6, 3.55)
    viz.stacked_gap_bars_h(ax, [f"{b}   (n={int(n[b]):,})" for b in C.RES_BAND_LABELS], {k: ct[k].values for k in C.SENTIMENT_ORDER}, sc, thick_px=24)
    viz.legend_keys(ax, [(k, sc[k]) for k in C.SENTIMENT_ORDER], loc="lower center", ncol=3, bbox=(0.5, -0.22))
    fig.subplots_adjust(left=0.2, right=0.97, top=viz.top(fig, 0.78), bottom=0.2)
    viz.heading(fig, "Negative share climbs steeply with resolution time", "Share of feedback by sentiment within each resolution-time band (100% stacked)")
    return viz.save(fig, "fig09_sentiment_by_resolution_band")


def fig10_correlation_heatmap() -> str:
    sp = pd.read_csv(C.TABLE_DIR / "eda_spearman.csv", index_col=0)
    names = {"rating": "Rating", "csat_score": "CSAT", "customer_revenue_inr": "Revenue", "order_count": "Orders", "repeat_purchase_flag": "Repeat flag", "resolution_time_hours": "Resolution h"}
    sp = sp.rename(index=names, columns=names)
    fig, ax, _ = viz.new_fig(5.8, 4.7)
    im = ax.imshow(sp.values, cmap=viz.div_cmap("light"), vmin=-1, vmax=1)
    ax.set_xticks(range(len(sp))); ax.set_yticks(range(len(sp)))
    ax.set_xticklabels(sp.columns, rotation=35, ha="right", fontsize=8, color=T["ink2"]); ax.set_yticklabels(sp.index, fontsize=8, color=T["ink2"])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(sp)):
        for j in range(len(sp)):
            v = sp.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if abs(v) > 0.55 else T["ink"])
    for k in range(len(sp) + 1):
        ax.axhline(k - 0.5, color=T["surface"], lw=2); ax.axvline(k - 0.5, color=T["surface"], lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, colors=T["muted"], length=0); cb.set_label("Spearman rho (blue +, red -)", fontsize=8, color=T["ink2"])
    fig.subplots_adjust(left=0.2, right=0.86, top=viz.top(fig, 0.82), bottom=0.2)
    viz.heading(fig, "What moves together, and what does not", "Spearman rank correlations; revenue and order count are unrelated to everything else")
    return viz.save(fig, "fig10_correlation_heatmap")


def fig11_fdr_hotspot_scan() -> str:
    h = tbl("eda_hotspot_scan"); n = len(h)
    fig, ax, _ = viz.new_fig(6.4, 3.7)
    ax.scatter(h["rank"], h.p_value, s=12, color=ACC, edgecolor=T["surface"], linewidth=0.6, zorder=4)
    ax.plot(h["rank"], h.bh_threshold, color=SENT["Negative"], lw=1.8, zorder=3)
    ax.axhline(0.05, color=T["muted"], lw=1.0, zorder=2)
    ax.text(n * 0.99, 0.065, "raw p = 0.05", fontsize=7.6, color=T["muted"], ha="right", va="bottom")
    viz.style_axes(ax, "light", grid="y")
    ax.set_xlim(0, n + 5); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cells ranked by p-value (smallest first)", fontsize=8, color=T["ink2"]); ax.set_ylabel("Fisher exact p-value", fontsize=8, color=T["ink2"])
    hs = load_metrics("eda")["hotspot"]
    ax.text(0.03, 0.80, f"{hs['tests']} cells tested\n{hs['raw_p_lt_0.05']} have raw p < 0.05 (chance alone: about {hs['expected_by_chance']:.0f})\n{hs['significant_after_fdr']} survive FDR control (smallest q = {hs['min_q_bh']:.2f})",
            transform=ax.transAxes, fontsize=8.4, color=T["ink"], linespacing=1.6, va="center")
    viz.legend_keys(ax, [("Segment cell (one- and two-way)", ACC), ("Benjamini-Hochberg threshold (q = 0.05)", SENT["Negative"])], loc="lower right", kind="line", bbox=(1.0, 0.1))
    fig.subplots_adjust(left=0.11, right=0.97, top=viz.top(fig, 0.78), bottom=0.14)
    viz.heading(fig, "Multiple-testing check: the apparent 'hotspots' are chance", "Every blue dot sits above the red line, so no cell is significant after false-discovery-rate control")
    return viz.save(fig, "fig11_fdr_hotspot_scan")


def make_part1() -> list[str]:
    return [f() for f in [fig01_pipeline_architecture, fig02_star_schema_erd, fig03_data_profile, fig04_dq_findings, fig05_sentiment_mix,
                          fig06_negative_rate_by_dimension, fig07_sentiment_vs_outcomes, fig08_resolution_distribution,
                          fig09_sentiment_by_resolution_band, fig10_correlation_heatmap, fig11_fdr_hotspot_scan]]


def make_all() -> list[str]:
    from . import figures2
    return make_part1() + figures2.make_part2()


if __name__ == "__main__":  # pragma: no cover
    for p in make_part1():
        print(p)
