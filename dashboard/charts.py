"""Plotly chart builders for the dashboard (interactive Streamlit app and the static HTML export).

Every builder returns ``(figure, table_dataframe)`` - the table is the accessible 'table view' twin of the chart.
Colour tokens come from ``cvi.viz`` (validated palette).  Bars use a 3px corner radius (Plotly rounds all four
corners; the print figures keep the stricter square-baseline spec).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvi import config as C  # noqa: E402
from cvi import kpis as K  # noqa: E402
from cvi import viz  # noqa: E402

SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
SC = "sentiment_pred"


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _tok(theme: str):
    return viz.THEMES[theme], viz.SENTIMENT[theme], viz.SEGMENT[theme], viz.SERIES[theme], viz.ACCENT[theme]


def _layout(fig: go.Figure, theme: str, height: int = 320, legend: bool = False, title: str | None = None, margin: dict | None = None, legend_pos: str = "top") -> go.Figure:
    t = viz.THEMES[theme]
    top = 36 if (legend and legend_pos == "top") else (40 if title else 14)
    leg = dict(orientation="h", font=dict(color=t["ink2"], size=12), bgcolor="rgba(0,0,0,0)", traceorder="normal")
    leg.update(dict(yanchor="bottom", y=1.02, xanchor="left", x=0) if legend_pos == "top" else dict(yanchor="top", y=-0.22, xanchor="left", x=0))
    fig.update_layout(
        template=None, paper_bgcolor=t["surface"], plot_bgcolor=t["surface"], height=height,
        font=dict(family=SANS, size=12, color=t["ink2"]),
        margin=margin or dict(l=12, r=18, t=top, b=14),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=t["ink"])) if title else None,
        showlegend=legend, legend=leg,
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(family=SANS, color=t["ink"], size=12)),
        bargap=0.5, barcornerradius=3,
    )
    ax = dict(gridcolor=t["grid"], linecolor=t["axis"], zeroline=False, tickfont=dict(color=t["muted"], size=11), title_font=dict(color=t["ink2"], size=12), showline=True, gridwidth=1, automargin=True)
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    return fig


def _empty(theme: str, msg: str = "No records match the current filters") -> tuple[go.Figure, pd.DataFrame]:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=viz.THEMES[theme]["muted"], size=13))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout(fig, theme, 160), pd.DataFrame()


# ---------------------------------------------------------------- charts ----
def sentiment_mix(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, S, _, _, _ = _tok(theme)
    shares = df[SC].value_counts(normalize=True).reindex(C.SENTIMENT_ORDER).fillna(0) * 100
    n = df[SC].value_counts().reindex(C.SENTIMENT_ORDER).fillna(0).astype(int)
    fig = go.Figure()
    for s in C.SENTIMENT_ORDER:
        fig.add_bar(y=[""], x=[shares[s]], orientation="h", name=s, marker=dict(color=S[s], line=dict(color=t["surface"], width=2)), text=[f"{shares[s]:.1f}%" if shares[s] > 6 else ""],
                    textposition="inside", insidetextanchor="start", textfont=dict(color="#0b0b0b" if s == "Neutral" else "#ffffff", size=13), customdata=[[int(n[s])]],
                    hovertemplate=f"<b>%{{x:.1f}}%</b> {s}<br>%{{customdata[0]:,}} records<extra></extra>")
    fig.update_layout(barmode="stack", barcornerradius=4)
    fig.update_xaxes(range=[0, 100], visible=False); fig.update_yaxes(visible=False)
    _layout(fig, theme, 96, legend=True, margin=dict(l=0, r=0, t=4, b=0))
    fig.update_layout(bargap=0.0, legend=dict(orientation="h", yanchor="top", y=-0.05, x=0, traceorder="normal"))
    return fig, pd.DataFrame({"sentiment": C.SENTIMENT_ORDER, "records": n.values, "share_pct": shares.round(2).values})


def neg_rate_by(df: pd.DataFrame, col: str, label: str, theme: str):
    if df.empty:
        return _empty(theme)
    t, S, _, _, _ = _tok(theme)
    k = K.kpi_table(df, col, SC).sort_values("negative_rate", ascending=True)
    overall = float((df[SC] == "Negative").mean()) * 100
    ylab = [f"{lvl}   {100 * r:.1f}%" for lvl, r in zip(k[col], k.negative_rate)]
    fig = go.Figure(go.Bar(y=ylab, x=100 * k.negative_rate, orientation="h", marker=dict(color=S["Negative"]),
                           error_x=dict(type="data", symmetric=False, array=100 * (k.negative_rate_hi - k.negative_rate), arrayminus=100 * (k.negative_rate - k.negative_rate_lo),
                                        color=t["ink2"], thickness=1.2, width=3),
                           customdata=np.c_[k.feedback_count, 100 * k.negative_rate_lo, 100 * k.negative_rate_hi],
                           hovertemplate="<b>%{x:.1f}%</b> negative<br>95% CI %{customdata[1]:.1f}-%{customdata[2]:.1f}%<br>n = %{customdata[0]:,}<extra>" + label + "</extra>"))
    fig.add_vline(x=overall, line_color=t["muted"], line_width=1)
    fig.add_annotation(x=overall, y=1.0, yref="paper", text=f"all {overall:.1f}%", showarrow=False, yanchor="bottom", font=dict(color=t["muted"], size=11))
    fig.update_xaxes(range=[0, max(45, 100 * k.negative_rate_hi.max() * 1.08)], ticksuffix="%")
    _layout(fig, theme, 70 + 30 * len(k), margin=dict(l=12, r=18, t=26, b=12))
    return fig, k[[col, "feedback_count", "negative_rate", "negative_rate_lo", "negative_rate_hi", "revenue_at_risk_inr"]].rename(columns={col: label})


def outcomes(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, S, _, _, _ = _tok(theme)
    k = K.kpi_table(df, SC, SC).set_index(SC).reindex(C.SENTIMENT_ORDER)
    g = df.groupby(SC, observed=True).customer_revenue_inr.agg(["mean", "std", "size"]).reindex(C.SENTIMENT_ORDER)
    se_rev = 1.96 * g["std"] / np.sqrt(g["size"])
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Repeat purchase rate (%)", "Avg resolution time (h)", "Avg revenue (INR thousand)"], horizontal_spacing=0.09)
    cols = [S[s] for s in C.SENTIMENT_ORDER]
    spec = [(100 * k.repeat_rate, 100 * (k.repeat_rate_hi - k.repeat_rate), 100 * (k.repeat_rate - k.repeat_rate_lo), "{:.0f}%"),
            (k.avg_resolution_hours, k.avg_resolution_hi - k.avg_resolution_hours, k.avg_resolution_hours - k.avg_resolution_lo, "{:.1f} h"),
            (g["mean"] / 1000, se_rev / 1000, se_rev / 1000, "{:.1f}K")]
    for i, (val, up, dn, fmt) in enumerate(spec, 1):
        fig.add_bar(x=C.SENTIMENT_ORDER, y=val, marker=dict(color=cols), showlegend=False, error_y=dict(type="data", symmetric=False, array=up.fillna(0), arrayminus=dn.fillna(0), color=t["ink2"], thickness=1.2, width=3),
                    text=[fmt.format(v) if pd.notna(v) else "" for v in val], textposition="outside", textfont=dict(color=t["ink"], size=12), cliponaxis=False,
                    hovertemplate="%{x}: <b>%{y:.2f}</b><extra></extra>", row=1, col=i)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=t["ink"]), x=a.x, xanchor="center")
    _layout(fig, theme, 340, margin=dict(l=12, r=12, t=44, b=14))
    for i, (val, _, _, _) in enumerate(spec, 1):
        fig.update_yaxes(range=[0, float(np.nanmax(val)) * 1.3], row=1, col=i)
    tbl = pd.DataFrame({"sentiment": C.SENTIMENT_ORDER, "records": k.feedback_count.values, "repeat_rate": k.repeat_rate.values, "avg_resolution_hours": k.avg_resolution_hours.values, "avg_revenue_inr": g["mean"].values})
    return fig, tbl


def resolution_hist(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, S, _, _, _ = _tok(theme)
    neg = df[df[SC] == "Negative"].resolution_time_hours; non = df[df[SC] != "Negative"].resolution_time_hours
    bins = np.arange(0, 33, 1.5)
    fig = go.Figure()
    rows = []
    for ser, col, nm in [(non, S["Positive"], "Neutral / Positive"), (neg, S["Negative"], "Negative")]:
        if len(ser) == 0:
            continue
        h, e = np.histogram(ser, bins=bins, density=True)
        fig.add_scatter(x=np.repeat(e, 2)[1:-1], y=np.repeat(h, 2), mode="lines", line=dict(color=col, width=2, shape="linear"), fill="tozeroy", fillcolor=_rgba(col, 0.10),
                        name=f"{nm}  (mean {ser.mean():.1f}h)", hovertemplate=f"{nm}<br>%{{x:.1f}}h: density %{{y:.3f}}<extra></extra>")
        rows += [{"group": nm, "bin_start_h": a, "density": b} for a, b in zip(e[:-1], h)]
    for x_, lab in [(12, "12h SLA"), (18, "18h critical")]:
        fig.add_vline(x=x_, line_color=t["ink2"], line_width=1)
        fig.add_annotation(x=x_, y=1.0, yref="paper", text=lab, showarrow=False, xanchor="left", yanchor="top", xshift=4, font=dict(color=t["ink2"], size=11))
    fig.update_xaxes(title_text="Resolution time (hours)", range=[0, 32]); fig.update_yaxes(title_text="Density", rangemode="tozero")
    _layout(fig, theme, 350, legend=True)
    return fig, pd.DataFrame(rows)


def sentiment_by_band(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, S, _, _, _ = _tok(theme)
    ct = pd.crosstab(df.resolution_band, df[SC], normalize="index").reindex(columns=C.SENTIMENT_ORDER).fillna(0) * 100
    n = df.resolution_band.value_counts().reindex(C.RES_BAND_LABELS).fillna(0).astype(int)
    ct = ct.reindex(C.RES_BAND_LABELS).fillna(0)
    ylab = [f"{b}  (n={int(n[b]):,})" for b in C.RES_BAND_LABELS]
    fig = go.Figure()
    for s in C.SENTIMENT_ORDER:
        fig.add_bar(y=ylab, x=ct[s], orientation="h", name=s, marker=dict(color=S[s], line=dict(color=t["surface"], width=2)), text=[f"{v:.0f}%" if v >= 7.5 else "" for v in ct[s]], textposition="inside",
                    insidetextanchor="middle", textfont=dict(color="#0b0b0b" if s == "Neutral" else "#ffffff", size=12), hovertemplate="%{y}<br><b>%{x:.1f}%</b> " + s + "<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
    _layout(fig, theme, 320, legend=True)
    fig.update_layout(bargap=0.35)
    return fig, ct.round(2).reset_index().rename(columns={"resolution_band": "resolution band"})


def theme_overview(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, _, _, cat, _ = _tok(theme)
    k = K.kpi_table(df, "issue_category", SC).sort_values("topic_share", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=k.issue_category, x=100 * k.topic_share, orientation="h", name="Topic share of feedback", marker=dict(color=cat[0]), hovertemplate="%{y}<br><b>%{x:.1f}%</b> of feedback<extra></extra>")
    fig.add_bar(y=k.issue_category, x=100 * k.rar_share_of_total, orientation="h", name="Share of Revenue at Risk", marker=dict(color=cat[1]), hovertemplate="%{y}<br><b>%{x:.1f}%</b> of Revenue at Risk<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12)
    fig.update_xaxes(ticksuffix="%", rangemode="tozero")
    _layout(fig, theme, 430, legend=True)
    fig.update_layout(bargap=0.35, bargroupgap=0.12)
    return fig, k[["issue_category", "feedback_count", "topic_share", "negative_rate", "revenue_at_risk_inr", "rar_share_of_total"]]


def _seq_scale(theme: str):
    ramp = viz.BLUE_RAMP if theme == "light" else viz.BLUE_RAMP[::-1]
    return [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]


def theme_heatmap(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t = viz.THEMES[theme]
    order = df.issue_category.value_counts().index
    ct = (pd.crosstab(df.issue_category, df.channel_name, normalize="columns") * 100).reindex(order)
    n = pd.crosstab(df.issue_category, df.channel_name).reindex(order)
    fig = go.Figure(go.Heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index), colorscale=_seq_scale(theme), zmin=0, zmax=20, xgap=2, ygap=2, customdata=n.values,
                               text=[[f"{v:.1f}%" for v in r] for r in ct.values], texttemplate="%{text}", textfont=dict(size=11, color="#ffffff" if theme == "light" else "#0b0b0b"),
                               hovertemplate="%{y} in %{x}<br><b>%{z:.1f}%</b> of the channel's feedback<br>n = %{customdata}<extra></extra>",
                               colorbar=dict(ticksuffix="%", tickfont=dict(color=t["muted"], size=10), outlinewidth=0, thickness=10)))
    fig.update_yaxes(autorange="reversed")
    _layout(fig, theme, 400, margin=dict(l=12, r=12, t=10, b=14))
    return fig, ct.round(2).reset_index()


def revenue_split(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, S, _, _, _ = _tok(theme)
    rev = df.groupby(SC, observed=True).customer_revenue_inr.sum().reindex(C.SENTIMENT_ORDER).fillna(0)
    tot = rev.sum()
    fig = go.Figure()
    for s in C.SENTIMENT_ORDER:
        fig.add_bar(y=[""], x=[100 * rev[s] / tot if tot else 0], orientation="h", name=s, marker=dict(color=S[s], line=dict(color=t["surface"], width=2)),
                    text=[f"INR {rev[s] / 1e6:.2f}M  {100 * rev[s] / tot:.1f}%" if tot and rev[s] / tot > 0.12 else ""], textposition="inside", insidetextanchor="start",
                    textfont=dict(color="#0b0b0b" if s == "Neutral" else "#ffffff", size=12), customdata=[[rev[s]]], hovertemplate=f"{s}: <b>INR %{{customdata[0]:,.0f}}</b> (%{{x:.1f}}%)<extra></extra>")
    fig.update_layout(barmode="stack", barcornerradius=4)
    fig.update_xaxes(range=[0, 100], visible=False); fig.update_yaxes(visible=False)
    _layout(fig, theme, 92, legend=True, margin=dict(l=0, r=0, t=4, b=0))
    fig.update_layout(bargap=0.0, legend=dict(orientation="h", yanchor="top", y=-0.05, x=0, traceorder="normal"))
    return fig, pd.DataFrame({"sentiment": C.SENTIMENT_ORDER, "revenue_inr": rev.values, "share_pct": (100 * rev / tot).round(2).values if tot else 0})


def segment_scatter(df: pd.DataFrame, theme: str):
    if df.empty or df.pc1.isna().all():
        return _empty(theme)
    t, _, SEG, _, _ = _tok(theme)
    rng = np.random.default_rng(C.SEED)
    fig = go.Figure()
    for nm in viz.SEGMENT_NAMES:
        g = df[df.segment_name == nm]
        if g.empty:
            continue
        fig.add_scattergl(x=g.pc1 + rng.normal(0, 0.03, len(g)), y=g.pc2 + rng.normal(0, 0.03, len(g)), mode="markers", name=f"{nm}  (n={len(g):,})", marker=dict(color=SEG[nm], size=5, opacity=0.55),
                          customdata=np.c_[g.csat_score, g.resolution_time_hours, g.repeat_purchase_flag * 100], hovertemplate=f"<b>{nm}</b><br>CSAT %{{customdata[0]:.1f}} | %{{customdata[1]:.1f}} h | repeat %{{customdata[2]:.0f}}%<extra></extra>")
    fig.update_xaxes(title_text="Principal component 1"); fig.update_yaxes(title_text="Principal component 2")
    _layout(fig, theme, 380, legend=True)
    return fig, df.groupby("segment_name").size().rename("records").reset_index()


def segment_profile(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t, _, SEG, _, _ = _tok(theme)
    p = df.groupby("segment_name").agg(records=("feedback_id", "size"), csat=("csat_score", "mean"), res=("resolution_time_hours", "mean"), repeat=("repeat_purchase_flag", "mean")).reindex(viz.SEGMENT_NAMES).dropna()
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Average CSAT (1-5)", "Avg resolution (hours)", "Repeat purchase rate (%)"], horizontal_spacing=0.08)
    for i, (col, fmt, scale) in enumerate([("csat", "{:.1f}", 1), ("res", "{:.1f}", 1), ("repeat", "{:.0f}%", 100)], 1):
        for nm in p.index:
            v = p.loc[nm, col] * scale
            fig.add_bar(x=[nm], y=[v], name=nm, legendgroup=nm, showlegend=(i == 1), marker=dict(color=SEG[nm]), text=[fmt.format(v)], textposition="outside", cliponaxis=False,
                        textfont=dict(color=t["ink"], size=12), hovertemplate=f"{nm}<br><b>%{{y:.2f}}</b><extra></extra>", row=1, col=i)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=t["ink"]))
    fig.update_xaxes(showticklabels=False)
    for i, (col, scale) in enumerate([("csat", 1), ("res", 1), ("repeat", 100)], 1):
        fig.update_yaxes(range=[0, float(p[col].max() * scale) * 1.32], row=1, col=i)
    _layout(fig, theme, 320, legend=True, legend_pos="bottom", margin=dict(l=12, r=12, t=44, b=56))
    fig.update_layout(bargap=0.35)
    return fig, p.reset_index().round(3)


def risk_matrix(df: pd.DataFrame, theme: str):
    if df.empty:
        return _empty(theme)
    t = viz.THEMES[theme]
    d = df.assign(_rar=df.customer_revenue_inr.astype(float) * (df[SC] == "Negative"))
    pv = d.pivot_table(index="segment_name", columns="revenue_tier", values="_rar", aggfunc="sum", observed=False).reindex(index=viz.SEGMENT_NAMES, columns=C.REVENUE_TIER_LABELS).fillna(0) / 1e3
    fig = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale=_seq_scale(theme), xgap=3, ygap=3, text=[[f"{v:,.0f}K" for v in r] for r in pv.values], texttemplate="%{text}",
                               textfont=dict(size=12, color="#ffffff" if theme == "light" else "#0b0b0b"), hovertemplate="%{y} x %{x}<br>Revenue at Risk <b>INR %{z:,.0f}K</b><extra></extra>",
                               colorbar=dict(ticksuffix="K", tickfont=dict(color=t["muted"], size=10), outlinewidth=0, thickness=10)))
    fig.add_shape(type="rect", x0=1.5, x1=3.5, y0=-0.5, y1=0.5, line=dict(color=t["ink"], width=2), fillcolor="rgba(0,0,0,0)")
    fig.add_annotation(x=2.5, y=-0.5, text="Critical tier", showarrow=False, yshift=12, font=dict(color=t["ink"], size=11))
    fig.update_yaxes(autorange="reversed")
    _layout(fig, theme, 270, margin=dict(l=12, r=12, t=30, b=14))
    return fig, pv.round(0).reset_index()


def scenarios(m: dict, theme: str):
    t, _, _, cat, acc = _tok(theme)
    sc = pd.DataFrame(m["scenarios"])
    lab = [f"{k.capitalize()} ({int(100 * v)}% of gap)   INR {r / 1e3:,.0f}K" for k, v, r in zip(sc.scenario, sc.share_of_gap_recovered, sc.recoverable_revenue_inr)]
    fig = go.Figure(go.Bar(y=lab[::-1], x=sc.recoverable_revenue_inr[::-1] / 1e3, orientation="h", marker=dict(color=acc),
                           error_x=dict(type="data", symmetric=False, array=(sc.ci_hi - sc.recoverable_revenue_inr)[::-1] / 1e3, arrayminus=(sc.recoverable_revenue_inr - sc.ci_lo)[::-1] / 1e3, color=t["ink2"], thickness=1.2, width=3),
                           hovertemplate="%{y}<br>bootstrap 95% CI shown<extra></extra>"))
    fig.update_xaxes(title_text="Recoverable revenue (INR thousand)", range=[0, 1000])
    _layout(fig, theme, 220, margin=dict(l=12, r=18, t=10, b=14))
    return fig, sc[["scenario", "share_of_gap_recovered", "recoverable_revenue_inr", "ci_lo", "ci_hi"]]


def rank_stability(tbl: pd.DataFrame, theme: str):
    t, S, _, _, acc = _tok(theme)
    r = tbl.sort_values("revenue_at_risk_inr", ascending=True)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Revenue at Risk (INR thousand)", "Chance of ranking #1 (%; line = chance 12.5%)"], horizontal_spacing=0.06)
    fig.add_bar(y=r.level, x=r.revenue_at_risk_inr / 1e3, orientation="h", marker=dict(color=S["Negative"]), showlegend=False, hovertemplate="%{y}<br><b>INR %{x:,.0f}K</b><extra></extra>", row=1, col=1)
    fig.add_bar(y=r.level, x=100 * r.p_rank1_rar, orientation="h", marker=dict(color=acc), showlegend=False, text=[f"{100 * v:.0f}%" for v in r.p_rank1_rar], textposition="outside", cliponaxis=False,
                textfont=dict(color=t["ink"]), hovertemplate="%{y}<br><b>%{x:.1f}%</b> of 2,000 resamples<extra></extra>", row=1, col=2)
    fig.add_vline(x=12.5, line_color=t["muted"], line_width=1, row=1, col=2)
    for a in fig.layout.annotations[:2]:
        a.update(font=dict(size=12, color=t["ink"]))
    fig.update_xaxes(rangemode="tozero")
    _layout(fig, theme, 420, margin=dict(l=12, r=40, t=44, b=14))
    return fig, r[["level", "n", "negative_rate", "revenue_at_risk_inr", "p_rank1_rar", "p_top3_rar"]]


def forest(tbl: pd.DataFrame, theme: str):
    t, _, _, _, acc = _tok(theme)
    order = ["Negative vs Positive sentiment", "Neutral vs Positive sentiment", "Resolution time (per +1 SD)", "Resolution time alone (per +1 SD, unadjusted)", "Revenue (per +INR 1,000)", "Order count (per +1)"]
    d = tbl.set_index("label").loc[order].reset_index()
    short = {"Negative vs Positive sentiment": "Negative vs Positive", "Neutral vs Positive sentiment": "Neutral vs Positive", "Resolution time (per +1 SD)": "Resolution (+1 SD), adjusted",
             "Resolution time alone (per +1 SD, unadjusted)": "Resolution (+1 SD), alone", "Revenue (per +INR 1,000)": "Revenue (+INR 1K)", "Order count (per +1)": "Orders (+1)"}
    d["label_full"] = d["label"]
    d["label"] = d["label"].map(short)
    fig = go.Figure()
    for sig, col, nm in [(True, acc, "p < 0.05"), (False, t["muted"], "not significant")]:
        g = d[(d.p_value < 0.05) == sig]
        if g.empty:
            continue
        fig.add_scatter(x=g.odds_ratio, y=g.label, mode="markers", name=nm, marker=dict(color=col, size=11, line=dict(color=t["surface"], width=2)),
                        error_x=dict(type="data", symmetric=False, array=g.ci_hi - g.odds_ratio, arrayminus=g.odds_ratio - g.ci_lo, color=col, thickness=2, width=0),
                        customdata=np.c_[g.ci_lo, g.ci_hi, g.p_value], hovertemplate="%{y}<br>OR <b>%{x:.2f}</b> (%{customdata[0]:.2f}-%{customdata[1]:.2f})<br>p = %{customdata[2]:.3g}<extra></extra>")
    fig.add_vline(x=1, line_color=t["muted"], line_width=1.2)
    fig.update_xaxes(type="log", range=[np.log10(0.1), np.log10(1.6)], tickvals=[0.1, 0.2, 0.5, 1], ticktext=["0.1", "0.2", "0.5", "1"], title_text="Odds ratio for repeat purchase (log scale; 1 = no effect)")
    fig.update_yaxes(autorange="reversed")
    _layout(fig, theme, 350, legend=True)
    return fig, d[["label_full", "odds_ratio", "ci_lo", "ci_hi", "p_value"]].rename(columns={"label_full": "term"})


def model_comparison(tbl: pd.DataFrame, theme: str):
    t, _, _, cat, _ = _tok(theme)
    order = ["Baseline: majority class", "VADER lexicon (+/-0.05)", "TF-IDF + Multinomial NB", "TF-IDF + Linear SVM", "TF-IDF + Logistic Regression", "Hybrid: TF-IDF + VADER features (LR)"]
    cv = tbl[tbl.regime == "cv5"].set_index("model").macro_f1.reindex(order)
    lo = tbl[tbl.regime == "lodo"].set_index("model").macro_f1.reindex(order)
    nice = {"Baseline: majority class": "Majority baseline", "VADER lexicon (+/-0.05)": "VADER lexicon", "TF-IDF + Multinomial NB": "TF-IDF + Naive Bayes", "TF-IDF + Linear SVM": "TF-IDF + Linear SVM",
            "TF-IDF + Logistic Regression": "TF-IDF + Logistic Reg. (chosen)", "Hybrid: TF-IDF + VADER features (LR)": "Hybrid TF-IDF + VADER"}
    lab = [nice[o] for o in order][::-1]
    fig = go.Figure()
    fig.add_bar(y=lab, x=cv.values[::-1], orientation="h", name="In-distribution (5-fold CV)", marker=dict(color=cat[0]), text=[f"{v:.2f}" for v in cv.values[::-1]], textposition="outside", cliponaxis=False, textfont=dict(color=t["ink"]))
    fig.add_bar(y=lab, x=lo.values[::-1], orientation="h", name="Unseen opinion words (leave-one-descriptor-out)", marker=dict(color=cat[1]), text=[f"{v:.2f}" for v in lo.values[::-1]], textposition="outside", cliponaxis=False, textfont=dict(color=t["ink"]))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1)
    fig.update_xaxes(range=[0, 1.12], title_text="Macro-F1")
    _layout(fig, theme, 400, legend=True, margin=dict(l=12, r=36, t=36, b=14))
    fig.update_layout(bargap=0.35, bargroupgap=0.1)
    return fig, pd.DataFrame({"model": order, "macro_f1_cv5": cv.values, "macro_f1_lodo": lo.values})


def priority(rec: pd.DataFrame, theme: str):
    t, _, _, _, acc = _tok(theme)
    r = rec.sort_values("priority_rank", ascending=False)      # table's own total order (ties by impact, then confidence); rank 1 plots last = top
    fig = go.Figure(go.Bar(y=[f"{i}  {tt}" for i, tt in zip(r.id, r.short)], x=r.priority_score, orientation="h", marker=dict(color=acc),
                           text=[f"{s:.1f}   I{a} x C{c} / E{e}   {p}" for s, a, c, e, p in zip(r.priority_score, r.impact_score, r.confidence, r.effort, r.phase)], textposition="outside", cliponaxis=False, textfont=dict(color=t["ink"], size=11),
                           customdata=np.c_[r.title, r.expected_impact], hovertemplate="<b>%{customdata[0]}</b><br>score %{x:.1f}<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(range=[0, 17], title_text="Priority score = Impact x Confidence / Effort")
    _layout(fig, theme, 350, margin=dict(l=12, r=20, t=10, b=14))
    return fig, r.sort_values("priority_rank")[["id", "title", "priority_score", "impact_score", "confidence", "effort", "phase"]]


# ---------------------------------------------------------------- tiles -----
STATUS_SVG = {
    "good": '<svg width="12" height="12" viewBox="0 0 12 12"><polygon points="6,1 11,10 1,10" fill="{c}"/></svg>',
    "warning": '<svg width="12" height="12" viewBox="0 0 12 12"><polygon points="6,0.5 11.5,6 6,11.5 0.5,6" fill="{c}"/></svg>',
    "serious": '<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="5.2" fill="{c}"/></svg>',
    "critical": '<svg width="12" height="12" viewBox="0 0 12 12"><polygon points="1,2 11,2 6,11" fill="{c}"/></svg>',
}


def kpi_tiles_html(df: pd.DataFrame) -> str:
    """Stat-tile row (label, value, proposed target, status = icon + label).  Styled by the CSS classes in css()."""
    from cvi import decision as D
    if df.empty:
        return "<div class='cvi-note'>No records match the current filters.</div>"
    k = K.overall_kpis(df, SC)
    T = C.PROPOSED_TARGETS
    items = [
        ("Negative Sentiment Rate", f"{100 * k['negative_sentiment_rate']:.1f}%", f"target &le; {100 * T['negative_sentiment_rate']:.0f}%", D._status(k["negative_sentiment_rate"], T["negative_sentiment_rate"], False)),
        ("Repeat Purchase Rate", f"{100 * k['repeat_purchase_rate']:.1f}%", f"target &ge; {100 * T['repeat_purchase_rate']:.0f}%", D._status(k["repeat_purchase_rate"], T["repeat_purchase_rate"], True)),
        ("Avg Resolution Time", f"{k['avg_resolution_hours']:.1f} h", f"target &le; {T['avg_resolution_hours']:.0f} h", D._status(k["avg_resolution_hours"], T["avg_resolution_hours"], False)),
        ("Revenue at Risk", f"INR {k['revenue_at_risk_inr'] / 1e6:.2f}M", f"{100 * k['revenue_at_risk_share']:.1f}% of revenue &middot; target &le; 20%", D._status(k["revenue_at_risk_share"], 0.20, False)),
        ("Net Sentiment Score", f"{k['net_sentiment_score']:+.1f}", f"target &ge; {T['net_sentiment_score']:.0f}", D._status(k["net_sentiment_score"], T["net_sentiment_score"], True)),
        ("Average CSAT", f"{k['avg_csat']:.2f}", f"target &ge; {T['avg_csat']:.1f}", D._status(k["avg_csat"], T["avg_csat"], True)),
    ]
    lab = {"good": "On target", "warning": "Near target", "serious": "Off target", "critical": "Well off target"}
    out = []
    for name, val, tgt, st in items:
        icon = STATUS_SVG[st].format(c=viz.STATUS[st])
        out.append(f"<div class='cvi-tile'><div class='cvi-tl'>{name}</div><div class='cvi-tv'>{val}</div><div class='cvi-tt'>{tgt}</div><div class='cvi-ts'>{icon}<span>{lab[st]}</span></div></div>")
    return "<div class='cvi-tiles'>" + "".join(out) + "</div>"


def css(theme: str) -> str:
    t = viz.THEMES[theme]
    return f"""
<style>
.cvi-tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:4px 0 6px}}
.cvi-tile{{background:{t['surface']};border:1px solid {t['axis']};border-radius:10px;padding:12px 14px 10px;font-family:{SANS}}}
.cvi-tl{{font-size:12px;color:{t['ink2']}}}
.cvi-tv{{font-size:26px;font-weight:600;color:{t['ink']};margin:6px 0 2px;line-height:1.1}}
.cvi-tt{{font-size:11px;color:{t['muted']}}}
.cvi-ts{{display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:{t['ink']}}}
.cvi-hero{{font-family:{SANS};color:{t['ink']}}}
.cvi-hero .n{{font-size:56px;font-weight:600;line-height:1}}
.cvi-hero .l{{font-size:13px;color:{t['ink2']};margin-top:4px}}
.cvi-note{{color:{t['ink2']};font-size:13px;font-family:{SANS}}}
.cvi-card{{background:{t['surface']};border:1px solid {t['axis']};border-radius:10px;padding:14px 16px;font-family:{SANS};margin-bottom:10px}}
.cvi-card h4{{margin:0 0 4px;font-size:14px;color:{t['ink']}}}
.cvi-card p{{margin:2px 0;font-size:12.5px;color:{t['ink2']}}}
.cvi-chip{{display:inline-block;border:1px solid {t['axis']};border-radius:999px;padding:1px 9px;font-size:11px;color:{t['ink2']};margin-right:6px}}
</style>"""
