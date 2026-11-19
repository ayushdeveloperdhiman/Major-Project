"""Static figures - part 2: NLP, topics, data mining, integration and decision layer (see figures.py for part 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import config as C
from . import diagrams as D
from . import viz
from .figures import ACC, SENT, T, silver, tbl
from .utils import fmt_inr, load_metrics

SEG = viz.SEGMENT["light"]
CAT = viz.SERIES["light"]


def _ticks_inr_m(ax, axis="x"):
    f = lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K" if v else "0"  # noqa: E731
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(f))


# ------------------------------------------------------------------- NLP ----
def fig12_nlp_model_comparison() -> str:
    c = tbl("sentiment_model_comparison")
    order = ["Baseline: majority class", "VADER lexicon (+/-0.05)", "TF-IDF + Multinomial NB", "TF-IDF + Linear SVM", "TF-IDF + Logistic Regression", "Hybrid: TF-IDF + VADER features (LR)"]
    short = {"Baseline: majority class": "Majority-class baseline", "VADER lexicon (+/-0.05)": "VADER (general lexicon)", "TF-IDF + Multinomial NB": "TF-IDF + Naive Bayes",
             "TF-IDF + Linear SVM": "TF-IDF + Linear SVM", "TF-IDF + Logistic Regression": "TF-IDF + Logistic Regression (chosen)", "Hybrid: TF-IDF + VADER features (LR)": "Hybrid TF-IDF + VADER"}
    cv = c[c.regime == "cv5"].set_index("model").macro_f1.reindex(order)
    lo = c[c.regime == "lodo"].set_index("model").macro_f1.reindex(order)
    fig, ax, _ = viz.new_fig(6.9, 3.9)
    n = len(order)
    ypos = np.arange(n)[::-1]
    ax.set_xlim(0, 1.18); ax.set_ylim(-0.7, n - 0.3)
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    ax.set_yticks(ypos); ax.set_yticklabels([short[o] for o in order], fontsize=8, color=T["ink2"])
    py, px = viz._px_per_unit(ax, "y"), viz._px_per_unit(ax, "x")
    for y, a, b in zip(ypos, cv.values, lo.values):
        viz.rounded_bar(ax, 0, y + 0.19, a, 11 / py, CAT[0], "h"); viz.rounded_bar(ax, 0, y - 0.19, b, 11 / py, CAT[1], "h")
        ax.text(a + 5 / px, y + 0.19, f"{a:.2f}", va="center", fontsize=7.8, color=T["ink"]); ax.text(b + 5 / px, y - 0.19, f"{b:.2f}", va="center", fontsize=7.8, color=T["ink"])
    ax.set_xlabel("Macro-F1 (1.0 = perfect)", fontsize=8, color=T["ink2"])
    viz.legend_keys(ax, [("In-distribution: 5-fold CV", CAT[0]), ("Unseen opinion words: leave-one-descriptor-out", CAT[1])], loc="upper right", bbox=(1.0, 1.0))
    fig.subplots_adjust(left=0.3, right=0.97, top=viz.top(fig, 0.78), bottom=0.13)
    viz.heading(fig, "Perfect in-distribution scores, but no mechanism for new vocabulary", "The text is templated (19 opinion words): supervised models memorise them and fall back to the prior on unseen words")
    return viz.save(fig, "fig12_nlp_model_comparison")


def _confusion_panel(ax, cm: pd.DataFrame, title: str, theme="light"):
    vals = cm.values
    rown = vals / vals.sum(1, keepdims=True)
    cmap = viz.seq_cmap(theme)
    ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
    labs = C.SENTIMENT_ORDER
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labs, fontsize=8, color=T["ink2"]); ax.set_yticklabels(labs, fontsize=8, color=T["ink2"])
    ax.set_xlabel("Predicted", fontsize=8, color=T["ink2"]); ax.set_ylabel("Reference (rating-implied)", fontsize=8, color=T["ink2"])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            v = rown[i, j]
            ax.text(j, i, f"{vals[i, j]:,}\n{100 * v:.0f}%", ha="center", va="center", fontsize=8.4, color="#ffffff" if v > 0.55 else T["ink"])
    for k in range(4):
        ax.axhline(k - 0.5, color=T["surface"], lw=2); ax.axvline(k - 0.5, color=T["surface"], lw=2)
    ax.set_title(title, fontsize=9)


def fig13_confusion_matrices() -> str:
    a = pd.read_csv(C.TABLE_DIR / "sentiment_confusion_chosen.csv", index_col=0)
    b = pd.read_csv(C.TABLE_DIR / "sentiment_confusion_vader.csv", index_col=0)
    s = load_metrics("sentiment")
    fig, axes, _ = viz.new_grid(1, 2, w=7.0, h=3.7, theme="light", gridspec_kw={"wspace": 0.45})
    _confusion_panel(axes[0], a, f"TF-IDF + LR   accuracy {100 * s['cv5']['accuracy']:.1f}%")
    _confusion_panel(axes[1], b, f"VADER   accuracy {100 * s['vader']['cv5_accuracy']:.1f}%")
    fig.subplots_adjust(left=0.12, right=0.98, top=viz.top(fig, 0.9), bottom=0.16)
    viz.heading(fig, "Domain-trained model vs general-purpose lexicon", "Confusion matrices (cells: records and % of each reference class); out-of-fold predictions, n = 2,400")
    return viz.save(fig, "fig13_confusion_matrices")


def fig14_vader_by_descriptor() -> str:
    v = tbl("sentiment_vader_by_descriptor")
    v = v.sort_values(["true", "vader_accuracy"], key=lambda s: s.map({"Negative": 0, "Neutral": 1, "Positive": 2}) if s.name == "true" else s).reset_index(drop=True)
    fig, ax, _ = viz.new_fig(6.6, 5.0)
    n = len(v)
    ypos = np.arange(n)[::-1]
    ax.set_xlim(0, 1.32); ax.set_ylim(-0.7, n - 0.3)
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    ax.set_yticks(ypos); ax.set_yticklabels(v.descriptor, fontsize=7.8, color=T["ink2"])
    py, px = viz._px_per_unit(ax, "y"), viz._px_per_unit(ax, "x")
    for y, r in zip(ypos, v.itertuples()):
        viz.rounded_bar(ax, 0, y, max(r.vader_accuracy, 0.002), 10 / py, SENT[r.true], "h")
        note = "  word scores exactly 0" if r.vader_word_score == 0 else ""
        ax.text(max(r.vader_accuracy, 0) + 5 / px, y, f"{100 * r.vader_accuracy:.0f}%{note}", va="center", fontsize=7.4, color=T["ink"])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of that word's records VADER classifies correctly", fontsize=8, color=T["ink2"])
    viz.legend_keys(ax, [("Negative word", SENT["Negative"]), ("Neutral word", SENT["Neutral"]), ("Positive word", SENT["Positive"])], loc="lower right", bbox=(1.0, 0.02))
    fig.subplots_adjust(left=0.15, right=0.97, top=viz.top(fig, 0.78), bottom=0.09)
    viz.heading(fig, "Where the general lexicon fails: domain opinion words", "VADER accuracy by the sentiment word in the text; 8 of 19 words are invisible to it")
    return viz.save(fig, "fig14_vader_by_descriptor")


def fig15_learned_lexicon() -> str:
    lx = tbl("sentiment_learned_lexicon")
    lx = lx[lx.coefficient > 3]
    fig, axes, _ = viz.new_grid(1, 3, w=7.0, h=3.3, theme="light", gridspec_kw={"wspace": 0.7})
    for ax, cls in zip(axes, C.SENTIMENT_ORDER):
        g = lx[lx["class"] == cls].sort_values("coefficient", ascending=False)
        viz.hbars(ax, list(g.term), g.coefficient.values, SENT[cls], fmt="{:.1f}", thick_px=13, xmax=g.coefficient.max() * 1.32)
        ax.set_title(f"{cls} class", fontsize=9)
        ax.tick_params(labelsize=7.4)
    fig.subplots_adjust(left=0.1, right=0.985, top=viz.top(fig, 0.88), bottom=0.12)
    fig.text(0.1, 0.02, "Terms are lemmas: break = broken, disappoint = disappointed, confuse = confusing, satisfy = satisfied.", fontsize=7.2, color=T["muted"])
    viz.heading(fig, "What the model learned: a domain opinion lexicon", "Largest logistic-regression coefficients per class; boilerplate and topic words carry ~0 weight")
    return viz.save(fig, "fig15_learned_lexicon")


def fig16_topic_selection() -> str:
    sw = tbl("topics_k_sweep")
    views = list(sw.view.unique())
    cols = dict(zip(views, [CAT[0], CAT[1], CAT[2]]))
    nice = {views[0]: "A. Full text", views[1]: "B. Opinion clause", views[2]: "C. Aspect words only"}
    fig, axes, _ = viz.new_grid(1, 2, w=7.0, h=3.5, theme="light", gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.38})
    ax = axes[0]
    viz.style_axes(ax, "light", grid="y")
    for v in views:
        g = sw[sw.view == v]
        ax.plot(g.k, g.coherence_npmi, color=cols[v], lw=2, marker="o", ms=5.2, mec=T["surface"], mew=1.4, zorder=4)
        sel = g[g.selected]
        ax.plot(sel.k, sel.coherence_npmi, marker="o", ms=10, mfc="none", mec=cols[v], mew=1.6, zorder=5)
    ax.set_xlabel("Number of topics k", fontsize=8, color=T["ink2"]); ax.set_ylabel("NPMI coherence", fontsize=8, color=T["ink2"])
    ax.set_xticks(range(4, 13)); ax.set_ylim(-0.05, 1.02)
    viz.legend_keys(ax, [(nice[v], cols[v]) for v in views], loc="upper left", kind="line", bbox=(0.0, 1.0))
    ax.set_title("Coherence by k (ring = selected)", fontsize=9)
    ax = axes[1]
    sel = sw[sw.selected].set_index("view").reindex(views)
    viz.vbars(ax, ["A", "B", "C"], sel.purity_vs_issue.values, [cols[v] for v in views], fmt="{:.2f}", thick_px=30, ymax=1.2)
    ax.set_title("Purity vs known issues", fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.subplots_adjust(left=0.08, right=0.985, top=viz.top(fig, 0.88), bottom=0.15)
    viz.heading(fig, "Separating aspects from opinions lets NMF recover all 8 issues", "k chosen on internal criteria only (coherence, stability, no empty topic); issue labels used purely for validation")
    return viz.save(fig, "fig16_topic_selection")


def fig17_theme_overview() -> str:
    p = tbl("topics_prevalence_relevance").sort_values("topic_share", ascending=False)
    fig, ax, _ = viz.new_fig(6.8, 3.9)
    n = len(p)
    ypos = np.arange(n)[::-1]
    ax.set_xlim(0, 0.2); ax.set_ylim(-0.7, n - 0.3)
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    ax.set_yticks(ypos); ax.set_yticklabels(p.topic_issue, fontsize=8, color=T["ink2"])
    py, px = viz._px_per_unit(ax, "y"), viz._px_per_unit(ax, "x")
    for y, r in zip(ypos, p.itertuples()):
        viz.rounded_bar(ax, 0, y + 0.19, r.topic_share, 10 / py, CAT[0], "h"); viz.rounded_bar(ax, 0, y - 0.19, r.rar_share_of_total, 10 / py, CAT[1], "h")
        ax.text(r.topic_share + 5 / px, y + 0.19, f"{100 * r.topic_share:.1f}%", va="center", fontsize=7.6, color=T["ink"])
        ax.text(r.rar_share_of_total + 5 / px, y - 0.19, f"{100 * r.rar_share_of_total:.1f}%", va="center", fontsize=7.6, color=T["ink"])
    ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2]); ax.set_xticklabels(["0%", "5%", "10%", "15%", "20%"])
    viz.legend_keys(ax, [("Topic share of feedback (Metric Guidance)", CAT[0]), ("Share of Revenue at Risk", CAT[1])], loc="upper center", ncol=2, bbox=(0.45, -0.09))
    fig.subplots_adjust(left=0.2, right=0.97, top=viz.top(fig, 0.78), bottom=0.15)
    viz.heading(fig, "Eight balanced themes; Customer Support and App Experience over-index on risk", "Share of feedback vs share of Revenue at Risk by theme (taxonomy classification = 100% match to Issue_Category)")
    return viz.save(fig, "fig17_theme_overview")


def fig18_theme_by_channel() -> str:
    d = silver()
    ct = pd.crosstab(d.topic_issue, d.channel, normalize="columns") * 100
    order = d.topic_issue.value_counts().index
    ct = ct.reindex(order)
    p = tbl("topics_composition_tests").set_index("segment_dimension").loc["channel", "p_value"]
    fig, ax, _ = viz.new_fig(6.2, 4.3)
    im = ax.imshow(ct.values, cmap=viz.seq_cmap("light"), vmin=0, vmax=20, aspect="auto")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, fontsize=8, color=T["ink2"]); ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index, fontsize=8, color=T["ink2"])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="#ffffff" if v > 9 else T["ink"])
    for k in range(ct.shape[0] + 1):
        ax.axhline(k - 0.5, color=T["surface"], lw=2)
    for k in range(ct.shape[1] + 1):
        ax.axvline(k - 0.5, color=T["surface"], lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, colors=T["muted"], length=0); cb.set_label("Share of channel's feedback (%)", fontsize=8, color=T["ink2"])
    fig.subplots_adjust(left=0.24, right=0.86, top=viz.top(fig, 0.82), bottom=0.07)
    viz.heading(fig, "Themes are spread evenly across channels", f"Theme mix within each channel; chi-square test of independence p = {p:.2f} (same for region, product, source after FDR)")
    return viz.save(fig, "fig18_theme_by_channel")


# ---------------------------------------------------------------- mining ----
def fig19_cluster_selection() -> str:
    s = tbl("mining_cluster_selection"); v = tbl("mining_value_space_null_check")
    fig, axes, _ = viz.new_grid(1, 2, w=7.0, h=3.5, theme="light", gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.3})
    for ax, d, ttl in [(axes[0], s, "Experience features (CSAT, resolution, repeat)"), (axes[1], v, "Value features (revenue, orders)")]:
        viz.style_axes(ax, "light", grid="y")
        ax.fill_between(d.k, d.null_silhouette_mean, d.null_silhouette_p95, color=T["deemph"], alpha=0.5, lw=0)
        ax.plot(d.k, d.null_silhouette_mean, color=T["muted"], lw=1.6)
        ax.plot(d.k, d.silhouette, color=ACC, lw=2, marker="o", ms=5.4, mec=T["surface"], mew=1.4, zorder=4)
        ax.set_xticks(list(d.k)); ax.set_xlabel("Clusters k", fontsize=8, color=T["ink2"]); ax.set_ylim(0.3, 0.5)
        ax.set_title(ttl, fontsize=8.6)
    sel = s[s.k == 3].iloc[0]
    axes[0].annotate("k = 3 selected", xy=(3, sel.silhouette), xytext=(3.9, 0.485), fontsize=8, color=T["ink"], arrowprops=dict(arrowstyle="-", color=T["muted"], lw=0.9))
    axes[0].set_ylabel("Silhouette", fontsize=8, color=T["ink2"])
    viz.legend_keys(axes[0], [("Observed data", ACC), ("Shuffled-column null (mean to 95th pct)", T["muted"])], loc="lower left", kind="line")
    axes[1].text(0.5, 0.06, "observed = null:\nno cluster structure", transform=axes[1].transAxes, ha="center", fontsize=8, color=T["ink2"], linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.985, top=viz.top(fig, 0.88), bottom=0.15)
    viz.heading(fig, "A good-looking silhouette is not evidence of clusters", "Experience space beats a shuffled-column null; value space (uniform noise) scores 0.41 yet equals its null, so it is tiered by rule")
    return viz.save(fig, "fig19_cluster_selection")


def fig20_segments() -> str:
    d = silver()
    prof = tbl("mining_segment_profile").set_index("segment_name")
    order = ["At-Risk", "Satisfied Non-Repeaters", "Loyal Repeaters"]
    fig = viz.apply_style("light") and None
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.95, wspace=0.42, left=0.07, right=0.97, top=viz.top(fig, 0.92), bottom=0.09)
    ax = fig.add_subplot(gs[:, 0])
    viz.style_axes(ax, "light", grid="y")
    rng = np.random.default_rng(C.SEED)
    for name in order:
        g = d[d.segment_name == name]
        ax.scatter(g.pc1 + rng.normal(0, 0.03, len(g)), g.pc2 + rng.normal(0, 0.03, len(g)), s=9, color=SEG[name], alpha=0.55, linewidths=0, zorder=3)
        ax.scatter([g.pc1.mean()], [g.pc2.mean()], s=95, color=SEG[name], edgecolor=T["surface"], linewidth=2, zorder=5)
    ax.set_xlabel("Principal component 1 (58% of variance)", fontsize=8, color=T["ink2"]); ax.set_ylabel("Principal component 2 (26%)", fontsize=8, color=T["ink2"])
    ax.set_title("Three stable segments (PCA view, large dots = centres)", fontsize=8.6)
    viz.legend_keys(ax, [(f"{n}  (n = {int(prof.loc[n, 'feedback_count']):,})", SEG[n]) for n in order], loc="upper right")
    # the At-Risk centre sits in a gap because one of the three features (repeat purchase) is a 0/1 flag: say so on the figure
    rr = 100 * prof.loc["At-Risk", "repeat_rate"]
    ax.text(0.02, 0.025, f"At-Risk has two arms (repeat = 1\nfor {rr:.0f}%, 0 for the rest), so its\ncentre (mean) falls between them",
            transform=ax.transAxes, fontsize=6.6, color=T["ink2"], va="bottom", ha="left", linespacing=1.35, zorder=6,
            bbox=dict(facecolor=T["surface"], edgecolor="none", pad=2.0))      # hides the gridline that would run through the text
    for i, (key, lab, fmt, scale) in enumerate([("avg_csat", "Average CSAT (1-5)", "{:.1f}", 1), ("avg_resolution_hours", "Avg resolution (hours)", "{:.1f}", 1), ("repeat_rate", "Repeat purchase rate (%)", "{:.0f}%", 100)]):
        a = fig.add_subplot(gs[i, 1])
        vals = [prof.loc[n, key] * scale for n in order]
        viz.hbars(a, ["At-Risk", "Sat. non-rep.", "Loyal rep."], vals, [SEG[n] for n in order], fmt=fmt, thick_px=11, xmax=max(vals) * 1.35 + 0.01)
        a.set_title(lab, fontsize=8.2)
        a.tick_params(labelsize=7.2)
        if key == "repeat_rate":
            a.set_xticks([0, 25, 50, 75, 100])
    viz.heading(fig, "Segments: dissatisfied core, satisfied non-repeaters, loyal repeaters", "K-means on standardised CSAT, resolution time and repeat flag (k = 3, silhouette 0.46, bootstrap ARI 0.99)")
    return viz.save(fig, "fig20_segments")


def fig21_association_rules() -> str:
    r = tbl("mining_association_rules_robust").sort_values("lift")
    fig, ax, _ = viz.new_fig(6.9, 3.6)
    n = len(r)
    ypos = np.arange(n)
    ax.set_xlim(0, 6.1); ax.set_ylim(-0.7, n - 0.3)
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    lab = [f"{a}  ->  {c.split('=')[1] if c.startswith('Sentiment') else c}" for a, c in zip(r.antecedent.str.replace("Resolution=", "Resolution ").str.replace("Sentiment=", ""), r.consequent)]
    lab = [f"{a}  ->  {'Negative' if 'Sentiment=Negative' in c else 'Repeat = No'}" for a, c in zip(r.antecedent.str.replace("Resolution=", "Resolution ").str.replace("Sentiment=", ""), r.consequent)]
    ax.set_yticks(ypos); ax.set_yticklabels(lab, fontsize=7.6, color=T["ink2"])
    py, px = viz._px_per_unit(ax, "y"), viz._px_per_unit(ax, "x")
    for y, x in zip(ypos, r.itertuples()):
        viz.rounded_bar(ax, 0, y, x.lift, 11 / py, ACC, "h")
        ax.text(x.lift + 5 / px, y, f"lift {x.lift:.2f}   conf. {100 * x.confidence:.0f}%   supp. {100 * x.support:.1f}%", va="center", fontsize=7.2, color=T["ink"])
    ax.axvline(1.0, color=T["muted"], lw=1.0, zorder=2)
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel("Lift (1.0 = no association);  conf. = confidence, supp. = support", fontsize=8, color=T["ink2"])
    fig.subplots_adjust(left=0.37, right=0.99, top=viz.top(fig, 0.8), bottom=0.14)
    viz.heading(fig, "Only 7 of 158 candidate rules survive significance and productivity tests", "None involve issue, channel, region, product or source: the robust patterns are resolution time, repeat purchase and sentiment")
    return viz.save(fig, "fig21_association_rules")


# ----------------------------------------------------------- integration ----
def fig22_repeat_driver_forest() -> str:
    t = tbl("integrated_repeat_driver_odds_ratios")
    order = ["Negative vs Positive sentiment", "Neutral vs Positive sentiment", "Resolution time (per +1 SD)", "Resolution time alone (per +1 SD, unadjusted)", "Revenue (per +INR 1,000)", "Order count (per +1)"]
    t = t.set_index("label").loc[order].reset_index()
    fig, ax, _ = viz.new_fig(6.9, 3.6)
    n = len(t)
    ypos = np.arange(n)[::-1]
    ax.set_xscale("log"); ax.set_xlim(0.1, 1.9); ax.set_ylim(-0.7, n - 0.2)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    ax.axvline(1.0, color=T["muted"], lw=1.2, zorder=2)
    ax.set_yticks(ypos); ax.set_yticklabels(t.label, fontsize=7.8, color=T["ink2"])
    for y, r in zip(ypos, t.itertuples()):
        sig = r.p_value < 0.05
        col = ACC if sig else T["muted"]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2, zorder=3, solid_capstyle="round")
        ax.plot([r.odds_ratio], [y], marker="o", ms=8, color=col, mec=T["surface"], mew=1.3, zorder=5)
        ax.text(1.98, y, f"OR {r.odds_ratio:.2f}  ({r.ci_lo:.2f}-{r.ci_hi:.2f})   {'p<0.001' if r.p_value < 0.001 else f'p={r.p_value:.2f}'}", va="center", fontsize=7.4, color=T["ink"], clip_on=False)
    ax.set_xticks([0.1, 0.2, 0.5, 1.0]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0"])
    ax.set_xlabel("Odds ratio for repeat purchase (log scale; 1 = no effect)", fontsize=8, color=T["ink2"])
    ax.text(0.1, -0.62, "lower repeat  <", fontsize=7.4, color=T["muted"], ha="left"); ax.text(1.9, -0.62, ">  higher repeat", fontsize=7.4, color=T["muted"], ha="right")
    viz.legend_keys(ax, [("p < 0.05", ACC), ("not significant", T["muted"])], loc="center left", bbox=(0.0, 0.2))
    fig.subplots_adjust(left=0.34, right=0.69, top=viz.top(fig, 0.8), bottom=0.15)
    viz.heading(fig, "Sentiment, not speed or value, drives repeat purchase", "Adjusted odds ratios (logistic regression with channel, region and issue controls); resolution loses its effect once sentiment is known")
    return viz.save(fig, "fig22_repeat_driver_forest")


def fig23_revenue_at_risk() -> str:
    d = silver(); m = load_metrics("integrate")
    rar = m["revenue_at_risk"]; sc = pd.DataFrame(m["scenarios"])
    rev = d.groupby("sent_pred").customer_revenue_inr.sum().reindex(C.SENTIMENT_ORDER)
    W, H = 7.0, 3.9
    fig, ax = D.canvas(W, H)
    x0, x1, yb, hb = 0.25, 6.75, 2.35, 0.29
    span, x = x1 - x0, x0
    ax.text(x0, H - 0.75, "Total revenue by sentiment (INR 15.12M)", fontsize=8.8, fontweight=500, color=T["ink"], va="top")
    for name in C.SENTIMENT_ORDER:
        w = span * rev[name] / rev.sum()
        ax.add_patch(FancyBboxPatch((x, yb), max(w - 0.025, 0), hb, boxstyle="round,pad=0,rounding_size=0.04", fc=SENT[name], ec="none", zorder=3))
        ax.text(x + 0.1, yb + hb / 2, f"{fmt_inr(rev[name])}   {100 * rev[name] / rev.sum():.1f}%", ha="left", va="center", fontsize=8.4, color="#0b0b0b" if name == "Neutral" else "#ffffff", fontweight=500, zorder=5)
        if name == "Negative":
            ax.text(x + 0.0, yb - 0.1, f"Revenue at Risk {fmt_inr(rar['revenue_at_risk_inr'])}  (95% CI {fmt_inr(rar['rar_ci'][0])} to {fmt_inr(rar['rar_ci'][1])})", fontsize=8, color=T["ink"], va="top")
            ax.text(x + 0.0, yb - 0.32, f"of which lapsed (no repeat) {fmt_inr(m['revenue_at_risk']['lapsed_unhappy']['revenue_inr'])} / {m['revenue_at_risk']['lapsed_unhappy']['records']} records;   still active {fmt_inr(m['revenue_at_risk']['active_unhappy']['revenue_inr'])} / {m['revenue_at_risk']['active_unhappy']['records']} records", fontsize=7.4, color=T["ink2"], va="top")
        x += w
    xs = x0
    for name in C.SENTIMENT_ORDER:
        ax.add_patch(FancyBboxPatch((xs, yb + hb + 0.1), 0.12, 0.12, boxstyle="square,pad=0", fc=SENT[name], ec="none"))
        ax.text(xs + 0.19, yb + hb + 0.16, name, fontsize=7.8, color=T["ink2"], va="center")
        xs += 1.15
    # scenario bars (accent) drawn in a sub-axes
    sub = fig.add_axes([0.20, 0.07, 0.74, 0.30])
    viz.style_axes(sub, "light", grid="x", spines_left=False)
    labels = [f"{k.capitalize()}  ({int(100 * v)}% of gap)" for k, v in zip(sc.scenario, sc.share_of_gap_recovered)]
    viz.hbars(sub, labels[::-1], (sc.recoverable_revenue_inr.values / 1e3)[::-1], ACC, fmt="{:,.0f}K", thick_px=13, err=np.array([sc.ci_lo.values[::-1] / 1e3, sc.ci_hi.values[::-1] / 1e3]), xmax=1100)
    sub.set_title(f"Recoverable revenue if x% of the {rar['repeat_gap_pp']:.0f}pp repeat gap closes (INR thousand, bootstrap 95% CI)", fontsize=8, loc="left", pad=6)
    sub.tick_params(labelsize=7.2)
    viz.heading(fig, "Revenue at Risk: INR 3.70M, a quarter of all revenue", "Scenarios, not forecasts: the repeat gap is observed in cross-sectional data, so how much is causal is unknown")
    return viz.save(fig, "fig23_revenue_at_risk")


def fig24_rank_stability() -> str:
    r = tbl("integrated_rank_stability_issue").sort_values("revenue_at_risk_inr", ascending=False)
    fig, axes, _ = viz.new_grid(1, 2, w=7.0, h=3.9, theme="light", gridspec_kw={"wspace": 0.14})
    ax = axes[0]
    viz.hbars(ax, list(r.level), r.revenue_at_risk_inr.values / 1e3, SENT["Negative"], fmt="{:,.0f}K", thick_px=14, xmax=700, ref=r.revenue_at_risk_inr.mean() / 1e3, ref_label="mean of 8 issues")
    ax.set_title("Revenue at Risk (INR thousand)", fontsize=8.8)
    ax = axes[1]
    viz.hbars(ax, list(r.level), 100 * r.p_rank1_rar.values, ACC, fmt="{:.0f}%", thick_px=14, xmax=62, ref=12.5, ref_label="chance 12.5%")
    ax.set_yticklabels([]); ax.set_title("Chance of ranking #1 (bootstrap)", fontsize=8.8)
    fig.subplots_adjust(left=0.2, right=0.985, top=viz.top(fig, 0.88), bottom=0.07)
    viz.heading(fig, "Customer Support leads, but the ranking is not firm", "2,000 bootstrap resamples: the top issue changes in more than half of them (leading candidates, not proven hotspots)")
    return viz.save(fig, "fig24_rank_stability")


def fig25_risk_matrix() -> str:
    rm = tbl("mining_risk_matrix_revenue_at_risk").set_index("segment_name")
    rm = rm.reindex(["At-Risk", "Satisfied Non-Repeaters", "Loyal Repeaters"])[["Q1 Low", "Q2 Mid-Low", "Q3 Mid-High", "Q4 Premium"]] / 1e3
    fig, ax, _ = viz.new_fig(6.4, 3.3)
    im = ax.imshow(rm.values, cmap=viz.seq_cmap("light"), vmin=0, vmax=rm.values.max(), aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(rm.columns, fontsize=8, color=T["ink2"]); ax.set_yticks(range(3)); ax.set_yticklabels(rm.index, fontsize=8, color=T["ink2"])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(4):
            v = rm.values[i, j]
            ax.text(j, i, f"{v:,.0f}K", ha="center", va="center", fontsize=9, color="#ffffff" if v > rm.values.max() * 0.45 else T["ink"], fontweight=500 if i == 0 and j >= 2 else "normal")
    for k in range(4):
        ax.axhline(k - 0.5, color=T["surface"], lw=2.5)
    for k in range(5):
        ax.axvline(k - 0.5, color=T["surface"], lw=2.5)
    ax.add_patch(Rectangle((1.5, -0.5), 2, 1, fill=False, ec=T["ink"], lw=1.6, zorder=6))
    ax.text(2.5, -0.62, "Critical tier", ha="center", va="bottom", fontsize=8, color=T["ink"], fontweight=500)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, colors=T["muted"], length=0); cb.set_label("Revenue at Risk (INR thousand)", fontsize=8, color=T["ink2"])
    fig.subplots_adjust(left=0.27, right=0.87, top=viz.top(fig, 0.95), bottom=0.12)
    viz.heading(fig, "Where the Revenue at Risk sits: segment x revenue tier", "The At-Risk segment in the top half of revenue (outlined) holds 72% of all Revenue at Risk")
    return viz.save(fig, "fig25_risk_matrix")


# -------------------------------------------------------------- decision ----
def fig26_priority_ranking() -> str:
    # plot in the table's own total order (score, then impact, then confidence): re-sorting on score alone scrambles ties
    r = tbl("decision_recommendations").sort_values("priority_rank", ascending=False)   # y runs bottom-up, so rank 1 is last
    fig, ax, _ = viz.new_fig(7.0, 4.2)
    n = len(r)
    ypos = np.arange(n)
    ax.set_xlim(0, 13.5); ax.set_ylim(-0.7, n - 0.3)
    viz.style_axes(ax, "light", grid="x", spines_left=False)
    ax.set_yticks(ypos); ax.set_yticklabels([f"{i}  {t}" for i, t in zip(r.id, r.short)], fontsize=7.8, color=T["ink2"])
    py, px = viz._px_per_unit(ax, "y"), viz._px_per_unit(ax, "x")
    for y, x in zip(ypos, r.itertuples()):
        viz.rounded_bar(ax, 0, y, x.priority_score, 12 / py, ACC, "h")
        ax.text(x.priority_score + 5 / px, y, f"{x.priority_score:.1f}   I{x.impact_score} x C{x.confidence} / E{x.effort}   {x.phase}", va="center", fontsize=7.4, color=T["ink"])
    ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_xlabel("Priority score = Impact (I) x Confidence (C) / Effort (E)", fontsize=8, color=T["ink2"])
    fig.subplots_adjust(left=0.36, right=0.99, top=viz.top(fig, 0.8), bottom=0.13)
    viz.heading(fig, "Prioritised actions", "Rubric: impact = base-scenario recoverable revenue; confidence = chance the action delivers; effort = judgement (assumption)")
    return viz.save(fig, "fig26_priority_ranking")


def _status_icon(ax, x, y, status, r=0.075):
    from matplotlib.patches import Circle, Polygon
    col = viz.STATUS[status]
    if status == "good":
        ax.add_patch(Polygon([(x - r, y - r * 0.8), (x + r, y - r * 0.8), (x, y + r)], closed=True, fc=col, ec="none", zorder=5))
    elif status == "warning":
        ax.add_patch(Polygon([(x - r, y), (x, y + r), (x + r, y), (x, y - r)], closed=True, fc=col, ec="none", zorder=5))
    elif status == "serious":
        ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=5))
    else:
        ax.add_patch(Polygon([(x - r, y + r * 0.8), (x + r, y + r * 0.8), (x, y - r)], closed=True, fc=col, ec="none", zorder=5))


def fig27_kpi_scorecard() -> str:
    sc = tbl("decision_kpi_scorecard")
    W, H = 7.0, 4.0
    fig, ax = D.canvas(W, H)
    tw, th, gx, gy = 1.56, 1.38, 0.14, 0.16
    x0, ytop = 0.25, H - 0.82
    short = {"Revenue at Risk (share of revenue)": "Revenue at Risk (% of revenue)", "Net Sentiment Score (pts)": "Net Sentiment Score (pts)", "Average CSAT (1-5)": "Average CSAT (1-5)",
             "Resolution time, negative cases (h)": "Resolution, negative cases", "Repeat rate, negative-sentiment customers": "Repeat rate, negative feedback",
             "Average Resolution Time (h)": "Avg Resolution Time"}
    fmt = {"pct": lambda v: f"{100 * v:.1f}%", "hours": lambda v: f"{v:.1f} h", "pts": lambda v: f"{v:+.1f}", "num": lambda v: f"{v:.2f}"}
    for i, r in enumerate(sc.itertuples()):
        c, rw = i % 4, i // 4
        x, yt = x0 + c * (tw + gx), ytop - rw * (th + gy)
        ax.add_patch(FancyBboxPatch((x, yt - th), tw, th, boxstyle="round,pad=0,rounding_size=0.06", fc=T["surface"], ec=T["axis"], lw=0.9, zorder=2))
        ax.text(x + 0.1, yt - 0.1, short.get(r.kpi, r.kpi), fontsize=7.4, color=T["ink2"], va="top", zorder=4)
        ax.text(x + 0.1, yt - 0.52, fmt[r.unit](r.current), fontsize=16, color=T["ink"], va="center", fontweight=500, zorder=4)
        tgt = ("<= " if not r.higher_is_better else ">= ") + fmt[r.unit](r.proposed_target).replace("+", "")
        ax.text(x + 0.1, yt - 0.84, f"Proposed target {tgt}", fontsize=6.9, color=T["muted"], va="center", zorder=4)
        _status_icon(ax, x + 0.17, yt - 1.14, r.status)
        ax.text(x + 0.31, yt - 1.14, r.status_label, fontsize=7.8, color=T["ink"], va="center", zorder=4)
    ax.text(x0, 0.2, "Targets are PROPOSED assumptions (none were supplied) and must be agreed with stakeholders.   Status = icon + label: well off / off / near / on target.", fontsize=6.9, color=T["muted"], va="center")
    viz.heading(fig, "Management KPI scorecard", "Metric Guidance KPIs plus Net Sentiment, CSAT and negative-case measures; sentiment = NLP prediction")
    return viz.save(fig, "fig27_kpi_scorecard")


def make_part2() -> list[str]:
    fns = [fig12_nlp_model_comparison, fig13_confusion_matrices, fig14_vader_by_descriptor, fig15_learned_lexicon, fig16_topic_selection, fig17_theme_overview,
           fig18_theme_by_channel, fig19_cluster_selection, fig20_segments, fig21_association_rules, fig22_repeat_driver_forest, fig23_revenue_at_risk,
           fig24_rank_stability, fig25_risk_matrix, fig26_priority_ranking, fig27_kpi_scorecard]
    return [f() for f in fns]


if __name__ == "__main__":  # pragma: no cover
    for p in make_part2():
        print(p)
