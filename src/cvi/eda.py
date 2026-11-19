"""Stage 5 - Exploratory business analytics (structured data).

Principle: never report a pattern without (a) an effect size, (b) an uncertainty statement and (c) a
multiple-testing correction when many cells are scanned.  Null results are reported as findings.
All outputs are CSV tables in ``outputs/tables`` plus ``outputs/metrics/eda.json``.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests

from . import config as C
from . import kpis as K
from .utils import cohens_d, cramers_v, get_logger, save_metrics, wilson_ci

log = get_logger("cvi.eda")

DIMS = {"Issue": "issue_category", "Channel": "channel", "Region": "region",
        "Product": "product_category", "Source": "feedback_source"}


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rating", "csat_score", "customer_revenue_inr", "order_count", "repeat_purchase_flag", "resolution_time_hours"]
    d = df[cols].astype("float64")
    t = d.describe().T
    t["skew"] = d.skew()
    t["kurtosis"] = d.kurt()
    t["ks_p_vs_uniform"] = [stats.kstest(d[c], "uniform", args=(d[c].min(), d[c].max() - d[c].min())).pvalue for c in cols]
    return t.reset_index().rename(columns={"index": "variable"})


def dimension_tests(df: pd.DataFrame, sent_col: str) -> pd.DataFrame:
    rows = []
    neg = df[df[sent_col] == "Negative"]
    for label, col in DIMS.items():
        ct = pd.crosstab(df[col], df[sent_col])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rate = df.groupby(col)[sent_col].apply(lambda s: (s == "Negative").mean())
        kw = stats.kruskal(*[g.resolution_time_hours.values for _, g in neg.groupby(col)])
        rows.append({"dimension": label, "levels": ct.shape[0], "chi2": chi2, "dof": dof, "p_value": p,
                     "cramers_v": cramers_v(ct), "neg_rate_min": rate.min(), "neg_rate_max": rate.max(),
                     "neg_rate_range_pp": 100 * (rate.max() - rate.min()),
                     "highest_neg_level": rate.idxmax(), "lowest_neg_level": rate.idxmin(),
                     "resolution_within_negative_kruskal_p": kw.pvalue, "significant_at_5pct": bool(p < 0.05)})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.minimum(out.p_value * len(out), 1.0)
    return out


def hotspot_scan(df: pd.DataFrame, sent_col: str, min_cell: int = 20) -> tuple[pd.DataFrame, dict]:
    """Every one-way and two-way cell vs. the rest (Fisher exact), Benjamini-Hochberg FDR across ALL tests."""
    d = df.assign(_neg=(df[sent_col] == "Negative").astype(int))
    tot_neg, tot_n = int(d._neg.sum()), len(d)
    cols = list(DIMS.values())
    rows = []

    def test(label: str, cell: str, sub: pd.DataFrame):
        a = int(sub._neg.sum()); b = len(sub) - a
        c = tot_neg - a; dd = (tot_n - len(sub)) - c
        p = stats.fisher_exact([[a, b], [c, dd]])[1]
        lo, hi = wilson_ci(a, len(sub))
        rows.append({"family": label, "cell": cell, "n": len(sub), "neg_rate": a / len(sub), "ci_lo": lo, "ci_hi": hi, "p_value": p})

    for c in cols:
        for v, g in d.groupby(c):
            test(c, str(v), g)
    for c1, c2 in itertools.combinations(cols, 2):
        for (v1, v2), g in d.groupby([c1, c2]):
            if len(g) >= min_cell:
                test(f"{c1} x {c2}", f"{v1} | {v2}", g)
    R = pd.DataFrame(rows)
    R["q_bh"] = multipletests(R.p_value, method="fdr_bh")[1]
    R["significant_after_fdr"] = R.q_bh < C.FDR_ALPHA
    R = R.sort_values("p_value").reset_index(drop=True)
    R["rank"] = np.arange(1, len(R) + 1)
    R["bh_threshold"] = C.FDR_ALPHA * R["rank"] / len(R)
    summary = {"tests": int(len(R)), "raw_p_lt_0.05": int((R.p_value < 0.05).sum()),
               "expected_by_chance": float(0.05 * len(R)), "min_p": float(R.p_value.min()), "min_q_bh": float(R.q_bh.min()),
               "significant_after_fdr": int(R.significant_after_fdr.sum()),
               "most_extreme_cell": f"{R.iloc[0].family}: {R.iloc[0].cell} ({R.iloc[0].neg_rate:.1%}, n={R.iloc[0].n})"}
    return R, summary


def sentiment_outcomes(df: pd.DataFrame, sent_col: str) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    d["_rep"] = d.repeat_purchase_flag.astype("float64")
    g = d.groupby(sent_col, observed=True).agg(n=("rating", "size"), repeat_rate=("_rep", "mean"), avg_resolution_hours=("resolution_time_hours", "mean"),
                                               sd_res=("resolution_time_hours", "std"), avg_csat=("csat_score", "mean"), avg_rating=("rating", "mean"),
                                               avg_revenue_inr=("customer_revenue_inr", "mean"), sd_rev=("customer_revenue_inr", "std"),
                                               median_revenue_inr=("customer_revenue_inr", "median"), avg_orders=("order_count", "mean"))
    g = g.reindex(C.SENTIMENT_ORDER).reset_index()
    n = g.n.astype(float)
    g["share"] = n / n.sum()
    ci = np.array([wilson_ci(r * m, m) for r, m in zip(g.repeat_rate, n)])
    g["repeat_lo"], g["repeat_hi"] = ci[:, 0], ci[:, 1]
    g["res_lo"] = g.avg_resolution_hours - 1.96 * g.sd_res / np.sqrt(n)
    g["res_hi"] = g.avg_resolution_hours + 1.96 * g.sd_res / np.sqrt(n)
    g["rev_lo"] = g.avg_revenue_inr - 1.96 * g.sd_rev / np.sqrt(n)
    g["rev_hi"] = g.avg_revenue_inr + 1.96 * g.sd_rev / np.sqrt(n)
    grp = lambda col: [x[col].values for _, x in d.groupby(sent_col)]  # noqa: E731
    neg = d[d[sent_col] == "Negative"]
    non = d[d[sent_col] != "Negative"]
    ct_rep = pd.crosstab(d[sent_col], d.repeat_purchase_flag)
    pos = d[d[sent_col] == "Positive"]
    t = {
        "repeat_vs_sentiment_chi2": float(stats.chi2_contingency(ct_rep)[0]),
        "repeat_vs_sentiment_p": float(stats.chi2_contingency(ct_rep)[1]),
        "repeat_vs_sentiment_cramers_v": cramers_v(ct_rep),
        "repeat_gap_pos_minus_neg_pp": float(100 * (pos._rep.mean() - neg._rep.mean())),
        "resolution_kruskal_p": float(stats.kruskal(*grp("resolution_time_hours")).pvalue),
        "resolution_cohens_d_neg_vs_nonneg": cohens_d(neg.resolution_time_hours, non.resolution_time_hours),
        "resolution_mannwhitney_p": float(stats.mannwhitneyu(neg.resolution_time_hours, non.resolution_time_hours).pvalue),
        "revenue_kruskal_p": float(stats.kruskal(*grp("customer_revenue_inr")).pvalue),
        "revenue_cohens_d_neg_vs_nonneg": cohens_d(neg.customer_revenue_inr, non.customer_revenue_inr),
        "orders_kruskal_p": float(stats.kruskal(*grp("order_count")).pvalue),
    }
    return g, t


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["rating", "csat_score", "customer_revenue_inr", "order_count", "repeat_purchase_flag", "resolution_time_hours"]
    return df[cols].astype("float64").corr(method="spearman")


def customer_structure(df: pd.DataFrame, sent_col: str, n_perm: int = 2000) -> dict:
    vc = df.customer_id.value_counts()
    multi = df[df.customer_id.isin(vc[vc > 1].index)].copy()
    g = multi.groupby("customer_id").rating
    k, N = g.ngroups, len(multi)
    n_i = g.size().values.astype(float)
    grand = multi.rating.astype(float).mean()
    ssb = (n_i * (g.mean().values - grand) ** 2).sum()
    ssw = ((multi.rating.astype(float) - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    icc = float((msb - msw) / (msb + (n0 - 1) * msw))
    F = msb / msw
    p_icc = float(1 - stats.f.cdf(F, k - 1, N - k))
    neg = (multi[sent_col] == "Negative").astype(int).values
    labels = multi.customer_id.values
    obs = int((pd.Series(neg).groupby(labels).sum() >= 2).sum())
    rng = np.random.default_rng(C.SEED)
    sims = np.array([(pd.Series(rng.permutation(neg)).groupby(labels).sum() >= 2).sum() for _ in range(n_perm)])
    inst = {}
    for col in ["region", "customer_revenue_inr", "order_count", "repeat_purchase_flag", "channel", "product_category"]:
        inst[col] = float((multi.groupby("customer_id")[col].nunique() > 1).mean())
    return {"customers": int(df.customer_id.nunique()), "multi_record_customers": int(k), "records_in_multi": int(N),
            "records_per_customer_max": int(vc.max()), "icc_rating": icc, "icc_p": p_icc,
            "customers_with_2plus_negative_observed": obs, "customers_with_2plus_negative_expected": float(sims.mean()),
            "customers_with_2plus_negative_perm_p": float((np.sum(sims >= obs) + 1) / (n_perm + 1)),
            "attribute_instability_share": inst}


def sequence_drift(df: pd.DataFrame, sent_col: str) -> dict:
    seq = df.feedback_id.str[3:].astype(int)
    neg = (df[sent_col] == "Negative").astype(int)
    blk = pd.cut(seq, 8, labels=False)
    rho, p = stats.spearmanr(seq, neg)
    return {"spearman_seq_vs_negative": float(rho), "p": float(p),
            "chi2_blocks_p": float(stats.chi2_contingency(pd.crosstab(blk, neg))[1]),
            "block_negative_rates": neg.groupby(blk).mean().round(4).tolist()}


def resolution_threshold(df: pd.DataFrame, sent_col: str) -> tuple[pd.DataFrame, dict]:
    y = (df[sent_col] == "Negative").astype(int).values
    x = df.resolution_time_hours.astype(float).values
    fpr, tpr, thr = roc_curve(y, x)
    j = tpr - fpr
    i = int(j.argmax())
    rows = []
    for th in [6, 8, 10, 12, 14, 15, 18, 24]:
        slow = x > th
        rows.append({"threshold_hours": th, "share_of_cases_slow": slow.mean(), "negative_rate_if_slow": y[slow].mean() if slow.any() else np.nan,
                     "negative_rate_if_not_slow": y[~slow].mean(), "share_of_negatives_captured": slow[y == 1].mean()})
    return pd.DataFrame(rows), {"auc_resolution_predicts_negative": float(roc_auc_score(y, x)),
                                "youden_threshold_hours": float(thr[i]), "youden_tpr": float(tpr[i]), "youden_fpr": float(fpr[i])}


def run(df: pd.DataFrame | None = None, sent_col: str = "reference_sentiment", save: bool = True) -> dict:
    if df is None:
        from .transform import load_silver
        df = load_silver()
    C.ensure_dirs()
    kp = K.overall_kpis(df.assign(sentiment_pred=df[sent_col]), "sentiment_pred")
    num = numeric_summary(df)
    dim = dimension_tests(df, sent_col)
    hot, hot_sum = hotspot_scan(df, sent_col)
    outc, outc_t = sentiment_outcomes(df, sent_col)
    sp = spearman_matrix(df)
    cust = customer_structure(df, sent_col)
    drift = sequence_drift(df, sent_col)
    thr_tbl, thr = resolution_threshold(df, sent_col)

    by_dim = {}
    for label, col in DIMS.items():
        t = K.kpi_table(df.assign(sentiment_pred=df[sent_col]), col, "sentiment_pred")
        t.insert(0, "dimension", label)
        t = t.rename(columns={col: "level"})
        by_dim[label] = t
    kpi_dims = pd.concat(by_dim.values(), ignore_index=True)

    patterns = [
        {"id": "P1", "type": "signal", "pattern": "Negative feedback is strongly tied to lower repeat purchase",
         "evidence": f"repeat rate Negative {outc.loc[0, 'repeat_rate']:.1%} vs Neutral {outc.loc[1, 'repeat_rate']:.1%} vs Positive {outc.loc[2, 'repeat_rate']:.1%} "
                     f"(chi2={outc_t['repeat_vs_sentiment_chi2']:.0f}, p={outc_t['repeat_vs_sentiment_p']:.1e}, Cramer's V={outc_t['repeat_vs_sentiment_cramers_v']:.2f})"},
        {"id": "P2", "type": "signal", "pattern": "Negative cases take about twice as long to resolve",
         "evidence": f"mean resolution {kp['negative_avg_resolution_hours']:.1f}h (negative) vs {kp['non_negative_avg_resolution_hours']:.1f}h (non-negative); "
                     f"Cohen's d={outc_t['resolution_cohens_d_neg_vs_nonneg']:.2f}; resolution time alone separates them with AUC={thr['auc_resolution_predicts_negative']:.2f}"},
        {"id": "P3", "type": "signal", "pattern": "Rating and CSAT are near-redundant measures of the same experience",
         "evidence": f"Spearman rho(rating, CSAT)={sp.loc['rating', 'csat_score']:.2f}; both fall with resolution time (rho={sp.loc['rating', 'resolution_time_hours']:.2f})"},
        {"id": "P4", "type": "null", "pattern": "No issue, channel, region, product or source is significantly worse than the rest",
         "evidence": f"chi-square p-values {', '.join(f'{r.dimension} {r.p_value:.2f}' for r in dim.itertuples())}; {hot_sum['tests']} cells scanned, "
                     f"{hot_sum['raw_p_lt_0.05']} raw p<0.05 (expected {hot_sum['expected_by_chance']:.0f} by chance), {hot_sum['significant_after_fdr']} survive FDR (min q={hot_sum['min_q_bh']:.2f})"},
        {"id": "P5", "type": "null", "pattern": "Revenue and order count are unrelated to sentiment (negative feedback is not confined to low-value records)",
         "evidence": f"mean revenue Negative INR {outc.loc[0, 'avg_revenue_inr']:,.0f} vs Positive INR {outc.loc[2, 'avg_revenue_inr']:,.0f} (Kruskal p={outc_t['revenue_kruskal_p']:.2f}); "
                     f"revenue is statistically uniform (KS p={num.set_index('variable').loc['customer_revenue_inr', 'ks_p_vs_uniform']:.2f})"},
    ]

    metrics = {"kpis_reference": kp, "dimension_tests": dim.to_dict("records"), "hotspot": hot_sum, "sentiment_outcomes": outc_t,
               "resolution_threshold": thr, "customer_structure": cust, "sequence_drift": drift, "material_patterns": patterns,
               "spearman": {a: {b: float(sp.loc[a, b]) for b in sp.columns} for a in sp.index}}
    if save:
        num.to_csv(C.TABLE_DIR / "eda_numeric_summary.csv", index=False)
        dim.to_csv(C.TABLE_DIR / "eda_dimension_tests.csv", index=False)
        hot.to_csv(C.TABLE_DIR / "eda_hotspot_scan.csv", index=False)
        outc.to_csv(C.TABLE_DIR / "eda_sentiment_outcomes.csv", index=False)
        sp.to_csv(C.TABLE_DIR / "eda_spearman.csv")
        thr_tbl.to_csv(C.TABLE_DIR / "eda_resolution_thresholds.csv", index=False)
        kpi_dims.to_csv(C.TABLE_DIR / "eda_kpis_by_dimension.csv", index=False)
        pd.DataFrame(patterns).to_csv(C.TABLE_DIR / "eda_material_patterns.csv", index=False)
        save_metrics("eda", metrics)
    log.info("EDA done | hotspot cells=%d, significant after FDR=%d | AUC(resolution->negative)=%.3f",
             hot_sum["tests"], hot_sum["significant_after_fdr"], thr["auc_resolution_predicts_negative"])
    return metrics


if __name__ == "__main__":  # pragma: no cover
    run()
