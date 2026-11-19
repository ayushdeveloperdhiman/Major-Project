"""Stage 10 - Integrated customer intelligence: link NLP / mining outputs to commercial KPIs.

Contents
  * sentiment x commercial KPIs (repeat purchase, resolution, revenue, CSAT)
  * driver model for repeat purchase (logistic regression, adjusted odds ratios; incremental-AUC check)
  * Revenue at Risk (Metric Guidance definition) + decomposition + bootstrap CIs
  * repeat-gap exposure and recovery SCENARIOS (cross-sectional data => scenarios, not forecasts)
  * bootstrap RANK STABILITY of issues / channels / regions (is the "top pain point" statistically firm?)
  * segment x value risk matrix (high-risk and high-opportunity cells)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C
from . import kpis as K
from .utils import get_logger, save_metrics

warnings.filterwarnings("ignore")
log = get_logger("cvi.integrate")

DIM_COLS = {"Issue": "issue_category", "Theme group": "issue_theme_group", "Channel": "channel", "Region": "region",
            "Product": "product_category", "Source": "feedback_source"}


def load_analytics_frame() -> pd.DataFrame:
    s = pd.read_parquet(C.SILVER_FILE)
    parts = [pd.read_parquet(C.SILVER_DIR / f) for f in ["nlp_sentiment.parquet", "nlp_topics.parquet", "mining_segments.parquet"]]
    for p in parts:
        s = s.merge(p, on="feedback_id", how="left", suffixes=("", "_dup"))
    s = s[[c for c in s.columns if not c.endswith("_dup")]]
    for c in ["resolution_band", "revenue_tier", "order_band", "csat_band", "sentiment_ref"]:
        s[c] = s[c].astype("string")
    return s


# --------------------------------------------------------------- drivers ----
def driver_model(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["repeat"] = d.repeat_purchase_flag.astype(int)
    d["res_z"] = (d.resolution_time_hours - d.resolution_time_hours.mean()) / d.resolution_time_hours.std()
    d["rev_k"] = d.customer_revenue_inr / 1000.0
    d["sent"] = pd.Categorical(d.sent_pred, categories=["Positive", "Neutral", "Negative"])    # Positive = reference level
    for c in ["channel", "region", "issue_category"]:
        d[c] = d[c].astype(str).astype("category")
    # NB: categorical-dtype columns are treated as categorical by patsy, so no C(...) is needed
    #     (the name C is this project's config module and would shadow patsy's C()).
    f_sent = "repeat ~ sent"
    f_res = "repeat ~ sent + res_z"
    f_full = "repeat ~ sent + res_z + rev_k + order_count + channel + region + issue_category"
    f_ctx0 = "repeat ~ sent + res_z + rev_k + order_count"
    m_res_only = smf.logit("repeat ~ res_z", d).fit(disp=0)
    m_sent = smf.logit(f_sent, d).fit(disp=0)
    m_res = smf.logit(f_res, d).fit(disp=0)
    m_full = smf.logit(f_full, d).fit(disp=0)
    m_ctx0 = smf.logit(f_ctx0, d).fit(disp=0)

    def lr_test(big, small):
        lr = 2 * (big.llf - small.llf)
        dof = big.df_model - small.df_model
        return {"chi2": float(lr), "dof": int(dof), "p": float(1 - stats.chi2.cdf(lr, dof))}

    ci = m_full.conf_int()
    tab = pd.DataFrame({"term": m_full.params.index, "odds_ratio": np.exp(m_full.params.values), "ci_lo": np.exp(ci[0].values),
                        "ci_hi": np.exp(ci[1].values), "p_value": m_full.pvalues.values})
    label = {"sent[T.Negative]": "Negative vs Positive sentiment", "sent[T.Neutral]": "Neutral vs Positive sentiment",
             "res_z": "Resolution time (per +1 SD)", "rev_k": "Revenue (per +INR 1,000)", "order_count": "Order count (per +1)"}
    tab["label"] = tab.term.map(label)
    key = tab[tab.label.notna()].copy()
    uni = pd.DataFrame({"term": ["res_z (alone)"], "odds_ratio": [float(np.exp(m_res_only.params["res_z"]))],
                        "ci_lo": [float(np.exp(m_res_only.conf_int().loc["res_z", 0]))], "ci_hi": [float(np.exp(m_res_only.conf_int().loc["res_z", 1]))],
                        "p_value": [float(m_res_only.pvalues["res_z"])], "label": ["Resolution time alone (per +1 SD, unadjusted)"]})
    key = pd.concat([uni, key], ignore_index=True)

    # honest out-of-sample discrimination (5-fold CV AUC) for nested feature sets
    y = d.repeat.values
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)

    def cv_auc(cols_num, cols_cat):
        X = pd.get_dummies(d[cols_cat], drop_first=True).astype(float) if cols_cat else pd.DataFrame(index=d.index)
        X = pd.concat([d[cols_num].astype(float), X], axis=1)
        p = cross_val_predict(LogisticRegression(max_iter=2000, C=1e6), X.values, y, cv=skf, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    auc = {"sentiment_only": cv_auc([], ["sent_pred"]), "sentiment_plus_resolution": cv_auc(["res_z"], ["sent_pred"]),
           "sentiment_plus_all_context": cv_auc(["res_z", "rev_k", "order_count"], ["sent_pred", "channel", "region", "issue_category"]),
           "resolution_only": cv_auc(["res_z"], [])}
    out = {"key_terms": key, "full_table": tab,
           "lr_tests": {"resolution_given_sentiment": lr_test(m_res, m_sent),
                        "revenue_orders_channel_region_issue_given_sentiment_resolution": lr_test(m_full, m_res),
                        "channel_region_issue_given_rest": lr_test(m_full, m_ctx0)},
           "cv_auc": auc, "mcfadden_r2": {"resolution_only": float(m_res_only.prsquared), "sentiment_only": float(m_sent.prsquared),
                                          "sentiment_plus_resolution": float(m_res.prsquared), "full": float(m_full.prsquared)}}
    # within-sentiment repeat rate by fast / slow resolution
    w = d.assign(slow=np.where(d.sla_breach == 1, "Slow (>12h)", "Within SLA (<=12h)")).groupby(["sent_pred", "slow"]).repeat.agg(["size", "mean"]).reset_index()
    w.columns = ["sentiment", "resolution", "n", "repeat_rate"]
    out["within_sentiment_by_sla"] = w
    return out


# ------------------------------------------------------- revenue at risk ----
def revenue_at_risk(df: pd.DataFrame, n_boot: int = C.N_BOOT) -> dict:
    d = df.copy()
    neg = (d.sent_pred == "Negative").values
    pos = (d.sent_pred == "Positive").values
    rev = d.customer_revenue_inr.astype(float).values
    rep = d.repeat_purchase_flag.astype(float).values
    total_rev, rar = rev.sum(), rev[neg].sum()
    gap = rep[pos].mean() - rep[neg].mean()
    exposure = rar * gap
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    neg_b, pos_b, rev_b, rep_b = neg[idx], pos[idx], rev[idx], rep[idx]
    rar_b = (rev_b * neg_b).sum(1)
    gap_b = (rep_b * pos_b).sum(1) / pos_b.sum(1) - (rep_b * neg_b).sum(1) / neg_b.sum(1)
    exp_b = rar_b * gap_b
    q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]  # noqa: E731
    active = neg & (rep == 1)
    lapsed = neg & (rep == 0)
    scen = pd.DataFrame([{"scenario": k, "share_of_gap_recovered": v, "recoverable_revenue_inr": exposure * v,
                          "ci_lo": float(np.quantile(exp_b, 0.025) * v), "ci_hi": float(np.quantile(exp_b, 0.975) * v),
                          "pct_of_total_revenue": exposure * v / total_rev} for k, v in C.RECOVERY_SCENARIOS.items()])
    decomposition = pd.DataFrame([
        {"component": "Not at risk (positive / neutral)", "records": int((~neg).sum()), "revenue_inr": float(rev[~neg].sum())},
        {"component": "Negative, still repeat-purchasing ('active but unhappy')", "records": int(active.sum()), "revenue_inr": float(rev[active].sum())},
        {"component": "Negative, no repeat purchase ('lapsed unhappy')", "records": int(lapsed.sum()), "revenue_inr": float(rev[lapsed].sum())}])
    decomposition["share_of_total_revenue"] = decomposition.revenue_inr / total_rev
    return {"total_revenue_inr": float(total_rev), "revenue_at_risk_inr": float(rar), "rar_ci": q(rar_b), "rar_share": float(rar / total_rev),
            "negative_records": int(neg.sum()), "repeat_gap_pp": float(100 * gap), "repeat_gap_ci_pp": [100 * x for x in q(gap_b)],
            "exposure_inr": float(exposure), "exposure_ci": q(exp_b), "exposure_pct_of_revenue": float(exposure / total_rev),
            "scenarios": scen, "decomposition": decomposition,
            "active_unhappy": {"records": int(active.sum()), "revenue_inr": float(rev[active].sum())},
            "lapsed_unhappy": {"records": int(lapsed.sum()), "revenue_inr": float(rev[lapsed].sum())}}


# ------------------------------------------------------- rank stability ----
def rank_stability(df: pd.DataFrame, col: str, n_boot: int = C.N_BOOT) -> pd.DataFrame:
    """Bootstrap the records; how often does each level rank #1 / top-3 on Revenue-at-Risk and on negative rate?"""
    d = df.copy()
    codes, levels = pd.factorize(d[col], sort=True)
    G = len(levels)
    neg = (d.sent_pred == "Negative").values.astype(float)
    rar_w = d.customer_revenue_inr.astype(float).values * neg
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    rar_b = np.stack([np.bincount(codes[i], weights=rar_w[i], minlength=G) for i in idx])
    neg_b = np.stack([np.bincount(codes[i], weights=neg[i], minlength=G) for i in idx])
    n_b = np.stack([np.bincount(codes[i], minlength=G) for i in idx]).astype(float)
    rate_b = neg_b / np.maximum(n_b, 1)

    def ranks(M):  # 1 = highest
        return (-M).argsort(1).argsort(1) + 1
    r_rar, r_rate = ranks(rar_b), ranks(rate_b)
    obs = K.kpi_table(d, col, "sent_pred").set_index(col).reindex(levels)
    out = pd.DataFrame({"level": levels, "n": obs.feedback_count.values, "negative_rate": obs.negative_rate.values, "revenue_at_risk_inr": obs.revenue_at_risk_inr.values})
    out["observed_rank_rar"] = out.revenue_at_risk_inr.rank(ascending=False).astype(int)
    out["p_rank1_rar"] = (r_rar == 1).mean(0)
    out["p_top3_rar"] = (r_rar <= 3).mean(0)
    out["rank_rar_lo"] = np.quantile(r_rar, 0.025, axis=0)
    out["rank_rar_hi"] = np.quantile(r_rar, 0.975, axis=0)
    out["observed_rank_rate"] = out.negative_rate.rank(ascending=False).astype(int)
    out["p_rank1_rate"] = (r_rate == 1).mean(0)
    out["p_top3_rate"] = (r_rate <= 3).mean(0)
    out["uniform_chance_rank1"] = 1.0 / G
    return out.sort_values("revenue_at_risk_inr", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------- run -------
def run(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    df = load_analytics_frame() if df is None else df
    C.ensure_dirs()
    kp = K.overall_kpis(df, "sent_pred")
    by_sent = K.kpi_table(df, "sent_pred").set_index("sent_pred").reindex(C.SENTIMENT_ORDER).reset_index()
    drv = driver_model(df)
    rar = revenue_at_risk(df)
    ranks = {name: rank_stability(df, col) for name, col in DIM_COLS.items()}
    by_dim = {}
    for name, col in DIM_COLS.items():
        t = K.kpi_table(df, col, "sent_pred")
        t.insert(0, "dimension", name)
        by_dim[name] = t.rename(columns={col: "level"})
    kdims = pd.concat(by_dim.values(), ignore_index=True)

    # segment x value tier & risk tiers
    risk_tier = K.kpi_table(df, "risk_tier", "sent_pred").set_index("risk_tier").reindex(["Critical", "High", "Opportunity", "Maintain"]).reset_index()
    seg_tier = (df.assign(_rar=df.customer_revenue_inr.astype(float) * (df.sent_pred == "Negative"))
                .groupby(["segment_name", "revenue_tier"]).agg(records=("feedback_id", "size"), revenue_inr=("customer_revenue_inr", "sum"), revenue_at_risk_inr=("_rar", "sum"),
                                                              repeat_rate=("repeat_purchase_flag", "mean")).reset_index())
    seg_tier["pct_of_total_rar"] = seg_tier.revenue_at_risk_inr / rar["revenue_at_risk_inr"]
    opp = df[df.segment_name == "Satisfied Non-Repeaters"]
    opp_rev = float(opp.customer_revenue_inr.sum())
    opportunity = {"records": int(len(opp)), "share": float(len(opp) / len(df)), "revenue_inr": opp_rev, "avg_csat": float(opp.csat_score.mean()),
                   "avg_resolution_hours": float(opp.resolution_time_hours.mean()),
                   "scenario_conversion_pct": {"5%": 0.05 * opp_rev, "10%": 0.10 * opp_rev, "20%": 0.20 * opp_rev}}
    crit = df[df.risk_tier == "Critical"]
    critical = {"records": int(len(crit)), "revenue_inr": float(crit.customer_revenue_inr.sum()),
                "revenue_at_risk_inr": float((crit.customer_revenue_inr.astype(float) * (crit.sent_pred == "Negative")).sum()),
                "share_of_total_rar": float((crit.customer_revenue_inr.astype(float) * (crit.sent_pred == "Negative")).sum() / rar["revenue_at_risk_inr"]),
                "avg_resolution_hours": float(crit.resolution_time_hours.mean()), "repeat_rate": float(crit.repeat_purchase_flag.mean())}
    # topic x sentiment (text intelligence linked to KPIs)
    theme_kpi = K.kpi_table(df, "topic_issue", "sent_pred").sort_values("revenue_at_risk_inr", ascending=False)
    # CSAT/rating consistency of predicted sentiment
    csat_by_pred = df.groupby("sent_pred").csat_score.agg(["mean", "std"]).reindex(C.SENTIMENT_ORDER)

    top_issue = ranks["Issue"].iloc[0]
    metrics = {
        "kpis_predicted": kp, "driver": {"key_terms": drv["key_terms"].to_dict("records"), "lr_tests": drv["lr_tests"], "cv_auc": drv["cv_auc"], "mcfadden_r2": drv["mcfadden_r2"]},
        "revenue_at_risk": {k: v for k, v in rar.items() if k not in ("scenarios", "decomposition")},
        "scenarios": rar["scenarios"].to_dict("records"), "decomposition": rar["decomposition"].to_dict("records"),
        "rank_stability": {"top_issue_by_rar": {"level": top_issue.level, "p_rank1": float(top_issue.p_rank1_rar), "p_top3": float(top_issue.p_top3_rar), "uniform_chance": 1 / len(ranks["Issue"])},
                           "top_issue_by_rate": ranks["Issue"].sort_values("negative_rate", ascending=False).iloc[0][["level", "negative_rate", "p_rank1_rate"]].to_dict(),
                           "by_dimension_max_p_rank1_rar": {k: float(v.p_rank1_rar.max()) for k, v in ranks.items()}},
        "critical_tier": critical, "opportunity_segment": opportunity,
        "risk_tier_counts": risk_tier[["risk_tier", "feedback_count", "revenue_at_risk_inr", "total_revenue_inr"]].to_dict("records"),
        "csat_by_predicted_sentiment": csat_by_pred["mean"].round(3).to_dict(),
        "sentiment_kpis": by_sent[["sent_pred", "feedback_count", "topic_share", "repeat_rate", "avg_resolution_hours", "avg_csat", "avg_revenue_inr"]].to_dict("records"),
    }
    if save:
        by_sent.to_csv(C.TABLE_DIR / "integrated_kpis_by_sentiment.csv", index=False)
        kdims.to_csv(C.TABLE_DIR / "integrated_kpis_by_dimension.csv", index=False)
        drv["key_terms"].to_csv(C.TABLE_DIR / "integrated_repeat_driver_odds_ratios.csv", index=False)
        drv["full_table"].to_csv(C.TABLE_DIR / "integrated_repeat_driver_full_model.csv", index=False)
        drv["within_sentiment_by_sla"].to_csv(C.TABLE_DIR / "integrated_repeat_by_sentiment_and_sla.csv", index=False)
        rar["scenarios"].to_csv(C.TABLE_DIR / "integrated_recovery_scenarios.csv", index=False)
        rar["decomposition"].to_csv(C.TABLE_DIR / "integrated_revenue_decomposition.csv", index=False)
        for k, v in ranks.items():
            v.to_csv(C.TABLE_DIR / f"integrated_rank_stability_{k.lower().replace(' ', '_')}.csv", index=False)
        risk_tier.to_csv(C.TABLE_DIR / "integrated_risk_tiers.csv", index=False)
        seg_tier.to_csv(C.TABLE_DIR / "integrated_segment_by_value_tier.csv", index=False)
        theme_kpi.to_csv(C.TABLE_DIR / "integrated_theme_kpis.csv", index=False)
        df.to_parquet(C.ANALYTICS_FILE, index=False)
        save_metrics("integrate", metrics)
    log.info("Integrate | RaR INR %.2fM (%.1f%% of revenue) | repeat gap %.1fpp | top issue by RaR '%s' ranks #1 in %.0f%% of bootstraps (chance %.1f%%)",
             rar["revenue_at_risk_inr"] / 1e6, 100 * rar["rar_share"], rar["repeat_gap_pp"], top_issue.level, 100 * top_issue.p_rank1_rar, 100 / len(ranks["Issue"]))
    return {"metrics": metrics, "driver": drv, "rar": rar, "ranks": ranks, "df": df}


if __name__ == "__main__":  # pragma: no cover
    run()
