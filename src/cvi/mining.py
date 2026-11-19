"""Stage 6 - Data mining: segmentation (K-means) and pattern discovery (association rules).

Why these techniques
  * K-means on standardised experience/retention features - fast, interpretable centroids that managers can act on.
    Validity is NOT assumed from a silhouette score: K-means will happily partition pure noise (revenue is
    uniform), so every k is compared with a COLUMN-PERMUTATION NULL (columns shuffled independently: marginals
    kept, joint structure destroyed).  Structure is real only if the observed silhouette beats the null.
  * Association rules (Apriori) - transparent "if X then outcome" patterns.  Raw rule sets are huge and mostly
    redundant, so rules pass a statistical funnel: lift filter -> Fisher exact + Benjamini-Hochberg FDR ->
    productivity test (an extra antecedent item must *significantly* improve on its parent rule).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from . import config as C
from . import kpis as K
from .utils import cramers_v, get_logger, save_metrics

warnings.filterwarnings("ignore")
log = get_logger("cvi.mining")
FF = "%.12g"      # 12 significant digits: removes last-bit floating-point noise from threaded reductions, keeps tiny p-values


# ============================================================ clustering ====
def _scale(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(df[feats].astype("float64"))


def select_k(X: np.ndarray, ks=C.CLUSTER_K_RANGE, n_null: int = 30, seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=seed).fit(X)
        lab = km.labels_
        null = []
        for _ in range(n_null):
            Xs = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
            nl = KMeans(k, n_init=5, random_state=seed).fit(Xs).labels_
            null.append(silhouette_score(Xs, nl))
        sil = silhouette_score(X, lab)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "calinski_harabasz": calinski_harabasz_score(X, lab),
                     "davies_bouldin": davies_bouldin_score(X, lab), "null_silhouette_mean": float(np.mean(null)),
                     "null_silhouette_p95": float(np.quantile(null, 0.95)), "beats_null": bool(sil > np.quantile(null, 0.95)),
                     "silhouette_lift_over_null": sil - float(np.mean(null)), "smallest_cluster_share": float(np.bincount(lab).min() / len(lab))})
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, k: int, B: int = 50, seed: int = C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    ref = KMeans(k, n_init=20, random_state=seed).fit(X)
    aris = []
    for b in range(B):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(k, n_init=10, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(ref.labels_, km.predict(X)))
    return {"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris)), "p05_ari": float(np.quantile(aris, 0.05)), "bootstrap_runs": B}


def compare_methods(X: np.ndarray, k: int, seed: int = C.SEED) -> pd.DataFrame:
    km = KMeans(k, n_init=50, random_state=seed).fit(X).labels_
    ward = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    gmm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(X).predict(X)
    rows = []
    for name, lab in [("K-means", km), ("Ward hierarchical", ward), ("Gaussian mixture", gmm)]:
        rows.append({"method": name, "silhouette": silhouette_score(X, lab), "calinski_harabasz": calinski_harabasz_score(X, lab),
                     "davies_bouldin": davies_bouldin_score(X, lab), "ari_vs_kmeans": adjusted_rand_score(km, lab), "smallest_cluster_share": np.bincount(lab).min() / len(lab)})
    return pd.DataFrame(rows)


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Deterministic, profile-based naming (cluster ids are arbitrary)."""
    if len(profile) != 3:
        return {i: f"Segment {i + 1}" for i in profile.index}
    at_risk = profile.csat.idxmin()
    rest = [i for i in profile.index if i != at_risk]
    loyal = profile.loc[rest, "repeat"].idxmax()
    other = [i for i in rest if i != loyal][0]
    return {at_risk: "At-Risk", loyal: "Loyal Repeaters", other: "Satisfied Non-Repeaters"}


SEGMENT_DESCRIPTION = {
    "At-Risk": "Low CSAT, slow resolution, mostly no repeat purchase - the dissatisfied core; priority for service recovery.",
    "Satisfied Non-Repeaters": "Satisfied (CSAT ~4) and resolved quickly but did NOT repeat-purchase - high-opportunity conversion segment.",
    "Loyal Repeaters": "Satisfied, resolved quickly and repeat-purchasing - protect and reward.",
}


def segment(df: pd.DataFrame, save: bool = True) -> dict:
    feats = C.CLUSTER_FEATURES
    X = _scale(df, feats)
    sel = select_k(X)
    ok = sel[sel.beats_null]
    kbest = int(ok.sort_values(["silhouette", "calinski_harabasz"], ascending=False).iloc[0].k) if len(ok) else 3
    stab = bootstrap_stability(X, kbest)
    cmp_ = compare_methods(X, kbest)
    km = KMeans(kbest, n_init=50, random_state=C.SEED).fit(X)
    d = df.copy()
    d["cluster"] = km.labels_
    prof = d.groupby("cluster").agg(n=("feedback_id", "size"), csat=("csat_score", "mean"), res=("resolution_time_hours", "mean"),
                                    repeat=("repeat_purchase_flag", "mean"), revenue=("customer_revenue_inr", "mean"), orders=("order_count", "mean"),
                                    neg_share=("sent_pred", lambda s: (s == "Negative").mean()))
    names = name_segments(prof)
    d["segment_name"] = d.cluster.map(names)
    d["risk_tier"] = np.select(
        [(d.segment_name == "At-Risk") & d.revenue_tier.isin(["Q3 Mid-High", "Q4 Premium"]),
         (d.segment_name == "At-Risk"), (d.segment_name == "Satisfied Non-Repeaters")], ["Critical", "High", "Opportunity"], default="Maintain")
    pca = PCA(2, random_state=C.SEED).fit(X)
    pcs = pca.transform(X)
    d["pc1"], d["pc2"] = pcs[:, 0], pcs[:, 1]
    # centroids in original units + z-scores
    cent = pd.DataFrame(StandardScaler().fit(df[feats].astype(float)).inverse_transform(km.cluster_centers_), columns=feats)
    cent["segment_name"] = [names[i] for i in range(kbest)]
    cent_z = pd.DataFrame(km.cluster_centers_, columns=[f"z_{f}" for f in feats])
    cent = pd.concat([cent, cent_z], axis=1)
    # a centroid of a 0/1 flag can come out as 4e-16 instead of 0 (threaded reductions): absolute rounding, not relative,
    # is what removes it (12 significant digits cannot clean a value whose magnitude is itself the noise)
    num_cols = cent.select_dtypes("number").columns
    cent[num_cols] = cent[num_cols].round(9) + 0.0   # "+ 0.0" turns any -0.0 into 0.0
    seg_prof = K.kpi_table(d, "segment_name", "sent_pred")
    extra = d.groupby("segment_name").agg(avg_orders=("order_count", "mean"), avg_csat_=("csat_score", "mean"), sla_breach_=("sla_breach", "mean"))
    seg_prof = seg_prof.merge(extra, on="segment_name").merge(
        d.groupby("segment_name").size().rename("n").reset_index(), on="segment_name")
    seg_prof["description"] = seg_prof.segment_name.map(SEGMENT_DESCRIPTION)
    # external validation + independence from categorical context
    nmi = normalized_mutual_info_score(d.sent_pred, d.segment_name)
    purity_at_risk = float((d[d.segment_name == "At-Risk"].sent_pred == "Negative").mean())
    cap_neg_in_at_risk = float((d[(d.sent_pred == "Negative")].segment_name == "At-Risk").mean())
    ctx = []
    for col in ["issue_category", "channel", "region", "product_category", "feedback_source"]:
        ct = pd.crosstab(d[col], d.segment_name)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        ctx.append({"dimension": col, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v(ct)})
    ctx = pd.DataFrame(ctx)
    ctx["q_bh"] = multipletests(ctx.p_value, method="fdr_bh")[1]
    # risk matrix: segment x revenue tier
    rm = d.assign(_rar=d.customer_revenue_inr.astype(float) * (d.sent_pred == "Negative")).pivot_table(
        index="segment_name", columns="revenue_tier", values="_rar", aggfunc="sum", observed=False).fillna(0.0)
    rm_n = d.pivot_table(index="segment_name", columns="revenue_tier", values="feedback_id", aggfunc="size", observed=False).fillna(0)
    out = {"k": kbest, "selection": sel, "stability": stab, "methods": cmp_, "profile": seg_prof, "centroids": cent, "assign": d,
           "context_tests": ctx, "risk_matrix_rar": rm, "risk_matrix_n": rm_n,
           "explained_variance_pca": [float(x) for x in pca.explained_variance_ratio_],
           "summary": {"features": feats, "k": kbest, "silhouette": float(sel.set_index("k").loc[kbest, "silhouette"]),
                       "calinski_harabasz": float(sel.set_index("k").loc[kbest, "calinski_harabasz"]), "davies_bouldin": float(sel.set_index("k").loc[kbest, "davies_bouldin"]),
                       "null_silhouette_p95": float(sel.set_index("k").loc[kbest, "null_silhouette_p95"]),
                       "k_values_beating_null": [int(x) for x in sel[sel.beats_null].k], "stability": stab,
                       "nmi_segment_vs_sentiment": float(nmi), "at_risk_negative_purity": purity_at_risk, "share_of_negatives_in_at_risk": cap_neg_in_at_risk,
                       "segment_sizes": {n: int(v) for n, v in d.segment_name.value_counts().items()},
                       "context_tests_min_q": float(ctx.q_bh.min()), "pca_explained": [float(x) for x in pca.explained_variance_ratio_],
                       "methods_ari_vs_kmeans": cmp_.set_index("method").ari_vs_kmeans.to_dict()}}
    # additional null check on the VALUE space (revenue x orders): K-means on pure noise
    Xv = _scale(df, ["customer_revenue_inr", "order_count"])
    selv = select_k(Xv, ks=[3, 4, 5], n_null=20)
    out["value_space_check"] = selv
    out["summary"]["value_space_beats_null"] = bool(selv.beats_null.any())
    out["summary"]["value_space_silhouette_k4"] = float(selv.set_index("k").loc[4, "silhouette"])
    out["summary"]["value_space_null_k4"] = float(selv.set_index("k").loc[4, "null_silhouette_mean"])
    if save:
        sel.to_csv(C.TABLE_DIR / "mining_cluster_selection.csv", index=False, float_format=FF)
        cmp_.to_csv(C.TABLE_DIR / "mining_cluster_methods.csv", index=False, float_format=FF)
        seg_prof.to_csv(C.TABLE_DIR / "mining_segment_profile.csv", index=False, float_format=FF)
        cent.to_csv(C.TABLE_DIR / "mining_cluster_centroids.csv", index=False, float_format=FF)
        ctx.to_csv(C.TABLE_DIR / "mining_segment_context_tests.csv", index=False, float_format=FF)
        rm.to_csv(C.TABLE_DIR / "mining_risk_matrix_revenue_at_risk.csv", float_format=FF)
        rm_n.to_csv(C.TABLE_DIR / "mining_risk_matrix_counts.csv", float_format=FF)
        selv.to_csv(C.TABLE_DIR / "mining_value_space_null_check.csv", index=False, float_format=FF)
        d[["feedback_id", "segment_name", "risk_tier", "pc1", "pc2"]].to_parquet(C.SILVER_DIR / "mining_segments.parquet", index=False)
    return out


# ================================================== association rules =======
def build_transactions(df: pd.DataFrame, sent_col: str = "sent_pred") -> pd.DataFrame:
    res = pd.cut(df.resolution_time_hours, [-np.inf, 12, 18, np.inf], labels=["<=12h", "12-18h", ">18h"]).astype(str)
    items = pd.DataFrame({
        "Sentiment": "Sentiment=" + df[sent_col].astype(str),
        "Resolution": "Resolution=" + res,
        "Repeat": np.where(df.repeat_purchase_flag == 1, "Repeat=Yes", "Repeat=No"),
        "Revenue": "Revenue=" + df.revenue_tier.astype(str),
        "Orders": "Orders=" + df.order_band.astype(str),
        "Issue": "Issue=" + df.issue_category, "Channel": "Channel=" + df.channel, "Region": "Region=" + df.region,
        "Product": "Product=" + df.product_category, "Source": "Source=" + df.feedback_source})
    return pd.get_dummies(items.stack()).groupby(level=0).sum().astype(bool)


def mine_rules(df: pd.DataFrame, sent_col: str = "sent_pred", save: bool = True) -> dict:
    oh = build_transactions(df, sent_col)
    n = len(oh)
    fi = apriori(oh, min_support=C.APRIORI_MIN_SUPPORT, use_colnames=True, max_len=C.APRIORI_MAX_LEN)
    rules = association_rules(fi, metric="lift", min_threshold=1.0, num_itemsets=len(fi))
    rules["ante"] = rules.antecedents.map(lambda s: tuple(sorted(s)))
    rules["cons"] = rules.consequents.map(lambda s: " & ".join(sorted(s)))
    targets = ["Sentiment=Negative", "Repeat=No"]
    R = rules[rules.cons.isin(targets)].copy()
    R = R[~R.apply(lambda r: any(i.split("=")[0] == r.cons.split("=")[0] for i in r.ante), axis=1)]   # no trivial same-variable rules
    funnel = {"frequent_itemsets": int(len(fi)), "rules_all_lift_ge_1": int(len(rules)), "rules_with_target_consequent": int(len(R))}
    R = R[R.lift >= C.RULE_MIN_LIFT].copy()
    funnel["rules_lift_ge_1.2"] = int(len(R))

    col = {c: oh[c].values for c in oh.columns}

    def mask(items) -> np.ndarray:
        m = np.ones(n, bool)
        for i in items:
            m &= col[i]
        return m

    def fisher_greater(a_mask, c_mask, b_mask=None):
        """H1: P(c | a) > P(c | b); b defaults to 'not a'."""
        b_mask = ~a_mask if b_mask is None else b_mask
        t = [[int((a_mask & c_mask).sum()), int((a_mask & ~c_mask).sum())], [int((b_mask & c_mask).sum()), int((b_mask & ~c_mask).sum())]]
        return stats.fisher_exact(t, alternative="greater")[1]

    p_assoc, p_prod, parent_best = [], [], []
    for r in R.itertuples():
        a, c = mask(r.ante), col[r.cons]
        p_assoc.append(fisher_greater(a, c))
        if len(r.ante) == 1:
            p_prod.append(0.0)
            parent_best.append(np.nan)
        else:
            worst, pconf = 0.0, 0.0
            for drop in r.ante:
                parent = tuple(x for x in r.ante if x != drop)
                pm = mask(parent)
                worst = max(worst, fisher_greater(a, c, pm & ~a))     # rule vs. parent-but-not-rule
                pconf = max(pconf, (pm & c).sum() / max(pm.sum(), 1))
            p_prod.append(worst)
            parent_best.append(pconf)
    R["p_assoc"], R["p_productive"], R["best_parent_confidence"] = p_assoc, p_prod, parent_best
    R["p_final"] = np.maximum(R.p_assoc, R.p_productive)
    R["q_assoc_bh"] = multipletests(R.p_assoc, method="fdr_bh")[1]
    R["q_final_bh"] = multipletests(R.p_final, method="fdr_bh")[1]
    R["antecedent_size"] = R.ante.map(len)
    R["antecedent"] = R.ante.map(lambda t: " & ".join(t))
    funnel["rules_significant_fdr_assoc"] = int((R.q_assoc_bh < C.FDR_ALPHA).sum())
    robust = R[(R.q_final_bh < C.FDR_ALPHA) & (R.confidence > R.best_parent_confidence.fillna(0))].copy()
    funnel["rules_robust_after_productivity"] = int(len(robust))
    keep = ["antecedent", "cons", "antecedent_size", "support", "confidence", "lift", "leverage", "conviction", "best_parent_confidence", "p_assoc", "p_productive", "q_final_bh"]
    # TOTAL sort order (explicit tie-breakers): frozenset iteration order depends on the process hash seed, so a sort on
    # lift/support alone is not reproducible when two rules tie (e.g. A->B and B->A have identical lift and support).
    robust = robust.sort_values(["lift", "support", "antecedent", "cons"], ascending=[False, False, True, True])[keep].rename(columns={"cons": "consequent"})
    all_cols = R.sort_values(["lift", "support", "antecedent", "cons"], ascending=[False, False, True, True])[keep + ["q_assoc_bh"]].rename(columns={"cons": "consequent"})
    # does ANY rule built on pure context (issue/channel/region/product/source) survive?
    ctx_items = ("Issue=", "Channel=", "Region=", "Product=", "Source=")
    context_only = robust[robust.antecedent.map(lambda a: all(i.startswith(ctx_items) for i in a.split(" & ")))]
    funnel["robust_rules_using_only_context_dimensions"] = int(len(context_only))
    ctx_with_any = robust[robust.antecedent.map(lambda a: any(i.startswith(ctx_items) for i in a.split(" & ")))]
    funnel["robust_rules_containing_any_context_item"] = int(len(ctx_with_any))
    summary = {"funnel": funnel, "min_support": C.APRIORI_MIN_SUPPORT, "max_len": C.APRIORI_MAX_LEN, "min_lift": C.RULE_MIN_LIFT,
               "robust_rules": robust.head(12).to_dict("records"), "n_transactions": int(n), "n_items": int(oh.shape[1])}
    if save:
        all_cols.to_csv(C.TABLE_DIR / "mining_association_rules_candidates.csv", index=False, float_format=FF)
        robust.to_csv(C.TABLE_DIR / "mining_association_rules_robust.csv", index=False, float_format=FF)
        pd.DataFrame([funnel]).T.rename(columns={0: "count"}).to_csv(C.TABLE_DIR / "mining_association_funnel.csv", float_format=FF)
    return {"summary": summary, "robust": robust, "candidates": all_cols, "funnel": funnel}


# ------------------------------------------------------------------- run #
def run(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        from .transform import load_silver
        s = load_silver()
        sc = pd.read_parquet(C.SILVER_DIR / "nlp_sentiment.parquet")
        df = s.merge(sc, on="feedback_id")
    C.ensure_dirs()
    seg = segment(df, save)
    rul = mine_rules(df, "sent_pred", save)
    metrics = {"segmentation": seg["summary"], "association_rules": rul["summary"]}
    if save:
        save_metrics("mining", metrics)
    s_ = seg["summary"]
    log.info("Mining | k=%d silhouette=%.3f (null p95 %.3f) stability ARI=%.2f | rules: %d candidates -> %d robust",
             s_["k"], s_["silhouette"], s_["null_silhouette_p95"], s_["stability"]["mean_ari"], rul["funnel"]["rules_lift_ge_1.2"], rul["funnel"]["rules_robust_after_productivity"])
    return {"seg": seg, "rules": rul, "metrics": metrics}


if __name__ == "__main__":  # pragma: no cover
    run()
