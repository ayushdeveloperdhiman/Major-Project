"""Stage 9 - Topic / theme intelligence.

Two complementary approaches
  (a) CLASSIFY  - a transparent keyword taxonomy maps text to the eight business themes
                  (usable on new free text, auditable by the business).
  (b) EXTRACT   - unsupervised NMF (primary) and LDA (comparator) discover topics without labels.
                  k is chosen on INTERNAL criteria only (NPMI coherence + seed stability); the known
                  Issue_Category labels are used purely as EXTERNAL validation (NMI / ARI / purity).

Design decision: topic discovery runs on the *opinion clause* (first sentence).  The second sentence merely
restates channel and product - verified 100 % redundant with the structured columns (rule DQ-K03) - so
including it would make the model rediscover structure we already hold in the warehouse.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests

from . import config as C
from . import kpis as K
from .nlp_prep import BoilerplateFilter, TextPreprocessor
from .utils import cramers_v, get_logger, save_metrics

warnings.filterwarnings("ignore")
log = get_logger("cvi.topics")

# Keyword taxonomy (lemmatised terms).  Business-owned and extensible.
# Weight 1.0 = term specific to the theme; 0.5 = ambiguous term shared across themes (e.g. 'delay', 'late').
# A theme needs a total score >= 1.0 to be assigned; ties between themes -> 'Multiple'.
THEME_KEYWORDS: dict[str, dict[str, float]] = {
    "Delivery Delay": {"delivery": 1, "deliver": 1, "courier": 1, "shipping": 1, "shipment": 1, "dispatch": 1, "tracking": 1,
                       "arrival": 1, "arrive": 0.5, "delay": 0.5, "late": 0.5},
    "Packaging": {"packaging": 1, "package": 1, "packing": 1, "box": 1, "parcel": 1, "damage": 0.5, "wrap": 0.5, "seal": 0.5},
    "Product Availability": {"availability": 1, "stock": 1, "unavailable": 1, "restock": 1, "sold": 0.5, "available": 0.5},
    "Product Quality": {"quality": 1, "defect": 1, "defective": 1, "faulty": 1, "durable": 1, "durability": 1, "material": 0.5, "break": 0.5, "broken": 0.5},
    "Pricing": {"price": 1, "pricing": 1, "expensive": 1, "overpriced": 1, "discount": 1, "cheap": 1, "cost": 1, "charge": 0.5, "fee": 0.5},
    "Customer Support": {"support": 1, "agent": 1, "helpline": 1, "representative": 1, "staff": 1, "customer": 0.5, "service": 0.5, "call": 0.5},
    "Refund Delay": {"refund": 1, "reimburse": 1, "chargeback": 1, "return": 0.5, "delay": 0.5},
    "App Experience": {"app": 1, "application": 1, "website": 1, "crash": 1, "login": 1, "checkout": 1, "interface": 1, "bug": 1, "update": 0.5},
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def opinion_clause(text: str) -> str:
    """First sentence = the opinion about the topic (second sentence restates channel/product)."""
    return _SENT_SPLIT.split(str(text).strip(), maxsplit=1)[0]


def classify_theme(text: str, clause_only: bool = True, pre: TextPreprocessor | None = None) -> str:
    pre = pre or TextPreprocessor(negation=False, remove_stopwords=True)
    toks = set(pre.tokens(opinion_clause(text) if clause_only else text))
    scores = {th: sum(w for k, w in kws.items() if k in toks) for th, kws in THEME_KEYWORDS.items()}
    best = max(scores.values())
    if best < 1.0:
        return "Unclassified"
    winners = [th for th, v in scores.items() if v == best]
    return winners[0] if len(winners) == 1 else "Multiple"


# ------------------------------------------------------------- coherence #
def npmi_coherence(topics_top_words: list[list[str]], binary_dt, vocab: dict[str, int]) -> float:
    """Mean NPMI over top-word pairs, using document co-occurrence (range -1..1; higher = more coherent)."""
    N = binary_dt.shape[0]
    co = (binary_dt.T @ binary_dt).toarray().astype(float)
    df_ = np.diag(co)
    scores = []
    for words in topics_top_words:
        idx = [vocab[w] for w in words if w in vocab]
        vals = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                p12 = co[idx[i], idx[j]] / N
                p1, p2 = df_[idx[i]] / N, df_[idx[j]] / N
                if p12 <= 0:
                    vals.append(-1.0)
                else:
                    vals.append(np.log(p12 / (p1 * p2)) / (-np.log(p12)) if p12 < 1 else 1.0)
        if vals:
            scores.append(np.mean(vals))
    return float(np.mean(scores)) if scores else float("nan")


def purity(y_true, y_pred) -> float:
    ct = pd.crosstab(pd.Series(y_pred, name="p"), pd.Series(y_true, name="t"))
    return float(ct.max(axis=1).sum() / ct.values.sum())


def _top_words(H: np.ndarray, feats: np.ndarray, n: int = 8, rel: float = 0.0) -> list[list[str]]:
    """Top-n words per topic; optionally keep only words whose weight is >= rel * the topic's maximum."""
    out = []
    for h in H:
        idx = np.argsort(h)[::-1][:n]
        if rel > 0:
            idx = [i for i in idx if h[i] >= rel * h.max()]
        out.append([feats[i] for i in idx])
    return out


def opinion_terms(docs: list[str], y_sent: pd.Series, alpha: float = C.FDR_ALPHA, min_v: float = 0.10, min_df: int = 5) -> pd.DataFrame:
    """Aspect-opinion separation: tokens whose presence is significantly (FDR) and materially (Cramer's V) associated
    with sentiment are OPINION words (how customers feel); the rest are ASPECT words (what they talk about)."""
    cv = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, min_df=min_df)
    B = cv.fit_transform(docs)
    y = np.asarray(y_sent)
    rows = []
    for j, w in enumerate(cv.get_feature_names_out()):
        col = np.asarray(B[:, j].todense()).ravel() > 0
        ct = pd.crosstab(col, y)
        if ct.shape[0] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"term": w, "docs": int(col.sum()), "chi2": chi2, "p_value": p, "cramers_v": cramers_v(ct)})
    t = pd.DataFrame(rows)
    t["q_bh"] = multipletests(t.p_value, method="fdr_bh")[1]
    t["is_opinion_word"] = (t.q_bh < alpha) & (t.cramers_v >= min_v)
    return t.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def _prep_tokens(texts: pd.Series) -> list[str]:
    tp = TextPreprocessor(negation=False)
    docs = tp.transform(texts)
    bf = BoilerplateFilter().fit(docs)
    return bf.transform(docs)


def discover(texts: pd.Series, y_issue: pd.Series, k_range=C.TOPIC_K_RANGE, view: str = "opinion", drop: set[str] | None = None) -> dict:
    """k-sweep with NMF on TF-IDF; k is chosen on INTERNAL criteria only (coherence + seed stability)."""
    docs = _prep_tokens(texts)
    if drop:
        docs = [" ".join(t for t in d.split() if t not in drop) for d in docs]
    tf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2)
    X = tf.fit_transform(docs)
    feats = np.array(tf.get_feature_names_out())
    cv = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, min_df=2, vocabulary=list(feats))
    B = cv.fit_transform(docs)
    vocab = {w: i for i, w in enumerate(feats)}
    rows = []
    for k in [k for k in k_range if k <= len(feats)]:      # NMF cannot have more components than features
        m = NMF(k, init="nndsvda", random_state=C.SEED, max_iter=800).fit(X)
        lab = m.transform(X).argmax(1)
        coh = npmi_coherence(_top_words(m.components_, feats, 4, rel=0.25), B, vocab)
        a = NMF(k, init="random", random_state=1, max_iter=800).fit(X).transform(X).argmax(1)
        b = NMF(k, init="random", random_state=2, max_iter=800).fit(X).transform(X).argmax(1)
        rows.append({"view": view, "k": k, "vocabulary": len(feats), "reconstruction_err": m.reconstruction_err_, "coherence_npmi": coh,
                     "empty_topics": int(k - len(set(lab))), "stability_ari_seed1_vs_seed2": adjusted_rand_score(a, b),
                     "nmi_vs_issue": normalized_mutual_info_score(y_issue, lab), "ari_vs_issue": adjusted_rand_score(y_issue, lab),
                     "purity_vs_issue": purity(y_issue, lab)})
    sweep = pd.DataFrame(rows)
    sweep["internal_score"] = (sweep.coherence_npmi + sweep.stability_ari_seed1_vs_seed2) / 2     # reported diagnostic
    # Selection (INTERNAL criteria only): highest NPMI coherence among k with no empty topic (a component that attracts
    # no document is not a topic) and seed stability >= 0.8; fall back gracefully if nothing qualifies.
    elig = sweep[(sweep.empty_topics == 0) & (sweep.stability_ari_seed1_vs_seed2 >= 0.8)]
    if elig.empty:
        elig = sweep[sweep.empty_topics == 0] if (sweep.empty_topics == 0).any() else sweep
    kbest = int(elig.sort_values(["coherence_npmi", "k"], ascending=[False, True]).iloc[0].k)
    sweep["selected"] = sweep.k == kbest
    return {"sweep": sweep, "k": kbest, "X": X, "feats": feats, "B": B, "vocab": vocab, "docs": docs, "tfidf": tf}


def fit_final_nmf(art: dict, k: int):
    m = NMF(k, init="nndsvda", random_state=C.SEED, max_iter=1000).fit(art["X"])
    W = m.transform(art["X"])
    return m, W


# ------------------------------------------------------------------- run #
def run(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    """df must contain silver columns + sent_pred (from the sentiment stage)."""
    if df is None:
        from .transform import load_silver
        s = load_silver()
        sc = pd.read_parquet(C.SILVER_DIR / "nlp_sentiment.parquet")
        df = s.merge(sc, on="feedback_id")
    C.ensure_dirs()
    pre = TextPreprocessor(negation=False)
    clause = df.text_clean.map(opinion_clause)

    # (a) taxonomy classification
    theme = df.text_clean.map(lambda t: classify_theme(t, True, pre))
    tax_acc = float((theme == df.issue_category).mean())
    tax_cm = pd.crosstab(df.issue_category, theme)

    # (b) unsupervised discovery.  Three views (documented comparison); the ASPECT view is the design choice:
    #     topics describe what customers talk ABOUT, so opinion words (learned from sentiment association) are removed.
    base_docs = _prep_tokens(clause)
    op = opinion_terms(base_docs, df.sent_pred)
    opinion_set = set(op[op.is_opinion_word].term)
    art_full = discover(df.text_clean, df.issue_category, view="A. full text")
    art_op = discover(clause, df.issue_category, view="B. opinion clause")
    art = discover(clause, df.issue_category, view="C. opinion clause, aspect words only", drop=opinion_set)
    sweep = pd.concat([art_full["sweep"], art_op["sweep"], art["sweep"]], ignore_index=True)
    k = art["k"]
    nmf, W = fit_final_nmf(art, k)
    nmf_lab = W.argmax(1)
    tops = _top_words(nmf.components_, art["feats"], 5, rel=0.25)
    topics_tbl = []
    for t in range(k):
        mask = nmf_lab == t
        maj = df.issue_category[mask].mode().iloc[0] if mask.any() else None
        topics_tbl.append({"nmf_topic_id": t, "top_words": ", ".join(tops[t]), "documents": int(mask.sum()), "share": float(mask.mean()),
                           "majority_issue_category": maj, "purity": float((df.issue_category[mask] == maj).mean()) if mask.any() else np.nan})
    topics_tbl = pd.DataFrame(topics_tbl).sort_values("nmf_topic_id")
    nmf_ext = {"k": k, "nmi_vs_issue": normalized_mutual_info_score(df.issue_category, nmf_lab), "ari_vs_issue": adjusted_rand_score(df.issue_category, nmf_lab),
               "purity_vs_issue": purity(df.issue_category, nmf_lab)}
    # LDA comparator at the same k
    cvec = CountVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2)
    Xc = cvec.fit_transform(art["docs"])
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=C.SEED, max_iter=60).fit(Xc)
    lda_lab = lda.transform(Xc).argmax(1)
    lda_feats = np.array(cvec.get_feature_names_out())
    Bc = CountVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, vocabulary=list(lda_feats)).fit_transform(art["docs"])
    lda_ext = {"k": k, "nmi_vs_issue": normalized_mutual_info_score(df.issue_category, lda_lab), "ari_vs_issue": adjusted_rand_score(df.issue_category, lda_lab),
               "purity_vs_issue": purity(df.issue_category, lda_lab), "perplexity": float(lda.perplexity(Xc)),
               "coherence_npmi": npmi_coherence(_top_words(lda.components_, lda_feats, 4, rel=0.25), Bc, {w: i for i, w in enumerate(lda_feats)})}
    nmf_ext["coherence_npmi"] = float(art["sweep"].set_index("k").loc[k, "coherence_npmi"])
    # label each NMF topic by its majority issue (for the warehouse; purity reported)
    topic_label = {t: f"T{t}: " + ", ".join(tops[t][:3]) for t in range(k)}

    out = df[["feedback_id"]].copy()
    out["topic_issue"] = theme.values
    out["nmf_topic_id"] = nmf_lab
    out["nmf_topic_label"] = [topic_label[t] for t in nmf_lab]
    out["opinion_clause"] = clause.values

    # ---- prevalence & business relevance (themes = taxonomy topics; sentiment = NLP prediction)
    d = df.assign(topic_issue=theme.values)
    prev = K.kpi_table(d, "topic_issue", "sent_pred").sort_values("topic_share", ascending=False)
    ct = pd.crosstab(d.topic_issue, d.sent_pred)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    group = K.kpi_table(d.assign(issue_theme_group=d.issue_theme_group), "issue_theme_group", "sent_pred").sort_values("topic_share", ascending=False)
    # ---- theme x business segment composition
    seg_rows = []
    for dim in ["channel", "region", "product_category", "feedback_source"]:
        c2 = pd.crosstab(d[dim], d.topic_issue)
        chi, pp, dd, _ = stats.chi2_contingency(c2)
        seg_rows.append({"segment_dimension": dim, "chi2": chi, "dof": dd, "p_value": pp, "cramers_v": cramers_v(c2)})
        share = (c2.div(c2.sum(1), axis=0) * 100).round(2)
        share.to_csv(C.TABLE_DIR / f"theme_share_by_{dim}.csv") if save else None
    seg = pd.DataFrame(seg_rows)
    seg["q_bh"] = multipletests(seg.p_value, method="fdr_bh")[1]
    # ---- negative-driver descriptor themes (what customers say when negative)
    negd = df.assign(sent=df.sent_pred)
    negd = negd[negd.sent == "Negative"]
    desc_theme = negd.negative_descriptor_theme.value_counts(dropna=False).rename_axis("descriptor_theme").reset_index(name="negative_records")
    desc_theme["share"] = desc_theme.negative_records / desc_theme.negative_records.sum()
    nd_tests = {c: float(stats.kruskal(*[g[c].values for _, g in negd.groupby("negative_descriptor_theme")]).pvalue)
                for c in ["resolution_time_hours", "customer_revenue_inr", "order_count"]}
    nd_tests["repeat_chi2_p"] = float(stats.chi2_contingency(pd.crosstab(negd.negative_descriptor_theme, negd.repeat_purchase_flag))[1])

    summary = {
        "taxonomy_accuracy_vs_issue": tax_acc, "taxonomy_unclassified": int((theme == "Unclassified").sum()), "taxonomy_multiple": int((theme == "Multiple").sum()),
        "nmf": nmf_ext, "lda": lda_ext,
        "k_selected_internal": k, "opinion_words_removed": sorted(opinion_set), "n_opinion_words": len(opinion_set),
        "views_at_selected_k": {v: {kk: float(x) for kk, x in g[g.selected].iloc[0][
            ["k", "coherence_npmi", "stability_ari_seed1_vs_seed2", "nmi_vs_issue", "purity_vs_issue", "empty_topics"]].items()} for v, g in sweep.groupby("view")},
        "theme_vs_sentiment": {"chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": cramers_v(ct)},
        "top_theme_by_share": prev.iloc[0].topic_issue, "top_theme_share": float(prev.iloc[0].topic_share),
        "theme_share_range_pp": float(100 * (prev.topic_share.max() - prev.topic_share.min())),
        "theme_composition_tests": seg.to_dict("records"),
        "negative_descriptor_theme_shares": desc_theme.set_index("descriptor_theme").share.to_dict(),
        "negative_descriptor_theme_tests": nd_tests,
        "nmf_topic_summary": topics_tbl.to_dict("records"),
    }
    if save:
        sweep.to_csv(C.TABLE_DIR / "topics_k_sweep.csv", index=False)
        op.to_csv(C.TABLE_DIR / "topics_opinion_word_scan.csv", index=False)
        topics_tbl.to_csv(C.TABLE_DIR / "topics_nmf_topics.csv", index=False)
        tax_cm.to_csv(C.TABLE_DIR / "topics_taxonomy_confusion.csv")
        prev.to_csv(C.TABLE_DIR / "topics_prevalence_relevance.csv", index=False)
        group.to_csv(C.TABLE_DIR / "topics_theme_group_kpis.csv", index=False)
        seg.to_csv(C.TABLE_DIR / "topics_composition_tests.csv", index=False)
        desc_theme.to_csv(C.TABLE_DIR / "topics_negative_descriptor_themes.csv", index=False)
        pd.DataFrame([{"model": "NMF (TF-IDF)", **{k_: v for k_, v in nmf_ext.items()}}, {"model": "LDA (counts)", **lda_ext}]).to_csv(C.TABLE_DIR / "topics_nmf_vs_lda.csv", index=False)
        out.to_parquet(C.SILVER_DIR / "nlp_topics.parquet", index=False)
        save_metrics("topics", summary)
    log.info("Topics | taxonomy acc=%.3f | NMF k=%d NMI=%.3f purity=%.3f | LDA NMI=%.3f | theme~sentiment p=%.2f",
             tax_acc, k, nmf_ext["nmi_vs_issue"], nmf_ext["purity_vs_issue"], lda_ext["nmi_vs_issue"], p)
    return {"summary": summary, "sweep": sweep, "topics": topics_tbl, "prevalence": prev, "assign": out}


if __name__ == "__main__":  # pragma: no cover
    run()
