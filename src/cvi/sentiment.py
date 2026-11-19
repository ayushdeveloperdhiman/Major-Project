"""Stage 8 - Sentiment analysis: build, evaluate and compare approaches.

Target      3-class sentiment (Negative / Neutral / Positive) predicted from TEXT ONLY.
Reference   ``reference_sentiment`` (verified to be a deterministic function of Rating, i.e. a rating-implied label).
            Rating and CSAT are NEVER used as model features (label leakage).

Approaches  (1) majority-class baseline  (2) VADER general-purpose lexicon  (3) TF-IDF + Logistic Regression
            (4) TF-IDF + Linear SVM  (5) TF-IDF + Multinomial NB  (6) hybrid TF-IDF + VADER features (LR)

Evaluation regimes
  cv5      stratified 5-fold out-of-fold predictions (in-distribution; also used to score every record
           for downstream KPIs so that no record is scored by a model that saw its label)
  holdout  stratified 80/20 split with bootstrap 95% CIs
  lodo     leave-one-DESCRIPTOR-out: the sentiment-bearing word of the test fold never appears in training.
           This is the honest generalisation test because the supplied text is templated
           (one descriptor word from a closed list of 19 carries all the sentiment).
Plus        descriptor-masking ablation (is the model reading the opinion or the topic words?),
            an out-of-template probe set and a per-descriptor VADER error analysis.
"""
from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import config as C
from .nlp_prep import BoilerplateFilter, TextPreprocessor
from .quality import parse_template
from .utils import get_logger, save_metrics

warnings.filterwarnings("ignore")
log = get_logger("cvi.sentiment")
LABELS = C.SENTIMENT_ORDER
_SIA = None


def _sia():
    """The VADER analyser, imported on first use.

    VADER is only the general-purpose-lexicon baseline.  Importing it at module level made *loading the saved model*
    (and so the whole dashboard) depend on a package that the model itself never uses.
    """
    global _SIA
    if _SIA is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"vaderSentiment is not installed for this Python ({sys.executable}). Use the project environment "
                "(./.venv/bin/python -m ...) or run: python -m pip install -r requirements.txt") from err
        _SIA = SentimentIntensityAnalyzer()
    return _SIA

# Hand-written, hand-labelled sentences that are NOT in the supplied template - a qualitative smoke test of
# domain shift (n=30, analyst-labelled, NOT a statistically meaningful benchmark).
OOD_PROBE = [
    ("Delivery was super quick and the packaging was perfect, very happy!", "Positive"),
    ("Love the new app, checkout is easy and fast.", "Positive"),
    ("The support agent solved my issue in minutes, excellent service.", "Positive"),
    ("Great quality for the price, will definitely buy again.", "Positive"),
    ("Refund arrived the next day, really impressed.", "Positive"),
    ("Product exactly as described and works wonderfully.", "Positive"),
    ("Smooth experience from order to delivery.", "Positive"),
    ("The staff in the store were friendly and helpful.", "Positive"),
    ("Fantastic value, I am satisfied with this purchase.", "Positive"),
    ("Very pleased with how quickly the issue was resolved.", "Positive"),
    ("My order arrived two weeks late and nobody answered my calls.", "Negative"),
    ("The product was broken when it arrived, terrible quality.", "Negative"),
    ("Customer support was rude and unhelpful.", "Negative"),
    ("Still waiting for my refund after a month, very disappointed.", "Negative"),
    ("The app keeps crashing at checkout, so frustrating.", "Negative"),
    ("Packaging was damaged and items were missing.", "Negative"),
    ("Way too expensive for what you get, not worth it.", "Negative"),
    ("I would not recommend this, the experience was awful.", "Negative"),
    ("Item was out of stock but they took my payment anyway.", "Negative"),
    ("Worst service ever, never ordering again.", "Negative"),
    ("The order arrived on the expected date.", "Neutral"),
    ("The product is okay, nothing special.", "Neutral"),
    ("Delivery took about the usual time.", "Neutral"),
    ("It is an average app, does what it says.", "Neutral"),
    ("The price is standard for this category.", "Neutral"),
    ("Packaging was acceptable.", "Neutral"),
    ("The support call was fine, neither good nor bad.", "Neutral"),
    ("I received the item as ordered.", "Neutral"),
    ("Quality is normal for the price.", "Neutral"),
    ("The refund process took the usual few days.", "Neutral"),
]


# ------------------------------------------------------------------ VADER #
def vader_scores(texts) -> pd.DataFrame:
    sia = _sia()
    rows = [sia.polarity_scores("" if t is None else str(t)) for t in texts]
    return pd.DataFrame(rows).rename(columns={"neg": "vader_neg", "neu": "vader_neu", "pos": "vader_pos", "compound": "vader_compound"})


def vader_label(compound, th: float = C.VADER_THRESHOLD) -> np.ndarray:
    c = np.asarray(compound, float)
    return np.where(c >= th, "Positive", np.where(c <= -th, "Negative", "Neutral"))


class VaderFeaturizer(BaseEstimator, TransformerMixin):
    """Adds the four VADER scores as dense features (applied to the RAW text, as VADER expects)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        v = vader_scores(list(X))
        return v[["vader_compound", "vader_neg", "vader_neu", "vader_pos"]].values


# ----------------------------------------------------------------- models #
def _tfidf(ngram=(1, 1)) -> TfidfVectorizer:
    # boilerplate is removed upstream (BoilerplateFilter), so no max_df filter is needed here
    return TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True, token_pattern=r"\S+", lowercase=False)


def make_models(lr_c: float = 1.0, ngram=(1, 1)) -> dict[str, Pipeline]:
    front = lambda: [("prep", TextPreprocessor()), ("boiler", BoilerplateFilter())]  # noqa: E731
    lr = lambda: LogisticRegression(C=lr_c, max_iter=3000, class_weight="balanced", random_state=C.SEED)  # noqa: E731
    return {
        "TF-IDF + Logistic Regression": Pipeline(front() + [("vec", _tfidf(ngram)), ("clf", lr())]),
        "TF-IDF + Linear SVM": Pipeline(front() + [("vec", _tfidf(ngram)), ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=C.SEED))]),
        "TF-IDF + Multinomial NB": Pipeline(front() + [("vec", _tfidf(ngram)), ("clf", MultinomialNB(alpha=0.5))]),
        "Hybrid: TF-IDF + VADER features (LR)": Pipeline([("feats", FeatureUnion([
            ("tfidf", Pipeline(front() + [("vec", _tfidf(ngram))])), ("vader", VaderFeaturizer())])), ("clf", lr())]),
    }


def metric_block(y, yp, proba=None) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=LABELS, average="macro", zero_division=0)
    out = {"accuracy": accuracy_score(y, yp), "macro_precision": p, "macro_recall": r, "macro_f1": f,
           "weighted_f1": f1_score(y, yp, labels=LABELS, average="weighted", zero_division=0),
           "cohen_kappa": cohen_kappa_score(y, yp), "mcc": matthews_corrcoef(y, yp)}
    if proba is not None:
        out["log_loss"] = log_loss(y, proba, labels=LABELS)
        out["roc_auc_ovr_macro"] = roc_auc_score(y, proba, multi_class="ovr", labels=LABELS)
    return out


def boot_ci(y, yp, fn, B: int = 1000, seed: int = C.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, yp = np.asarray(y), np.asarray(yp)
    vals = [fn(y[i], yp[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def per_class_report(y, yp) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, yp, labels=LABELS, zero_division=0)
    return pd.DataFrame({"class": LABELS, "precision": p, "recall": r, "f1": f, "support": s})


def confusion(y, yp) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y, yp, labels=LABELS), index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])


def extract_descriptor(texts: pd.Series) -> pd.Series:
    return parse_template(texts)["descriptor"].str.casefold()


def mask_descriptor(texts: pd.Series, desc: pd.Series) -> pd.Series:
    return pd.Series([t.replace(f" was {d}.", " was .") if isinstance(d, str) else t for t, d in zip(texts, desc)], index=texts.index)


# ------------------------------------------------------------------ run #
def run(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        from .transform import load_silver
        df = load_silver()
    C.ensure_dirs()
    X = df["text_clean"].astype(str)
    y = df["reference_sentiment"].astype(str)
    desc = extract_descriptor(df["text_clean"]).fillna("unknown")
    skf = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    # ---- tuning on the TRAIN split only (hold-out untouched).  Selection rule (documented):
    #      keep configs within 0.005 of the best CV macro-F1, then prefer the simplest feature space (unigram) and the
    #      library-default regularisation (C=1).  CV log-loss is reported but NOT used as a tie-breaker: on separable
    #      training data it improves monotonically with C and would simply pick the most over-confident model.
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(X, y, desc, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    base = Pipeline([("prep", TextPreprocessor()), ("boiler", BoilerplateFilter()), ("vec", _tfidf()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=C.SEED))])
    grid = GridSearchCV(base, {"vec__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.1, 1, 10, 100]}, cv=skf,
                        scoring={"f1": "f1_macro", "nll": "neg_log_loss"}, refit=False, n_jobs=1)
    grid.fit(Xtr, ytr)
    tuning = pd.DataFrame({"ngram_range": [str(p["vec__ngram_range"]) for p in grid.cv_results_["params"]],
                           "C": [p["clf__C"] for p in grid.cv_results_["params"]],
                           "cv_macro_f1_mean": grid.cv_results_["mean_test_f1"], "cv_macro_f1_std": grid.cv_results_["std_test_f1"],
                           "cv_log_loss": -grid.cv_results_["mean_test_nll"]})
    elig = tuning[tuning.cv_macro_f1_mean >= tuning.cv_macro_f1_mean.max() - 0.005].copy()
    elig["_uni"] = (elig.ngram_range != "(1, 1)").astype(int)
    elig["_c_dist"] = (np.log10(elig.C.astype(float)) - 0.0).abs()          # distance from default C=1 on a log scale
    pick = elig.sort_values(["_uni", "_c_dist", "C"]).iloc[0]
    tuning["selected"] = (tuning.ngram_range == pick.ngram_range) & (tuning.C == pick.C)
    best_c, best_ng = float(pick.C), (1, 1) if pick.ngram_range == "(1, 1)" else (1, 2)
    models = make_models(best_c, best_ng)
    log.info("Tuning: %d of %d configs tie on macro-F1; selected C=%s ngram=%s (simplest + default regularisation)", len(elig), len(tuning), best_c, best_ng)

    rows, oof_proba, oof_pred = [], {}, {}
    # ---- baselines
    dummy = DummyClassifier(strategy="most_frequent")
    pred_dummy = cross_val_predict(dummy, X, y, cv=skf)
    rows.append({"model": "Baseline: majority class", "regime": "cv5", **metric_block(y, pred_dummy)})
    vs = vader_scores(X)
    v_pred = vader_label(vs.vader_compound)
    rows.append({"model": "VADER lexicon (+/-0.05)", "regime": "cv5", **metric_block(y, v_pred)})
    rows.append({"model": "VADER lexicon (+/-0.05)", "regime": "lodo", **metric_block(y, v_pred)})      # untrained: unaffected by vocabulary split
    ytest_v = vader_label(vader_scores(Xte).vader_compound)
    rows.append({"model": "VADER lexicon (+/-0.05)", "regime": "holdout", **metric_block(yte, ytest_v)})
    rows.append({"model": "Baseline: majority class", "regime": "holdout", **metric_block(yte, np.repeat(ytr.mode()[0], len(yte)))})
    rows.append({"model": "Baseline: majority class", "regime": "lodo", **metric_block(y, pred_dummy)})

    logo = LeaveOneGroupOut()
    holdout_ci = {}
    for name, est in models.items():
        has_proba = hasattr(est.named_steps["clf"], "predict_proba")
        # cv5 (OOF)
        pred = cross_val_predict(clone(est), X, y, cv=skf)
        proba = cross_val_predict(clone(est), X, y, cv=skf, method="predict_proba") if has_proba else None
        oof_pred[name], oof_proba[name] = pred, proba
        rows.append({"model": name, "regime": "cv5", **metric_block(y, pred, proba)})
        # holdout
        m = clone(est).fit(Xtr, ytr)
        yp = m.predict(Xte)
        pp = m.predict_proba(Xte) if has_proba else None
        rows.append({"model": name, "regime": "holdout", **metric_block(yte, yp, pp)})
        holdout_ci[name] = {"accuracy_ci": boot_ci(yte, yp, accuracy_score), "macro_f1_ci": boot_ci(yte, yp, lambda a, b: f1_score(a, b, average="macro", labels=LABELS))}
        # leave-one-descriptor-out
        pred_l = cross_val_predict(clone(est), X, y, groups=desc, cv=logo)
        rows.append({"model": name, "regime": "lodo", **metric_block(y, pred_l)})
    comp = pd.DataFrame(rows)

    # ---- model selection: best cv5 macro-F1; among models within 0.005 prefer the simplest (LR)
    cv = comp[(comp.regime == "cv5") & comp.model.isin(models)]
    best_f1 = cv.macro_f1.max()
    simple_order = ["TF-IDF + Logistic Regression", "TF-IDF + Linear SVM", "TF-IDF + Multinomial NB", "Hybrid: TF-IDF + VADER features (LR)"]
    chosen = next(m for m in simple_order if m in set(cv[cv.macro_f1 >= best_f1 - 0.005].model))
    log.info("Chosen production model: %s (cv5 macro-F1 %.4f)", chosen, cv[cv.model == chosen].macro_f1.iloc[0])

    # ---- reports for chosen model + VADER
    rep = pd.concat([per_class_report(y, oof_pred[chosen]).assign(model=chosen + " (cv5 OOF)"),
                     per_class_report(y, v_pred).assign(model="VADER lexicon"),
                     per_class_report(y, pred_dummy).assign(model="Majority baseline")])
    cm_chosen, cm_vader = confusion(y, oof_pred[chosen]), confusion(y, v_pred)

    # ---- descriptor-level VADER error analysis
    d_tbl = pd.DataFrame({"descriptor": desc, "true": y, "vader": v_pred, "vader_compound": vs.vader_compound.values})
    vd = d_tbl.groupby(["descriptor", "true"]).apply(lambda g: pd.Series({"n": len(g), "vader_accuracy": (g.true == g.vader).mean(),
                                                                         "mean_compound": g.vader_compound.mean(),
                                                                         "pred_Negative": (g.vader == "Negative").mean(), "pred_Neutral": (g.vader == "Neutral").mean(),
                                                                         "pred_Positive": (g.vader == "Positive").mean()})).reset_index()
    # word-level lexicon score (descriptor alone) - shows which words VADER knows
    vd["vader_word_score"] = [_sia().polarity_scores(w)["compound"] for w in vd.descriptor]
    vd = vd.sort_values(["true", "vader_accuracy"]).reset_index(drop=True)

    # ---- ablation: mask the descriptor (topic/channel/product words only) and descriptor-only
    Xm = mask_descriptor(X, desc)
    lr = models["TF-IDF + Logistic Regression"]
    pm = cross_val_predict(clone(lr), Xm, y, cv=skf)
    pd_only = cross_val_predict(clone(lr), desc.astype(str), y, cv=skf)
    abl = pd.DataFrame([
        {"input": "Full text", **{k: v for k, v in metric_block(y, oof_pred["TF-IDF + Logistic Regression"]).items() if k in ("accuracy", "macro_f1")}},
        {"input": "Descriptor word MASKED (topic / channel / product words only)", **{k: v for k, v in metric_block(y, pm).items() if k in ("accuracy", "macro_f1")}},
        {"input": "Descriptor word ONLY", **{k: v for k, v in metric_block(y, pd_only).items() if k in ("accuracy", "macro_f1")}},
        {"input": "Majority-class baseline", **{k: v for k, v in metric_block(y, pred_dummy).items() if k in ("accuracy", "macro_f1")}},
    ])

    # ---- final fit on all data (production model) + OOF scoring for downstream KPIs
    final = clone(models[chosen]).fit(X, y)
    joblib.dump(final, C.MODEL_DIR / "sentiment_model.joblib")
    cls = list(final.named_steps["clf"].classes_)
    assert cls == LABELS, cls
    P = oof_proba[chosen]
    scored = pd.DataFrame({"feedback_id": df.feedback_id.values, "descriptor": desc.values,
                           "sent_pred": np.array(LABELS)[P.argmax(1)], "p_negative": P[:, 0], "p_neutral": P[:, 1], "p_positive": P[:, 2],
                           "vader_compound": vs.vader_compound.values, "vader_label": v_pred})
    scored["negative_descriptor_theme"] = scored.descriptor.map(C.NEGATIVE_DESCRIPTOR_THEME).where(scored.sent_pred == "Negative")

    # ---- learned domain lexicon (LR coefficients)
    clf = final.named_steps["clf"]
    vec = final.named_steps["vec"]
    feats = np.array(vec.get_feature_names_out())
    lex = []
    for i, c in enumerate(LABELS):
        order = np.argsort(clf.coef_[i])[::-1][:8]
        lex += [{"class": c, "term": feats[j], "coefficient": float(clf.coef_[i][j])} for j in order]
    lex = pd.DataFrame(lex)

    # ---- out-of-template probe (qualitative)
    probe = pd.DataFrame(OOD_PROBE, columns=["text", "label"])
    probe["TF-IDF+LR (trained on supplied data)"] = final.predict(probe.text)
    hyb = clone(models["Hybrid: TF-IDF + VADER features (LR)"]).fit(X, y)
    probe["Hybrid TF-IDF+VADER"] = hyb.predict(probe.text)
    probe["VADER lexicon"] = vader_label(vader_scores(probe.text).vader_compound)
    probe_acc = {c: float((probe[c] == probe.label).mean()) for c in probe.columns[2:]}

    # ---- metrics summary
    ch = comp[(comp.model == chosen)].set_index("regime")
    summary = {
        "chosen_model": chosen, "reference_label_is_rating_derived": True,
        "cv5": {k: float(ch.loc["cv5", k]) for k in ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "roc_auc_ovr_macro", "log_loss"]},
        "holdout": {k: float(ch.loc["holdout", k]) for k in ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "roc_auc_ovr_macro"]},
        "holdout_ci": holdout_ci[chosen], "holdout_n": int(len(yte)),
        "lodo": {k: float(ch.loc["lodo", k]) for k in ["accuracy", "macro_f1", "cohen_kappa"]},
        "vader": {"cv5_accuracy": float(comp[(comp.model.str.startswith("VADER")) & (comp.regime == "cv5")].accuracy.iloc[0]),
                  "cv5_macro_f1": float(comp[(comp.model.str.startswith("VADER")) & (comp.regime == "cv5")].macro_f1.iloc[0])},
        "majority_baseline_accuracy": float(comp[(comp.model.str.startswith("Baseline")) & (comp.regime == "cv5")].accuracy.iloc[0]),
        "hybrid": {"cv5_macro_f1": float(comp[(comp.model.str.startswith("Hybrid")) & (comp.regime == "cv5")].macro_f1.iloc[0]),
                   "lodo_macro_f1": float(comp[(comp.model.str.startswith("Hybrid")) & (comp.regime == "lodo")].macro_f1.iloc[0]),
                   "lodo_accuracy": float(comp[(comp.model.str.startswith("Hybrid")) & (comp.regime == "lodo")].accuracy.iloc[0])},
        "ablation": abl.set_index("input").accuracy.to_dict(),
        "ood_probe_accuracy": probe_acc, "ood_probe_n": int(len(probe)),
        "tuning_best": {"params": {"C": best_c, "ngram_range": str(best_ng)}, "n_configs": int(len(tuning)), "n_tied_on_macro_f1": int(len(elig)),
                        "cv_macro_f1": float(pick.cv_macro_f1_mean), "cv_log_loss": float(pick.cv_log_loss)},
        "vader_words_scored_zero": sorted(vd[(vd.vader_word_score == 0)].descriptor.tolist()),
        "vader_lowest_accuracy_words": vd.sort_values("vader_accuracy").head(8)[["descriptor", "true", "vader_accuracy"]].to_dict("records"),
        "n_oof_misclassified": int((scored.sent_pred != y.values).sum()),
        "agreement_pred_vs_reference": float((scored.sent_pred.values == y.values).mean()),
    }
    if save:
        comp.to_csv(C.TABLE_DIR / "sentiment_model_comparison.csv", index=False)
        rep.to_csv(C.TABLE_DIR / "sentiment_classification_report.csv", index=False)
        cm_chosen.to_csv(C.TABLE_DIR / "sentiment_confusion_chosen.csv")
        cm_vader.to_csv(C.TABLE_DIR / "sentiment_confusion_vader.csv")
        vd.to_csv(C.TABLE_DIR / "sentiment_vader_by_descriptor.csv", index=False)
        abl.to_csv(C.TABLE_DIR / "sentiment_ablation.csv", index=False)
        lex.to_csv(C.TABLE_DIR / "sentiment_learned_lexicon.csv", index=False)
        probe.to_csv(C.TABLE_DIR / "sentiment_ood_probe.csv", index=False)
        tuning.to_csv(C.TABLE_DIR / "sentiment_tuning.csv", index=False)
        scored.to_parquet(C.SILVER_DIR / "nlp_sentiment.parquet", index=False)
        save_metrics("sentiment", summary)
    log.info("Sentiment | chosen=%s | cv5 acc=%.4f macroF1=%.4f | LODO macroF1=%.3f | VADER acc=%.3f | probe acc LR=%.2f VADER=%.2f",
             chosen, summary["cv5"]["accuracy"], summary["cv5"]["macro_f1"], summary["lodo"]["macro_f1"], summary["vader"]["cv5_accuracy"],
             probe_acc["TF-IDF+LR (trained on supplied data)"], probe_acc["VADER lexicon"])
    return {"summary": summary, "comparison": comp, "scored": scored}


def predict_sentiment(texts, model_path=C.MODEL_DIR / "sentiment_model.joblib") -> pd.DataFrame:
    """Score new free text with the saved production model (used by the dashboard 'try it' panel)."""
    m = joblib.load(model_path)
    texts = list(texts)
    P = m.predict_proba(texts)
    return pd.DataFrame({"text": texts, "sentiment": np.array(LABELS)[P.argmax(1)], "p_negative": P[:, 0], "p_neutral": P[:, 1], "p_positive": P[:, 2]})


if __name__ == "__main__":  # pragma: no cover
    run()
