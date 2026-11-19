"""NLP pre-processing, sentiment, topics, data mining and reproducibility tests."""
import numpy as np
import pandas as pd
import pytest

from cvi import config as C
from cvi import mining, nlp_prep, sentiment, topics
from cvi.utils import load_metrics


# --------------------------------------------------------------- pre-processing
@pytest.mark.case("T-NLP-01", "NLP pre-processing", "\"The delivery wasn't good, and the courier didn't call!\"", "Contractions expanded and negation scope marked: tokens 'not', 'NOT_good', 'NOT_call'")
def test_negation(actual):
    toks = nlp_prep.TextPreprocessor().tokens("The delivery wasn't good, and the courier didn't call!")
    actual(toks)
    assert "NOT_good" in toks and "NOT_call" in toks and "good" not in toks


@pytest.mark.case("T-NLP-02", "NLP pre-processing", "Text with HTML, URL, e-mail, @mention and #hashtag", "All noise removed; only content words remain")
def test_noise_removal(actual):
    toks = nlp_prep.TextPreprocessor().tokens("Great <b>service</b> see https://x.co mail me a@b.com @brand #love")
    actual(toks)
    assert toks == ["great", "service", "see", "mail"] or set(toks) == {"great", "service", "see", "mail"}


@pytest.mark.case("T-NLP-03", "NLP pre-processing", "Stop-word removal on 'This is not the best' and 'It is very good'; negated intensifier in 'not very good'", "Function words dropped; negator 'not' retained; intensifier 'very' retained; inside a negation scope it becomes NOT_very")
def test_stopwords(actual):
    pre = nlp_prep.TextPreprocessor()
    a, b, c = pre.tokens("This is not the best"), pre.tokens("It is very good"), pre.tokens("It is not very good")
    actual(f"{a} | {b} | {c}")
    assert "not" in a and "NOT_best" in a and not {"this", "is", "the"} & set(a)
    assert b == ["very", "good"] and c == ["not", "NOT_very", "NOT_good"]


@pytest.mark.case("T-NLP-04", "NLP pre-processing", "Lemmatisation of 'delays delayed refunds packaging'", "Consolidated to 'delay', 'delay', 'refund', 'package'")
def test_lemma(actual):
    toks = nlp_prep.TextPreprocessor().tokens("delays delayed refunds packaging")
    actual(toks)
    assert toks[:3] == ["delay", "delay", "refund"] and "package" in toks


@pytest.mark.case("T-NLP-05", "NLP pre-processing", "BoilerplateFilter fitted on a training corpus then applied to unseen text", "Learns only tokens with document frequency >= 90% from the TRAINING data; unseen text is filtered with the stored list (no leakage)")
def test_boilerplate_no_leak(actual):
    train = ["order was good", "order was bad", "order was slow", "order was fine", "order was ok", "order was nice", "order was late", "order was fast", "order was poor", "delivery order was great"]
    f = nlp_prep.BoilerplateFilter(0.9).fit(train)
    out = f.transform(["the order was awful delivery"])
    actual(f"stop_={sorted(f.stop_)}; transformed={out}")
    assert f.stop_ == frozenset({"order", "was"}) and out == ["the awful delivery"]


@pytest.mark.case("T-NLP-06", "NLP pre-processing", "Corpus statistics on the supplied text", "Vocabulary of 44 normalised tokens; exactly 6 boilerplate terms with DF >= 90% (affect, channel, experience, order, overall, use)")
def test_corpus_stats(silver, actual):
    st = nlp_prep.corpus_stats(nlp_prep.TextPreprocessor(return_tokens=True).transform(silver.text_clean))
    actual(f"vocab={st['vocabulary_size']}; boilerplate={st['boilerplate_terms_df_ge_90pct']}")
    assert st["vocabulary_size"] == 44 and st["boilerplate_terms_df_ge_90pct"] == ["affect", "channel", "experience", "order", "overall", "use"]


# --------------------------------------------------------------------- sentiment
@pytest.mark.case("T-SEN-01", "Sentiment", "Chosen model, 5-fold CV and 20% hold-out vs rating-implied reference labels", "Accuracy and macro-F1 >= 0.99 in-distribution (documented as a templated-text ceiling effect)")
def test_in_distribution(actual):
    m = load_metrics("sentiment")
    actual(f"CV acc={m['cv5']['accuracy']:.4f} macroF1={m['cv5']['macro_f1']:.4f}; hold-out acc={m['holdout']['accuracy']:.4f} (n={m['holdout_n']})")
    assert m["cv5"]["accuracy"] >= 0.99 and m["cv5"]["macro_f1"] >= 0.99 and m["holdout"]["accuracy"] >= 0.99


@pytest.mark.case("T-SEN-02", "Sentiment", "Baselines: majority class and general-purpose VADER", "Majority accuracy 57.1%; VADER well below the trained model (60.6%, < 70%) and 8 of 19 opinion words score exactly 0")
def test_baselines(actual):
    m = load_metrics("sentiment")
    actual(f"majority={m['majority_baseline_accuracy']:.4f}; VADER={m['vader']['cv5_accuracy']:.4f}; zero-scored words={len(m['vader_words_scored_zero'])}")
    assert abs(m["majority_baseline_accuracy"] - 0.5708) < 0.001 and 0.55 < m["vader"]["cv5_accuracy"] < 0.70 and len(m["vader_words_scored_zero"]) == 8


@pytest.mark.case("T-SEN-03", "Sentiment", "Leave-one-descriptor-out validation (opinion word unseen in training)", "Documented limitation: accuracy collapses to the majority baseline (57.1%) - the model has no mechanism for unseen vocabulary")
def test_lodo_limitation(actual):
    m = load_metrics("sentiment")
    actual(f"LODO accuracy={m['lodo']['accuracy']:.4f} kappa={m['lodo']['cohen_kappa']:.3f}")
    assert abs(m["lodo"]["accuracy"] - m["majority_baseline_accuracy"]) < 0.01 and m["lodo"]["cohen_kappa"] < 0.05


@pytest.mark.case("T-SEN-04", "Sentiment", "Leakage ablation: mask the sentiment word, keep topic / channel / product words", "Accuracy near chance (< 0.5): the model reads the opinion, not the topic")
def test_ablation(actual):
    a = load_metrics("sentiment")["ablation"]
    key = [k for k in a if k.startswith("Descriptor word MASKED")][0]
    actual(f"masked accuracy={a[key]:.3f}; descriptor-only={a['Descriptor word ONLY']:.3f}")
    assert a[key] < 0.5 and a["Descriptor word ONLY"] >= 0.99


@pytest.mark.case("T-SEN-05", "Sentiment", "Saved production model scores three templated sentences", "Negative / Neutral / Positive respectively; probabilities in [0,1] and sum to 1")
def test_model_smoke(actual):
    texts = ["The delivery delay experience was slow. I used the app channel for my home order and this affected my overall experience.",
             "The pricing experience was average. I used the web channel for my beauty order and this affected my overall experience.",
             "The packaging experience was excellent. I used the store channel for my grocery order and this affected my overall experience."]
    out = sentiment.predict_sentiment(texts)
    actual(out.sentiment.tolist())
    assert out.sentiment.tolist() == ["Negative", "Neutral", "Positive"]
    P = out[["p_negative", "p_neutral", "p_positive"]].values
    assert ((P >= 0) & (P <= 1)).all() and np.allclose(P.sum(1), 1)


@pytest.mark.case("T-SEN-06", "Sentiment", "VADER labelling thresholds at +/-0.05", "compound 0.05 -> Positive, -0.05 -> Negative, 0.049 -> Neutral")
def test_vader_thresholds(actual):
    lab = sentiment.vader_label([0.05, -0.05, 0.049, -0.049, 0.9]).tolist()
    actual(lab)
    assert lab == ["Positive", "Negative", "Neutral", "Neutral", "Positive"]


@pytest.mark.case("T-SEN-07", "Sentiment", "Out-of-fold predictions used downstream", "2,400 scored rows; agreement with reference 100%; no missing scores")
def test_oof(analytics, silver, actual):
    agree = (analytics.sent_pred == silver.reference_sentiment).mean()
    actual(f"rows={len(analytics)}; agreement={agree:.4f}; nulls={int(analytics[['sent_pred', 'p_negative']].isna().sum().sum())}")
    assert len(analytics) == 2400 and agree == 1.0 and analytics[["sent_pred", "p_negative", "p_neutral", "p_positive"]].notna().all().all()


# ------------------------------------------------------------------------ topics
@pytest.mark.case("T-TOP-01", "Topics", "Keyword-taxonomy classifier vs the known Issue_Category", "Accuracy >= 99% with 0 unclassified / multi-topic records")
def test_taxonomy(actual):
    m = load_metrics("topics")
    actual(f"accuracy={m['taxonomy_accuracy_vs_issue']:.4f}; unclassified={m['taxonomy_unclassified']}; multiple={m['taxonomy_multiple']}")
    assert m["taxonomy_accuracy_vs_issue"] >= 0.99 and m["taxonomy_unclassified"] == 0 and m["taxonomy_multiple"] == 0


@pytest.mark.case("T-TOP-02", "Topics", "Taxonomy on hand-written sentences", "'The refund is still pending' -> Refund Delay; 'courier delivered late' -> Delivery Delay; 'checkout keeps crashing in the app' -> App Experience")
def test_taxonomy_free_text(actual):
    out = [topics.classify_theme(t, clause_only=False) for t in ["The refund is still pending", "The courier delivered it very late", "Checkout keeps crashing in the app"]]
    actual(out)
    assert out == ["Refund Delay", "Delivery Delay", "App Experience"]


@pytest.mark.case("T-TOP-03", "Topics", "Aspect-opinion separation: terms associated with sentiment (chi-square + FDR + Cramer's V >= 0.10)", "Exactly the 19 opinion words are removed; no aspect word (delivery, refund, price, support ...) is removed")
def test_opinion_words(actual):
    m = load_metrics("topics")
    removed = set(m["opinion_words_removed"])
    aspect = {"delivery", "delay", "refund", "price", "support", "customer", "app", "package", "product", "availability", "quality"}
    actual(f"{m['n_opinion_words']} removed; aspect words removed={sorted(aspect & removed)}")
    assert m["n_opinion_words"] == 19 and not (aspect & removed)


@pytest.mark.case("T-TOP-04", "Topics", "NMF on the aspect view (k selected by internal criteria only)", "k = 8, no empty topic, NMI and purity vs Issue_Category = 1.0; beats LDA (NMI < 1)")
def test_nmf(actual):
    m = load_metrics("topics")
    actual(f"k={m['nmf']['k']}; NMI={m['nmf']['nmi_vs_issue']:.3f}; purity={m['nmf']['purity_vs_issue']:.3f}; LDA NMI={m['lda']['nmi_vs_issue']:.3f}")
    assert m["nmf"]["k"] == 8 and m["nmf"]["nmi_vs_issue"] > 0.99 and m["nmf"]["purity_vs_issue"] > 0.99 and m["lda"]["nmi_vs_issue"] < m["nmf"]["nmi_vs_issue"]
    t = pd.read_csv(C.TABLE_DIR / "topics_nmf_topics.csv")
    assert (t.documents > 0).all() and abs(t.share.sum() - 1) < 1e-9


@pytest.mark.case("T-TOP-05", "Topics", "Theme x business segment independence (channel, region, product, source)", "No theme mix differs significantly after FDR (min q > 0.05)")
def test_theme_independence(actual):
    t = pd.read_csv(C.TABLE_DIR / "topics_composition_tests.csv")
    actual(f"min q={t.q_bh.min():.3f}")
    assert t.q_bh.min() > 0.05


# ------------------------------------------------------------------------ mining
@pytest.mark.case("T-MIN-01", "Data mining", "K-means segmentation selection vs column-permutation null", "k = 3; observed silhouette above the null 95th percentile; bootstrap ARI >= 0.9")
def test_clusters(actual):
    m = load_metrics("mining")["segmentation"]
    actual(f"k={m['k']}; silhouette={m['silhouette']:.3f} vs null p95={m['null_silhouette_p95']:.3f}; ARI={m['stability']['mean_ari']:.3f}")
    assert m["k"] == 3 and m["silhouette"] > m["null_silhouette_p95"] and m["stability"]["mean_ari"] >= 0.9


@pytest.mark.case("T-MIN-02", "Data mining", "Same null test on the value space (revenue x orders: uniform noise)", "Silhouette does NOT beat the shuffled-column null - no cluster structure, so value is tiered by rule")
def test_value_space_null(actual):
    m = load_metrics("mining")["segmentation"]
    actual(f"beats null={m['value_space_beats_null']}; silhouette(k=4)={m['value_space_silhouette_k4']:.3f} vs null mean {m['value_space_null_k4']:.3f}")
    assert m["value_space_beats_null"] is False and abs(m["value_space_silhouette_k4"] - m["value_space_null_k4"]) < 0.01


@pytest.mark.case("T-MIN-03", "Data mining", "Segment assignment completeness", "2,400 assignments, no nulls; risk tiers within {Critical, High, Opportunity, Maintain}; Critical tier = At-Risk in revenue Q3/Q4")
def test_segment_assignment(analytics, actual):
    ok_tiers = set(analytics.risk_tier.unique()) <= {"Critical", "High", "Opportunity", "Maintain"}
    crit = analytics[analytics.risk_tier == "Critical"]
    actual(f"rows={analytics.segment_name.notna().sum()}; tiers={analytics.risk_tier.value_counts().to_dict()}")
    assert analytics.segment_name.notna().all() and ok_tiers and (crit.segment_name == "At-Risk").all() and crit.revenue_tier.isin(["Q3 Mid-High", "Q4 Premium"]).all()


@pytest.mark.case("T-MIN-04", "Data mining", "Association rules: statistical funnel (lift >= 1.2 -> Fisher + BH-FDR -> productivity)", "Robust rules all have lift >= 1.2 and q < 0.05; none contain issue / channel / region / product / source")
def test_rules(actual):
    r = pd.read_csv(C.TABLE_DIR / "mining_association_rules_robust.csv")
    f = load_metrics("mining")["association_rules"]["funnel"]
    ctx = r.antecedent.str.contains("Issue=|Channel=|Region=|Product=|Source=")
    actual(f"{f['rules_lift_ge_1.2']} candidates -> {len(r)} robust; context items in robust rules={int(ctx.sum())}")
    assert len(r) >= 5 and (r.lift >= C.RULE_MIN_LIFT).all() and (r.q_final_bh < 0.05).all() and not ctx.any()


@pytest.mark.case("T-MIN-05", "Data mining", "Re-run K-means selection twice with the same seed", "Identical silhouette values (deterministic)")
def test_mining_determinism(analytics, actual):
    X = mining._scale(analytics, C.CLUSTER_FEATURES)
    a = mining.select_k(X, ks=[3], n_null=4)
    b = mining.select_k(X, ks=[3], n_null=4)
    actual(f"{a.silhouette.iloc[0]:.6f} == {b.silhouette.iloc[0]:.6f}")
    assert a.silhouette.iloc[0] == b.silhouette.iloc[0] and a.null_silhouette_mean.iloc[0] == b.null_silhouette_mean.iloc[0]


# ------------------------------------------------------------------ reproducibility
@pytest.mark.case("T-REP-01", "Reproducibility", "Run manifest and environment check", "Manifest records the input SHA-256 (matches the file), seed 42, Python and package versions, stage timings; the environment check flags a missing package and a version mismatch against a pin file and ignores unpinned lines")
def test_manifest(actual, tmp_path):
    import json
    from cvi.utils import environment_issues, sha256_file
    man = json.load(open(C.OUT_DIR / "run_manifest.json"))
    pins = tmp_path / "pins.txt"
    pins.write_text("# comment\npandas==0.0.1\nnot-a-real-package-xyz==1.0\nnumpy\n", encoding="utf-8")
    found = environment_issues(pins)
    actual(f"sha256 {man['input']['sha256'][:12]}...; python {man['environment']['python']}; stages={len(man['stage_seconds'])}; "
           f"environment check on a fake pin file -> {len(found)} issues: {found}")
    assert man["input"]["sha256"] == sha256_file(C.RAW_XLSX) and man["seed"] == 42 and "pandas" in man["environment"]["packages"] and len(man["stage_seconds"]) >= 5
    assert len(found) == 2 and any("pandas" in f and "pinned 0.0.1" in f for f in found) and any("not-a-real-package-xyz is not installed" in f for f in found)
    assert environment_issues(pins, only={"numpy"}) == []           # an unpinned line is ignored and `only` filters


@pytest.mark.case("T-REP-02", "Reproducibility", "Re-run the exploratory statistics twice", "Identical hot-spot scan table (fixed seeds, deterministic tests)")
def test_eda_determinism(silver, actual):
    from cvi import eda
    a, _ = eda.hotspot_scan(silver, "reference_sentiment")
    b, _ = eda.hotspot_scan(silver, "reference_sentiment")
    actual(f"{len(a)} cells; identical={a.equals(b)}")
    assert a.equals(b)
