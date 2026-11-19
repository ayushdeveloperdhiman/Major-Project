"""Stage 7 - NLP pre-processing.

A single, configurable, sklearn-compatible transformer so that the *same* text
workflow runs in training, evaluation, the batch pipeline and the dashboard.

Pipeline (each step: what it does -> why it is needed)
------------------------------------------------------
1  Unicode NFKC + whitespace collapse   -> one canonical character representation
2  Strip HTML / URLs / e-mails / @ #    -> channel noise (reviews, social) with no sentiment content
3  Lowercase                            -> 'Slow' and 'slow' become the same feature (less sparsity)
4  Contraction expansion (n't -> not)   -> makes negation explicit ("wasn't good" -> "was not good")
5  Collapse repeated adjacent words     -> repairs the source template glitch 'experience experience'
6  Clause split + tokenise (letters)    -> negation scope is bounded by clause punctuation; digits/punctuation dropped
7  Negation marking (NOT_ prefix, 3 tokens) -> "not good" must not look like "good"
8  Stop-word removal (negators kept)    -> removes function words, keeps polarity-bearing 'not/no/never/very'
9  Lemmatisation (WordNet)              -> 'delays/delayed' -> 'delay' (vocabulary consolidation)
10 Data-driven boilerplate removal      -> tokens in >=90% of documents (max_df) carry no information

Steps 1-9 live in ``TextPreprocessor``; step 10 is ``BoilerplateFilter`` (fit on training data only -> no leakage).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C
from .utils import get_logger

log = get_logger("cvi.nlp")

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MENTION = re.compile(r"[@#]\w+")
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)
_CLAUSE_SPLIT = re.compile(r"[.!?;,:()\[\]\n]+")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CONTRACTIONS = [
    (re.compile(r"\bcan't\b"), "can not"), (re.compile(r"\bwon't\b"), "will not"), (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"), (re.compile(r"'re\b"), " are"), (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"), (re.compile(r"'m\b"), " am"), (re.compile(r"'d\b"), " would"),
]
KEEP_REPEATS = {"very", "really", "so", "no", "not", "never", "too", "much", "more", "most", "had", "that"}
KEEP_WORDS = set(C.NEGATION_WORDS) | {"very", "too", "more", "most", "only", "against", "few", "less"}

STEP_DOC = [
    ("1", "Unicode NFKC + whitespace collapse", "Canonical character representation"),
    ("2", "Strip HTML, URLs, e-mails, @mentions/#hashtags", "Channel noise (reviews, social) carries no sentiment"),
    ("3", "Lowercase", "Merge case variants; reduce sparsity"),
    ("4", "Expand contractions (n't -> not)", "Make negation explicit for step 7"),
    ("5", "Collapse repeated adjacent words (intensifiers/negators excepted)", "Repair template glitch 'experience experience'"),
    ("6", "Clause split + regex tokenisation (letters/apostrophes only)", "Negation scope bounded by punctuation; drop digits/punctuation"),
    ("7", "Negation marking: NOT_ prefix on next 3 tokens in the clause", "'not good' must not be confused with 'good'"),
    ("8", "Stop-word removal (NLTK English; negators/intensifiers kept)", "Remove function words, keep polarity cues"),
    ("9", "WordNet lemmatisation (noun, then verb)", "Consolidate inflections (delays/delayed -> delay)"),
    ("10", "Boilerplate removal by document frequency (BoilerplateFilter, max_df=0.90, before n-grams)", "Terms present in >=90% of texts have no discriminative value; learned on training data only"),
]

_resources = {"stop": None, "lem": None, "ready": False}


def _load_resources() -> None:
    """Load NLTK resources from the project-local folder; degrade gracefully (documented) when unavailable."""
    if _resources["ready"]:
        return
    import nltk
    nltk.data.path.insert(0, str(C.NLTK_DIR))
    try:
        from nltk.corpus import stopwords
        _resources["stop"] = set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", download_dir=str(C.NLTK_DIR), quiet=True)
            from nltk.corpus import stopwords
            _resources["stop"] = set(stopwords.words("english"))
        except Exception:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            _resources["stop"] = set(ENGLISH_STOP_WORDS)
            log.warning("NLTK stopwords unavailable - falling back to scikit-learn's list")
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")      # triggers the corpus load
        _resources["lem"] = lem
    except LookupError:
        try:
            import nltk
            nltk.download("wordnet", download_dir=str(C.NLTK_DIR), quiet=True)
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            lem.lemmatize("tests")
            _resources["lem"] = lem
        except Exception:
            _resources["lem"] = None
            log.warning("WordNet unavailable - lemmatisation disabled")
    _resources["ready"] = True


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Stateless text -> normalised-token-string transformer (sklearn compatible, picklable)."""

    def __init__(self, lowercase=True, strip_noise=True, expand_contractions=True, collapse_repeats=True,
                 negation=True, negation_window=3, remove_stopwords=True, lemmatize=True, min_token_len=2,
                 return_tokens=False):
        self.lowercase = lowercase
        self.strip_noise = strip_noise
        self.expand_contractions = expand_contractions
        self.collapse_repeats = collapse_repeats
        self.negation = negation
        self.negation_window = negation_window
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_len = min_token_len
        self.return_tokens = return_tokens

    # -- single document
    def tokens(self, text: str) -> list[str]:
        _load_resources()
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return []
        t = unicodedata.normalize("NFKC", str(text))                         # 1
        if self.strip_noise:                                                 # 2
            t = _HTML.sub(" ", t); t = _URL.sub(" ", t); t = _EMAIL.sub(" ", t); t = _MENTION.sub(" ", t)
        if self.lowercase:                                                   # 3
            t = t.lower()
        t = t.replace("’", "'").replace("`", "'")
        if self.expand_contractions:                                         # 4
            for pat, rep in _CONTRACTIONS:
                t = pat.sub(rep, t)
        if self.collapse_repeats:                                            # 5
            t = _REPEAT.sub(lambda m: m.group(0) if m.group(1).lower() in KEEP_REPEATS else m.group(1), t)
        stop = _resources["stop"] - KEEP_WORDS if self.remove_stopwords else set()
        lem = _resources["lem"] if self.lemmatize else None
        out: list[str] = []
        for clause in _CLAUSE_SPLIT.split(t):                                # 6
            toks = _TOKEN.findall(clause)
            neg_left = 0
            for w in toks:
                is_neg = w in C.NEGATION_WORDS
                if len(w) < self.min_token_len and not is_neg:
                    continue
                if w in stop:                                                # 8
                    continue
                if lem is not None and not is_neg:                           # 9
                    w = lem.lemmatize(lem.lemmatize(w, "n"), "v")
                if self.negation and is_neg:                                 # 7
                    neg_left = self.negation_window
                    out.append(w)
                    continue
                if self.negation and neg_left > 0:
                    out.append("NOT_" + w)
                    neg_left -= 1
                else:
                    out.append(w)
        return out

    # -- sklearn API
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        seq = X.tolist() if hasattr(X, "tolist") else list(X)
        toks = [self.tokens(x) for x in seq]
        return toks if self.return_tokens else [" ".join(t) for t in toks]

    def __getstate__(self):               # NLTK objects are re-loaded lazily; keep pickles small and portable
        return self.__dict__.copy()


class BoilerplateFilter(BaseEstimator, TransformerMixin):
    """Step 10 - drop tokens that occur in >= ``max_df`` of the TRAINING documents (no discriminative value).

    Done on the token stream *before* n-grams are formed, so no junk bigrams such as 'experience poor' arise.
    Learned in ``fit`` (training data only -> no leakage) and applied identically in ``transform``.
    """

    def __init__(self, max_df: float = C.BOILERPLATE_DF_THRESHOLD):
        self.max_df = max_df

    def fit(self, X, y=None):
        docs = [set(str(x).split()) for x in X]
        n = max(len(docs), 1)
        counts = Counter(t for d in docs for t in d)
        self.stop_ = frozenset(t for t, c in counts.items() if c / n >= self.max_df)
        return self

    def transform(self, X):
        return [" ".join(t for t in str(x).split() if t not in self.stop_) for x in X]


def describe_steps() -> pd.DataFrame:
    return pd.DataFrame(STEP_DOC, columns=["step", "operation", "why_it_is_required"])


def corpus_stats(token_lists: list[list[str]], df_threshold: float = C.BOILERPLATE_DF_THRESHOLD) -> dict:
    n = len(token_lists)
    tf = Counter(t for toks in token_lists for t in toks)
    dfc = Counter(t for toks in token_lists for t in set(toks))
    boiler = sorted([t for t, c in dfc.items() if c / n >= df_threshold])
    return {
        "documents": n, "tokens_total": int(sum(tf.values())), "vocabulary_size": len(tf),
        "avg_tokens_per_doc": round(sum(tf.values()) / n, 2), "hapax_terms": int(sum(1 for c in tf.values() if c == 1)),
        "boilerplate_terms_df_ge_90pct": boiler,
        "top_terms": tf.most_common(15),
    }


def before_after_examples(texts: list[str], pre: TextPreprocessor | None = None) -> pd.DataFrame:
    pre = pre or TextPreprocessor()
    return pd.DataFrame({"raw_text": texts, "processed_tokens": [" ".join(pre.tokens(t)) for t in texts]})
