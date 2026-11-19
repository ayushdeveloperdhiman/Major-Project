"""Stage 2 - Data-quality rule engine.

A small declarative engine: each :class:`Rule` returns a boolean mask of
*violating* rows.  Severity drives the policy:

* ``error`` - row is quarantined (cannot be trusted for analytics)
* ``warn``  - row is kept but flagged in ``dq_flags`` (and fixed where a safe
              deterministic fix exists, see :mod:`cvi.transform`)
* ``info``  - monitoring signal only

The same rule set is executed on the raw (Bronze) data to quantify problems and
again on the cleaned (Silver) data to prove that the fixes worked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config as C
from .utils import get_logger

log = get_logger("cvi.quality")

TEMPLATE_RE = re.compile(
    r"^The (?P<issue>.+?) experience was (?P<descriptor>[\w\- ]+?)\. I used the (?P<channel>\w+) channel "
    r"for my (?P<product>\w+) order and this affected my overall experience\.$", re.I)
REPEATED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.I)

NUMERIC_COLS = ["Rating", "Customer_Revenue_INR", "Order_Count", "Repeat_Purchase_Flag",
                "Resolution_Time_Hours", "CSAT_Score"]
CATEGORICAL_COLS = list(C.ALLOWED_VALUES.keys())
STRING_COLS = ["Feedback_ID", "Customer_ID", "Feedback_Text"] + CATEGORICAL_COLS


@dataclass
class Rule:
    rule_id: str
    dimension: str
    severity: str               # error | warn | info
    description: str
    check: Callable[[pd.DataFrame], pd.Series]   # -> True where the row VIOLATES the rule
    applies_to: str = "row"


# ------------------------------------------------------------------ helpers #
def typed_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numerics coerced (unparseable -> NaN) and strings left untouched."""
    v = df.copy()
    for c in NUMERIC_COLS:
        if c in v.columns:
            v[c + "__raw"] = v[c]
            v[c] = pd.to_numeric(v[c], errors="coerce")
    return v


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype("string").str.strip() == "")


def _in_domain(s: pd.Series, allowed: list[str]) -> pd.Series:
    return s.astype("string").isin(allowed)


UNKNOWN = "Unknown"   # documented placeholder for blank categoricals (see transform.py)


def _norm_in_domain(s: pd.Series, allowed: list[str]) -> pd.Series:
    """True when the value is blank (handled by DQ-C02), the 'Unknown' placeholder, or a case/whitespace variant of an allowed value."""
    cf = {a.casefold() for a in allowed} | {UNKNOWN.casefold()}
    z = s.astype("string").str.strip().str.casefold()
    return z.isin(list(cf)) | s.isna() | (z == "")


def parse_template(text: pd.Series) -> pd.DataFrame:
    """Extract issue/descriptor/channel/product from the templated sentence (NaN when not parseable)."""
    ext = text.astype("string").str.extract(TEMPLATE_RE)
    return ext


# ------------------------------------------------------------------- rules #
def build_rules() -> list[Rule]:
    R: list[Rule] = []
    add = lambda *a, **k: R.append(Rule(*a, **k))  # noqa: E731

    # ---- Completeness
    add("DQ-C01", "Completeness", "error", "Key fields Feedback_ID and Customer_ID are not blank",
        lambda d: _blank(d["Feedback_ID"]) | _blank(d["Customer_ID"]))
    add("DQ-C02", "Completeness", "warn", "Categorical fields (Region, Channel, Product, Issue, Source, Sentiment) are not blank (blank -> 'Unknown')",
        lambda d: pd.concat([_blank(d[c]) for c in CATEGORICAL_COLS], axis=1).any(axis=1))
    add("DQ-C03", "Completeness", "warn", "Numeric fields are present and parseable (missing kept as NaN, never imputed)",
        lambda d: pd.concat([d[c].isna() for c in NUMERIC_COLS], axis=1).any(axis=1))
    add("DQ-C04", "Completeness", "error", f"Feedback_Text is present and has at least {C.MIN_TEXT_CHARS} characters",
        lambda d: _blank(d["Feedback_Text"]) | (d["Feedback_Text"].astype("string").str.strip().str.len() < C.MIN_TEXT_CHARS))

    # ---- Uniqueness
    add("DQ-U01", "Uniqueness", "error", "Feedback_ID is unique (duplicates after the first occurrence are quarantined)",
        lambda d: d["Feedback_ID"].duplicated(keep="first"))
    add("DQ-U02", "Uniqueness", "warn", "No exact duplicate record (all fields identical apart from Feedback_ID)",
        lambda d: d.drop(columns=[c for c in d.columns if c.startswith("_") or c.endswith("__raw") or c == "Feedback_ID"]).duplicated(keep="first"))
    add("DQ-U03", "Uniqueness", "info", "Same customer + identical text (informational: templated text repeats legitimately)",
        lambda d: d.duplicated(["Customer_ID", "Feedback_Text"], keep="first"))

    # ---- Validity
    add("DQ-V01", "Validity", "error", "Feedback_ID matches pattern FB-NNNNN",
        lambda d: ~d["Feedback_ID"].astype("string").str.strip().str.fullmatch(C.ID_PATTERNS["Feedback_ID"]).fillna(False))
    add("DQ-V02", "Validity", "error", "Customer_ID matches pattern C-NNNNN",
        lambda d: ~d["Customer_ID"].astype("string").str.strip().str.fullmatch(C.ID_PATTERNS["Customer_ID"]).fillna(False))
    for col, allowed in C.ALLOWED_VALUES.items():
        add(f"DQ-V03-{col}", "Validity", "error", f"{col} belongs to the allowed domain {allowed} (after trim/case normalisation)",
            (lambda c, a: lambda d: ~_norm_in_domain(d[c], a))(col, allowed))
    add("DQ-V04", "Validity", "error", "Rating is an integer 1-5",
        lambda d: d["Rating"].notna() & ~(d["Rating"].between(1, 5) & (d["Rating"] == d["Rating"].round())))
    add("DQ-V05", "Validity", "error", "CSAT_Score lies in [1, 5]",
        lambda d: d["CSAT_Score"].notna() & ~d["CSAT_Score"].between(*C.RANGES["CSAT_Score"]))
    add("DQ-V06", "Validity", "error", "Repeat_Purchase_Flag is 0 or 1",
        lambda d: d["Repeat_Purchase_Flag"].notna() & ~d["Repeat_Purchase_Flag"].isin([0, 1]))
    add("DQ-V07", "Validity", "error", "Order_Count is an integer >= 1",
        lambda d: d["Order_Count"].notna() & ~((d["Order_Count"] >= 1) & (d["Order_Count"] == d["Order_Count"].round())))
    add("DQ-V08", "Validity", "error", "Customer_Revenue_INR is positive",
        lambda d: d["Customer_Revenue_INR"].notna() & ~(d["Customer_Revenue_INR"] > 0))
    add("DQ-V09", "Validity", "error", f"Resolution_Time_Hours is within [0, {C.RANGES['Resolution_Time_Hours'][1]:.0f}]",
        lambda d: d["Resolution_Time_Hours"].notna() & ~d["Resolution_Time_Hours"].between(*C.RANGES["Resolution_Time_Hours"]))
    add("DQ-V10", "Validity", "warn", "No leading/trailing whitespace in string fields (auto-trimmed)",
        lambda d: pd.concat([(d[c].astype("string") != d[c].astype("string").str.strip()).fillna(False) for c in STRING_COLS], axis=1).any(axis=1))
    add("DQ-V11", "Validity", "warn", "Categorical values use canonical casing (case variants are auto-mapped)",
        lambda d: pd.concat([(_norm_in_domain(d[c], a) & ~_in_domain(d[c].astype("string").str.strip(), a) & ~_blank(d[c]) & (d[c].astype("string").str.strip() != UNKNOWN)) for c, a in C.ALLOWED_VALUES.items()], axis=1).any(axis=1))
    add("DQ-V12", "Validity", "warn", f"Resolution_Time_Hours <= {C.RESOLUTION_WARN_HOURS:.0f}h (plausibility)",
        lambda d: d["Resolution_Time_Hours"] > C.RESOLUTION_WARN_HOURS)

    # ---- Consistency (cross-field)
    add("DQ-K01", "Consistency", "warn", "Reference_Sentiment agrees with Rating (1-2 Negative, 3 Neutral, 4-5 Positive)",
        lambda d: d["Rating"].map(C.RATING_TO_SENTIMENT).astype("string") != d["Reference_Sentiment"].astype("string").str.strip().str.title())
    add("DQ-K02", "Consistency", "warn", "CSAT_Score is within 1.0 point of Rating",
        lambda d: (d["CSAT_Score"] - d["Rating"]).abs() > 1.0)

    def _text_vs_structure(d: pd.DataFrame) -> pd.Series:
        """For templated texts, the issue/channel/product named in the sentence must match the structured fields.

        Containment (not phrase equality) is used so the rule is robust to the cosmetic
        'experience experience' glitch and to its repaired form.  Non-templated free text
        is out of scope for this rule (cannot be asserted)."""
        t = d["Feedback_Text"].astype("string").fillna("").str.casefold()
        parsed = parse_template(d["Feedback_Text"])["issue"].notna()
        g = lambda c: d[c].astype("string").fillna("").str.strip().str.casefold()  # noqa: E731
        ok = pd.Series([(i in x) and (f"the {c} channel" in x) and (f"my {p} order" in x)
                        for x, i, c, p in zip(t, g("Issue_Category"), g("Channel"), g("Product_Category"))], index=d.index)
        return parsed & ~ok
    add("DQ-K03", "Consistency", "warn", "Issue/channel/product named in the text agree with the structured fields", _text_vs_structure)
    add("DQ-K04", "Consistency", "warn", "Repeat_Purchase_Flag=1 requires Order_Count>=2 (logical consistency)",
        lambda d: (d["Repeat_Purchase_Flag"] == 1) & (d["Order_Count"] == 1))
    add("DQ-K05", "Consistency", "warn", "Text has no repeated adjacent word (e.g. 'experience experience')",
        lambda d: d["Feedback_Text"].map(lambda t: bool(REPEATED_WORD_RE.search(t)) if isinstance(t, str) else False))

    # ---- Plausibility / monitoring
    def _iqr_outlier(col: str, k: float) -> Callable[[pd.DataFrame], pd.Series]:
        def f(d: pd.DataFrame) -> pd.Series:
            q1, q3 = d[col].quantile([0.25, 0.75])
            lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
            return (d[col] < lo) | (d[col] > hi)
        return f
    add("DQ-P01", "Plausibility", "info", "Customer_Revenue_INR has no extreme outliers (3xIQR fence)", _iqr_outlier("Customer_Revenue_INR", 3.0))
    add("DQ-P02", "Plausibility", "info", "Resolution_Time_Hours has no mild outliers (1.5xIQR fence)", _iqr_outlier("Resolution_Time_Hours", 1.5))
    add("DQ-P03", "Plausibility", "info", "Resolution_Time_Hours is not censored at a floor of 1.0h",
        lambda d: d["Resolution_Time_Hours"] == 1.0)
    return R


# ------------------------------------------------------ dataset-level checks #
def dataset_checks(raw: pd.DataFrame) -> pd.DataFrame:
    """Profile-level (non row-level) findings: schema, entity stability, text realism."""
    rows = []
    cols = list(raw.columns)
    missing = [c for c in C.EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in C.EXPECTED_COLUMNS and not c.startswith("_")]
    rows.append(("DQ-S01", "Schema", "error", "All 15 contract columns present, none unexpected",
                 len(missing) + len(extra), len(C.EXPECTED_COLUMNS), f"missing={missing} unexpected={extra}"))

    t = typed_view(raw)
    vc = t["Customer_ID"].value_counts()
    multi = t[t["Customer_ID"].isin(vc[vc > 1].index)]
    if len(multi):
        for col, rid in [("Region", "DQ-E01"), ("Customer_Revenue_INR", "DQ-E02")]:
            nun = multi.groupby("Customer_ID")[col].nunique()
            rows.append((rid, "Entity integrity", "warn",
                         f"{col} is stable across a customer's records (entity-resolution risk if not)",
                         int((nun > 1).sum()), int(len(nun)),
                         f"{(nun > 1).mean():.1%} of multi-record customers have >1 distinct {col}"))
    p = parse_template(t["Feedback_Text"])
    share = float(p["issue"].notna().mean())
    rows.append(("DQ-X01", "Text realism", "warn", "Feedback_Text is free-form (not >95% one fixed sentence template)",
                 int(p["issue"].notna().sum()) if share > 0.95 else 0, len(t),
                 f"{share:.1%} of texts follow a single fixed template; sentiment is carried by one descriptor word "
                 f"({p['descriptor'].str.casefold().nunique()} distinct descriptors)"))
    out = pd.DataFrame(rows, columns=["rule_id", "dimension", "severity", "description", "rows_failed", "rows_checked", "detail"])
    out["scope"] = "dataset"
    return out


# ------------------------------------------------------------------ runner #
def run_rules(raw: pd.DataFrame, rules: list[Rule] | None = None) -> dict:
    """Execute all rules; return report, per-row flags, error mask and a scorecard."""
    rules = rules or build_rules()
    d = typed_view(raw)
    n = len(d)
    masks: dict[str, pd.Series] = {}
    report_rows = []
    for r in rules:
        try:
            m = r.check(d).fillna(False).astype(bool)
        except Exception as e:  # a broken rule must fail loudly, not silently pass
            raise RuntimeError(f"Rule {r.rule_id} failed to execute: {e}") from e
        masks[r.rule_id] = m
        failed = int(m.sum())
        status = "PASS" if failed == 0 else {"error": "FAIL", "warn": "WARN", "info": "INFO"}[r.severity]
        report_rows.append({"rule_id": r.rule_id, "scope": "row", "dimension": r.dimension, "severity": r.severity,
                            "description": r.description, "rows_checked": n, "rows_failed": failed,
                            "pct_failed": round(100 * failed / max(n, 1), 3), "status": status, "detail": ""})
    ds = dataset_checks(raw)
    for _, x in ds.iterrows():
        status = "PASS" if x.rows_failed == 0 else {"error": "FAIL", "warn": "WARN", "info": "INFO"}[x.severity]
        report_rows.append({"rule_id": x.rule_id, "scope": "dataset", "dimension": x.dimension, "severity": x.severity,
                            "description": x.description, "rows_checked": int(x.rows_checked), "rows_failed": int(x.rows_failed),
                            "pct_failed": round(100 * x.rows_failed / max(x.rows_checked, 1), 3), "status": status, "detail": x.detail})
    report = pd.DataFrame(report_rows)

    sev = {r.rule_id: r.severity for r in rules}
    err_ids = [k for k, s in sev.items() if s == "error"]
    warn_ids = [k for k, s in sev.items() if s == "warn"]
    error_mask = pd.concat([masks[k] for k in err_ids], axis=1).any(axis=1) if err_ids else pd.Series(False, index=d.index)
    flag_ids = err_ids + warn_ids
    flags = pd.Series([";".join(k for k in flag_ids if masks[k].iat[i]) for i in range(n)], index=d.index, name="dq_flags")
    err_reason = pd.Series([";".join(k for k in err_ids if masks[k].iat[i]) for i in range(n)], index=d.index, name="dq_error_reason")

    dim = {}
    for dname in sorted({r.dimension for r in rules}):
        ids = [r.rule_id for r in rules if r.dimension == dname and r.severity in ("error", "warn")]
        if ids:
            dim[dname] = float(1 - pd.concat([masks[k] for k in ids], axis=1).any(axis=1).mean())
    score = {
        "rows": n,
        "rows_failing_error_rules": int(error_mask.sum()),
        "pct_rows_passing_all_error_rules": round(float(100 * (1 - error_mask.mean())), 3),
        "rows_with_any_warning": int(((flags != "") & ~error_mask).sum()),
        "pct_rows_fully_clean": round(float(100 * (flags == "").mean()), 3),
        "dimension_pass_rate_pct": {k: round(100 * v, 2) for k, v in dim.items()},
        "n_rules": int(len(report)),
        "n_rules_pass": int((report.status == "PASS").sum()),
        "n_rules_fail": int((report.status == "FAIL").sum()),
        "n_rules_warn": int((report.status == "WARN").sum()),
        "n_rules_info": int((report.status == "INFO").sum()),
    }
    return {"report": report, "flags": flags, "error_reason": err_reason, "error_mask": error_mask,
            "masks": masks, "score": score}


def save_report(result: dict, stage: str) -> None:
    result["report"].to_csv(C.DQ_DIR / f"dq_report_{stage}.csv", index=False)
    from .utils import write_json
    write_json(result["score"], C.DQ_DIR / f"dq_scorecard_{stage}.json")
