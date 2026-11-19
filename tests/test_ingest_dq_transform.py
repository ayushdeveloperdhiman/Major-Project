"""Ingestion, data-quality (fault injection) and transformation tests."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from cvi import config as C
from cvi import ingest, quality, transform
from cvi.utils import sha256_file, sha256_frame


def _clean(df):
    return transform.clean(df)


# ------------------------------------------------------------------ ingestion
@pytest.mark.case("T-ING-01", "Ingestion", "Read sheet 'Customer Voice' from the client workbook", "2,400 rows x 15 columns")
def test_read_source(raw_excel, actual):
    actual(f"{raw_excel.shape[0]} rows x {raw_excel.shape[1]} columns")
    assert raw_excel.shape == (2400, 15)


@pytest.mark.case("T-ING-02", "Ingestion", "Compare source columns with the supplied data dictionary", "Same 15 fields, same order; no drift")
def test_schema_contract(raw_excel, actual):
    dd = pd.read_excel(C.DICTIONARY_XLSX, sheet_name="Data Dictionary")["Field"].tolist()
    drift = ingest.schema_drift(raw_excel)
    actual(f"dictionary match={dd == list(raw_excel.columns)}; drift={drift}")
    assert dd == list(raw_excel.columns) and not drift["missing_columns"] and not drift["unexpected_columns"] and drift["order_matches"]


@pytest.mark.case("T-ING-03", "Ingestion", "Bronze lineage columns", "5 lineage columns; row hash unique; batch id = first 12 hex of the file SHA-256")
def test_bronze_lineage(bronze, actual):
    batch = sha256_file(C.RAW_XLSX)[:12]
    actual(f"lineage={ingest.LINEAGE_COLS}; unique row hashes={bronze._row_hash.nunique()}; batch={bronze._batch_id.iloc[0]}")
    assert all(c in bronze.columns for c in ingest.LINEAGE_COLS)
    assert bronze._row_hash.nunique() == len(bronze) and (bronze._batch_id == batch).all()
    assert bronze._source_row.min() == 2 and bronze._source_row.max() == 2401        # Excel rows (header = row 1)


@pytest.mark.case("T-ING-04", "Ingestion", "Bronze keeps raw values (independent openpyxl read; numbers compared numerically, text exactly)", "All 15 fields stored as text; 2,400 rows; every cell equals the Excel value (only whole-number floats lose '.0': 4576.0 -> '4576')")
def test_bronze_raw_fidelity(bronze, actual):
    import openpyxl
    ws = openpyxl.load_workbook(C.RAW_XLSX, read_only=True, data_only=True)[C.RAW_SHEET]
    rows = list(ws.iter_rows(values_only=True))
    src = pd.DataFrame(rows[1:], columns=list(rows[0]))[C.EXPECTED_COLUMNS]
    bad = 0
    for c in C.EXPECTED_COLUMNS:
        if C.DICTIONARY_TYPES[c] == "Numeric":
            bad += int((bronze[c].astype(float).values != src[c].astype(float).values).sum())
        else:
            bad += int((bronze[c].astype(str).values != src[c].astype(str).values).sum())
    dtypes_ok = all(str(bronze[c].dtype) == "string" for c in C.EXPECTED_COLUMNS)
    actual(f"rows={len(bronze)} vs source {len(src)}; all string dtype={dtypes_ok}; mismatching cells={bad}")
    assert len(bronze) == len(src) and bad == 0 and dtypes_ok


@pytest.mark.case("T-ING-05", "Ingestion", "Missing file / wrong sheet name", "FileNotFoundError / ValueError with a clear message")
def test_ingest_errors(actual):
    with pytest.raises(FileNotFoundError):
        ingest.read_source(C.PROJECT_DATA / "does_not_exist.xlsx")
    with pytest.raises(ValueError):
        ingest.read_source(C.RAW_XLSX, sheet="Nope")
    actual("FileNotFoundError and ValueError raised")


@pytest.mark.case("T-ING-06", "Ingestion", "Source profile", "15 column profiles; both ID columns 100% pattern-conformant; 0 nulls")
def test_profile(raw_excel, actual):
    p = ingest.profile_source(raw_excel)
    actual(f"{len(p)} profiles; null cells={int(p.null_count.sum())}; ID conformance={p.dropna(subset=['pattern_conformance_pct']).pattern_conformance_pct.tolist()}")
    assert len(p) == 15 and p.null_count.sum() == 0 and (p.dropna(subset=["pattern_conformance_pct"]).pattern_conformance_pct == 100).all()


# --------------------------------------------------------------- data quality
@pytest.mark.case("T-DQ-01", "Data quality", "Run all rules on the supplied (clean) data", "0 rows fail any ERROR rule; 0 rows quarantined")
def test_dq_clean_data(bronze, actual):
    res = quality.run_rules(bronze[C.EXPECTED_COLUMNS])
    actual(f"rows failing error rules={res['score']['rows_failing_error_rules']}; rules={res['score']['n_rules']}")
    assert res["score"]["rows_failing_error_rules"] == 0 and res["score"]["n_rules_fail"] == 0


@pytest.mark.case("T-DQ-02", "Data quality", "Inject a blank Feedback_ID", "DQ-C01 raised; row quarantined")
def test_dq_null_key(bronze_sample, actual):
    bronze_sample.loc[5, "Feedback_ID"] = pd.NA
    out = _clean(bronze_sample)
    actual(f"quarantined={len(out['quarantine'])}; reason={out['quarantine'].dq_error_reason.tolist()}")
    assert len(out["quarantine"]) == 1 and "DQ-C01" in out["quarantine"].dq_error_reason.iloc[0] and len(out["silver"]) == 59


@pytest.mark.case("T-DQ-03", "Data quality", "Inject a duplicate Feedback_ID", "First occurrence kept; later one quarantined (DQ-U01)")
def test_dq_duplicate_id(bronze_sample, actual):
    bronze_sample.loc[10, "Feedback_ID"] = bronze_sample.loc[2, "Feedback_ID"]
    out = _clean(bronze_sample)
    actual(f"silver={len(out['silver'])}; quarantined reason={out['quarantine'].dq_error_reason.tolist()}")
    assert "DQ-U01" in out["quarantine"].dq_error_reason.iloc[0] and out["silver"].feedback_id.is_unique


@pytest.mark.case("T-DQ-04", "Data quality", "Inject an exact duplicate record under a new Feedback_ID", "Detected by DQ-U02 and quarantined (never silently dropped)")
def test_dq_exact_duplicate(bronze_sample, actual):
    row = bronze_sample.loc[3].copy()
    row["Feedback_ID"] = "FB-99999"
    out = _clean(pd.concat([bronze_sample, row.to_frame().T], ignore_index=True))
    actual(f"quarantined={len(out['quarantine'])}; reason={out['quarantine'].dq_error_reason.tolist()}")
    assert len(out["quarantine"]) == 1 and "DQ-U02" in out["quarantine"].dq_error_reason.iloc[0]


@pytest.mark.case("T-DQ-05", "Data quality", "Invalid category 'Mars' vs case variant 'north' and padded ' West '", "'Mars' quarantined (DQ-V03); variants canonicalised to 'North'/'West' and kept")
def test_dq_domain_and_canonicalisation(bronze_sample, actual):
    bronze_sample.loc[1, "Region"] = "Mars"
    bronze_sample.loc[2, "Region"] = "north"
    bronze_sample.loc[3, "Region"] = "  West "
    out = _clean(bronze_sample)
    s = out["silver"].set_index("feedback_id")
    actual(f"quarantined={len(out['quarantine'])} ({out['quarantine'].dq_error_reason.tolist()}); canonical values={s.loc[bronze_sample.loc[2, 'Feedback_ID'], 'region']}, {s.loc[bronze_sample.loc[3, 'Feedback_ID'], 'region']}")
    assert len(out["quarantine"]) == 1 and "DQ-V03-Region" in out["quarantine"].dq_error_reason.iloc[0]
    assert s.loc[bronze_sample.loc[2, "Feedback_ID"], "region"] == "North" and s.loc[bronze_sample.loc[3, "Feedback_ID"], "region"] == "West"


@pytest.mark.case("T-DQ-06", "Data quality", "Inject impossible values: Rating 7, CSAT 6.2, revenue -5, repeat flag 2, resolution -3", "Each violates its range rule; all 5 rows quarantined")
def test_dq_range_rules(bronze_sample, actual):
    for i, (c, v) in enumerate([("Rating", "7"), ("CSAT_Score", "6.2"), ("Customer_Revenue_INR", "-5"), ("Repeat_Purchase_Flag", "2"), ("Resolution_Time_Hours", "-3")]):
        bronze_sample.loc[i + 1, c] = v
    out = _clean(bronze_sample)
    reasons = sorted(out["quarantine"].dq_error_reason.tolist())
    actual(f"quarantined={len(out['quarantine'])}; reasons={reasons}")
    assert len(out["quarantine"]) == 5 and {"DQ-V04", "DQ-V05", "DQ-V06", "DQ-V08", "DQ-V09"} <= {r.split(";")[0] for r in reasons}


@pytest.mark.case("T-DQ-07", "Data quality", "Missing revenue and blank Channel", "Row kept; revenue stays NaN (never imputed) and is flagged DQ-C03; blank channel -> 'Unknown' and flagged DQ-C02")
def test_dq_missing_policy(bronze_sample, actual):
    bronze_sample.loc[4, "Customer_Revenue_INR"] = pd.NA
    bronze_sample.loc[6, "Channel"] = ""
    out = _clean(bronze_sample)
    s = out["silver"].set_index("feedback_id")
    a, b = s.loc[bronze_sample.loc[4, "Feedback_ID"]], s.loc[bronze_sample.loc[6, "Feedback_ID"]]
    actual(f"rows kept={len(out['silver'])}; revenue={a.customer_revenue_inr}; flags={a.dq_flags}; channel={b.channel}; flags={b.dq_flags}")
    assert len(out["silver"]) == 60 and np.isnan(a.customer_revenue_inr) and "DQ-C03" in a.dq_flags and b.channel == "Unknown" and "DQ-C02" in b.dq_flags


@pytest.mark.case("T-DQ-08", "Data quality", "Unparseable numeric ('abc') in Order_Count", "Coerced to NaN, flagged DQ-C03, row retained (transformation TR-03 logs 1 change)")
def test_dq_unparseable_numeric(bronze_sample, actual):
    bronze_sample.loc[7, "Order_Count"] = "abc"
    out = _clean(bronze_sample)
    tr03 = out["log"].set_index("step").loc["TR-03", "cells_changed"]
    s = out["silver"].set_index("feedback_id").loc[bronze_sample.loc[7, "Feedback_ID"]]
    actual(f"order_count={s.order_count}; TR-03 cells changed={tr03}; flags={s.dq_flags}")
    assert pd.isna(s.order_count) and tr03 == 1 and "DQ-C03" in s.dq_flags


@pytest.mark.case("T-DQ-09", "Data quality", "Text names a different issue/channel/product than the structured fields (3 faults + 1 free-text row)", "DQ-K03 flags exactly the 3 genuine mismatches; free text is out of scope")
def test_dq_text_vs_structure(bronze, actual):
    raw = bronze[C.EXPECTED_COLUMNS].head(30).copy()
    raw.loc[0, "Issue_Category"] = "Pricing"
    raw.loc[1, "Channel"] = "Store"
    raw.loc[2, "Product_Category"] = "Fashion"
    raw.loc[3, "Feedback_Text"] = "Totally free form comment with no template at all"
    m = quality.run_rules(raw)["masks"]["DQ-K03"]
    actual(f"K03 flagged rows={list(m[m].index)}")
    assert list(m[m].index) == [0, 1, 2]


@pytest.mark.case("T-DQ-10", "Data quality", "Supplied data: Repeat_Purchase_Flag=1 with Order_Count=1", "113 records flagged DQ-K04 (warning, retained)")
def test_dq_repeat_vs_orders(bronze, actual):
    r = quality.run_rules(bronze[C.EXPECTED_COLUMNS])["report"].set_index("rule_id")
    actual(f"K04 rows={r.loc['DQ-K04', 'rows_failed']}")
    assert r.loc["DQ-K04", "rows_failed"] == 113 and r.loc["DQ-K04", "status"] == "WARN"


@pytest.mark.case("T-DQ-11", "Data quality", "Template glitch 'experience experience'", "298 raw rows flagged; 0 after cleaning; intensifier 'very very' preserved")
def test_dq_repeated_word(silver, bronze, actual):
    raw_n = int(quality.run_rules(bronze[C.EXPECTED_COLUMNS])["report"].set_index("rule_id").loc["DQ-K05", "rows_failed"])
    fixed = silver.text_clean.str.contains(quality.REPEATED_WORD_RE).sum()
    kept = transform.collapse_repeated_words("It was very very slow experience experience")
    actual(f"raw={raw_n}; after={fixed}; sample -> '{kept}'")
    assert raw_n == 298 and fixed == 0 and "very very" in kept and "experience experience" not in kept


# ------------------------------------------------------------- transformation
@pytest.mark.case("T-TR-01", "Transformation", "Bronze -> Silver on the supplied data", "2,400 in, 2,400 out, 0 quarantined; no narrow integer dtypes (int8/int16/int32)")
def test_silver_counts_and_dtypes(silver, bronze, actual):
    narrow = [c for c in silver.columns if str(silver[c].dtype) in ("int8", "int16", "int32", "uint8")]
    actual(f"{len(bronze)} -> {len(silver)}; narrow int columns={narrow}")
    assert len(silver) == len(bronze) == 2400 and not narrow


@pytest.mark.case("T-TR-02", "Transformation", "Resolution-band boundaries 6.0, 6.1, 12.0, 12.1, 18.1, 24.1", "0-6h, 6-12h, 6-12h, 12-18h, 18-24h, >24h; sla_breach only above 12h")
def test_band_boundaries(actual):
    df = pd.DataFrame({"resolution_time_hours": [6.0, 6.1, 12.0, 12.1, 18.1, 24.1], "customer_revenue_inr": [1000.0] * 6, "order_count": [2] * 6, "csat_score": [3.0] * 6,
                       "reference_sentiment": ["Neutral"] * 6, "issue_category": ["Pricing"] * 6, "channel": ["App"] * 6, "feedback_source": ["Chat"] * 6,
                       "repeat_purchase_flag": [0] * 6, "customer_id": ["C-00001"] * 6, "feedback_id": [f"FB-0000{i}" for i in range(6)]})
    out = transform.derive_features(df, {"revenue_quartile_edges": [500.0, 1000.0, 1500.0], "sla_target_hours": 12.0, "critical_delay_hours": 18.0})
    bands = out.resolution_band.astype(str).tolist()
    actual(f"bands={bands}; sla_breach={out.sla_breach.tolist()}")
    assert bands == ["0-6h", "6-12h", "6-12h", "12-18h", "18-24h", ">24h"] and out.sla_breach.tolist() == [0, 0, 0, 1, 1, 1] and out.critical_delay.tolist() == [0, 0, 0, 0, 1, 1]


@pytest.mark.case("T-TR-03", "Transformation", "Revenue tiers on the supplied data", "Four quartile tiers of exactly 600 records each")
def test_revenue_tiers(silver, actual):
    vc = silver.revenue_tier.value_counts().to_dict()
    actual(vc)
    assert sorted(vc.values()) == [600] * 4


@pytest.mark.case("T-TR-04", "Transformation", "Run the transformation twice on the same Bronze batch", "Identical Silver content hash (deterministic and idempotent)")
def test_transform_idempotent(bronze, actual):
    h1 = sha256_frame(transform.clean(bronze)["silver"].drop(columns=["_batch_id"]))
    h2 = sha256_frame(transform.clean(bronze)["silver"].drop(columns=["_batch_id"]))
    actual(f"{h1[:16]} == {h2[:16]}")
    assert h1 == h2


@pytest.mark.case("T-TR-05", "Transformation", "Regression: int8 overflow when computing 100 * grouped sum of a flag column", "Grouped negative rate computed from Silver flags equals the value from raw Excel")
def test_no_overflow(silver, raw_excel, actual):
    g = silver.groupby("issue_category").is_negative_ref.agg(["sum", "size"])
    rate = (100 * g["sum"] / g["size"]).round(2)
    ref = raw_excel.groupby("Issue_Category").Reference_Sentiment.apply(lambda s: round(100 * (s == "Negative").mean(), 2))
    actual(f"max abs diff={float((rate - ref).abs().max()):.4f}")
    assert (rate - ref).abs().max() < 1e-9


@pytest.mark.case("T-TR-06", "Transformation", "Derived average order value", "avg_order_value_inr == revenue / order_count (2 dp)")
def test_aov(silver, actual):
    exp = (silver.customer_revenue_inr / silver.order_count).round(2)
    actual(f"max diff={float((silver.avg_order_value_inr - exp).abs().max())}")
    assert (silver.avg_order_value_inr - exp).abs().max() < 1e-9
