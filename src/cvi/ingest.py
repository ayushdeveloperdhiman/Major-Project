"""Stage 1 - Ingestion & profiling (Bronze layer).

Pattern: *schema-on-read landing zone*.  The source sheet is landed exactly as
read (all values stored as strings so nothing is silently coerced) together with
lineage columns.  Typing, cleaning and rules are applied later (Silver).

Outputs
-------
data/bronze/customer_voice_raw.parquet      raw rows + lineage
outputs/dq/source_profile.csv               column-level profile
outputs/reference/*.csv                     data dictionary, metric guidance, case notes
outputs/metrics/ingest.json                 headline ingestion metrics
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from . import config as C
from .utils import get_logger, save_metrics, sha256_file, write_json

log = get_logger("cvi.ingest")

LINEAGE_COLS = ["_source_file", "_source_sheet", "_source_row", "_batch_id", "_row_hash"]


def read_source(path: Path = C.RAW_XLSX, sheet: str = C.RAW_SHEET) -> pd.DataFrame:
    """Read the raw sheet *without* type inference beyond what Excel stores."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Source workbook not found: {path}")
    xl = pd.ExcelFile(path, engine="openpyxl")
    if sheet not in xl.sheet_names:
        raise ValueError(f"Sheet {sheet!r} not in workbook; found {xl.sheet_names}")
    return xl.parse(sheet, dtype=object)


def schema_drift(df: pd.DataFrame, expected: list[str] = C.EXPECTED_COLUMNS) -> dict:
    cols = list(df.columns)
    return {
        "missing_columns": [c for c in expected if c not in cols],
        "unexpected_columns": [c for c in cols if c not in expected],
        "order_matches": [c for c in cols if c in expected] == [c for c in expected if c in cols],
        "n_columns": len(cols),
    }


def _row_hash(values: tuple) -> str:
    return hashlib.sha256("\x1f".join("" if pd.isna(v) else str(v) for v in values).encode()).hexdigest()


def to_bronze(raw: pd.DataFrame, path: Path = C.RAW_XLSX, sheet: str = C.RAW_SHEET) -> pd.DataFrame:
    """Land raw data as strings + lineage; nothing is cleaned or dropped."""
    batch = sha256_file(path)[:12]
    b = raw.copy()
    cols = [c for c in b.columns]
    for c in cols:
        b[c] = b[c].map(lambda v: pd.NA if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)).astype("string")
    b["_source_file"] = Path(path).name
    b["_source_sheet"] = sheet
    b["_source_row"] = np.arange(2, len(b) + 2)          # Excel row number (header = row 1)
    b["_batch_id"] = batch
    b["_row_hash"] = [_row_hash(t) for t in raw.itertuples(index=False, name=None)]
    return b


def profile_source(raw: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile of the supplied data (completeness, cardinality, ranges, pattern conformance)."""
    rows = []
    for c in raw.columns:
        s = raw[c]
        nn = s.notna().sum()
        sn = pd.to_numeric(s, errors="coerce")
        is_num = C.DICTIONARY_TYPES.get(c) == "Numeric"
        txt = s.dropna().astype(str)
        row = {
            "column": c,
            "dictionary_type": C.DICTIONARY_TYPES.get(c, "n/a"),
            "dtype_read": str(s.map(type).mode().iloc[0].__name__) if len(s) else "n/a",
            "rows": len(s),
            "non_null": int(nn),
            "null_count": int(len(s) - nn),
            "null_pct": round(100 * (len(s) - nn) / max(len(s), 1), 3),
            "distinct": int(s.nunique(dropna=True)),
            "distinct_pct": round(100 * s.nunique(dropna=True) / max(len(s), 1), 2),
            "leading_trailing_ws": int((txt != txt.str.strip()).sum()),
            "min": float(sn.min()) if is_num else (txt.min() if len(txt) else None),
            "max": float(sn.max()) if is_num else (txt.max() if len(txt) else None),
            "mean": round(float(sn.mean()), 4) if is_num else None,
            "std": round(float(sn.std()), 4) if is_num else None,
            "median": float(sn.median()) if is_num else None,
            "top_value": txt.mode().iloc[0] if len(txt) and C.DICTIONARY_TYPES.get(c) != "Text" else None,
            "top_value_pct": round(100 * (txt == txt.mode().iloc[0]).mean(), 2) if len(txt) and C.DICTIONARY_TYPES.get(c) != "Text" else None,
        }
        if c in C.ID_PATTERNS:
            row["pattern_conformance_pct"] = round(100 * txt.str.fullmatch(C.ID_PATTERNS[c]).mean(), 3)
        if is_num:
            row["non_numeric_values"] = int(s.notna().sum() - sn.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def read_reference_tables() -> dict[str, pd.DataFrame]:
    """Data dictionary, metric guidance and case notes supplied with the project."""
    out = {}
    dd = pd.read_excel(C.DICTIONARY_XLSX, sheet_name="Data Dictionary")
    mg = pd.read_excel(C.DICTIONARY_XLSX, sheet_name="Metric Guidance")
    cn = pd.read_excel(C.RAW_XLSX, sheet_name=C.CASE_NOTES_SHEET)
    out["data_dictionary"], out["metric_guidance"], out["case_notes"] = dd, mg, cn
    return out


def ingest(save: bool = True) -> dict:
    """Run Stage 1 end-to-end and return headline metrics."""
    C.ensure_dirs()
    log.info("Reading source workbook: %s", C.RAW_XLSX.name)
    raw = read_source()
    drift = schema_drift(raw)
    refs = read_reference_tables()
    dd_fields = refs["data_dictionary"]["Field"].tolist()
    dictionary_matches = dd_fields == list(raw.columns)
    bronze = to_bronze(raw)
    profile = profile_source(raw)

    # Source inventory (every sheet in the three supplied workbooks)
    inventory = []
    for p in [C.RAW_XLSX, C.DICTIONARY_XLSX, C.TEMPLATE_XLSX]:
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        for ws in wb.worksheets:
            inventory.append({"file": p.name, "sheet": ws.title, "rows": ws.max_row, "columns": ws.max_column,
                              "sha256_12": sha256_file(p)[:12]})
        wb.close()
    inventory = pd.DataFrame(inventory)

    metrics = {
        "source_file": C.RAW_XLSX.name, "source_sheet": C.RAW_SHEET,
        "source_sha256": sha256_file(C.RAW_XLSX), "batch_id": sha256_file(C.RAW_XLSX)[:12],
        "rows": int(len(raw)), "columns": int(raw.shape[1]),
        "schema_drift": drift, "dictionary_fields_match_columns": bool(dictionary_matches),
        "total_null_cells": int(raw.isna().sum().sum()),
        "distinct_feedback_ids": int(raw["Feedback_ID"].nunique()),
        "distinct_customer_ids": int(raw["Customer_ID"].nunique()),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "competitor_data_present": False,
        "note_competitor_file": "'Competitor Alpha Data Set for Student.xlsx' contains only a Data Dictionary and "
                                "Metric Guidance - no competitor measurements - so no competitor benchmarking is possible.",
    }
    if save:
        bronze.to_parquet(C.BRONZE_FILE, index=False)
        profile.to_csv(C.DQ_DIR / "source_profile.csv", index=False)
        inventory.to_csv(C.DQ_DIR / "source_inventory.csv", index=False)
        for name, t in refs.items():
            t.to_csv(C.REFERENCE_DIR / f"{name}.csv", index=False)
        save_metrics("ingest", metrics)
        write_json(metrics, C.BRONZE_DIR / "ingest_log.json")
    log.info("Bronze landed: %d rows x %d cols (+%d lineage cols) | batch=%s", len(bronze), raw.shape[1],
             len(LINEAGE_COLS), metrics["batch_id"])
    return metrics


def load_bronze() -> pd.DataFrame:
    return pd.read_parquet(C.BRONZE_FILE)


if __name__ == "__main__":  # pragma: no cover
    ingest()
