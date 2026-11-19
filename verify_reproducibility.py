"""Whole-pipeline reproducibility check: run everything N times, compare SHA-256 of every result file.

    python verify_reproducibility.py            # 2 full runs (about 4 minutes); exit code 1 if anything differs
    python verify_reproducibility.py --runs 3

Compared (byte for byte): outputs/tables/*.csv, outputs/metrics/*.json, outputs/dq/*.csv, plus a content hash
of the four Silver parquet files (parquet metadata is not stable, the rows are).
Not compared, by design: outputs/metrics/ingest.json (holds the ingestion wall-clock time), run_manifest.json
(timings), the PNG figures and the pickled model.  The unit tests cover single stages (T-TR-04, T-WH-06,
T-MIN-05, T-REP-02); this script is the end-to-end version of the same promise.
Result is written to outputs/validation/reproducibility_check.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SILVER = ["customer_voice_clean.parquet", "nlp_sentiment.parquet", "nlp_topics.parquet", "mining_segments.parquet"]
EXCLUDED = ["outputs/metrics/ingest.json", "outputs/run_manifest.json", "outputs/figures/*.png", "outputs/models/*.joblib"]


def snapshot() -> dict[str, str]:
    """SHA-256 of every compared result file, keyed by project-relative path."""
    out: dict[str, str] = {}
    files = (sorted((ROOT / "outputs" / "tables").glob("*.csv")) + sorted((ROOT / "outputs" / "metrics").glob("*.json"))
             + sorted((ROOT / "outputs" / "dq").glob("*.csv")))
    for p in files:
        if p.name == "ingest.json":
            continue
        out[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in SILVER:
        df = pd.read_parquet(ROOT / "data" / "silver" / name)
        out[f"data/silver/{name}"] = hashlib.sha256(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=2, help="number of full pipeline runs to compare (default 2)")
    n_runs = max(2, ap.parse_args().runs)

    snaps, seconds = [], []
    for i in range(1, n_runs + 1):
        t0 = time.time()
        print(f"run {i}/{n_runs}: python run_pipeline.py ...", flush=True)
        r = subprocess.run([sys.executable, "run_pipeline.py"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
            print(f"run {i} FAILED (exit {r.returncode}); cannot compare")
            return 2
        seconds.append(round(time.time() - t0, 1))
        snaps.append(snapshot())
        print(f"   done in {seconds[-1]} s, {len(snaps[-1])} files hashed", flush=True)

    keys = sorted(set().union(*snaps))
    differing = [k for k in keys if len({s.get(k) for s in snaps}) > 1]
    identical = len(keys) - len(differing)
    print(f"\nfiles compared: {len(keys)} | identical across {n_runs} full runs: {identical} | different: {len(differing)}")
    for k in differing:
        print("   DIFF:", k)
    report = {"checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "runs": n_runs, "seconds_per_run": seconds,
              "files_compared": len(keys), "identical": identical, "different": differing, "excluded_by_design": EXCLUDED,
              "result": "PASS" if not differing else "FAIL"}
    out = ROOT / "outputs" / "validation" / "reproducibility_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['result']}  (report: {out.relative_to(ROOT)})")
    return 0 if not differing else 1


if __name__ == "__main__":
    sys.exit(main())
