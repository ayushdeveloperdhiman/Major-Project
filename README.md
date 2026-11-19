# AI-Powered Customer Voice Intelligence and Business Decision Platform

**Project MSDS-MAJOR-CVI-001 | Client: InsightRetail India (anonymised) | MSDS - Master of Science in Data Science | Major Project**

An end-to-end, reproducible data-to-decision workflow: raw customer feedback -> data-quality controls -> analytical warehouse -> EDA and KPIs -> data mining -> NLP (sentiment and themes) -> integrated intelligence -> dashboard and prioritised actions -> validation evidence.

## Headline results (all numbers come from the saved pipeline outputs)

| Finding | Evidence |
| --- | --- |
| Sentiment, not segment, drives repeat purchase | Repeat rate 28.2% (negative) vs 72.0% (positive); adjusted odds ratio 0.17 (95% CI 0.13-0.22) |
| Negative cases resolve about twice as slowly | 18.1 h vs 9.1 h (Cohen's d 1.90); 91% of negatives breach a 12 h SLA |
| Revenue at Risk | INR 3.70M = 24.5% of revenue (bootstrap 95% CI 3.43M-4.01M); 71.5% sits in a 306-record Critical tier |
| No segment is a proven sentiment hotspot | 0 of 314 segment cells survive false-discovery-rate control; the "top issue" (Customer Support) ranks first in only 43% of bootstrap resamples |
| Model scores are a ceiling effect | 100% in-distribution, but 57.1% (= baseline) on opinion words never seen in training; the text is templated |

## Three data facts that shape every result

1. The feedback text is **100% templated** (one sentence in which one of 19 words carries the sentiment), so model scores of 100% are an upper bound, not real-world accuracy.
2. `Reference_Sentiment` is a **deterministic function of `Rating`** (1-2 Negative, 3 Neutral, 4-5 Positive): it is not independent human labelling.
3. `Customer_ID` is **not a stable key** (revenue differs within a customer in 100% of multi-record customers), so all customer metrics are at feedback-record level. There are no timestamps, and the "Competitor Alpha" workbook contains only the data dictionary and metric guidance (no competitor data).

## Quick start

```bash
# 1. environment (Python 3.13 recommended; tested on 3.13.9)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # (pip install -e . is optional: the scripts and tests add src/ to the path themselves)

# 2. run everything (12 stages, ~2 minutes, deterministic, seed 42)
python run_pipeline.py                 # python run_pipeline.py --list  shows the stages; --from eda / --only decision figures

# 3. validate
pytest                                 # 65 tests; writes outputs/validation/test_results.csv
python verify_reproducibility.py       # optional (~4 min): runs the whole pipeline twice, compares 78 result files byte for byte

# 4. build the deliverables that depend on executed results
python make_deliverables.py            # deliverables/Completed_Data_Analysis_Template.xlsx
python make_notebook.py                # executes notebooks/Customer_Voice_Intelligence_End_to_End.ipynb

# 5. dashboard
python -m streamlit run dashboard/app.py   # interactive (localhost). Use `python -m streamlit`, not a bare `streamlit` (see Troubleshooting)
python dashboard/build_static.py       # single-file offline export -> dashboard/static/cvi_dashboard.html
python dashboard/capture_screenshots.py  # optional: needs Google Chrome + playwright
```

NLTK resources (stop-words, WordNet) are downloaded into `nltk_data/` on first use and need internet once.

### Troubleshooting: `ModuleNotFoundError: vaderSentiment` / `mlxtend` / `cvi`, or `'LogisticRegression' object has no attribute 'multi_class'`

All of these mean the command is running a **different Python from the one the project was built with**. A bare `streamlit`, `pytest` or `pip` command belongs to whichever Python installed it (often Anaconda), and that Python may lack packages or have another scikit-learn: the saved sentiment model needs scikit-learn 1.9.1. Check which Python you are using with `python -c "import sys; print(sys.executable)"`, then either

- use the project's interpreter: `./.venv/bin/python -m streamlit run dashboard/app.py` (likewise `./.venv/bin/python run_pipeline.py`, `./.venv/bin/python -m pytest`), or
- install the pinned packages into the Python you want to use: `python -m pip install -r requirements.txt` (a fresh virtual environment is cleanest).

The dashboard shows a warning banner, and `run_pipeline.py` stops with a message, naming the packages that differ from `requirements.txt`.

## Project structure

```
Project Data/            the three supplied workbooks (read-only inputs)
run_pipeline.py          one command, 12 stages          src/cvi/pipeline.py  orchestration, validation, run manifest
verify_reproducibility.py  end-to-end determinism check  (writes outputs/validation/reproducibility_check.json)
src/cvi/                 ingest | quality | transform | warehouse | eda | kpis | nlp_prep | sentiment | topics
                         mining | integrate | decision | figures | figures2 | viz | diagrams | template_xlsx
sql/                     schema.sql (star schema DDL) and views.sql (KPI semantic layer)
data/                    bronze/ (raw landing) silver/ (clean + model outputs) gold/ (warehouse + analytics frame)
outputs/                 tables/ figures/ models/ dq/ validation/ metrics/ run_manifest.json
dashboard/               app.py (Streamlit), charts.py, data.py, build_static.py, static/, screenshots/
tests/                   65 pytest cases; each records its own "actual result" for the Test Cases sheet
notebooks/               executed end-to-end notebook (outputs embedded)
deliverables/            Completed_Data_Analysis_Template.xlsx
```

## Workbook activities -> where the evidence is

| Activity | Code | Key outputs |
| --- | --- | --- |
| 1 Problem framing, KPIs | `kpis.py`, `config.py`, `decision.py` | `outputs/tables/decision_kpi_scorecard.csv` |
| 2 Data engineering, ingestion | `ingest.py`, `pipeline.py` | `data/bronze/`, `outputs/dq/source_profile.csv`, `outputs/run_manifest.json` |
| 3 Data quality, features | `quality.py`, `transform.py` | `outputs/dq/dq_report_*.csv`, `transformation_log.csv` |
| 4 Warehouse | `warehouse.py`, `sql/` | `data/gold/cvi_warehouse.db`, `outputs/validation/warehouse_reconciliation.csv` |
| 5 Exploratory analytics | `eda.py` | `outputs/tables/eda_*.csv`, figures 03-11 |
| 6 Data mining | `mining.py` | `outputs/tables/mining_*.csv`, figures 19-21, 25 |
| 7 NLP pre-processing | `nlp_prep.py` | tests T-NLP-01..06 |
| 8 Sentiment | `sentiment.py` | `outputs/tables/sentiment_*.csv`, `outputs/models/`, figures 12-15 |
| 9 Topics / themes | `topics.py` | `outputs/tables/topics_*.csv`, figures 16-18 |
| 10 Integrated intelligence | `integrate.py` | `outputs/tables/integrated_*.csv`, figures 22-24 |
| 11 Decision layer | `decision.py`, `dashboard/` | `decision_*.csv`, dashboard, screenshots, figures 26-27 |
| 12 Validation | `tests/`, `pipeline.py` | `outputs/validation/test_results.csv`, notebook |

## Method notes (why it is built this way)

- **Honest statistics:** every claim carries an effect size and an interval; segment scans use Fisher exact tests with Benjamini-Hochberg FDR; rankings are shown with bootstrap rank stability; clustering is compared with a column-permutation null; association rules pass a significance and productivity funnel.
- **No label leakage:** sentiment models see text only; Rating and CSAT are never features. A leakage ablation (opinion word masked -> chance) and a leave-one-opinion-word-out test expose the ceiling effect.
- **Single source of truth for KPIs:** definitions follow the supplied Metric Guidance and are implemented twice (pandas and SQL views), reconciled 28 of 28.
- **Reproducible to the byte:** seed 42, pinned requirements, and two consecutive full runs give identical tables, metrics and data-quality reports (78 files compared by SHA-256; the ingestion timestamp, figures and pickled model are excluded by design). Floating-point noise from threaded maths is removed at write time (12 significant digits; centroids rounded to 9 decimals) and every ranking has a total order, so ties cannot reorder.
- **Targets are proposed assumptions** (none were supplied). Impact figures are scenarios, not forecasts, because the data are cross-sectional.

## Submission pack (LMS)

Upload the completed Workbook, which is `deliverables/Student_Project_Workbook_Ayush_Dhiman.pdf` (US Letter, laid out exactly like the supplied `Project Data/Template.pdf`; the `.docx` beside it is the editable copy: add your enrolment number, batch and date, tick the declaration, then open it in Word and Save as PDF, which also renders the headings in true Calibri), together with: `deliverables/Completed_Data_Analysis_Template.xlsx`, `notebooks/`, `dashboard/screenshots/`, `outputs/validation/test_results.csv`, and a zip of the code (`src/`, `sql/`, `tests/`, `run_pipeline.py`, `README.md`, `requirements.txt`; exclude `.venv/`). The student declaration in the workbook is yours to complete.
