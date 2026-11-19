"""Central configuration for the Customer Voice Intelligence (CVI) platform.

Every path, threshold, mapping and business assumption used anywhere in the
pipeline is declared here so that it is documented once and reproducible.
Values marked ``ASSUMPTION`` are analyst decisions (the client supplied no
rule) and are listed again in the workbook's assumptions sections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA = ROOT / "Project Data"
RAW_XLSX = PROJECT_DATA / "Client Data Set for Student Project Workbook.xlsx"
RAW_SHEET = "Customer Voice"
CASE_NOTES_SHEET = "Case Notes"
DICTIONARY_XLSX = PROJECT_DATA / "Competitor Alpha Data Set for Student.xlsx"
TEMPLATE_XLSX = PROJECT_DATA / "Sample Student Data Analysis Template.xlsx"

DATA_DIR = ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = OUT_DIR / "models"
DQ_DIR = OUT_DIR / "dq"
METRIC_DIR = OUT_DIR / "metrics"
VALIDATION_DIR = OUT_DIR / "validation"
REFERENCE_DIR = OUT_DIR / "reference"
SQL_DIR = ROOT / "sql"
NLTK_DIR = ROOT / "nltk_data"
DELIVERABLES_DIR = ROOT / "deliverables"
DASHBOARD_DIR = ROOT / "dashboard"

BRONZE_FILE = BRONZE_DIR / "customer_voice_raw.parquet"
SILVER_FILE = SILVER_DIR / "customer_voice_clean.parquet"
QUARANTINE_FILE = SILVER_DIR / "customer_voice_quarantine.parquet"
NLP_FILE = SILVER_DIR / "customer_voice_nlp.parquet"
ANALYTICS_FILE = GOLD_DIR / "customer_voice_analytics.parquet"
DERIVATION_PARAMS_FILE = SILVER_DIR / "derivation_params.json"
WAREHOUSE_DB = GOLD_DIR / "cvi_warehouse.db"

ALL_DIRS = [BRONZE_DIR, SILVER_DIR, GOLD_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR,
            DQ_DIR, METRIC_DIR, VALIDATION_DIR, REFERENCE_DIR]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
N_BOOT = 2000          # bootstrap resamples for confidence intervals
CI_LEVEL = 0.95

# --------------------------------------------------------------------------- #
# Source schema contract (from the supplied data dictionary)
# --------------------------------------------------------------------------- #
EXPECTED_COLUMNS = [
    "Feedback_ID", "Customer_ID", "Region", "Channel", "Product_Category",
    "Issue_Category", "Rating", "Feedback_Text", "Reference_Sentiment",
    "Customer_Revenue_INR", "Order_Count", "Repeat_Purchase_Flag",
    "Resolution_Time_Hours", "CSAT_Score", "Feedback_Source",
]
DICTIONARY_TYPES = {
    "Feedback_ID": "Identifier", "Customer_ID": "Identifier", "Region": "Categorical",
    "Channel": "Categorical", "Product_Category": "Categorical", "Issue_Category": "Categorical",
    "Rating": "Numeric", "Feedback_Text": "Text", "Reference_Sentiment": "Categorical",
    "Customer_Revenue_INR": "Numeric", "Order_Count": "Numeric", "Repeat_Purchase_Flag": "Numeric",
    "Resolution_Time_Hours": "Numeric", "CSAT_Score": "Numeric", "Feedback_Source": "Categorical",
}
SNAKE = {c: c.lower() for c in EXPECTED_COLUMNS}   # Feedback_ID -> feedback_id

ALLOWED_VALUES = {
    "Region": ["Central", "East", "North", "South", "West"],
    "Channel": ["App", "Marketplace", "Store", "Web"],
    "Product_Category": ["Beauty", "Electronics", "Fashion", "Grocery", "Home"],
    "Issue_Category": ["App Experience", "Customer Support", "Delivery Delay", "Packaging",
                       "Pricing", "Product Availability", "Product Quality", "Refund Delay"],
    "Reference_Sentiment": ["Negative", "Neutral", "Positive"],
    "Feedback_Source": ["Chat", "Email", "Review", "Social", "Survey"],
}
RANGES = {  # (min, max) inclusive, hard validity bounds
    "Rating": (1, 5),
    "CSAT_Score": (1.0, 5.0),
    "Repeat_Purchase_Flag": (0, 1),
    "Order_Count": (1, 1000),
    "Customer_Revenue_INR": (0.01, 10_000_000),
    "Resolution_Time_Hours": (0.0, 720.0),     # ASSUMPTION: >30 days is not plausible
}
RESOLUTION_WARN_HOURS = 72.0                   # ASSUMPTION: plausibility warning level
ID_PATTERNS = {"Feedback_ID": r"FB-\d{5}", "Customer_ID": r"C-\d{5}"}
MIN_TEXT_CHARS = 10

# --------------------------------------------------------------------------- #
# Business mappings (ASSUMPTIONS - the client supplied no rule)
# --------------------------------------------------------------------------- #
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SENTIMENT_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}
# Verified in profiling: Reference_Sentiment is a deterministic function of Rating.
RATING_TO_SENTIMENT = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}

# Resolution-time (SLA) bands. No SLA was supplied; the 12h target is the
# Youden-optimal cut-off separating negative from non-negative feedback
# (12.5h in profiling) rounded to a round business number.
RES_BAND_EDGES = [0, 6, 12, 18, 24, np.inf]
RES_BAND_LABELS = ["0-6h", "6-12h", "12-18h", "18-24h", ">24h"]
SLA_TARGET_HOURS = 12.0
CRITICAL_DELAY_HOURS = 18.0

ORDER_BAND_EDGES = [0, 3, 8, np.inf]
ORDER_BAND_LABELS = ["Low (1-3)", "Medium (4-8)", "High (9+)"]
REVENUE_TIER_LABELS = ["Q1 Low", "Q2 Mid-Low", "Q3 Mid-High", "Q4 Premium"]
CSAT_BAND_EDGES = [0, 2.5, 3.5, np.inf]
CSAT_BAND_LABELS = ["Low (<2.5)", "Medium (2.5-3.5)", "High (>=3.5)"]

# Business roll-up of the eight issue categories into four theme groups.
ISSUE_THEME_GROUP = {
    "Delivery Delay": "Fulfilment & Logistics",
    "Packaging": "Fulfilment & Logistics",
    "Product Availability": "Fulfilment & Logistics",
    "Product Quality": "Product & Price",
    "Pricing": "Product & Price",
    "Customer Support": "Service & Support",
    "Refund Delay": "Service & Support",
    "App Experience": "Digital Experience",
}
CHANNEL_TYPE = {"App": "Digital", "Web": "Digital", "Marketplace": "Digital", "Store": "Physical"}
SOURCE_CLASS = {"Survey": "Solicited", "Chat": "Support interaction", "Email": "Support interaction",
                "Review": "Public voice", "Social": "Public voice"}

# Descriptor (sentiment-bearing word) families for negative feedback.
NEGATIVE_DESCRIPTOR_THEME = {
    "slow": "Speed / timeliness", "late": "Speed / timeliness",
    "poor": "Quality / defect", "broken": "Quality / defect",
    "unhelpful": "Usability / helpfulness", "confusing": "Usability / helpfulness",
    "disappointed": "Emotional reaction",
}

# --------------------------------------------------------------------------- #
# Text / NLP
# --------------------------------------------------------------------------- #
NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot",
                  "cant", "can't", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
                  "wasnt", "wasn't", "isnt", "isn't", "wont", "won't", "wouldnt", "wouldn't",
                  "couldnt", "couldn't", "shouldnt", "shouldn't", "hardly", "barely", "without"}
BOILERPLATE_DF_THRESHOLD = 0.90   # tokens present in >=90% of documents carry no information
TEST_SIZE = 0.20
CV_FOLDS = 5
VADER_THRESHOLD = 0.05            # standard +/-0.05 compound cut-off (Hutto & Gilbert, 2014)
TOPIC_K_RANGE = range(4, 13)

# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #
CLUSTER_FEATURES = ["csat_score", "resolution_time_hours", "repeat_purchase_flag"]
CLUSTER_K_RANGE = range(2, 9)
APRIORI_MIN_SUPPORT = 0.02
APRIORI_MAX_LEN = 3
RULE_MIN_LIFT = 1.2
FDR_ALPHA = 0.05

# --------------------------------------------------------------------------- #
# Decision layer: proposed KPI targets (ASSUMPTIONS to be agreed with the client)
# --------------------------------------------------------------------------- #
PROPOSED_TARGETS = {
    "negative_sentiment_rate": 0.20,
    "repeat_purchase_rate": 0.60,
    "avg_resolution_hours": 10.0,
    "net_sentiment_score": 40.0,
    "avg_csat": 3.7,
    "negative_resolution_hours": 12.0,
    "negative_repeat_rate": 0.40,
}
# Share of the observed repeat-purchase gap (positive vs negative) assumed to be
# causally recoverable.  The data are cross-sectional, so this is a scenario range.
RECOVERY_SCENARIOS = {"conservative": 0.10, "base": 0.25, "optimistic": 0.50}
