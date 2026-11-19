-- =============================================================================
-- CVI Analytical Warehouse  |  Project MSDS-MAJOR-CVI-001  |  Engine: SQLite 3
-- Pattern : Kimball star schema, Type-1 dimensions (no history - the source has
--           no date field), surrogate integer keys, enforced FK/CHECK constraints.
-- Grain   : fact_feedback = ONE ROW PER FEEDBACK RECORD (Feedback_ID).
-- NOTE    : Customer_ID is a weak entity key - revenue, orders, repeat flag and
--           region vary between the records of the same Customer_ID, so those
--           values live in the FACT (record level) and are never rolled up to a
--           customer attribute.
-- =============================================================================
PRAGMA foreign_keys = ON;

-- ------------------------------- DIMENSIONS ----------------------------------
CREATE TABLE dim_region (
    region_key           INTEGER PRIMARY KEY,
    region_name          TEXT    NOT NULL UNIQUE
);

CREATE TABLE dim_channel (
    channel_key          INTEGER PRIMARY KEY,
    channel_name         TEXT    NOT NULL UNIQUE,
    channel_type         TEXT    NOT NULL CHECK (channel_type IN ('Digital','Physical','Unknown'))
);

CREATE TABLE dim_product_category (
    product_category_key  INTEGER PRIMARY KEY,
    product_category_name TEXT    NOT NULL UNIQUE
);

CREATE TABLE dim_issue (                      -- feedback topic / touchpoint
    issue_key            INTEGER PRIMARY KEY,
    issue_category       TEXT    NOT NULL UNIQUE,
    issue_theme_group    TEXT    NOT NULL     -- 4-group business roll-up (hierarchy level 2)
);

CREATE TABLE dim_feedback_source (
    source_key           INTEGER PRIMARY KEY,
    source_name          TEXT    NOT NULL UNIQUE,
    source_class         TEXT    NOT NULL
);

CREATE TABLE dim_sentiment (
    sentiment_key        INTEGER PRIMARY KEY,
    sentiment_label      TEXT    NOT NULL UNIQUE,
    sentiment_score      INTEGER NOT NULL,    -- -1 / 0 / +1
    sort_order           INTEGER NOT NULL
);

CREATE TABLE dim_rating (
    rating_key           INTEGER PRIMARY KEY CHECK (rating_key BETWEEN 1 AND 5),
    rating_label         TEXT    NOT NULL,
    implied_sentiment    TEXT    NOT NULL
);

CREATE TABLE dim_resolution_band (
    band_key             INTEGER PRIMARY KEY,
    band_label           TEXT    NOT NULL UNIQUE,
    lower_hours          REAL    NOT NULL,
    upper_hours          REAL,                -- NULL = open-ended
    sla_status           TEXT    NOT NULL     -- Within SLA / Breach / Critical
);

CREATE TABLE dim_revenue_tier (
    tier_key             INTEGER PRIMARY KEY,
    tier_label           TEXT    NOT NULL UNIQUE,
    lower_inr            REAL,
    upper_inr            REAL
);

CREATE TABLE dim_customer (
    customer_key         INTEGER PRIMARY KEY,
    customer_id          TEXT    NOT NULL UNIQUE,
    feedback_record_count INTEGER NOT NULL CHECK (feedback_record_count >= 1),
    distinct_region_count INTEGER NOT NULL    -- >1 flags unstable customer master data
);

CREATE TABLE dim_segment (                    -- populated by the data-mining stage
    segment_key          INTEGER PRIMARY KEY,
    segment_name         TEXT    NOT NULL UNIQUE,
    segment_description  TEXT
);

-- ---------------------------------- FACTS ------------------------------------
CREATE TABLE fact_feedback (
    feedback_id          TEXT    PRIMARY KEY,            -- degenerate dimension = grain key
    customer_key         INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    region_key           INTEGER NOT NULL REFERENCES dim_region(region_key),
    channel_key          INTEGER NOT NULL REFERENCES dim_channel(channel_key),
    product_category_key INTEGER NOT NULL REFERENCES dim_product_category(product_category_key),
    issue_key            INTEGER NOT NULL REFERENCES dim_issue(issue_key),
    source_key           INTEGER NOT NULL REFERENCES dim_feedback_source(source_key),
    rating_key           INTEGER NOT NULL REFERENCES dim_rating(rating_key),
    ref_sentiment_key    INTEGER NOT NULL REFERENCES dim_sentiment(sentiment_key),
    resolution_band_key  INTEGER REFERENCES dim_resolution_band(band_key),
    revenue_tier_key     INTEGER REFERENCES dim_revenue_tier(tier_key),
    -- additive / semi-additive measures
    customer_revenue_inr   REAL    CHECK (customer_revenue_inr > 0),
    order_count            INTEGER CHECK (order_count >= 1),
    resolution_time_hours  REAL    CHECK (resolution_time_hours >= 0),
    -- non-additive measures
    rating                 INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    csat_score             REAL    CHECK (csat_score BETWEEN 1 AND 5),
    avg_order_value_inr    REAL,
    repeat_purchase_flag   INTEGER CHECK (repeat_purchase_flag IN (0,1)),
    -- flags
    is_negative_ref            INTEGER NOT NULL CHECK (is_negative_ref IN (0,1)),
    sla_breach                 INTEGER NOT NULL CHECK (sla_breach IN (0,1)),
    critical_delay             INTEGER NOT NULL CHECK (critical_delay IN (0,1)),
    flag_repeat_without_orders INTEGER NOT NULL CHECK (flag_repeat_without_orders IN (0,1)),
    -- lineage / quality
    dq_flags             TEXT,
    batch_id             TEXT    NOT NULL,
    row_hash             TEXT    NOT NULL,
    source_row           INTEGER NOT NULL
);

CREATE TABLE feedback_text (                  -- unstructured text kept out of the narrow fact
    feedback_id          TEXT    PRIMARY KEY REFERENCES fact_feedback(feedback_id),
    feedback_text        TEXT    NOT NULL,
    text_clean           TEXT    NOT NULL,
    text_len_chars       INTEGER NOT NULL,
    text_len_words       INTEGER NOT NULL
);

CREATE TABLE fact_feedback_nlp (              -- model output, 1:1 with the fact
    feedback_id          TEXT    PRIMARY KEY REFERENCES fact_feedback(feedback_id),
    pred_sentiment_key   INTEGER NOT NULL REFERENCES dim_sentiment(sentiment_key),
    p_negative           REAL    NOT NULL CHECK (p_negative BETWEEN 0 AND 1),
    p_neutral            REAL    NOT NULL CHECK (p_neutral  BETWEEN 0 AND 1),
    p_positive           REAL    NOT NULL CHECK (p_positive BETWEEN 0 AND 1),
    vader_compound       REAL,
    vader_sentiment_key  INTEGER REFERENCES dim_sentiment(sentiment_key),
    topic_issue_key      INTEGER REFERENCES dim_issue(issue_key),   -- theme extracted from TEXT
    nmf_topic_id         INTEGER,
    descriptor           TEXT,
    negative_descriptor_theme TEXT,
    model_version        TEXT    NOT NULL
);

CREATE TABLE fact_feedback_segment (          -- data-mining output, 1:1 with the fact
    feedback_id          TEXT    PRIMARY KEY REFERENCES fact_feedback(feedback_id),
    segment_key          INTEGER NOT NULL REFERENCES dim_segment(segment_key),
    risk_tier            TEXT    NOT NULL
);

-- -------------------------------- INDEXES ------------------------------------
CREATE INDEX ix_fact_customer   ON fact_feedback(customer_key);
CREATE INDEX ix_fact_region     ON fact_feedback(region_key);
CREATE INDEX ix_fact_channel    ON fact_feedback(channel_key);
CREATE INDEX ix_fact_product    ON fact_feedback(product_category_key);
CREATE INDEX ix_fact_issue      ON fact_feedback(issue_key);
CREATE INDEX ix_fact_source     ON fact_feedback(source_key);
CREATE INDEX ix_fact_refsent    ON fact_feedback(ref_sentiment_key);
CREATE INDEX ix_nlp_pred        ON fact_feedback_nlp(pred_sentiment_key);
CREATE INDEX ix_seg_segment     ON fact_feedback_segment(segment_key);
