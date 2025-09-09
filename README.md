# RRF Analytics

This project analyzes equity in Restaurant Revitalization Fund awards using a comprehensive dataset of over 100,000 restaurant grants. The analysis pipeline processes raw grant data, standardizes it, and generates equity metrics to understand funding distribution patterns between disadvantaged and non-disadvantaged businesses.

The system follows a standard ETL → Analysis workflow, handling the full dataset without sampling limitations to ensure statistical accuracy.

---

## Project Structure and Flow

- Workflow: ETL (ingest and standardize) → Analysis (metrics and visuals)
- Scale: Designed for 100k+ rows; runs on the full dataset by default for unbiased results

---

## config.py — Central configuration and field mapping

config.py centralizes file paths, the database connection, and column-name mappings. This protects the pipeline from upstream schema drift and common typos in raw data.

Key elements:
- Paths for project, raw data, and processed data (pathlib-based)
- DB connection string (SQLAlchemy-style DSN) for local PostgreSQL
- COLUMN_ALIASES to correct raw column names (e.g., SocioeconmicIndicator → SocioeconomicIndicator, HubzoneIndicator → HubZoneIndicator)
- DEMOGRAPHIC_FIELDS and RURAL_FIELD_MAPPING for readable labels in reports
- GRANT_PURPOSE_FIELDS listing raw Y/N purpose columns (ETL converts them to {name}_binary)

Why it matters: downstream code can rely on stable, corrected names, which prevents subtle bugs when upstream files change.

---

## etl.py — Ingest, clean, and enrich

etl.py loads the main CSV, normalizes its schema, derives analysis-ready features, and saves results to both disk and PostgreSQL.

Highlights:
- Input selection: uses the largest CSV in data/raw as the primary dataset
- Robust load: treats all columns as strings first, harmonizes common NA markers, and falls back to latin-1 on encoding errors
- Column normalization: applies COLUMN_ALIASES immediately so all later steps use corrected names
- Type coercion: converts ApprovalDate to datetime and GrantAmount from strings (e.g., "$10,000") to numeric
- Indicator normalization: standardizes Y/N columns and the RuralUrbanIndicator so later binary features are unambiguous
- Feature engineering:
  - is_rural: 1 if RuralUrbanIndicator == 'R', else 0
  - is_disadvantaged_core: 1 if any of {SocioeconomicIndicator, WomenOwnedIndicator, VeteranIndicator, LMIIndicator, HubZoneIndicator} == 'Y'
  - is_disadvantaged: logical OR of is_disadvantaged_core and is_rural
  - {purpose}_binary: one binary column per grant purpose listed in GRANT_PURPOSE_FIELDS
- Persistence: writes a timestamped CSV to data/processed and a full snapshot to PostgreSQL (table: rrf_data, replace on each run)

Non‑elementary logic worth noting:
- Normalization guardrails: missing/blank values in Y/N indicators are treated as 'N' to avoid accidental positives
- GrantAmount parsing removes currency symbols and commas, then coerces to numeric with errors='coerce' (invalid values become NaN)

---

## analysis.py — Metrics and visualizations

analysis.py loads the processed data (prefers the database; falls back to the latest processed CSV) and produces descriptive stats, an equity ratio, and charts.

What it computes:
- Descriptives: counts and shares for each demographic indicator and the rural flag
- Equity ratio: average GrantAmount for disadvantaged businesses divided by the average for non‑disadvantaged businesses
- Visuals: distribution of grant amounts, disadvantaged vs. non‑disadvantaged share, top states, and grant purpose usage

Important implementation details:
- Full‑table reads (no LIMIT) ensure unbiased averages at this dataset size
- Defensive re‑coercion keeps GrantAmount numeric before group means
- The pie chart explicitly locks label order to avoid swapped labels

---

## Key definitions
- is_disadvantaged_core: any of SocioeconomicIndicator, WomenOwnedIndicator, VeteranIndicator, LMIIndicator, or HubZoneIndicator is 'Y'
- is_disadvantaged: is_disadvantaged_core OR is_rural
- Equity ratio: mean(GrantAmount | is_disadvantaged == 1) / mean(GrantAmount | is_disadvantaged == 0)

---

## Getting started

Requirements: Python 3.10+, Docker

1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
# Optional (uv):
# uv pip install -r requirements.txt
```

3) Start PostgreSQL locally
```bash
docker-compose up -d
```

4) Place the raw data at
```
data/raw/rrf.csv
```

5) Run ETL (full dataset)
```bash
python etl.py
```

- Optional: process a sample
```bash
python -c "import etl; etl.run_etl(1000)"
```

6) Run analysis
```bash
python analysis.py
```

7) Inspect outputs
```bash
ls -lt data/processed | head -5
```

Notes
- DB settings live in config.py (DB_CONFIG) and align with docker-compose defaults (user=postgres, password=postgres, db=rrf_analytics)
- ETL writes a timestamped CSV and replaces the rrf_data table on each run
- Analysis prefers the DB and falls back to the latest processed CSV if needed

