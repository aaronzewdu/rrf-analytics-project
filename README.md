# RRF Analytics

**Equity analysis on 100K+ Restaurant Revitalization Fund (RRF) grants.**

_**Goal:**_ compare funding for disadvantaged vs. non‑disadvantaged businesses.

_**Finding:**_ disadvantaged businesses receive about 87% of non‑disadvantaged average funding.

_**RRF in one line:**_ a pandemic-era SBA grant that helped restaurants cover losses.

This project examines how those dollars were distributed.

---

## Quick start
Prereqs: Python 3.10+, Docker

1) Start PostgreSQL
```bash
docker-compose up -d
```

2) Install dependencies
```bash
pip install -r requirements.txt
# (optional) uv: uv pip install -r requirements.txt
```

3) Add data. Place the raw CSV at:
```
data/raw/rrf.csv
```

4) Run the pipeline
```bash
python etl.py               # Process full dataset → DB + CSV
python analysis.py          # Compute metrics + save visuals
python equity_analysis.py   # Deeper statistical equity (CIs, robustness)
```

5) See results
- Database:
  - table rrf_data (local Postgres)
- Files in "data/processed/..."
  - timestamped and cleaned CSV,
  - analysis and equity analysis JSONs,
  - PNG plots)

---

## Essential files (what they do)
- config.py
  - Paths and database settings
  - Standardizes raw column names
- etl.py
  - Loads raw CSV
  - Cleans data
  - Creates flags (disadvantaged, rural, grant purposes)
  - Saves to DB and CSV
- analysis.py
  - Reads processed data
  - Computes equity metrics
  - Saves charts and a summary JSON
- equity_analysis.py
  - Statistical equity
    - Confidence intervals
    - Geographic comparisons
    - Robustness and Sensitivity
  - Writes equity_analysis_results.json and extra plots
- purpose_helpers.py
  - Helpers to list and label grant purpose columns
- utils.py
  - Simple logger and shared utilities
- main.py **(optional)**
  - Prints project summary and quick command tips

---

## Notes
- Data source:
  - SBA RRF FOIA dataset (100K+ grants) - https://data.sba.gov/dataset/rrf-foia
- Defaults are safe:
  - analysis prefers the database and falls back to the latest processed CSV
- Docker only for the database; the Python programs run without flags

