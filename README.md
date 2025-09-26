# RRF Analytics

Evidence-based analysis of the Restaurant Revitalization Fund (RRF) with a focus on funding equity and a high-accuracy purpose classifier.

Why it matters
- Public funds should be allocated equitably. This project quantifies equity and reveals patterns that inform policy and program design.

Key findings (from current runs)
- Equity gap: disadvantaged businesses receive about 64% of non‑disadvantaged median funding (95% CI in outputs)
- Geographic disparities: large spread by state (e.g., NY ≈ 55%, FL ≈ 98%)
- ML success: ~95% micro‑F1 multi‑label classifier for grant purposes

How it works
- Data → ETL → Database/CSV → Analysis → Visualizations/JSON → ML classifier + artifacts
- DB preferred, CSV fallback is automatic

Quick start
Prereqs: Python 3.10+. PostgreSQL optional (scripts fall back to CSV if DB is unavailable).

1) Install dependencies
```bash path=null start=null
pip install -r requirements.txt
```
2) Add data (raw CSV)
```bash path=null start=null
data/raw/rrf.csv
```
3) Run
```bash path=null start=null
python etl.py               # process → DB + CSV
python analysis.py          # summary + purpose visuals
python equity_analysis.py   # equity ratios + CIs + plots
python prediction.py        # 95% purpose classifier + metrics
```

Outputs
- data/processed/
  - rrf_processed_*.csv — cleaned dataset
  - analysis_summary.json — core metrics and distributions
  - equity_analysis_results.json — CI‑backed equity summary
  - 01_purpose_profile.png, 02_purpose_cooccurrence.png — purpose visuals
  - equity_01_state_comparison.png, equity_02_allocation_gap.png — equity visuals
  - purposes_model.joblib — trained classifier
  - predictions.csv — purpose probabilities and flags
  - metrics.json — ML performance and insights

Essential scripts
- etl.py — load raw CSV, clean/normalize, create indicators, save to DB/CSV
- analysis.py — summary metrics + purpose profiles/co‑occurrence
- equity_analysis.py — equity ratios with bootstrap CIs, geo and demographic breakdowns
- prediction.py — multi‑label purpose classifier (~95% micro‑F1)
- config.py — paths, DB URL, column mappings

Data
- SBA RRF FOIA dataset (100K+ grants): https://data.sba.gov/dataset/rrf-foia

Repro notes
- If PostgreSQL is running and config.DB_CONFIG is reachable, analysis uses the DB; otherwise it falls back to the latest processed CSV.
- SHOW_PLOTS=1 enables interactive figures; RRF_LOG_LEVEL controls logging verbosity.

