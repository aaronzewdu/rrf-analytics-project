# RRF Analytics

Evidence-based analysis of the Restaurant Revitalization Fund (RRF) with a focus on funding equity and business insights. Surfaces clear, public-ready visuals that compare purpose selections across business types and contexts.

Why it matters
- Public funds should be allocated equitably. This project quantifies equity with statistical rigor (bootstrap confidence intervals) and reveals non-obvious business patterns that inform policy and program design.

Key findings (from current runs)
- Equity gap: disadvantaged businesses receive about 64% of non‑disadvantaged median funding (95% CI in outputs)
- Geographic disparities: large spread by state (e.g., NY ≈ 55%, FL ≈ 98%)
- Business insights: statistical analysis reveals non-obvious patterns in purpose selection across business types

How it works
- Data → ETL → Database/CSV → Analysis → Visualizations/JSON + statistical insights
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
python etl.py                               # process → DB + CSV
python analysis.py                          # summary + purpose profiles/co‑occurrence
python equity_analysis.py                   # equity ratios + CIs + plots
python business_purpose_analysis.py         # business-type purpose insights (JSON)
python business_purpose_analysis_visuals.py # public-ready purpose visuals from insights JSON
```

Outputs
- data/processed/
  - rrf_processed_*.csv — cleaned dataset
  - analysis_summary.json — core metrics and distributions
  - equity_analysis_results.json — CI‑backed equity summary
  - business_purpose_insights.json — per‑type purpose differences and key insights
  - 01_purpose_profile.png, 02_purpose_cooccurrence.png — core purpose visuals
  - 03_purpose_vs_peers_dumbbell.png — bakeries/brewpubs vs industry average
  - 04_bakery_vs_peers_bars.png, 04_brewpub_vs_peers_bars.png — difference vs peers
  - 05_rural_urban_rent_by_type.png — urban vs rural rent selection
  - 06_equity_bakeries_supplies.png — supplies selection: disadvantaged vs standard bakeries
  - equity_01_state_comparison.png, equity_02_allocation_gap.png — equity visuals
  - key_findings.txt — plain text summary of actionable insights

Essential scripts
- etl.py — load raw CSV, clean/normalize, create indicators, save to DB/CSV
- analysis.py — summary metrics + purpose profiles/co‑occurrence
- equity_analysis.py — equity ratios with bootstrap CIs, geo and demographic breakdowns
- business_purpose_analysis.py — detects credible purpose differences by business type, geography, and disadvantaged status; writes business_purpose_insights.json
- business_purpose_analysis_visuals.py — generates figures (03–06) from insights JSON
- config.py — paths, DB URL, column mappings

Data
- SBA RRF FOIA dataset (100K+ grants): https://data.sba.gov/dataset/rrf-foia

Repro notes
- If PostgreSQL is running and config.DB_CONFIG is reachable, analysis uses the DB; otherwise it falls back to the latest processed CSV.
- SHOW_PLOTS=1 enables interactive figures; RRF_LOG_LEVEL controls logging verbosity.

