# Restaurant Revitalization Fund Analytics Project

## Overview
Analytics pipeline combining equity analysis and predictive modeling for the Restaurant Revitalization
Fund (RRF) dataset.

### Dataset
- CSV with >100,000 entries obtained from the Office of Capital Access on the
U.S. Small Business Administration website.
    https://data.sba.gov/dataset/rrf-foia

### Goals
1. **Equity Analysis:** To assess if disadvantaged businesses received proportional funding

2. **Predictive Modeling:** Predict which grant purposes restaurants need based on their characteristics

### Tech Stack
- Python 3.13.7 (venv)
- pandas, scikit-learn, sqlalchemy
- PostgreSQL (via Docker)
- Streamlit (for dashboards and visualizations)

### DB Config
- **Host:** localhost
- **Port:** 5432
- **DB:** rrf_analytics
- **User/Psswd:** postgres/postgres

### Todo List
- ~~python venv~~
- ~~docker setup~~
- ~~postgre setup within docker~~
- ~~test db connection~~
- create proj struct
- build etl pipeline
- equity analysis (1)
- train models (2)
- containerization checks
- streamlist setup and visualizations
- continue proj, etc... tbd...