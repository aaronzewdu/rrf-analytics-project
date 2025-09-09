#!/usr/bin/env python3
"""
Restaurant Revitalization Fund Analytics Project

Equity analysis and predictive modeling on 100K+ SBA records.
Dataset: https://data.sba.gov/dataset/rrf-foia

Goals:
1. Assess if disadvantaged businesses received proportional funding
2. Predict grant purposes based on business characteristics

Tech Stack: Python, PostgreSQL, Docker, pandas, scikit-learn

Current Status: Phase 1 Complete (ETL + Basic Analysis)
Finding: Disadvantaged businesses receive 87% of non-disadvantaged funding
"""

def main():
    print(__doc__)
    print("\n" + "="*60)
    print("QUICK START")
    print("="*60)
    print("1. docker-compose up -d     # Start PostgreSQL")
    print("2. python etl.py test       # Test connection")
    print("3. python etl.py            # Process 100K records")
    print("4. python analysis.py       # Generate insights")
    print("\n Sample Data:")
    print("python -c 'import etl; etl.run_etl(1000)'")
    print("\n Files:")
    print("- etl.py        → Data processing")
    print("- analysis.py   → Equity analysis & plots")
    print("- game_plan.md  → Project roadmap")
    print("- config.py     → Settings")
    
if __name__ == "__main__":
    main()
