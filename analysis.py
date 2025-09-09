#!/usr/bin/env python3
"""
Analysis for RRF Analytics Project
"""
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from config import PROCESSED_DATA_DIR, DB_CONFIG, DEMOGRAPHIC_FIELDS, RURAL_FIELD_MAPPING

def load_data():
    """Load processed data"""
    try:
        engine = create_engine(DB_CONFIG)
        df = pd.read_sql("SELECT * FROM rrf_data", engine)
        print(f"Loaded {len(df)} records from database")
        return df
    except:
        # Fallback to CSV
        processed_files = list(PROCESSED_DATA_DIR.glob("rrf_processed_*.csv"))
        if not processed_files:
            raise FileNotFoundError("No processed data found. Run ETL first.")
        
        latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} records from {latest_file}")
        return df

def analyze_demographics(df):
    """Analyze demographics"""
    print("\n=== DEMOGRAPHICS ===")
    if 'is_disadvantaged' in df.columns:
        disadvantaged = df['is_disadvantaged'].sum()
        total = len(df)
        print(f"Disadvantaged businesses: {disadvantaged:,} ({disadvantaged/total:.1%})")
        
        # Core SBA demographic categories (Y/N fields)
        for col, label in DEMOGRAPHIC_FIELDS.items():
            if col in df.columns:
                count = (df[col] == 'Y').sum()
                print(f"{label}: {count:,} ({count/total:.1%})")
        
        # Rural category (binary field)
        for col, label in RURAL_FIELD_MAPPING.items():
            if col in df.columns:
                count = df[col].sum()
                print(f"{label}: {count:,} ({count/total:.1%})")

def analyze_equity(df):
    """Core equity analysis"""
    print("\n=== EQUITY ANALYSIS ===")
    
    if 'is_disadvantaged' not in df.columns or 'GrantAmount' not in df.columns:
        print("Missing required columns")
        return
    
    # GrantAmount should already be numeric from ETL, but ensure it's clean
    grant_amounts = pd.to_numeric(df['GrantAmount'], errors='coerce')
    
    disadvantaged = grant_amounts[df['is_disadvantaged'] == 1].dropna()
    non_disadvantaged = grant_amounts[df['is_disadvantaged'] == 0].dropna()
    
    print(f"DISADVANTAGED: {len(disadvantaged):,} businesses, avg grant: ${disadvantaged.mean():,.0f}")
    print(f"NON-DISADVANTAGED: {len(non_disadvantaged):,} businesses, avg grant: ${non_disadvantaged.mean():,.0f}")
    
    if non_disadvantaged.mean() > 0:
        ratio = disadvantaged.mean() / non_disadvantaged.mean()
        print(f"EQUITY RATIO: {ratio:.2f} (disadvantaged get {ratio:.0%} of non-disadvantaged funding)")

def create_plots(df):
    """Create basic plots"""
    print("\n=== CREATING PLOTS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Grant amounts (should already be numeric from ETL)
    if 'GrantAmount' in df.columns:
        grant_data = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
        axes[0,0].hist(grant_data / 1000, bins=30, alpha=0.7)
        axes[0,0].set_title('Grant Distribution ($K)')
        axes[0,0].set_xlabel('Grant Amount (thousands)')
    
    # Demographics
    if 'is_disadvantaged' in df.columns:
        demo_counts = df['is_disadvantaged'].value_counts().reindex([0, 1], fill_value=0)
        axes[0,1].pie(demo_counts.values, labels=['Non-Disadvantaged', 'Disadvantaged'], autopct='%1.1f%%')
        axes[0,1].set_title('Business Demographics')
    
    # States
    if 'BusinessState' in df.columns:
        states = df['BusinessState'].value_counts().head(8)
        axes[1,0].bar(range(len(states)), states.values)
        axes[1,0].set_title('Top States')
        axes[1,0].set_xticks(range(len(states)))
        axes[1,0].set_xticklabels(states.index, rotation=45)
    
    # Grant purposes
    purpose_cols = [col for col in df.columns if 'purpose' in col and col.endswith('_binary')]
    if purpose_cols:
        purposes = df[purpose_cols].sum().sort_values(ascending=False)
        purpose_names = [col.replace('_binary', '').replace('grant_purpose_', '').title() for col in purposes.index]
        axes[1,1].barh(range(len(purposes)), purposes.values)
        axes[1,1].set_title('Grant Purposes')
        axes[1,1].set_yticks(range(len(purposes)))
        axes[1,1].set_yticklabels(purpose_names)
    
    plt.tight_layout()
    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    plot_file = PROCESSED_DATA_DIR / "analysis_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plots: {plot_file}")
    plt.show()

def run_analysis():
    """Run complete analysis"""
    print("=== RRF ANALYSIS ===")
    df = load_data()
    analyze_demographics(df)
    analyze_equity(df)
    
    try:
        create_plots(df)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\n=== COMPLETE ===")
    return df

if __name__ == "__main__":
    run_analysis()
