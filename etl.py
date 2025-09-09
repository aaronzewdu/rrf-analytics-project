#!/usr/bin/env python3
"""
ETL Pipeline for RRF Analytics Project
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, GRANT_PURPOSE_FIELDS, DEMOGRAPHIC_FIELDS, DB_CONFIG, COLUMN_ALIASES
import psycopg2

def find_csv_file():
    """Find the RRF CSV file"""
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {RAW_DATA_DIR}")
    return max(csv_files, key=lambda f: f.stat().st_size)

def load_and_clean_data(sample_size=None):
    """Load and clean RRF data"""
    file_path = find_csv_file()
    print(f"Loading data from: {file_path}")
    
    # Load CSV with encoding handling
    try:
        df = pd.read_csv(file_path, dtype=str, na_values=['', 'NULL', 'N/A'], nrows=sample_size)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, dtype=str, na_values=['', 'NULL', 'N/A'], encoding='latin-1', nrows=sample_size)
    
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Normalize column names to handle common variations and typos
    df = df.rename(columns=COLUMN_ALIASES)
    normalized_columns = [col for col in COLUMN_ALIASES.keys() if col in df.columns]
    if normalized_columns:
        print(f"Normalized column names: {normalized_columns}")
    
    # Basic cleaning
    if 'ApprovalDate' in df.columns:
        df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'], errors='coerce')
    
    if 'GrantAmount' in df.columns:
        df['GrantAmount'] = pd.to_numeric(df['GrantAmount'].str.replace(r'[\\$,]', '', regex=True), errors='coerce')
    
    # Handle RuralUrbanIndicator separately (R/U values, not Y/N)
    if 'RuralUrbanIndicator' in df.columns:
        df['RuralUrbanIndicator'] = (
            df['RuralUrbanIndicator']
            .astype(str)
            .str.strip()
            .str.upper()
        )
    
    # Fill missing binary indicators and normalize case (Y/N fields only)
    binary_fields = [field for field in list(DEMOGRAPHIC_FIELDS.keys()) + GRANT_PURPOSE_FIELDS 
                     if field != 'RuralUrbanIndicator']
    for field in binary_fields:
        if field in df.columns:
            df[field] = (
                df[field]
                .astype(str)
                .str.strip()
                .str.upper()
                .fillna('N')
                .replace(['', 'NAN', 'NONE'], 'N')
            )
    
    # Create disadvantaged business flag
    # Core SBA disadvantaged categories: Socioeconomic, Women-Owned, Veteran, LMI, HubZone
    # Note: Using corrected column names after alias processing
    disadvantaged_indicators = ['SocioeconomicIndicator', 'WomenOwnedIndicator', 'VeteranIndicator', 'LMIIndicator', 'HubZoneIndicator']
    available_indicators = [col for col in disadvantaged_indicators if col in df.columns]
    
    if available_indicators:
        # Create binary columns and overall flag for core SBA categories
        for col in available_indicators:
            df[f'{col}_binary'] = (df[col] == 'Y').astype(int)
        df['is_disadvantaged_core'] = df[[f'{col}_binary' for col in available_indicators]].any(axis=1).astype(int)
        
        # Include rural as disadvantaged (per project definition)
        # This creates broader definition: is_disadvantaged = core SBA + rural
        if 'RuralUrbanIndicator' in df.columns:
            df['is_rural'] = (df['RuralUrbanIndicator'] == 'R').astype(int)
            df['is_disadvantaged'] = (df['is_disadvantaged_core'] | df['is_rural']).astype(int)
        else:
            df['is_disadvantaged'] = df['is_disadvantaged_core']
    
    # Create binary columns for grant purposes (for plotting)
    for col in GRANT_PURPOSE_FIELDS:
        if col in df.columns:
            df[f'{col}_binary'] = (df[col] == 'Y').astype(int)
    
    return df

def save_data(df):
    """Save processed data"""
    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = PROCESSED_DATA_DIR / f"rrf_processed_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved to: {csv_file}")
    
    # Save to database
    try:
        engine = create_engine(DB_CONFIG)
        df.to_sql('rrf_data', engine, if_exists='replace', index=False)
        print(f"Saved {len(df)} records to database")
    except Exception as e:
        print(f"Database save failed: {e}")

def run_etl(sample_size=None):
    """Run complete ETL pipeline"""
    print("=== RRF ETL Pipeline ===")
    df = load_and_clean_data(sample_size)
    save_data(df)
    print(f"=== Complete: {len(df)} records ===")
    return df

def test_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(
            host="localhost", database="rrf_analytics",
            user="postgres", password="postgres"
        )
        print("PostgreSQL connection successful")
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
    else:
        run_etl()
