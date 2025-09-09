"""
Simple configuration for RRF Analytics Project
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database
DB_CONFIG = "postgresql://postgres:postgres@localhost:5432/rrf_analytics"

# Field mappings (using corrected column names after alias processing)
DEMOGRAPHIC_FIELDS = {
    'SocioeconomicIndicator': 'Socioeconomic',  # Corrected from 'SocioeconmicIndicator' by COLUMN_ALIASES
    'WomenOwnedIndicator': 'Women-Owned',
    'VeteranIndicator': 'Veteran',
    'LMIIndicator': 'Low-Income',
    'HubZoneIndicator': 'HubZone'  # Corrected from 'HubzoneIndicator' by COLUMN_ALIASES
}

# Rural is handled separately since it uses R/U values, not Y/N
RURAL_FIELD_MAPPING = {
    'is_rural': 'Rural'
}

# Column name aliases to fix typos and standardize names
# Maps FROM current (possibly misspelled) names TO corrected names
COLUMN_ALIASES = {
    'SocioeconmicIndicator': 'SocioeconomicIndicator',  # Fix missing 'o' typo
    'HubzoneIndicator': 'HubZoneIndicator',  # Fix capitalization
}

GRANT_PURPOSE_FIELDS = [
    "grant_purpose_payroll", "grant_purpose_rent", "grant_purpose_debt",
    "grant_purpose_food", "grant_purpose_operations", "grant_purpose_supplies", 
    "grant_purpose_utility", "grant_purp_cons_outdoor_seating",
    "grant_purpose_covered_supplier", "grant_purpose_maintenance_indoor"
]
