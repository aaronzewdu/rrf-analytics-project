# simple configs

from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# db
DB_CONFIG = "postgresql://postgres:postgres@localhost:5432/rrf_analytics"

# field mappings (using corrected column names after alias processing)
DEMOGRAPHIC_FIELDS = {
    'SocioeconomicIndicator': 'Socioeconomic',  # corrected from 'SocioeconmicIndicator' by COLUMN_ALIASES
    'WomenOwnedIndicator': 'Women-Owned',
    'VeteranIndicator': 'Veteran',
    'LMIIndicator': 'Low-Income',
    'HubZoneIndicator': 'HubZone'  # corrected from 'HubzoneIndicator' by COLUMN_ALIASES
}

# rural is handled separately since it uses R/U values, not Y/N
RURAL_FIELD_MAPPING = {
    'is_rural': 'Rural'
}

# column name aliases to fix typos and standardize names
# maps FROM current (possibly misspelled) names TO corrected names
COLUMN_ALIASES = {
    'SocioeconmicIndicator': 'SocioeconomicIndicator',  # fix missing 'o' typo
    'HubzoneIndicator': 'HubZoneIndicator',  # fix capitalization
}

# grant purpose fields from the raw data (10 total)
# note: outdoor seating has a different naming pattern (grant_purp_ instead of grant_purpose_)
GRANT_PURPOSE_FIELDS = [
    "grant_purpose_payroll", "grant_purpose_rent", "grant_purpose_debt",
    "grant_purpose_food", "grant_purpose_operations", "grant_purpose_supplies", 
    "grant_purpose_utility", "grant_purp_cons_outdoor_seating",  # different prefix pattern
    "grant_purpose_covered_supplier", "grant_purpose_maintenance_indoor"
]
