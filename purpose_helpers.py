#!/usr/bin/env python3

# purpose-related helper functions

def list_purpose_binary_cols(df):
    return [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]

def clean_purpose_label(col_name, keep_underscore=False):
    name = col_name.replace('_binary', '')
    name = name.replace('grant_purpose_', '').replace('grant_purp_', '')
    
    # special handling
    name = name.replace('cons_outdoor_seating', 'outdoor_seating_construction')
    
    # underscores to spaces
    if not keep_underscore:
        name = name.replace('_', ' ')
    
    return name.title()