#!/usr/bin/env python3
# finds where business types select grant purposes more or less than peers

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    DB_CONFIG, PROCESSED_DATA_DIR, GRANT_PURPOSE_FIELDS,
    DEMOGRAPHIC_FIELDS, COLUMN_ALIASES
)
from purpose_helpers import list_purpose_binary_cols, clean_purpose_label
from utils import setup_logger

logger = setup_logger(__name__)

# comparable business groups (mutually exclusive)
COMPARABLE_GROUPS = {
    'mobile': ['Food Truck', 'Food Cart', 'Food Stand'],
    'caterer': ['Caterer'],
    'fixed_food': ['Restaurant', 'Bakery', 'Snack and Nonalcoholic Beverage Bar'],
    'fixed_beverage': ['Bar', 'Saloon', 'Lounge', 'Tavern', 'Brewpub', 'Tasting Room', 'Taproom'],
    'producers': ['Brewery', 'Distillery', 'Winery', 'Licensed Alcohol Producer'],
    'hospitality': ['Inn'],
    'other': ['Other', 'Mixed']
}

# structural expectations matrix
# 'high' expected more often; 'low' expected rare; 'exclude' not applicable; 'neutral' no prior
STRUCTURAL_EXPECTATIONS = {
    'mobile': {  # Food trucks, carts, stands
        'payroll': 'neutral',
        'rent': 'exclude',
        'debt': 'neutral',
        'food': 'high',
        'operations': 'high',
        'supplies': 'high',
        'utility': 'exclude',
        'outdoor_seating': 'exclude',
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'exclude'
    },
    'caterer': {
        'payroll': 'neutral',
        'rent': 'low',
        'debt': 'neutral',
        'food': 'high',
        'operations': 'high',
        'supplies': 'high',
        'utility': 'low',
        'outdoor_seating': 'exclude',
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'low'
    },
    'bakery': {
        'payroll': 'neutral',
'rent': 'neutral',  # changed from 'high' - some own property
        'debt': 'neutral',
'food': 'neutral',  # changed from 'high' - may have had inventory
'operations': 'neutral',  # changed from 'high'
'supplies': 'neutral',  # changed from 'high' - may have equipment
'utility': 'neutral',  # changed from 'high' - varies by size
        'outdoor_seating': 'low',
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'neutral'
    },
'bar': {  # bars, taverns, lounges
        'payroll': 'high',
        'rent': 'high',
        'debt': 'neutral',
        'food': 'low',
        'operations': 'high',
        'supplies': 'high',
        'utility': 'high',
        'outdoor_seating': 'neutral',
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'neutral'
    },
    'brewpub': {
'payroll': 'neutral',  # changed from 'high'
'rent': 'neutral',  # changed from 'high'
        'debt': 'neutral',
'food': 'neutral',  # changed from 'high' - varies by food program
'operations': 'neutral',  # changed from 'high'
'supplies': 'neutral',  # changed from 'high' - may have equipment
'utility': 'neutral',  # changed from 'high'
'outdoor_seating': 'neutral',  # changed from 'high' - not all have patios
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'neutral'
    },
'producer': {  # breweries, distilleries, wineries
        'payroll': 'neutral',
        'rent': 'high',
        'debt': 'high',
        'food': 'low',
        'operations': 'high',
        'supplies': 'high',
        'utility': 'high',
        'outdoor_seating': 'high',
        'covered_supplier': 'high',
        'maintenance_indoor': 'neutral'
    },
    'restaurant': {
'payroll': 'neutral',  # changed from 'high' - varies by family vs corporate
'rent': 'neutral',  # changed from 'high' - some own
        'debt': 'neutral',
'food': 'neutral',  # changed from 'high' - inventory varies
'operations': 'neutral',  # changed from 'high'
        'supplies': 'neutral',
'utility': 'neutral',  # changed from 'high'
        'outdoor_seating': 'neutral',
        'covered_supplier': 'neutral',
'maintenance_indoor': 'neutral'  # changed from 'high'
    },
'coffee_shop': {  # snack and nonalcoholic beverage bars
        'payroll': 'neutral',
        'rent': 'high',
        'debt': 'neutral',
        'food': 'high',
        'operations': 'high',
        'supplies': 'neutral',
        'utility': 'high',
        'outdoor_seating': 'low',
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'neutral'
    },
    'inn': {
        'payroll': 'high',
        'rent': 'high',
        'debt': 'high',
        'food': 'neutral',
        'operations': 'high',
        'supplies': 'neutral',
        'utility': 'high',
        'outdoor_seating': 'low',
        'covered_supplier': 'neutral',
        'maintenance_indoor': 'high'
    }
}

def load_data() -> pd.DataFrame:
    # load processed data from db or csv fallback
    try:
        import psycopg2
        import sqlalchemy
        engine = sqlalchemy.create_engine(DB_CONFIG)
        df = pd.read_sql_query("SELECT * FROM rrf_data", engine)
        logger.info(f"Loaded {len(df)} records from database")
    except Exception as e:
        logger.warning(f"Database load failed: {e}. Falling back to CSV.")
        csv_files = sorted(Path(PROCESSED_DATA_DIR).glob('rrf_processed_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No processed data found in {PROCESSED_DATA_DIR}")
        df = pd.read_csv(csv_files[-1])
        logger.info(f"Loaded {len(df)} records from {csv_files[-1].name}")
    
    return df

def classify_business_type(restaurant_type: str) -> Tuple[str, str]:
    # classify restaurant type into (group, specific type)
    if pd.isna(restaurant_type):
        return ('other', 'unknown')
    
    rt_lower = restaurant_type.lower()
    
    # check mobile vendors first (most restrictive)
    if any(term in rt_lower for term in ['food truck', 'food cart', 'food stand']):
        return ('mobile', 'mobile')
    
    # caterer
    if 'caterer' in rt_lower:
        return ('caterer', 'caterer')
    
    # producers
    if any(term in rt_lower for term in ['brewery', 'distillery', 'winery', 'licensed alcohol producer']):
        if 'brewpub' in rt_lower:
            return ('fixed_beverage', 'brewpub')
        return ('producers', 'producer')
    
    # hospitality
    if 'inn' in rt_lower:
        return ('hospitality', 'inn')
    
    # fixed beverage
    if any(term in rt_lower for term in ['bar', 'saloon', 'lounge', 'tavern', 'taproom', 'tasting room']):
        if 'brewpub' in rt_lower:
            return ('fixed_beverage', 'brewpub')
        return ('fixed_beverage', 'bar')
    
    # bakery
    if 'bakery' in rt_lower:
        return ('fixed_food', 'bakery')
    
    # coffee shops
    if 'snack and nonalcoholic beverage bar' in rt_lower:
        return ('fixed_food', 'coffee_shop')
    
    # restaurant (catch‑all for fixed food service)
    if 'restaurant' in rt_lower:
        return ('fixed_food', 'restaurant')
    
    return ('other', 'other')

def get_expectations(business_type: str) -> Dict[str, str]:
    # get structural expectations for a business type
    return STRUCTURAL_EXPECTATIONS.get(business_type, {k: 'neutral' for k in STRUCTURAL_EXPECTATIONS['restaurant'].keys()})

def calculate_surprise_score(
    segment_rate: float,
    baseline_rate: float,
    segment_n: int,
    baseline_n: int,
    expectation: str
) -> Optional[Dict]:
    # compute z score and filter by expectations
    # return none if obvious/expected or not strong enough
    # minimum sample size
    if segment_n < 30:
        return None
    
    # pooled proportion and standard error
    pooled_p = (segment_rate * segment_n + baseline_rate * baseline_n) / (segment_n + baseline_n)
    
    # avoid division by zero
    if pooled_p == 0 or pooled_p == 1:
        return None
    
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/segment_n + 1/baseline_n))
    
    if se == 0:
        return None
    
    # z score
    z = (segment_rate - baseline_rate) / se
    
    # expectation filter
    if expectation == 'high' and z > 0:
        return None  # Expected pattern, don't report
    elif expectation in ['low', 'exclude'] and z < 0:
        return None  # Expected pattern, don't report
    elif abs(z) < 3.0:  # Standardized threshold
        return None
    
    # confidence interval
    ci_lower = segment_rate - 1.96 * se
    ci_upper = segment_rate + 1.96 * se
    
    # effect size (odds ratio)
    if baseline_rate > 0 and baseline_rate < 1 and segment_rate > 0 and segment_rate < 1:
        odds_segment = segment_rate / (1 - segment_rate)
        odds_baseline = baseline_rate / (1 - baseline_rate)
        odds_ratio = odds_segment / odds_baseline
    else:
        odds_ratio = None
    
    return {
        'z_score': z,
        'p_value': 2 * (1 - scipy_stats.norm.cdf(abs(z))),
        'segment_rate': segment_rate,
        'baseline_rate': baseline_rate,
        'difference': segment_rate - baseline_rate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'odds_ratio': odds_ratio,
        'segment_n': segment_n,
        'baseline_n': baseline_n
    }

def apply_multiple_testing_correction(results: Dict) -> Dict:
    # apply benjamini–hochberg fdr to all p-values
    # collect all p-values and their locations
    all_tests = []
    
    # from business types
    for bt, bt_data in results['business_types'].items():
        for purpose, stats in bt_data['purposes'].items():
            all_tests.append({
                'type': 'business',
                'business_type': bt,
                'purpose': purpose,
                'p_value': stats['p_value'],
                'z_score': stats['z_score']
            })
    
    # from geographic
    for bt, geo_list in results['geographic'].items():
        for geo_data in geo_list:
            # Calculate p-value from z-score if not present
            p_val = 2 * (1 - scipy_stats.norm.cdf(abs(geo_data['z_score'])))
            all_tests.append({
                'type': 'geographic',
                'business_type': bt,
                'purpose': geo_data['purpose'],
                'p_value': p_val,
                'z_score': geo_data['z_score']
            })
    
    # from disadvantaged
    for bt, disadv_list in results['disadvantaged'].items():
        for disadv_data in disadv_list:
            p_val = 2 * (1 - scipy_stats.norm.cdf(abs(disadv_data['z_score'])))
            all_tests.append({
                'type': 'disadvantaged',
                'business_type': bt,
                'purpose': disadv_data['purpose'],
                'p_value': p_val,
                'z_score': disadv_data['z_score']
            })
    
    if not all_tests:
        return results
    
    # apply benjamini–hochberg correction manually
    p_values = np.array([t['p_value'] for t in all_tests])
    n_tests = len(p_values)
    
    # sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # calculate adjusted p-values using bh method
    p_adjusted = np.zeros(n_tests)
    for i in range(n_tests-1, -1, -1):
        if i == n_tests - 1:
            p_adjusted[i] = sorted_p[i]
        else:
            p_adjusted[i] = min(p_adjusted[i+1], sorted_p[i] * n_tests / (i+1))
    
    # reorder to match original order
    p_adjusted_orig = np.zeros(n_tests)
    p_adjusted_orig[sorted_indices] = p_adjusted
    
    # determine which are significant
    rejected = p_adjusted_orig <= 0.05
    
    # create lookup for adjusted p-values
    adjustment_lookup = {}
    for i, test in enumerate(all_tests):
        key = (test['type'], test['business_type'], test['purpose'])
        adjustment_lookup[key] = {
            'p_adjusted': float(p_adjusted_orig[i]),
            'significant_after_correction': bool(rejected[i])
        }
    
    # add adjusted p-values to results
    for bt, bt_data in results['business_types'].items():
        for purpose, stats in bt_data['purposes'].items():
            key = ('business', bt, purpose)
            if key in adjustment_lookup:
                stats['p_adjusted'] = adjustment_lookup[key]['p_adjusted']
                stats['significant_fdr'] = adjustment_lookup[key]['significant_after_correction']
    
    # add test statistics to results
    results['test_statistics'] = {
        'total_tests_performed': len(all_tests),
        'significant_before_correction': sum(1 for t in all_tests if t['p_value'] < 0.05),
        'significant_after_fdr': sum(rejected),
        'correction_method': 'Benjamini-Hochberg FDR'
    }
    
    return results

def analyze_purpose_preferences(df: pd.DataFrame) -> Dict:
    # main analysis of purpose preferences
    results = {
        'business_types': {},
        'geographic': {},
        'disadvantaged': {},
        'key_insights': [],
        'diagnostics': {
            'tests_performed': 0,
            'filtered_by_expectations': 0,
            'filtered_by_significance': 0
        }
    }
    
    # add business classification
    df['business_group'], df['business_type'] = zip(*df['RestaurantType'].apply(classify_business_type))
    
    # log business type distribution
    bt_counts = df['business_type'].value_counts()
    logger.info(f"Business type distribution: {bt_counts.head(10).to_dict()}")
    
    # get purpose columns
    purpose_cols = list_purpose_binary_cols(df)
    purpose_names = {col: clean_purpose_label(col.replace('_binary', '')) for col in purpose_cols}
    
    # calculate overall baseline rates
    overall_rates = {col: df[col].mean() for col in purpose_cols}
    
    # analyze by business type within comparable groups
    for group_name, group_types in COMPARABLE_GROUPS.items():
        if group_name == 'other':
            continue
            
        group_df = df[df['business_group'] == group_name]
        if len(group_df) < 50:
            continue
        
        # group baseline
        group_rates = {col: group_df[col].mean() for col in purpose_cols}
        
        # analyze each specific type within group
        for business_type in group_df['business_type'].unique():
            type_df = group_df[group_df['business_type'] == business_type]
            
            if len(type_df) < 30:
                logger.debug(f"Skipping {business_type} - insufficient sample size: {len(type_df)}")
                continue
            
            logger.debug(f"Analyzing {business_type} (n={len(type_df)}) against {group_name} baseline (n={len(group_df)})")
            
            type_results = {
                'n': len(type_df),
                'group': group_name,
                'purposes': {}
            }
            
            expectations = get_expectations(business_type)
            
            # analyze each purpose
            for purpose_col in purpose_cols:
                purpose_short = purpose_col.replace('grant_purpose_', '').replace('_binary', '')
                purpose_short = purpose_short.replace('grant_purp_cons_', '')  # handle outdoor seating
                
                segment_rate = type_df[purpose_col].mean()
                baseline_rate = group_rates[purpose_col]
                
                expectation = expectations.get(purpose_short, 'neutral')
                
                results['diagnostics']['tests_performed'] += 1
                
                surprise = calculate_surprise_score(
                    segment_rate, baseline_rate,
                    len(type_df), len(group_df),
                    expectation
                )
                
                # track why patterns are filtered
                if surprise is None:
                    # check if filtered by expectation or significance
                    z_test = (segment_rate - baseline_rate) / np.sqrt(
                        baseline_rate * (1 - baseline_rate) * (1/len(type_df) + 1/len(group_df))
                    ) if baseline_rate > 0 and baseline_rate < 1 else 0
                    
                    if expectation == 'high' and z_test > 0:
                        results['diagnostics']['filtered_by_expectations'] += 1
                    elif expectation in ['low', 'exclude'] and z_test < 0:
                        results['diagnostics']['filtered_by_expectations'] += 1
                    elif abs(z_test) < 3.0:
                        results['diagnostics']['filtered_by_significance'] += 1
                
                if surprise:
                    type_results['purposes'][purpose_names[purpose_col]] = surprise
                    
                    # Add to key insights if surprising (standardized threshold)
                    if abs(surprise['z_score']) > 3.0:
                        insight = generate_insight(
                            business_type, purpose_names[purpose_col],
                            surprise, expectation
                        )
                        if insight:
                            results['key_insights'].append(insight)
            
            if type_results['purposes']:  # Only add if non-obvious patterns found
                results['business_types'][business_type] = type_results
    
    # Geographic analysis (rural vs urban)
    analyze_geographic_patterns(df, purpose_cols, purpose_names, results)
    
    # Disadvantaged status analysis
    analyze_disadvantaged_patterns(df, purpose_cols, purpose_names, results)
    
    # Apply multiple testing correction
    results = apply_multiple_testing_correction(results)
    
    # Filter key insights to only those significant after correction
    if 'test_statistics' in results:
        # re-filter key insights based on fdr correction
        filtered_insights = []
        for insight in results['key_insights']:
            # Check if this insight is still significant after correction
            if insight.get('type') == 'geographic' or insight.get('type') == 'equity':
                # keep geographic and equity insights with high z scores
                if abs(insight.get('z_score', 0)) > 3.5:
                    filtered_insights.append(insight)
            else:
                # check business type insights for fdr significance
                bt = insight.get('business_type')
                purpose = insight.get('purpose')
                if bt in results['business_types'] and purpose in results['business_types'][bt]['purposes']:
                    if results['business_types'][bt]['purposes'][purpose].get('significant_fdr', False):
                        filtered_insights.append(insight)
        
        results['key_insights'] = filtered_insights
    
    # sort and limit key insights
    results['key_insights'] = sorted(
        results['key_insights'], 
        key=lambda x: abs(x.get('z_score', 0)),
        reverse=True
    )[:15]
    
    return results

def pluralize_business_type(business_type: str) -> str:
    # simple pluralization
    plurals = {
        'bakery': 'bakeries',
        'brewery': 'breweries', 
        'distillery': 'distilleries',
        'winery': 'wineries',
        'inn': 'inns',
        'bar': 'bars',
        'brewpub': 'brewpubs',
        'restaurant': 'restaurants',
        'producer': 'producers',
        'caterer': 'caterers',
        'mobile': 'mobile vendors',
        'coffee_shop': 'coffee shops'
    }
    return plurals.get(business_type, business_type + 's')

def analyze_geographic_patterns(df: pd.DataFrame, purpose_cols: List[str], 
                                purpose_names: Dict[str, str], results: Dict):
    # rural vs urban patterns within types
    
    for business_type in df['business_type'].unique():
        if business_type == 'other' or business_type == 'unknown':
            continue
            
        type_df = df[df['business_type'] == business_type]
        
        rural_df = type_df[type_df['is_rural'] == 1]
        urban_df = type_df[type_df['is_rural'] == 0]
        
        if len(rural_df) < 30 or len(urban_df) < 30:
            continue
        
        expectations = get_expectations(business_type)
        
        for purpose_col in purpose_cols:
            purpose_short = purpose_col.replace('grant_purpose_', '').replace('_binary', '')
            purpose_short = purpose_short.replace('grant_purp_cons_', '')
            
            rural_rate = rural_df[purpose_col].mean()
            urban_rate = urban_df[purpose_col].mean()
            
            expectation = expectations.get(purpose_short, 'neutral')
            
            # for geographic we allow differences regardless of expectations
            if abs(rural_rate - urban_rate) > 0.1:  # 10% difference threshold
                surprise = calculate_surprise_score(
                    rural_rate, urban_rate,
                    len(rural_df), len(urban_df),
                    'neutral'  # Override expectation for geographic comparison
                )
                
                if surprise and abs(surprise['z_score']) > 3.0:  # threshold
                    # decide which location is higher and compute display ratio
                    if rural_rate > urban_rate:
                        location = 'rural'
                        if surprise['odds_ratio']:
                            # Rural is higher, so ratio should be rural/urban > 1
                            odds_ratio_display = surprise['odds_ratio'] if surprise['odds_ratio'] > 1 else 1/surprise['odds_ratio']
                        else:
                            odds_ratio_display = None
                    else:
                        location = 'urban'
                        if surprise['odds_ratio']:
                            # Urban is higher, so ratio should be urban/rural > 1
                            odds_ratio_display = 1/surprise['odds_ratio'] if surprise['odds_ratio'] < 1 else surprise['odds_ratio']
                        else:
                            odds_ratio_display = None
                    
                    bt_plural = pluralize_business_type(business_type)
                    
                    insight = {
                        'type': 'geographic',
                        'business_type': business_type,
                        'pattern': f"{location.capitalize()} {bt_plural}",
                        'purpose': purpose_names[purpose_col],
                        'rural_rate': rural_rate,
                        'urban_rate': urban_rate,
                        'difference': rural_rate - urban_rate,
                        'z_score': surprise['z_score'],
                        'interpretation': f"{location.capitalize()} {bt_plural} are {odds_ratio_display:.1f}x as likely to select {purpose_names[purpose_col]}" if odds_ratio_display and odds_ratio_display != 1 else f"{location.capitalize()} {bt_plural} show {abs(rural_rate-urban_rate):.1%} higher selection of {purpose_names[purpose_col]}"
                    }
                    
                    if business_type not in results['geographic']:
                        results['geographic'][business_type] = []
                    results['geographic'][business_type].append(insight)
                    
                    if abs(surprise['z_score']) > 3.5:  # higher threshold for key insights
                        results['key_insights'].append(insight)

def analyze_disadvantaged_patterns(df: pd.DataFrame, purpose_cols: List[str],
                                  purpose_names: Dict[str, str], results: Dict):
    # disadvantaged vs non‑disadvantaged patterns within types
    
    for business_type in df['business_type'].unique():
        if business_type == 'other' or business_type == 'unknown':
            continue
            
        type_df = df[df['business_type'] == business_type]
        
        disadv_df = type_df[type_df['is_disadvantaged'] == 1]
        non_disadv_df = type_df[type_df['is_disadvantaged'] == 0]
        
        if len(disadv_df) < 30 or len(non_disadv_df) < 30:
            continue
        
        expectations = get_expectations(business_type)
        
        for purpose_col in purpose_cols:
            purpose_short = purpose_col.replace('grant_purpose_', '').replace('_binary', '')
            purpose_short = purpose_short.replace('grant_purp_cons_', '')
            
            disadv_rate = disadv_df[purpose_col].mean()
            non_disadv_rate = non_disadv_df[purpose_col].mean()
            
            # look for meaningful differences
            if abs(disadv_rate - non_disadv_rate) > 0.08:  # 8% difference threshold
                surprise = calculate_surprise_score(
                    disadv_rate, non_disadv_rate,
                    len(disadv_df), len(non_disadv_df),
                    'neutral'  # Override for disadvantaged comparison
                )
                
                if surprise and abs(surprise['z_score']) > 3.0:  # Standardized threshold
                    
                    bt_plural = pluralize_business_type(business_type)
                    
                    insight = {
                        'type': 'equity',
                        'business_type': business_type,
                        'pattern': f"Disadvantaged {bt_plural}",
                        'purpose': purpose_names[purpose_col],
                        'disadvantaged_rate': disadv_rate,
                        'non_disadvantaged_rate': non_disadv_rate,
                        'difference': disadv_rate - non_disadv_rate,
                        'z_score': surprise['z_score'],
                        'interpretation': interpret_disadvantaged_pattern(
                            business_type, purpose_names[purpose_col],
                            disadv_rate, non_disadv_rate, surprise
                        )
                    }
                    
                    if business_type not in results['disadvantaged']:
                        results['disadvantaged'][business_type] = []
                    results['disadvantaged'][business_type].append(insight)
                    
                    if abs(surprise['z_score']) > 3.5:  # higher threshold for key insights
                        results['key_insights'].append(insight)

def generate_insight(business_type: str, purpose: str, 
                     surprise: Dict, expectation: str) -> Optional[Dict]:
    # build a short interpretation for a finding
    
    if not surprise or not surprise.get('odds_ratio'):
        return None
    
    z = surprise['z_score']
    odds_ratio = surprise['odds_ratio']
    
    # skip if not strong enough
    if abs(z) < 3:
        return None
    
    bt_plural = pluralize_business_type(business_type)
    direction = "more" if z > 0 else "less"
    
    # pick wording based on expectation
    if expectation in ['high', 'exclude'] and z < -3:
        interpretation = f"Surprisingly, {bt_plural} are {1/odds_ratio:.1f}x LESS likely to select {purpose} despite structural expectations"
    elif expectation in ['low', 'exclude'] and z > 3:
        interpretation = f"Unexpectedly, {bt_plural} are {odds_ratio:.1f}x MORE likely to select {purpose} despite typical constraints"
    elif expectation == 'neutral' and abs(z) > 3.0:  # Standardized threshold
        interpretation = f"{bt_plural} are {abs(odds_ratio):.1f}x {direction} likely to select {purpose} compared to similar businesses"
    else:
        return None
    
    return {
        'business_type': business_type,
        'purpose': purpose,
        'z_score': z,
        'odds_ratio': odds_ratio,
        'interpretation': interpretation,
        'segment_rate': surprise['segment_rate'],
        'baseline_rate': surprise['baseline_rate']
    }

def interpret_disadvantaged_pattern(business_type: str, purpose: str,
                                   disadv_rate: float, non_disadv_rate: float,
                                   surprise: Dict) -> str:
    # interpretation for disadvantaged patterns
    
    diff_pct = (disadv_rate - non_disadv_rate) * 100
    bt_plural = pluralize_business_type(business_type)
    
    if purpose in ['Payroll', 'Utility', 'Rent']:
        if disadv_rate > non_disadv_rate:
            return f"Disadvantaged {bt_plural} prioritize immediate operational needs ({purpose}) {abs(diff_pct):.0f}% more"
        else:
            return f"Disadvantaged {bt_plural} select {purpose} {abs(diff_pct):.0f}% less, possibly indicating different cost structures"
    
    elif purpose in ['Outdoor Seating Construction', 'Supplies']:
        if disadv_rate < non_disadv_rate:
            return f"Disadvantaged {bt_plural} invest in {purpose} {abs(diff_pct):.0f}% less, suggesting capital constraints"
        else:
            return f"Despite constraints, disadvantaged {bt_plural} invest in {purpose} {abs(diff_pct):.0f}% more"
    
    else:
        direction = "more" if disadv_rate > non_disadv_rate else "less"
        return f"Disadvantaged {bt_plural} select {purpose} {abs(diff_pct):.0f}% {direction} than non-disadvantaged peers"

def save_results(results: Dict):
    # save analysis results to json
    
    # convert numpy types for json serialization
    def clean_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        return obj
    
    results_clean = clean_for_json(results)
    
    output_path = Path(PROCESSED_DATA_DIR) / 'business_purpose_insights.json'
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    logger.info(f"Saved insights to {output_path}")
    
    # also save a short text summary
    summary_path = Path(PROCESSED_DATA_DIR) / 'key_findings.txt'
    with open(summary_path, 'w') as f:
        f.write("KEY NON-OBVIOUS FINDINGS FROM RRF PURPOSE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TOP INSIGHTS (Surprising Patterns Only):\n")
        f.write("-" * 40 + "\n\n")
        
        for i, insight in enumerate(results_clean['key_insights'][:10], 1):
            f.write(f"{i}. {insight.get('interpretation', 'N/A')}\n")
            if 'segment_rate' in insight:
                f.write(f"   - Rate: {insight['segment_rate']:.1%} vs baseline {insight['baseline_rate']:.1%}\n")
            f.write("\n")
        
        f.write("\nMETHODOLOGY NOTE:\n")
        f.write("-" * 40 + "\n")
        f.write("All findings shown are statistically significant (p < 0.01) and\n")
        f.write("represent departures from structural expectations. Obvious patterns\n")
        f.write("(e.g., food trucks not paying rent) have been filtered out.\n")
    
    logger.info(f"Saved summary to {summary_path}")

def main():
    # entry point
    logger.info("Starting business purpose preference analysis")
    
    # Load data
    df = load_data()
    logger.info(f"Analyzing {len(df)} businesses")
    
    # Run analysis
    results = analyze_purpose_preferences(df)
    
    # Report findings
    logger.info(f"Found {len(results['business_types'])} business types with non-obvious patterns")
    logger.info(f"Identified {len(results['key_insights'])} key insights")
    
    # Report diagnostics
    if 'diagnostics' in results:
        logger.info(f"Tests performed: {results['diagnostics']['tests_performed']}")
        logger.info(f"Filtered by expectations: {results['diagnostics']['filtered_by_expectations']}")
        logger.info(f"Filtered by significance: {results['diagnostics']['filtered_by_significance']}")
    
    if 'test_statistics' in results:
        logger.info(f"Significant before correction: {results['test_statistics']['significant_before_correction']}")
        logger.info(f"Significant after FDR: {results['test_statistics']['significant_after_fdr']}")
    
    
    # Save results
    save_results(results)
    
    logger.info("Analysis complete!")
    
    # Print top insights
    print("\nTOP 5 NON-OBVIOUS INSIGHTS:")
    print("=" * 60)
    for i, insight in enumerate(results['key_insights'][:5], 1):
        print(f"\n{i}. {insight.get('interpretation', 'N/A')}")
        if 'z_score' in insight:
            print(f"   Statistical significance: z = {insight['z_score']:.2f}")

if __name__ == "__main__":
    main()