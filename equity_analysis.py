#!/usr/bin/env python3
# equity analysis with bootstrap confidence intervals
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import matplotlib
if os.environ.get("SHOW_PLOTS", "0") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR, DEMOGRAPHIC_FIELDS
from analysis import load_data
from utils import setup_logger

logger = setup_logger(__name__)


def maybe_show():
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()

def compute_equity_ratio_with_ci(disadv_amounts, non_disadv_amounts):
    # bootstrap confidence interval for equity ratio
    if len(disadv_amounts) == 0 or len(non_disadv_amounts) == 0:
        return None, None, None
    
    point = disadv_amounts.mean() / non_disadv_amounts.mean()
    
    ratios = []
    for _ in range(200):
        d_sample = np.random.choice(disadv_amounts, size=len(disadv_amounts), replace=True)
        nd_sample = np.random.choice(non_disadv_amounts, size=len(non_disadv_amounts), replace=True)
        if nd_sample.mean() > 0:
            ratios.append(d_sample.mean() / nd_sample.mean())
    
    lower = np.percentile(ratios, 2.5)
    upper = np.percentile(ratios, 97.5)
    
    return point, lower, upper

def analyze_equity_overall(df):
    # overall equity with confidence intervals
    
    df_clean = df.copy()
    df_clean['amount_num'] = pd.to_numeric(df_clean['GrantAmount'], errors='coerce')
    df_clean = df_clean.dropna(subset=['amount_num'])
    
    total = len(df_clean)
    n_disadv = (df_clean['is_disadvantaged'] == 1).sum()
    n_non_disadv = (df_clean['is_disadvantaged'] == 0).sum()
    
    disadv_amounts = df_clean[df_clean['is_disadvantaged'] == 1]['amount_num']
    non_disadv_amounts = df_clean[df_clean['is_disadvantaged'] == 0]['amount_num']
    
    ratio, lower, upper = compute_equity_ratio_with_ci(disadv_amounts.values, non_disadv_amounts.values)
    
    total_funds = df_clean['amount_num'].sum()
    disadv_funds = df_clean[df_clean['is_disadvantaged'] == 1]['amount_num'].sum()
    non_disadv_funds = df_clean[df_clean['is_disadvantaged'] == 0]['amount_num'].sum()
    
    share_recipients_disadv = n_disadv / total
    share_funds_disadv = disadv_funds / total_funds
    
    return {
        'equity_ratio': ratio,
        'equity_ratio_ci': (lower, upper),
        'n_disadvantaged': int(n_disadv),
        'n_non_disadvantaged': int(n_non_disadv),
        'mean_disadvantaged': float(disadv_amounts.mean()),
        'mean_non_disadvantaged': float(non_disadv_amounts.mean()),
        'share_funds_disadvantaged': float(share_funds_disadv),
        'share_recipients_disadvantaged': float(share_recipients_disadv)
    }

def analyze_equity_by_geography(df, amount_num):
    # equity by state and rural/urban
    
    results = {}
    
    state_counts = df['BusinessState'].value_counts().head(10)
    
    state_equity = []
    for state in state_counts.index:
        state_mask = df['BusinessState'] == state
        state_df = df[state_mask]
        state_amounts = amount_num[state_mask]
        
        disadv_mask = state_df['is_disadvantaged'] == 1
        non_disadv_mask = state_df['is_disadvantaged'] == 0
        
        disadv_amounts = state_amounts[disadv_mask]
        non_disadv_amounts = state_amounts[non_disadv_mask]
        
        ratio, lower, upper = compute_equity_ratio_with_ci(disadv_amounts, non_disadv_amounts)
        if ratio is not None:
            n = len(state_df)
            state_equity.append({
                'state': state,
                'n': n,
                'ratio': ratio,
                'ci_lower': lower,
                'ci_upper': upper
            })
    
    state_equity.sort(key=lambda x: x['ratio'])
    
    results['state_equity'] = state_equity
    
    if 'is_rural' in df.columns:
        for is_rural, label in [(0, "Urban"), (1, "Rural")]:
            location_mask = df['is_rural'] == is_rural
            location_df = df[location_mask]
            if len(location_df) > 0:
                location_amounts = amount_num[location_mask]
                
                disadv_mask = location_df['is_disadvantaged'] == 1
                non_disadv_mask = location_df['is_disadvantaged'] == 0
                
                disadv_amounts = location_amounts[disadv_mask]
                non_disadv_amounts = location_amounts[non_disadv_mask]
                
                ratio, lower, upper = compute_equity_ratio_with_ci(disadv_amounts, non_disadv_amounts)
                if ratio is not None:
                    results[f'{label.lower()}_equity_ratio'] = ratio
    
    return results

def analyze_purpose_differences(df):
    # purpose selection differences between groups
    purpose_cols = [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]
    if not purpose_cols:
        return {}
    
    disadv_df = df[df['is_disadvantaged'] == 1]
    non_disadv_df = df[df['is_disadvantaged'] == 0]
    
    purpose_diffs = []
    for col in purpose_cols:
        rate_disadv = disadv_df[col].mean() * 100
        rate_non_disadv = non_disadv_df[col].mean() * 100
        
        diff = rate_disadv - rate_non_disadv
        
        name = col.replace('_binary', '').replace('grant_purpose_', '').replace('grant_purp_', '')
        
        purpose_diffs.append({
            'purpose': name,
            'disadvantaged_rate': rate_disadv,
            'non_disadvantaged_rate': rate_non_disadv,
            'difference_pp': diff
        })
    
    purpose_diffs.sort(key=lambda x: abs(x['difference_pp']), reverse=True)
    
    return purpose_diffs

def run_robustness_check(df):
    # robustness check with location controls
    
    df_reg = df.copy()
    df_reg['log_amount'] = np.log(pd.to_numeric(df_reg['GrantAmount'], errors='coerce'))
    df_reg = df_reg.dropna(subset=['log_amount'])
    
    from sklearn.linear_model import LinearRegression
    
    state_dummies = pd.get_dummies(df_reg['BusinessState'], prefix='state', drop_first=True)
    
    X = pd.DataFrame()
    X['is_disadvantaged'] = df_reg['is_disadvantaged']
    if 'is_rural' in df_reg.columns:
        X['is_rural'] = df_reg['is_rural']
    X = pd.concat([X, state_dummies], axis=1)
    
    y = df_reg['log_amount'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    disadv_coef = model.coef_[0]
    
    return {
        'disadvantaged_coef': disadv_coef,
        'pct_difference': (np.exp(disadv_coef)-1)*100
    }

def sensitivity_analysis(df, amount_num):
    # test excluding rural from disadvantaged definition
    df_sens = df.copy()
    if 'is_disadvantaged_core' in df_sens.columns:
        
        disadv_mask_orig = df_sens['is_disadvantaged'] == 1
        non_disadv_mask_orig = df_sens['is_disadvantaged'] == 0
        orig_ratio, orig_lower, orig_upper = compute_equity_ratio_with_ci(
            amount_num[disadv_mask_orig], amount_num[non_disadv_mask_orig]
        )
        
        disadv_mask_core = df_sens['is_disadvantaged_core'] == 1
        non_disadv_mask_core = df_sens['is_disadvantaged_core'] == 0
        core_ratio, core_lower, core_upper = compute_equity_ratio_with_ci(
            amount_num[disadv_mask_core], amount_num[non_disadv_mask_core]
        )
        
        n_rural_only = ((df_sens['is_disadvantaged'] == 1) & 
                       (df_sens['is_disadvantaged_core'] == 0)).sum()
        
        return {
            'original_ratio': orig_ratio,
            'core_only_ratio': core_ratio,
            'n_rural_only_disadvantaged': int(n_rural_only)
        }
    else:
        return {}

def _to_binary(series):
    # convert y/n patterns to 1/0
    def map_val(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return 1 if v == 1 else 0
        s = str(v).strip().upper()
        return 1 if s in {"Y", "YES", "TRUE", "1"} else 0
    return series.apply(map_val)

def analyze_demographic_breakdowns(df):
    # equity by demographic groups
    results = {}

    fields = [f for f in DEMOGRAPHIC_FIELDS.keys() if f in df.columns]
    if 'MinorityOwnedIndicator' in df.columns and 'MinorityOwnedIndicator' not in fields:
        fields.append('MinorityOwnedIndicator')

    amount = pd.to_numeric(df['GrantAmount'], errors='coerce') if 'GrantAmount' in df.columns else pd.Series(dtype=float)

    for field in fields:
        grp = _to_binary(df[field])
        mask1 = grp == 1
        mask0 = grp == 0
        x1 = amount[mask1].dropna()
        x0 = amount[mask0].dropna()
        if x1.empty or x0.empty:
            continue
        label = DEMOGRAPHIC_FIELDS.get(field, field)
        stats = {
            'label': label,
            'field': field,
            'count_1': int(mask1.sum()),
            'count_0': int(mask0.sum()),
            'mean_1': float(x1.mean()),
            'mean_0': float(x0.mean()),
            'median_1': float(x1.median()),
            'median_0': float(x0.median()),
            'equity_ratio': float(x1.mean() / x0.mean()) if x0.mean() > 0 else None
        }
        results[field] = stats

    return results

def create_equity_visualizations(df, results):
    # create equity plots
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Set2")
    
    plots_created = []
    
    if 'state_equity' in results.get('geographic', {}):
        fig, ax = plt.subplots(figsize=(14, 10))
        
        state_data = results['geographic']['state_equity']
        states = [item['state'] for item in state_data]
        ratios = [item['ratio'] for item in state_data]
        ci_lowers = [item['ci_lower'] for item in state_data]
        ci_uppers = [item['ci_upper'] for item in state_data]
        
        def get_color(r):
            if r < 0.7: return '#E74C3C'
            elif r < 0.9: return '#F39C12'
            else: return '#27AE60'
        
        colors = [get_color(r) for r in ratios]
        
        y_pos = np.arange(len(states))
        ax.barh(y_pos, ratios, color=colors)
        
        for i, (ratio, lower, upper) in enumerate(zip(ratios, ci_lowers, ci_uppers)):
            ax.errorbar(ratio, i, xerr=[[ratio - lower], [upper - ratio]], fmt='none', color='black')
            ax.text(max(ratio + 0.05, upper + 0.02), i, f'{ratio:.3f}', va='center', ha='left')
        
        ax.axvline(x=1.0, color='black', linestyle='--')
        ax.text(1.02, len(states) * 0.95, 'Perfect\nEquity', ha='left', va='top')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(states)
        ax.set_xlabel('Equity Ratio (Disadvantaged ÷ Non-Disadvantaged)')
        ax.set_ylabel('State')
        ax.set_title('State-Level Funding Equity Analysis\nTop 10 States by Grant Count with 95% Confidence Intervals\n\nRed: Severe Disparity (<0.7)  •  Orange: Moderate Disparity (0.7-0.9)  •  Green: Near Parity (≥0.9)')
        plt.tight_layout()
        
        plot_file = PROCESSED_DATA_DIR / "equity_01_state_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        maybe_show()
        plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    overall = results['overall']
    categories = ['Share of\nBusinesses', 'Share of\nTotal Funding']
    disadvantaged_shares = [overall['share_recipients_disadvantaged'] * 100, 
                           overall['share_funds_disadvantaged'] * 100]
    non_disadvantaged_shares = [100 - overall['share_recipients_disadvantaged'] * 100,
                               100 - overall['share_funds_disadvantaged'] * 100]
    
    disadv_color = '#E8503A'
    non_disadv_color = '#45B7D1'
    
    x = np.arange(len(categories))
    width = 0.32
    
    bars1 = ax.bar(x - width/2, disadvantaged_shares, width, 
                   label='Disadvantaged Businesses', color=disadv_color)
    bars2 = ax.bar(x + width/2, non_disadvantaged_shares, width,
                   label='Non-Disadvantaged Businesses', color=non_disadv_color)
    for bars, values in [(bars1, disadvantaged_shares), (bars2, non_disadvantaged_shares)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            y_pos = height + 1 if height < 50 else height - 5
            color = 'black' if height < 50 else 'white'
            ax.text(bar.get_x() + bar.get_width()/2., y_pos, f'{height:.1f}%', 
                   ha='center', va='bottom' if height < 50 else 'center', color=color)
    
    expected_share = overall['share_recipients_disadvantaged'] * 100
    ax.axhline(y=expected_share, color='#2ECC71', linestyle='--')
    ax.text(1.05, expected_share + 1, f'Proportional Line\n({expected_share:.1f}%)', 
           ha='left', va='bottom')
    
    ax.set_xlabel('Allocation Category')
    ax.set_ylabel('Percentage Share (%)')
    ax.set_title('Restaurant Relief Fund: Equity in Resource Distribution\n' +
                'Comparing Business Representation vs. Funding Allocation\n\n' +
                'Red: Disadvantaged Businesses  •  Blue: Non-Disadvantaged Businesses')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    
    plot_file = PROCESSED_DATA_DIR / "equity_02_allocation_gap.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plots_created.append(plot_file.name)
    maybe_show()
    plt.close()
    
    return plots_created

def save_results(results):
    # save results to json
    output_file = PROCESSED_DATA_DIR / 'equity_analysis_results.json'
    comprehensive = {
        "equity_analysis": {
            "overall_ratio": round(results['overall']['equity_ratio'], 3),
            "confidence_interval": [round(x, 3) for x in results['overall']['equity_ratio_ci']],
            "disadvantaged_businesses": {
                "count": results['overall']['n_disadvantaged'],
                "pct_of_total": round(results['overall']['share_recipients_disadvantaged'] * 100, 1),
                "mean_grant": int(results['overall']['mean_disadvantaged']),
                "share_of_funds": round(results['overall']['share_funds_disadvantaged'] * 100, 1)
            },
            "non_disadvantaged_businesses": {
                "count": results['overall']['n_non_disadvantaged'], 
                "pct_of_total": round((1 - results['overall']['share_recipients_disadvantaged']) * 100, 1),
                "mean_grant": int(results['overall']['mean_non_disadvantaged']),
                "share_of_funds": round((1 - results['overall']['share_funds_disadvantaged']) * 100, 1)
            }
        }
    }
    
    if 'geographic' in results and 'state_equity' in results['geographic']:
        state_data = results['geographic']['state_equity']
        comprehensive["geographic_equity"] = {
            "worst_states": [
                {"state": s['state'], "ratio": round(s['ratio'], 3), "ci_lower": round(s['ci_lower'], 3), "ci_upper": round(s['ci_upper'], 3)} 
                for s in sorted(state_data, key=lambda x: x['ratio'])[:5]
            ],
            "best_states": [
                {"state": s['state'], "ratio": round(s['ratio'], 3), "ci_lower": round(s['ci_lower'], 3), "ci_upper": round(s['ci_upper'], 3)} 
                for s in sorted(state_data, key=lambda x: x['ratio'], reverse=True)[:5]
            ]
        }
    
    if 'demographic_breakdowns' in results:
        demo_summary = {}
        for field, data in results['demographic_breakdowns'].items():
            if 'equity_ratio' in data:
                demo_summary[data['label']] = round(data['equity_ratio'], 3)
        if demo_summary:
            comprehensive["demographic_equity_ratios"] = demo_summary
    
    if results.get('robustness'):
        comprehensive["robustness_check"] = {
            "controlled_effect_pct": round(results['robustness']['pct_difference'], 1)
        }
    if results.get('sensitivity'):
        comprehensive["sensitivity_analysis"] = {
            "equity_ratio_with_rural": round(results['sensitivity']['original_ratio'], 3),
            "equity_ratio_core_only": round(results['sensitivity']['core_only_ratio'], 3),
            "rural_only_disadvantaged_count": results['sensitivity']['n_rural_only_disadvantaged']
        }
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive, f, indent=2)

def main():
    logger.info("Loading data...")
    
    df = load_data()
    
    required = ['is_disadvantaged', 'GrantAmount', 'BusinessState']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.info("Preprocessing data...")
    amount_num = pd.to_numeric(df['GrantAmount'], errors='coerce')
    valid_mask = ~amount_num.isna()
    df = df[valid_mask]
    amount_num = amount_num[valid_mask]
    
    results = {}
    
    logger.info("Overall equity analysis...")
    results['overall'] = analyze_equity_overall(df)
    
    logger.info("Geographic equity analysis...")
    results['geographic'] = analyze_equity_by_geography(df, amount_num)
    
    logger.info("Purpose analysis...")
    results['purpose_differences'] = analyze_purpose_differences(df)

    logger.info("Demographic analysis...")
    results['demographic_breakdowns'] = analyze_demographic_breakdowns(df)
    
    logger.info("Robustness check...")
    try:
        results['robustness'] = run_robustness_check(df)
    except Exception as e:
        logger.warning(f"Robustness check failed: {e}")
        results['robustness'] = None
    
    logger.info("Sensitivity analysis...")
    results['sensitivity'] = sensitivity_analysis(df, amount_num)

    logger.info("Creating visualizations...")
    try:
        create_equity_visualizations(df, results)
    except Exception as e:
        logger.warning(f"Visualization creation failed: {e}")
    
    logger.info("Saving results...")
    save_results(results)
    
    logger.info("Phase 2 analysis complete")
    
    return results

if __name__ == "__main__":
    main()