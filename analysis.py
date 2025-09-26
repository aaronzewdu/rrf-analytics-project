#!/usr/bin/env python3
# analysis for rrf analytics project

import os
import pandas as pd
import matplotlib
# use non-interactive backend unless SHOW_PLOTS=1
if os.environ.get("SHOW_PLOTS", "0") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import json
import textwrap

from config import PROCESSED_DATA_DIR, DB_CONFIG, DEMOGRAPHIC_FIELDS, RURAL_FIELD_MAPPING, GRANT_PURPOSE_FIELDS
from utils import setup_logger
from purpose_helpers import list_purpose_binary_cols, clean_purpose_label

# show plots only if explicitly requested via env var
# export SHOW_PLOTS=1 to enable interactive windows

# setup logger
logger = setup_logger(__name__)

def maybe_show():
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()


def load_data():
    # load processed data
    db_error = None
    try:
        engine = create_engine(DB_CONFIG)
        df = pd.read_sql("SELECT * FROM rrf_data", engine)
        return df
    except SQLAlchemyError as e:
        db_error = e
    except Exception as e:
        # capture any other runtime errors during db load (e.g., driver issues)
        db_error = e

    # fallback to latest processed csv if db load failed
    processed_files = list(PROCESSED_DATA_DIR.glob("rrf_processed_*.csv"))
    if not processed_files:
        raise FileNotFoundError(
            f"No processed data found and DB query failed. Run ETL first. Original DB error: {db_error}"
        ) from db_error
    latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    return df

def analyze_demographics(df):
    # analyze demographics
    if 'is_disadvantaged' not in df.columns:
        return {}
    
    total = len(df)
    disadvantaged = df['is_disadvantaged'].sum()
    
    results = {
        "total_businesses": int(total),
        "disadvantaged_count": int(disadvantaged),
        "disadvantaged_pct": round(disadvantaged/total * 100, 1)
    }
    
    # demographic categories
    demographics = {}
    for col, label in {**DEMOGRAPHIC_FIELDS, **RURAL_FIELD_MAPPING}.items():
        if col in df.columns:
            count = (df[col] == 'Y').sum() if col in DEMOGRAPHIC_FIELDS else df[col].sum()
            demographics[label] = {
                "count": int(count),
                "pct": round(count/total * 100, 1)
            }
    results["demographics"] = demographics
    return results

def analyze_descriptive_stats(df):
    # comprehensive descriptive statistics
    print("\n=== DESCRIPTIVE STATISTICS ===")
    if 'GrantAmount' not in df.columns:
        return
    
    grants = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
    stats = {'Mean': grants.mean(), 'Median': grants.median(), 'Std Dev': grants.std(),
             'Min': grants.min(), 'Max': grants.max(), 
             '25th percentile': grants.quantile(0.25), '75th percentile': grants.quantile(0.75)}
    
    print(f"\nGrant Amounts (${len(grants):,} records):")
    for stat, value in stats.items():
        print(f"  {stat}: ${value:,.0f}")

def analyze_data_quality(df):
    # data quality assessment
    print(f"\n=== DATA QUALITY ===")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # missing data
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count:,} ({count/len(df):.1%})")
    else:
        print("\nNo missing values detected")
    
    # outlier detection
    if 'GrantAmount' in df.columns:
        grants = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
        q1, q3 = grants.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = grants[(grants < q1 - 1.5*iqr) | (grants > q3 + 1.5*iqr)]
        print(f"\nGrant amount outliers: {len(outliers):,} ({len(outliers)/len(grants):.1%})")

def analyze_equity(df):
    # core equity analysis
    if not {'is_disadvantaged', 'GrantAmount'}.issubset(df.columns):
        return {}
    
    grant_amounts = pd.to_numeric(df['GrantAmount'], errors='coerce')
    groups = {'disadvantaged': df['is_disadvantaged'] == 1, 
              'non_disadvantaged': df['is_disadvantaged'] == 0}
    
    results = {}
    for name, mask in groups.items():
        data = grant_amounts[mask].dropna()
        results[name] = {
            "count": int(len(data)),
            "mean_grant": int(data.mean()),
            "median_grant": int(data.median())
        }
    
    if results['non_disadvantaged']['mean_grant'] > 0:
        ratio = results['disadvantaged']['mean_grant'] / results['non_disadvantaged']['mean_grant']
        results["equity_ratio"] = round(ratio, 3)
    
    return results

def analyze_geographic_patterns(df):
    # geographic analysis
    print("\n=== GEOGRAPHIC ANALYSIS ===")
    
    # state-level analysis
    if {'BusinessState', 'GrantAmount'}.issubset(df.columns):
        state_stats = df.groupby('BusinessState').agg({
            'GrantAmount': ['count', lambda x: pd.to_numeric(x, errors='coerce').mean()],
            'is_disadvantaged': 'mean'
        }).round(2)
        state_stats.columns = ['Grant_Count', 'Avg_Grant', 'Pct_Disadvantaged']
        
        print("\nTop 10 states by grant count:")
        for state, row in state_stats.nlargest(10, 'Grant_Count').iterrows():
            print(f"  {state}: {row['Grant_Count']:,} grants, ${row['Avg_Grant']:,.0f} avg, {row['Pct_Disadvantaged']:.1%} disadvantaged")
    
    # rural vs urban
    if {'is_rural', 'GrantAmount'}.issubset(df.columns):
        rural_stats = df.groupby('is_rural').agg({
            'GrantAmount': lambda x: pd.to_numeric(x, errors='coerce').mean(),
            'is_disadvantaged': ['count', 'mean']
        }).round(2)
        rural_stats.columns = ['Avg_Grant', 'Count', 'Pct_Disadvantaged']
        
        print("\nRural vs Urban comparison:")
        for is_rural, row in rural_stats.iterrows():
            location = "Rural" if is_rural == 1 else "Urban"
            print(f"  {location}: {row['Count']:,} grants, ${row['Avg_Grant']:,.0f} avg, {row['Pct_Disadvantaged']:.1%} disadvantaged")

def analyze_grant_purposes(df):
    # grant purpose analysis
    print("\n=== GRANT PURPOSE ANALYSIS ===")
    purpose_cols = list_purpose_binary_cols(df)
    if not purpose_cols:
        return
    
    total = len(df)
    purposes = df[purpose_cols].sum().sort_values(ascending=False)
    
    print("\nGrant purpose frequency:")
    for col, count in purposes.items():
        name = clean_purpose_label(col)
        print(f"  {name}: {count:,} ({count/total:.1%})")
    
    # co-occurrence analysis
    print("\nTop purpose combinations:")
    purpose_count_series = df[purpose_cols].sum(axis=1)
    for count, freq in purpose_count_series.value_counts().sort_index().items():
        print(f"  {count} purposes: {freq:,} businesses ({freq/total:.1%})")
    
    # purpose patterns by status
    if 'is_disadvantaged' in df.columns:
        print("\nPurpose patterns by disadvantaged status:")
        for status, label in [(0, "Non-disadvantaged"), (1, "Disadvantaged")]:
            subset = df[df['is_disadvantaged'] == status]
            if not subset.empty:
                print(f"  {label} top purposes:")
                for col, pct in subset[purpose_cols].mean().nlargest(3).items():
                    name = clean_purpose_label(col)
                    print(f"    {name}: {pct:.1%}")

def compute_and_save_summary_metrics(df):
    # compute and persist comprehensive analysis results
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # collect all analysis results
    summary = {
        "records": int(len(df))
    }
    
    # add demographics analysis
    demographics_results = analyze_demographics(df)
    if demographics_results:
        summary["demographics"] = demographics_results
    
    # add equity analysis
    equity_results = analyze_equity(df)
    if equity_results:
        summary["equity"] = equity_results
    
    # add basic grant statistics
    if 'GrantAmount' in df.columns:
        grants = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
        summary["grant_statistics"] = {
            "mean": int(grants.mean()),
            "median": int(grants.median()),
            "min": int(grants.min()),
            "max": int(grants.max())
        }

    
    # plot 01 data: enhanced grant equity details
    if {'GrantAmount', 'is_disadvantaged'}.issubset(df.columns):
        df_plot = df.copy()
        df_plot['GrantAmount_num'] = pd.to_numeric(df_plot['GrantAmount'], errors='coerce')
        df_plot = df_plot[df_plot['GrantAmount_num'] > 0].dropna(subset=['GrantAmount_num'])
        
        stats = df_plot.groupby('is_disadvantaged')['GrantAmount_num'].agg(['median', 'mean', 'count']).round(0)
        
        if 1 in stats.index and 0 in stats.index:
            equity_ratio_median = stats.loc[1, 'median'] / stats.loc[0, 'median']
            summary["grant_equity_details"] = {
                "equity_ratio_median": round(equity_ratio_median, 3),
                "interpretation": f"Disadvantaged businesses receive {equity_ratio_median:.0%} of non-disadvantaged median funding levels"
            }

    # plot 03 data: state distribution
    if 'BusinessState' in df.columns:
        states = df['BusinessState'].value_counts()
        states_data = []
        for state, count in states.items():
            states_data.append({
                'state': str(state),
                'grant_count': int(count)
            })
        
        summary["state_distribution"] = {
            "total_states": len(states),
            "states": states_data
        }

    # write summary json
    (PROCESSED_DATA_DIR / 'analysis_summary.json').write_text(json.dumps(summary, indent=2))
    return summary

def create_individual_plots(df):
    # create individual plots for better customization
    
    # set global style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    plots_created = []
    
    
    # purpose profile: minimal styling
    purpose_cols = list_purpose_binary_cols(df)
    if purpose_cols and 'GrantAmount' in df.columns and 'is_disadvantaged' in df.columns:
        df_temp = df.copy()
        df_temp['GrantAmount_num'] = pd.to_numeric(df_temp['GrantAmount'], errors='coerce')

        # calculate baselines and groups
        pooled_sel_baseline = df_temp[purpose_cols].sum().sum() / (len(df_temp) * len(purpose_cols)) * 100.0
        disadv_df = df_temp[df_temp['is_disadvantaged'] == 1]
        non_disadv_df = df_temp[df_temp['is_disadvantaged'] == 0]
        n_disadv, n_non_disadv = len(disadv_df), len(non_disadv_df)
        n_min_disadv = max(300, int(np.ceil(0.005 * n_disadv))) if n_disadv > 0 else 0
        n_min_non_disadv = max(300, int(np.ceil(0.005 * n_non_disadv))) if n_non_disadv > 0 else 0

        # calculate selection differences for each purpose
        results = []
        for col in purpose_cols:
            purpose_name = clean_purpose_label(col, keep_underscore=False)
            nD_sel = int(disadv_df[col].sum()) if n_disadv > 0 else 0
            nN_sel = int(non_disadv_df[col].sum()) if n_non_disadv > 0 else 0
            sel_rate_disadv = (nD_sel / n_disadv * 100.0) if n_disadv > 0 else 0.0
            sel_rate_non_disadv = (nN_sel / n_non_disadv * 100.0) if n_non_disadv > 0 else 0.0
            sel_diff_disadv = ((sel_rate_disadv - pooled_sel_baseline) / pooled_sel_baseline * 100.0) if pooled_sel_baseline > 0 else 0.0
            sel_diff_non_disadv = ((sel_rate_non_disadv - pooled_sel_baseline) / pooled_sel_baseline * 100.0) if pooled_sel_baseline > 0 else 0.0
            results.append({
                'purpose_label': purpose_name, 'sel_diff_disadv': sel_diff_disadv, 
                'sel_diff_non_disadv': sel_diff_non_disadv, 'nD_sel': nD_sel, 'nN_sel': nN_sel
            })

        out = pd.DataFrame(results)
        # filter and rank by effect size
        if n_disadv > 0 and n_non_disadv > 0:
            out = out[(out['nD_sel'] >= n_min_disadv) & (out['nN_sel'] >= n_min_non_disadv)]
        if out.empty:
            out = pd.DataFrame(results)
        out['effect_score'] = out[['sel_diff_disadv', 'sel_diff_non_disadv']].abs().max(axis=1)
        out = out.sort_values('effect_score', ascending=False).head(10)

        # create plot
        x = np.arange(len(out))
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - 0.15, out['sel_diff_disadv'], 0.3, color='#45B7D1', label='Disadvantaged Businesses')
        ax.bar(x + 0.15, out['sel_diff_non_disadv'], 0.3, color='#E8503A', label='Non-Disadvantaged Businesses')
        ax.axhline(y=0, color='black', linestyle='-')
        ax.set_ylabel('Percent Difference from Overall Average (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(lbl, width=15) for lbl in out['purpose_label']])
        ax.legend(loc='upper left', title='Business Type')
        ax.set_title('Grant Purpose Usage Patterns by Business Status\nSelection Rate Differences from Overall Average')
        plt.tight_layout()

        plot_file = PROCESSED_DATA_DIR / "01_purpose_profile.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        maybe_show()
        plt.close()
    
    
    
    # purpose co-occurrence heatmap
    purpose_cols = list_purpose_binary_cols(df)
    if len(purpose_cols) >= 5:  # only create if we have enough purposes
        plt.figure(figsize=(10, 8))
        
        # create correlation matrix for purposes
        purpose_data = df[purpose_cols]
        corr_matrix = purpose_data.corr()
        
        # clean up labels
        clean_labels = [clean_purpose_label(col) for col in purpose_cols]
        
        # create heatmap - mask only upper triangle, keep diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # k=1 keeps diagonal visible
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8},
                   xticklabels=clean_labels, yticklabels=clean_labels)
        
        plt.title('Grant Purpose Co-occurrence Analysis\n(How Often Purposes are Selected Together)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plot_file = PROCESSED_DATA_DIR / "02_purpose_cooccurrence.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        maybe_show()
        plt.close()
    
    return plots_created

def run_comprehensive_analysis():
    # run complete comprehensive analysis
    logger.info("Loading data...")
    df = load_data()
    
    logger.info("Running analysis...")
    # persist comprehensive results
    try:
        compute_and_save_summary_metrics(df)
        logger.info("Analysis complete")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    
    logger.info("Creating visualizations...")
    # visualizations with fallback
    try:
        create_individual_plots(df)
        logger.info("Visualizations complete")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    return df

def create_plots_only():
    # create just the individual plots without running full analysis
    print("=== CREATING PLOTS ONLY ===")
    df = load_data()
    plots_created = create_individual_plots(df)
    print(f"\n=== PLOTS COMPLETE: {len(plots_created)} files created ===")
    return plots_created

def run_analysis():
    # backward compatibility
    return run_comprehensive_analysis()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plots":
        create_plots_only()
    else:
        run_analysis()
