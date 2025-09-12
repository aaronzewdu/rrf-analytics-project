#!/usr/bin/env python3
"""
Analysis for RRF Analytics Project
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine

from config import PROCESSED_DATA_DIR, DB_CONFIG, DEMOGRAPHIC_FIELDS, RURAL_FIELD_MAPPING, GRANT_PURPOSE_FIELDS

# Helper function for cleaning purpose names
def _clean_purpose_name(col, keep_underscore=False):
    """Clean purpose column names for display"""
    # Remove common prefixes and suffixes
    name = col.replace('_binary', '')
    name = name.replace('grant_purpose_', '').replace('grant_purp_', '')
    
    # Special handling for outdoor seating construction
    name = name.replace('cons_outdoor_seating', 'outdoor_seating_construction')
    
    # Convert underscores to spaces unless requested to keep them
    if not keep_underscore:
        name = name.replace('_', ' ')
    
    return name.title()

def load_data():
    """Load processed data"""
    try:
        engine = create_engine(DB_CONFIG)
        df = pd.read_sql("SELECT * FROM rrf_data", engine)
        print(f"Loaded {len(df)} records from database")
        return df
    except:
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
    if 'is_disadvantaged' not in df.columns:
        return
    
    total = len(df)
    disadvantaged = df['is_disadvantaged'].sum()
    print(f"Disadvantaged businesses: {disadvantaged:,} ({disadvantaged/total:.1%})")
    
    # Demographic categories
    for col, label in {**DEMOGRAPHIC_FIELDS, **RURAL_FIELD_MAPPING}.items():
        if col in df.columns:
            count = (df[col] == 'Y').sum() if col in DEMOGRAPHIC_FIELDS else df[col].sum()
            print(f"{label}: {count:,} ({count/total:.1%})")

def analyze_descriptive_stats(df):
    """Comprehensive descriptive statistics"""
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
    """Data quality assessment"""
    print(f"\n=== DATA QUALITY ===")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Missing data
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count:,} ({count/len(df):.1%})")
    else:
        print("\nNo missing values detected")
    
    # Outlier detection
    if 'GrantAmount' in df.columns:
        grants = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
        q1, q3 = grants.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = grants[(grants < q1 - 1.5*iqr) | (grants > q3 + 1.5*iqr)]
        print(f"\nGrant amount outliers: {len(outliers):,} ({len(outliers)/len(grants):.1%})")

def analyze_equity(df):
    """Core equity analysis"""
    print("\n=== EQUITY ANALYSIS ===")
    if not {'is_disadvantaged', 'GrantAmount'}.issubset(df.columns):
        print("Missing required columns")
        return
    
    grant_amounts = pd.to_numeric(df['GrantAmount'], errors='coerce')
    groups = {'DISADVANTAGED': df['is_disadvantaged'] == 1, 
              'NON-DISADVANTAGED': df['is_disadvantaged'] == 0}
    
    stats = {}
    for name, mask in groups.items():
        data = grant_amounts[mask].dropna()
        stats[name] = (len(data), data.mean(), data.median())
        print(f"{name}: {len(data):,} businesses")
        print(f"  Mean grant: ${data.mean():,.0f}")
        print(f"  Median grant: ${data.median():,.0f}")
    
    if stats['NON-DISADVANTAGED'][1] > 0:
        ratio = stats['DISADVANTAGED'][1] / stats['NON-DISADVANTAGED'][1]
        print(f"\nEQUITY RATIO: {ratio:.2f} (disadvantaged get {ratio:.0%} of non-disadvantaged funding)")

def analyze_geographic_patterns(df):
    """Geographic analysis"""
    print("\n=== GEOGRAPHIC ANALYSIS ===")
    
    # State-level analysis
    if {'BusinessState', 'GrantAmount'}.issubset(df.columns):
        state_stats = df.groupby('BusinessState').agg({
            'GrantAmount': ['count', lambda x: pd.to_numeric(x, errors='coerce').mean()],
            'is_disadvantaged': 'mean'
        }).round(2)
        state_stats.columns = ['Grant_Count', 'Avg_Grant', 'Pct_Disadvantaged']
        
        print("\nTop 10 states by grant count:")
        for state, row in state_stats.nlargest(10, 'Grant_Count').iterrows():
            print(f"  {state}: {row['Grant_Count']:,} grants, ${row['Avg_Grant']:,.0f} avg, {row['Pct_Disadvantaged']:.1%} disadvantaged")
    
    # Rural vs Urban
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
    """Grant purpose analysis"""
    print("\n=== GRANT PURPOSE ANALYSIS ===")
    purpose_cols = [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]
    if not purpose_cols:
        return
    
    total = len(df)
    purposes = df[purpose_cols].sum().sort_values(ascending=False)
    
    print("\nGrant purpose frequency:")
    for col, count in purposes.items():
        name = col.replace('_binary', '').replace('grant_purpose_', '').replace('grant_purp_', '').title()
        print(f"  {name}: {count:,} ({count/total:.1%})")
    
    # Co-occurrence analysis
    print("\nTop purpose combinations:")
    df['purpose_count'] = df[purpose_cols].sum(axis=1)
    for count, freq in df['purpose_count'].value_counts().sort_index().items():
        print(f"  {count} purposes: {freq:,} businesses ({freq/total:.1%})")
    
    # Purpose patterns by status
    if 'is_disadvantaged' in df.columns:
        print("\nPurpose patterns by disadvantaged status:")
        for status, label in [(0, "Non-disadvantaged"), (1, "Disadvantaged")]:
            subset = df[df['is_disadvantaged'] == status]
            if not subset.empty:
                print(f"  {label} top purposes:")
                for col, pct in subset[purpose_cols].mean().nlargest(3).items():
                    name = col.replace('_binary', '').replace('grant_purpose_', '').title()
                    print(f"    {name}: {pct:.1%}")

def create_individual_plots(df):
    """Create individual plots for better customization"""
    print("\n=== CREATING INDIVIDUAL PLOTS ===")
    
    # Set global style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    plots_created = []
    
    # 1. Grant Amount Distribution
    if 'GrantAmount' in df.columns:
        plt.figure(figsize=(10, 6))
        grant_data = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
        # Focus on where most data actually is (0-2M)
        plt.hist(grant_data / 1000, bins=60, alpha=0.8, color='steelblue', edgecolor='black', range=(0, 2000))
        plt.title('Restaurant Revitalization Fund - Grant Amount Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Grant Amount (Thousands $)', fontsize=12)
        plt.ylabel('Number of Businesses', fontsize=12)
        plt.grid(True, alpha=0.3)
        # Add summary stats as text
        mean_val = grant_data.mean() / 1000
        median_val = grant_data.median() / 1000
        plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: ${mean_val:.0f}K')
        plt.axvline(median_val, color='orange', linestyle='--', alpha=0.7, label=f'Median: ${median_val:.0f}K')
        plt.legend()
        plt.xlim(0, 2000)  # Focus on relevant range
        plot_file = PROCESSED_DATA_DIR / "01_grant_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 2. Grant Amount Distribution by Status (3-Panel Comprehensive Analysis)
    if {'GrantAmount', 'is_disadvantaged'}.issubset(df.columns):
        # Create figure with three subplots
        fig = plt.figure(figsize=(18, 8))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        
        # Prepare data
        df_plot = df.copy()
        df_plot['GrantAmount_num'] = pd.to_numeric(df_plot['GrantAmount'], errors='coerce') / 1000  # Convert to thousands
        df_plot = df_plot.dropna(subset=['GrantAmount_num'])
        df_plot['Status'] = df_plot['is_disadvantaged'].map({0: 'Non-Disadvantaged', 1: 'Disadvantaged'})
        
        # Calculate statistics for annotations
        stats = df_plot.groupby('Status')['GrantAmount_num'].agg(['mean', 'median', 'std', 'count'])
        
        # LEFT PLOT: Violin plot showing distribution shape
        violin_parts = ax1.violinplot(
            [df_plot[df_plot['Status'] == 'Non-Disadvantaged']['GrantAmount_num'].values,
             df_plot[df_plot['Status'] == 'Disadvantaged']['GrantAmount_num'].values],
            positions=[0, 1],
            widths=0.7,
            showmeans=True,
            showmedians=True,
            showextrema=False
        )
        
        # Style the violin plot
        colors = ['lightcoral', 'lightblue']
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Style mean and median lines
        violin_parts['cmeans'].set_edgecolor('red')
        violin_parts['cmeans'].set_linewidth(2)
        violin_parts['cmedians'].set_edgecolor('black')
        violin_parts['cmedians'].set_linewidth(2)
        
        # Add labels and formatting for left plot
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Non-Disadv.', 'Disadvantaged'], fontsize=10)
        ax1.set_ylabel('Grant Amount ($K)', fontsize=11)
        ax1.set_title('Distribution Shape', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1500)  # Focus on main distribution
        
        # Add count labels
        for i, (status, row) in enumerate(stats.iterrows()):
            ax1.text(i, -100, f'n = {int(row["count"]):,}', ha='center', fontsize=10, style='italic')
        
        # Add statistical annotations on the violin plot
        for i, (status, row) in enumerate(stats.iterrows()):
            ax1.text(i, row['mean'] + 50, f'μ = ${row["mean"]:.0f}K', 
                    ha='center', fontsize=10, fontweight='bold')
            ax1.text(i, row['median'] - 50, f'M = ${row["median"]:.0f}K', 
                    ha='center', fontsize=10)
        
        # MIDDLE PLOT: Comparative bar chart with key metrics
        metrics = ['Mean', 'Median', '75th %ile']
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        # Calculate metrics
        non_disadv_metrics = [
            stats.loc['Non-Disadvantaged', 'mean'],
            stats.loc['Non-Disadvantaged', 'median'],
            df_plot[df_plot['Status'] == 'Non-Disadvantaged']['GrantAmount_num'].quantile(0.75)
        ]
        disadv_metrics = [
            stats.loc['Disadvantaged', 'mean'],
            stats.loc['Disadvantaged', 'median'],
            df_plot[df_plot['Status'] == 'Disadvantaged']['GrantAmount_num'].quantile(0.75)
        ]
        
        # Create grouped bar chart
        bars1 = ax2.bar(x_pos - width/2, non_disadv_metrics, width, 
                       label='Non-Disadvantaged', color='lightcoral', edgecolor='darkred', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, disadv_metrics, width,
                       label='Disadvantaged', color='lightblue', edgecolor='darkblue', alpha=0.7)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'${height:.0f}K', ha='center', va='bottom', fontweight='bold')
        
        # Format middle plot
        ax2.set_xlabel('Statistical Measure', fontsize=11)
        ax2.set_ylabel('Grant Amount ($K)', fontsize=11)
        ax2.set_title('Per-Business Metrics', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metrics, fontsize=10)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add equity ratio annotation
        equity_ratio = stats.loc['Disadvantaged', 'mean'] / stats.loc['Non-Disadvantaged', 'mean']
        ax2.text(0.5, 0.95, f'Equity Ratio: {equity_ratio:.2f}',
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                fontsize=10, fontweight='bold')
        
        # THIRD PLOT: Total Fund Allocation
        # Calculate total funds received by each group
        total_disadv = (df_plot[df_plot['is_disadvantaged'] == 1]['GrantAmount_num'].sum())
        total_non_disadv = (df_plot[df_plot['is_disadvantaged'] == 0]['GrantAmount_num'].sum())
        total_funds = total_disadv + total_non_disadv
        
        # Calculate percentages
        pct_disadv = (total_disadv / total_funds) * 100
        pct_non_disadv = (total_non_disadv / total_funds) * 100
        
        # Create nested pie chart (donut)
        sizes_outer = [pct_non_disadv, pct_disadv]
        colors_outer = ['lightcoral', 'lightblue']
        explode = (0.05, 0)  # Slightly separate non-disadvantaged
        
        wedges, texts, autotexts = ax3.pie(sizes_outer, explode=explode, labels=None,
                                           colors=colors_outer, autopct='%1.1f%%',
                                           shadow=False, startangle=90,
                                           wedgeprops=dict(width=0.5, edgecolor='black'))
        
        # Add center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.50, fc='white', linewidth=2, edgecolor='black')
        ax3.add_artist(centre_circle)
        
        # Add title and labels
        ax3.set_title('Total Fund Allocation', fontsize=12, fontweight='bold')
        
        # Add text in center with total
        ax3.text(0, 0.1, f'Total:\n${total_funds/1000:.0f}M', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax3.text(0, -0.15, f'({len(df_plot):,} grants)', 
                ha='center', va='center', fontsize=9, style='italic')
        
        # Add legend with actual dollar amounts
        legend_labels = [
            f'Non-Disadvantaged\n${total_non_disadv/1000:.0f}M ({pct_non_disadv:.1f}%)\n{int(stats.loc["Non-Disadvantaged", "count"]):,} businesses',
            f'Disadvantaged\n${total_disadv/1000:.0f}M ({pct_disadv:.1f}%)\n{int(stats.loc["Disadvantaged", "count"]):,} businesses'
        ]
        ax3.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9, frameon=True)
        
        # Make the axes equal for circular pie
        ax3.axis('equal')
        
        # Main title for entire figure
        fig.suptitle('Restaurant Revitalization Fund - Complete Equity Analysis\nIndividual Distribution | Per-Business Metrics | Total Fund Allocation', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plot_file = PROCESSED_DATA_DIR / "02_grant_distribution_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 3. Business Demographics Pie Chart
    if 'is_disadvantaged' in df.columns:
        plt.figure(figsize=(8, 8))
        demo_counts = df['is_disadvantaged'].value_counts().reindex([0, 1], fill_value=0)
        colors = ['#ff9999', '#66b3ff']
        wedges, texts, autotexts = plt.pie(demo_counts.values, labels=['Non-Disadvantaged', 'Disadvantaged'], 
                                          autopct='%1.1f%%', colors=colors, startangle=90, 
                                          textprops={'fontsize': 12})
        plt.title('Business Demographics Distribution', fontsize=14, fontweight='bold')
        plot_file = PROCESSED_DATA_DIR / "03_demographics_pie.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 4. Top States by Grant Count
    if 'BusinessState' in df.columns:
        plt.figure(figsize=(12, 8))
        states = df['BusinessState'].value_counts().head(10)
        bars = plt.barh(range(len(states)), states.values, color='lightgreen', edgecolor='darkgreen')
        plt.title('Top 10 States by Number of Grants', fontsize=14, fontweight='bold')
        plt.yticks(range(len(states)), states.index)
        plt.xlabel('Number of Grants', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        # Add value labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                    f'{int(bar.get_width()):,}', ha='left', va='center')
        plot_file = PROCESSED_DATA_DIR / "04_top_states.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 5. Grant Purposes Frequency
    purpose_cols = [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]
    if purpose_cols:
        plt.figure(figsize=(12, 8))
        purposes = df[purpose_cols].sum().sort_values(ascending=True)  # ascending for better layout
        purpose_names = [_clean_purpose_name(col, keep_underscore=True) for col in purposes.index]
        
        # Calculate percentages for better comparison
        total_businesses = len(df)
        percentages = (purposes / total_businesses) * 100
        
        # Create color gradient based on percentage
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(purposes)))
        bars = plt.barh(range(len(purposes)), percentages.values, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title('Grant Purpose Selection Rate by Businesses', fontsize=14, fontweight='bold')
        plt.yticks(range(len(purposes)), purpose_names)
        plt.xlabel('Percentage of Businesses Selecting Purpose (%)', fontsize=12)
        plt.ylabel('Grant Purpose', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.1f}%', ha='left', va='center', fontweight='bold')
        
        plt.xlim(0, 100)  # Set clear 0-100% range
        plot_file = PROCESSED_DATA_DIR / "05_grant_purposes.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 6. Rural vs Urban Comparison
    if {'is_rural', 'GrantAmount'}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        rural_data = df.copy()
        rural_data['GrantAmount_num'] = pd.to_numeric(rural_data['GrantAmount'], errors='coerce')
        rural_data['Location'] = rural_data['is_rural'].map({0: 'Urban', 1: 'Rural'})
        location_stats = rural_data.groupby('Location')['GrantAmount_num'].mean()
        bars = plt.bar(location_stats.index, location_stats.values, 
                      color=['skyblue', 'lightcoral'], edgecolor=['darkblue', 'darkred'], width=0.6)
        plt.title('Average Grant Amount by Location Type', fontsize=14, fontweight='bold')
        plt.ylabel('Average Grant Amount ($)', fontsize=12)
        plt.xlabel('Location Type', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5000,
                    f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        plot_file = PROCESSED_DATA_DIR / "06_rural_vs_urban.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    
    # 7. State-Level Equity Analysis
    if {'BusinessState', 'GrantAmount', 'is_disadvantaged'}.issubset(df.columns):
        plt.figure(figsize=(14, 8))
        
        # Calculate equity ratios by state (top 15 states)
        top_states = df['BusinessState'].value_counts().head(15).index
        state_equity = []
        
        for state in top_states:
            state_data = df[df['BusinessState'] == state]
            disadv_mean = pd.to_numeric(state_data[state_data['is_disadvantaged'] == 1]['GrantAmount'], errors='coerce').mean()
            non_disadv_mean = pd.to_numeric(state_data[state_data['is_disadvantaged'] == 0]['GrantAmount'], errors='coerce').mean()
            
            if pd.notna(disadv_mean) and pd.notna(non_disadv_mean) and non_disadv_mean > 0:
                state_equity.append((state, disadv_mean / non_disadv_mean, len(state_data)))
        
        # Sort by equity ratio
        state_equity.sort(key=lambda x: x[1])
        states, ratios, counts = zip(*state_equity)
        
        # Color code: red for ratios < 1.0, green for >= 1.0
        colors = ['lightcoral' if r < 1.0 else 'lightgreen' for r in ratios]
        
        bars = plt.barh(range(len(states)), ratios, color=colors, edgecolor='black', alpha=0.8)
        plt.title('State-Level Funding Equity Analysis\n(Disadvantaged ÷ Non-Disadvantaged Grant Ratio)', fontsize=14, fontweight='bold')
        plt.yticks(range(len(states)), states)
        plt.xlabel('Equity Ratio (1.0 = Equal Funding)', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Equity (1.0)')
        
        # Add ratio labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{bar.get_width():.2f}', ha='left', va='center', fontweight='bold')
        
        plt.legend()
        plt.xlim(0, max(ratios) * 1.1)
        plot_file = PROCESSED_DATA_DIR / "07_state_equity_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 8. Purpose Co-occurrence Heatmap
    purpose_cols = [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]
    if len(purpose_cols) >= 5:  # Only create if we have enough purposes
        plt.figure(figsize=(10, 8))
        
        # Create correlation matrix for purposes
        purpose_data = df[purpose_cols]
        corr_matrix = purpose_data.corr()
        
        # Clean up labels
        clean_labels = [_clean_purpose_name(col) for col in purpose_cols]
        
        # Create heatmap - mask only upper triangle, keep diagonal
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # k=1 keeps diagonal visible
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8},
                   xticklabels=clean_labels, yticklabels=clean_labels)
        
        plt.title('Grant Purpose Co-occurrence Analysis\n(How Often Purposes are Selected Together)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plot_file = PROCESSED_DATA_DIR / "08_purpose_cooccurrence.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 9. Purpose Patterns by Demographics
    purpose_cols = [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]
    if purpose_cols and 'is_disadvantaged' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Calculate purpose rates by demographic status
        disadv_purposes = df[df['is_disadvantaged'] == 1][purpose_cols].mean() * 100
        non_disadv_purposes = df[df['is_disadvantaged'] == 0][purpose_cols].mean() * 100
        
        # Clean labels
        clean_labels = [_clean_purpose_name(col) for col in purpose_cols]
        
        x = np.arange(len(clean_labels))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, disadv_purposes.values, width, 
                       label='Disadvantaged', color='lightblue', edgecolor='darkblue', alpha=0.8)
        bars2 = plt.bar(x + width/2, non_disadv_purposes.values, width,
                       label='Non-Disadvantaged', color='lightcoral', edgecolor='darkred', alpha=0.8)
        
        plt.title('Grant Purpose Selection by Business Status', fontsize=14, fontweight='bold')
        plt.xlabel('Grant Purpose', fontsize=12)
        plt.ylabel('Selection Rate (%)', fontsize=12)
        plt.xticks(x, clean_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        plt.ylim(0, 100)
        plt.tight_layout()
        
        plot_file = PROCESSED_DATA_DIR / "09_purpose_by_demographics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    # 10. Average Grant by Primary Purpose
    purpose_cols = [col for col in df.columns if col.endswith('_binary') and ('purpose' in col or 'purp' in col)]
    if purpose_cols and 'GrantAmount' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Find primary purpose (most common single purpose) for each business
        df_temp = df.copy()
        df_temp['purpose_count'] = df_temp[purpose_cols].sum(axis=1)
        df_temp['GrantAmount_num'] = pd.to_numeric(df_temp['GrantAmount'], errors='coerce')
        
        # Calculate average grant by purpose
        purpose_grants = {}
        for col in purpose_cols:
            businesses_with_purpose = df_temp[df_temp[col] == 1]
            avg_grant = businesses_with_purpose['GrantAmount_num'].mean()
            count = len(businesses_with_purpose)
            purpose_grants[col] = (avg_grant / 1000, count)  # Convert to thousands
        
        # Sort by average grant amount
        sorted_purposes = sorted(purpose_grants.items(), key=lambda x: x[1][0], reverse=True)
        
        purposes, amounts_counts = zip(*sorted_purposes)
        amounts, counts = zip(*amounts_counts)
        
        clean_labels = [_clean_purpose_name(col) for col in purposes]
        
        # Create color gradient based on amount
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(amounts)))
        bars = plt.bar(range(len(amounts)), amounts, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title('Average Grant Amount by Purpose Category', fontsize=14, fontweight='bold')
        plt.xlabel('Grant Purpose', fontsize=12)
        plt.ylabel('Average Grant Amount ($K)', fontsize=12)
        plt.xticks(range(len(clean_labels)), clean_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'${bar.get_height():.0f}K', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        plot_file = PROCESSED_DATA_DIR / "10_grant_by_purpose.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plots_created.append(plot_file.name)
        plt.show()
        plt.close()
    
    print(f"Created {len(plots_created)} individual plots:")
    for plot in plots_created:
        print(f"  - {plot}")
    
    return plots_created

def run_comprehensive_analysis():
    """Run complete comprehensive analysis"""
    print("=== RRF COMPREHENSIVE ANALYSIS ===")
    df = load_data()
    
    # Run all analyses
    for analyze_func in [analyze_data_quality, analyze_descriptive_stats, 
                         analyze_demographics, analyze_equity, 
                         analyze_geographic_patterns, analyze_grant_purposes]:
        analyze_func(df)
    
    # Visualizations with fallback
    try:
        create_individual_plots(df)
    except Exception as e:
        print(f"Individual plotting failed: {e}")
        try:
            create_basic_plots(df)
        except Exception as e2:
            print(f"Basic plotting also failed: {e2}")
    
    print("\n=== COMPREHENSIVE ANALYSIS COMPLETE ===")
    return df

def create_basic_plots(df):
    """Fallback basic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if 'GrantAmount' in df.columns:
        grant_data = pd.to_numeric(df['GrantAmount'], errors='coerce').dropna()
        axes[0,0].hist(grant_data / 1000, bins=30, alpha=0.7)
        axes[0,0].set_title('Grant Distribution ($K)')
    
    if 'is_disadvantaged' in df.columns:
        demo_counts = df['is_disadvantaged'].value_counts().reindex([0, 1], fill_value=0)
        axes[0,1].pie(demo_counts.values, labels=['Non-Disadvantaged', 'Disadvantaged'], autopct='%1.1f%%')
        axes[0,1].set_title('Business Demographics')
    
    plt.tight_layout()
    plot_file = PROCESSED_DATA_DIR / "basic_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved basic plots: {plot_file}")

def create_plots_only():
    """Create just the individual plots without running full analysis"""
    print("=== CREATING PLOTS ONLY ===")
    df = load_data()
    plots_created = create_individual_plots(df)
    print(f"\n=== PLOTS COMPLETE: {len(plots_created)} files created ===")
    return plots_created

def run_analysis():
    """Backward compatibility - runs comprehensive analysis"""
    return run_comprehensive_analysis()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plots":
        create_plots_only()
    else:
        run_analysis()
