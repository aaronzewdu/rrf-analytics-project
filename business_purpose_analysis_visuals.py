#!/usr/bin/env python3
# visuals for business purpose analysis

from __future__ import annotations

import json
from pathlib import Path
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR
from utils import setup_logger

logger = setup_logger(__name__)
sns.set_theme(style="whitegrid", context="talk")

INSIGHTS_FILENAME = "business_purpose_insights.json"

FOOTNOTE_COMPARABLES = (
    "Comparable businesses: fixed-food peers (restaurants, cafes, delis, bakeries) and "
    "fixed-beverage peers (bars, taprooms, brewpubs). Geographic items compare urban vs rural "
    "within the same business type. Disadvantaged items compare disadvantaged vs non-disadvantaged within the same type."
)


def _load_insights(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Insights JSON not found at {path}. Run `python business_purpose_analysis.py` first."
        )
    with open(path, "r") as f:
        return json.load(f)


def _get_biztype_df(insights: dict, biz_type: str) -> pd.DataFrame:
    # small helper to shape per‑type purpose data
    bt = insights.get("business_types", {}).get(biz_type)
    if not bt:
        return pd.DataFrame(columns=[
            "purpose", "segment_rate", "baseline_rate", "diff",
            "ci_lower", "ci_upper", "baseline_ci_lower", "baseline_ci_upper", "significant"
        ])

    rows = []
    for purpose, stats in bt.get("purposes", {}).items():
        seg = float(stats.get("segment_rate", np.nan))
        base = float(stats.get("baseline_rate", np.nan))
        diff = seg - base
        ci_l = float(stats.get("ci_lower", np.nan))
        ci_u = float(stats.get("ci_upper", np.nan))
        base_n = stats.get("baseline_n")
        # Approx baseline CI (binomial normal approx)
        if base_n and 0 < base < 1:
            se_b = math.sqrt(base * (1 - base) / float(base_n))
            base_ci_l = base - 1.96 * se_b
            base_ci_u = base + 1.96 * se_b
        else:
            base_ci_l = np.nan
            base_ci_u = np.nan
        significant = bool(stats.get("significant_fdr", False))
        rows.append({
            "purpose": purpose,
            "segment_rate": seg,
            "baseline_rate": base,
            "diff": diff,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "baseline_ci_lower": base_ci_l,
            "baseline_ci_upper": base_ci_u,
            "significant": significant,
        })

    df = pd.DataFrame(rows)
    return df


def _plot_dumbbell_for_types(insights: dict, out_dir: Path) -> None:
    # dumbbell chart for bakeries and brewpubs
    present_types = [bt for bt in ("bakery", "brewpub") if bt in insights.get("business_types", {})]
    if not present_types:
        logger.warning("No bakery/brewpub data present for dumbbell chart; skipping.")
        return

    n_panels = len(present_types)
    fig, axes = plt.subplots(1, n_panels, figsize=(14 if n_panels == 2 else 8, 8))
    if n_panels == 1:
        axes = [axes]

    # simple color palette
    COLOR_MORE = "#2E7D32"  # Professional green
    COLOR_LESS = "#D32F2F"  # Professional red
    COLOR_BASELINE = "#757575"  # Grey

    for ax, bt in zip(axes, present_types):
        df = _get_biztype_df(insights, bt)
        if df.empty:
            ax.axis("off")
            continue
        
        # sort by absolute difference
        df_sig = df[df["significant"]].copy()
        if df_sig.empty:
            df_sig = df.copy()
        df_sig = df_sig.sort_values("diff", key=lambda s: s.abs(), ascending=True)

        y = np.arange(len(df_sig))
        
        # connecting lines
        for i, (idx, row) in enumerate(df_sig.iterrows()):
            color = COLOR_MORE if row["diff"] > 0 else COLOR_LESS
            ax.plot([row["baseline_rate"], row["segment_rate"]], [i, i], 
                   color=color, lw=2.5, alpha=0.7, zorder=1)

        # baseline dots
        ax.scatter(df_sig["baseline_rate"], y, s=80, color=COLOR_BASELINE, 
                  edgecolors='white', linewidths=1, zorder=3, alpha=0.9)

        # segment dots
        colors = [COLOR_MORE if d > 0 else COLOR_LESS for d in df_sig["diff"]]
        ax.scatter(df_sig["segment_rate"], y, s=100, color=colors,
                  edgecolors='white', linewidths=1.5, zorder=4)

        # y labels
        ax.set_yticks(y)
        ax.set_yticklabels(df_sig["purpose"].tolist(), fontsize=11)
        
        # subtle gridlines
        ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # x ticks as decimals
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
        ax.tick_params(axis='x', labelsize=10, pad=5, length=4, width=1)
        
        # small titles
        title_bt = "Bakeries" if bt == "bakery" else "Brewpubs"
        peer_label = "Industry" if bt == "bakery" else "Industry"
        ax.set_title(f"{title_bt} vs {peer_label} Average", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Selection Rate", fontsize=11)

    # legend at top
    if n_panels > 0:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BASELINE, 
                      markersize=8, label='Industry average'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MORE, 
                      markersize=8, label='Above average'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_LESS, 
                      markersize=8, label='Below average'),
        ]
        fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, 
                  bbox_to_anchor=(0.5, 0.88), fontsize=10)

    fig.suptitle("Grant Purpose Selection Patterns", fontsize=16, fontweight='bold', y=0.95)

    out_path = out_dir / "03_purpose_vs_peers_dumbbell.png"
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_heatmap(insights: dict, out_dir: Path) -> None:
    # bar charts for bakery and brewpub vs peers
    bt_map = insights.get("business_types", {})
    if not bt_map:
        logger.warning("No business_types data for differences; skipping.")
        return

    def plot_single(bt: str, peer_label: str, filename: str) -> None:
        bt_data = bt_map.get(bt)
        if not bt_data:
            return
        rows = []
        for purpose, stats in bt_data.get("purposes", {}).items():
            seg = float(stats.get("segment_rate", np.nan))
            base = float(stats.get("baseline_rate", np.nan))
            if np.isnan(seg) or np.isnan(base):
                continue
            diff_pct = (seg - base) * 100.0  # percent points
            sig = bool(stats.get("significant_fdr", False))
            rows.append({"purpose": purpose, "diff_pct": diff_pct, "significant": sig})
        if not rows:
            return
        df = pd.DataFrame(rows)
        # Sort by absolute difference
        df = df.sort_values("diff_pct", key=lambda s: s.abs(), ascending=False)

        # figure sizing
        n = len(df)
        fig_h = max(4.5, min(0.6 * n + 2.0, 12))
        fig_w = 12
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

        colors = np.where(df["diff_pct"] >= 0, "#2C7BB6", "#D7191C")
        alphas = np.where(df["significant"], 0.95, 0.4)
        for purpose, diff_val, color, alpha in zip(df["purpose"], df["diff_pct"], colors, alphas):
            ax.barh(purpose, diff_val, color=color, alpha=alpha, edgecolor="#333333")

        # symmetric x limits rounded to 5pp
        max_abs = float(np.nanmax(np.abs(df["diff_pct"].values))) if len(df) else 10.0
        lim = max(5.0, (math.ceil(max_abs / 5.0) * 5.0))
        ax.set_xlim(-lim, lim)

        ax.axvline(0, color="#9E9E9E", lw=1)
        ax.set_xlabel("Difference vs peers (%)", labelpad=10)
        ax.set_ylabel("Purpose")

        title_bt = "Bakeries" if bt == "bakery" else "Brewpubs"
        ax.set_title(f"{title_bt} vs {peer_label} peers — purpose selection gap", pad=14, fontsize=16)

        # readability
        ax.grid(axis='x', linestyle='--', alpha=0.25)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)

        # allow wider left margin for labels
        fig.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.15)

        out_path = out_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved {out_path}")

    # generate bakery and brewpub panels
    plot_single("bakery", "fixed-food", "04_bakery_vs_peers_bars.png")
    plot_single("brewpub", "fixed-beverage", "04_brewpub_vs_peers_bars.png")


def _plot_urban_rural_rent(insights: dict, out_dir: Path) -> None:
    # urban vs rural rent comparison
    geo = insights.get("geographic", {})
    rows = []
    for bt, items in geo.items():
        for it in items:
            if it.get("purpose") == "Rent":
                rows.append({
                    "business_type": bt,
                    "Rural": float(it.get("rural_rate", np.nan)) * 100,  # Convert to percentage
                    "Urban": float(it.get("urban_rate", np.nan)) * 100,
                })

    if not rows:
        logger.warning("No urban/rural Rent items found; skipping urban-rural bar chart.")
        return

    df = pd.DataFrame(rows)
    df["gap"] = df["Urban"] - df["Rural"]
    df = df.sort_values("gap", ascending=True)  # Ascending for horizontal bars
    
    # styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # grouped bars
    y_pos = np.arange(len(df))
    bar_height = 0.35
    
    # plot bars
    urban_bars = ax.barh(y_pos - bar_height/2, df["Urban"], bar_height, 
                         label='Urban', color='#1565C0', alpha=0.9)
    rural_bars = ax.barh(y_pos + bar_height/2, df["Rural"], bar_height,
                         label='Rural', color='#78909C', alpha=0.9)
    
    # no individual bar labels
    
    # Styling with descriptive business labels
    business_labels = {
        'bar': 'Drinking Establishments',
        'producer': 'Beverage Producers'
    }
    ax.set_yticks(y_pos)
    ax.set_yticklabels([business_labels.get(bt, bt.title()) for bt in df["business_type"]], fontsize=12)
    ax.set_xlabel('Rent Selection Rate (%)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_title('Location Impact on Rent Selection', fontsize=14, fontweight='bold', pad=40)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend - positioned properly below the title
    ax.legend(loc='upper center', frameon=False, fontsize=11, 
              bbox_to_anchor=(0.5, 1.08), ncol=2)
    
    out_path = out_dir / "05_rural_urban_rent_by_type.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def _plot_equity_panel(insights: dict, out_dir: Path) -> None:
    # simple equity comparison
    disadv = insights.get("disadvantaged", {})

    rows = []
    for bt, items in disadv.items():
        for it in items:
            if it.get("purpose") == "Supplies":
                rows.append({
                    "business_type": bt.title(),
                    "Non-disadvantaged": float(it.get("non_disadvantaged_rate", np.nan)) * 100,
                    "Disadvantaged": float(it.get("disadvantaged_rate", np.nan)) * 100,
                })

    if not rows:
        logger.warning("No disadvantaged vs non-disadvantaged 'Supplies' items found; skipping equity bar chart.")
        return

    df = pd.DataFrame(rows)
    
    # simple horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    
    categories = ['Standard Bakeries', 'Disadvantaged Bakeries']
    values = [df.iloc[0]["Non-disadvantaged"], df.iloc[0]["Disadvantaged"]]
    colors = ['#9E9E9E', '#2E7D32']
    
    # create horizontal bars
    bars = ax.barh(categories, values, color=colors, height=0.5)
    
    # clean styling - no labels, no grid
    ax.set_xlim(0, 100)
    ax.set_xlabel('Supplies Selection Rate (%)', fontsize=12)
    ax.set_title('Bakeries: Supplies Selection by Business Status', fontsize=14, fontweight='bold', pad=15)
    
    # remove all spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # no grid
    ax.grid(False)
    
    out_path = out_dir / "06_equity_bakeries_supplies.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main() -> None:
    processed_dir = Path(PROCESSED_DATA_DIR)
    insights_path = processed_dir / INSIGHTS_FILENAME
    logger.info(f"Loading insights from {insights_path}")
    insights = _load_insights(insights_path)

    # Generate figures
    _plot_dumbbell_for_types(insights, processed_dir)
    _plot_heatmap(insights, processed_dir)
    _plot_urban_rural_rent(insights, processed_dir)
    _plot_equity_panel(insights, processed_dir)

    logger.info("All visuals generated.")


if __name__ == "__main__":
    main()
