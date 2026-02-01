"""
05_visualize.py
Generate maps and visualizations for transit desert analysis.

Outputs:
- Maps: CPTA, TDI, Transit Gap, LISA clusters, Classification
- Figures: Distribution plots, scatter plots, equity comparison
- Tables: Summary statistics

Usage:
    python src/05_visualize.py
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import contextily as ctx
    CTX_AVAILABLE = True
except ImportError:
    ctx = None
    CTX_AVAILABLE = False
    print("contextily not installed. Basemaps will not be added.")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results():
    """Load transit desert analysis results."""
    results_path = PROJECT_ROOT / "data" / "processed" / "transit_deserts.gpkg"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    gdf = gpd.read_file(results_path)
    print(f"  ✓ Loaded results for {len(gdf)} tracts")
    
    return gdf


def setup_plot_style():
    """Set up matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10


def create_choropleth_map(gdf, column, title, cmap='RdYlBu_r', output_path=None, 
                          legend_title=None):
    """Create a choropleth map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    gdf_plot = gdf.to_crs(epsg=3857)
    
    gdf_plot.plot(
        column=column,
        cmap=cmap,
        linewidth=0.5,
        edgecolor='white',
        legend=True,
        legend_kwds={
            'label': legend_title or column,
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.05
        },
        ax=ax
    )
    
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def create_classification_map(gdf, output_path=None, city_name=""):
    """Create map of transit desert classification."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    gdf_plot = gdf.to_crs(epsg=3857)
    
    colors = {
        'Transit Desert': '#d73027',
        'Transit Stressed': '#fc8d59',
        'Underserved': '#fee090',
        'Well-Served': '#91bfdb'
    }
    
    for category, color in colors.items():
        subset = gdf_plot[gdf_plot['classification_simple'] == category]
        if len(subset) > 0:
            subset.plot(color=color, linewidth=0.5, edgecolor='white', ax=ax)
    
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    
    patches = [mpatches.Patch(color=color, label=f"{cat} ({len(gdf_plot[gdf_plot['classification_simple']==cat])})") 
               for cat, color in colors.items()]
    ax.legend(handles=patches, loc='lower right', frameon=True, fontsize=10, title='Classification')
    
    title = 'Transit Desert Classification'
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def create_lisa_map(gdf, output_path=None, city_name=""):
    """Create LISA cluster map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    gdf_plot = gdf.to_crs(epsg=3857)
    
    colors = {
        'High-High': '#d7191c',
        'Low-Low': '#2c7bb6',
        'High-Low': '#fdae61',
        'Low-High': '#abd9e9',
        'Not Significant': '#eeeeee'
    }
    
    for cluster, color in colors.items():
        subset = gdf_plot[gdf_plot['lisa_cluster'] == cluster]
        if len(subset) > 0:
            subset.plot(color=color, linewidth=0.5, edgecolor='white', ax=ax)
    
    if CTX_AVAILABLE:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    
    patches = [mpatches.Patch(color=color, label=f"{cat} ({len(gdf_plot[gdf_plot['lisa_cluster']==cat])})") 
               for cat, color in colors.items()]
    ax.legend(handles=patches, loc='lower right', frameon=True, fontsize=10, title='LISA Clusters (p<0.05)')
    
    title = "Local Moran's I Clusters (Transit Gap)"
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def create_scatter_plot(gdf, output_path=None, city_name=""):
    """Create scatter plot of supply vs demand."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = {
        'Transit Desert': '#d73027',
        'Transit Stressed': '#fc8d59',
        'Underserved': '#fee090',
        'Well-Served': '#91bfdb'
    }
    
    for category, color in colors.items():
        subset = gdf[gdf['classification_simple'] == category]
        ax.scatter(subset['CPTA_normalized'], subset['TDI_normalized'],
                   c=color, label=f"{category} ({len(subset)})", alpha=0.7, s=50,
                   edgecolors='white', linewidth=0.5)
    
    cpta_median = gdf['CPTA_normalized'].median()
    tdi_median = gdf['TDI_normalized'].median()
    
    ax.axvline(x=cpta_median, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=tdi_median, color='gray', linestyle='--', alpha=0.5)
    
    ax.text(cpta_median/2, tdi_median + (100-tdi_median)/2, 'Transit\nDesert', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.text(cpta_median + (100-cpta_median)/2, tdi_median + (100-tdi_median)/2, 'Transit\nStressed', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.text(cpta_median/2, tdi_median/2, 'Under-\nserved', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    ax.text(cpta_median + (100-cpta_median)/2, tdi_median/2, 'Well-\nServed', 
            ha='center', va='center', fontsize=12, alpha=0.4, fontweight='bold')
    
    ax.set_xlabel('Transit Supply (CPTA Score)', fontsize=12)
    ax.set_ylabel('Transit Demand (TDI Score)', fontsize=12)
    title = 'Transit Supply vs. Demand by Census Tract'
    if city_name:
        title += f'\n{city_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def create_distribution_plots(gdf, output_path=None):
    """Create distribution plots for key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    sns.histplot(gdf['CPTA_normalized'], kde=True, ax=ax1, color='steelblue', bins=20)
    ax1.axvline(x=gdf['CPTA_normalized'].median(), color='red', linestyle='--', linewidth=2, 
                label=f'Median: {gdf["CPTA_normalized"].median():.1f}')
    ax1.set_xlabel('CPTA Score (0-100)')
    ax1.set_title('Transit Supply Distribution', fontweight='bold')
    ax1.legend()
    
    ax2 = axes[0, 1]
    sns.histplot(gdf['TDI_normalized'], kde=True, ax=ax2, color='coral', bins=20)
    ax2.axvline(x=gdf['TDI_normalized'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {gdf["TDI_normalized"].median():.1f}')
    ax2.set_xlabel('TDI Score (0-100)')
    ax2.set_title('Transit Demand Distribution', fontweight='bold')
    ax2.legend()
    
    ax3 = axes[1, 0]
    sns.histplot(gdf['transit_gap'], kde=True, ax=ax3, color='purple', bins=20)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(x=gdf['transit_gap'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {gdf["transit_gap"].median():.1f}')
    ax3.set_xlabel('Transit Gap (TDI - CPTA)')
    ax3.set_title('Transit Gap Distribution', fontweight='bold')
    ax3.legend()
    
    ax4 = axes[1, 1]
    class_counts = gdf['classification_simple'].value_counts()
    colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb']
    class_order = ['Transit Desert', 'Transit Stressed', 'Underserved', 'Well-Served']
    class_counts = class_counts.reindex(class_order).fillna(0)
    
    bars = ax4.bar(range(len(class_counts)), class_counts.values, color=colors)
    ax4.set_xticks(range(len(class_counts)))
    ax4.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax4.set_ylabel('Number of Tracts')
    ax4.set_title('Transit Desert Classification', fontweight='bold')
    
    for bar, count in zip(bars, class_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(count)), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def create_equity_comparison(gdf, output_path=None):
    """Create equity comparison bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    deserts = gdf[gdf['transit_desert'] == True]
    non_deserts = gdf[gdf['transit_desert'] == False]
    
    metrics = {
        'pct_zero_vehicle': 'Zero-Vehicle HH',
        'pct_poverty': 'Below Poverty',
        'pct_minority': 'Minority Pop.',
        'pct_elderly': 'Elderly (65+)'
    }
    
    available_metrics = {k: v for k, v in metrics.items() if k in gdf.columns}
    
    if not available_metrics:
        print("  ⚠ No demographic columns found")
        plt.close()
        return
    
    x = np.arange(len(available_metrics))
    width = 0.35
    
    desert_vals = [deserts[m].mean() * 100 for m in available_metrics.keys()]
    non_desert_vals = [non_deserts[m].mean() * 100 for m in available_metrics.keys()]
    
    bars1 = ax.bar(x - width/2, desert_vals, width, label='Transit Deserts', color='#d73027')
    bars2 = ax.bar(x + width/2, non_desert_vals, width, label='Non-Deserts', color='#91bfdb')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Demographic Comparison: Transit Deserts vs. Non-Deserts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics.values(), fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path.name}")
    
    plt.close()


def create_summary_table(gdf, output_path=None):
    """Create summary statistics table."""
    overall = {
        'Metric': ['Total Tracts', 'Transit Deserts', 'Transit Stressed', 'Underserved', 'Well-Served'],
        'Count': [
            len(gdf),
            len(gdf[gdf['classification_simple'] == 'Transit Desert']),
            len(gdf[gdf['classification_simple'] == 'Transit Stressed']),
            len(gdf[gdf['classification_simple'] == 'Underserved']),
            len(gdf[gdf['classification_simple'] == 'Well-Served'])
        ]
    }
    overall_df = pd.DataFrame(overall)
    overall_df['Percentage'] = (overall_df['Count'] / len(gdf) * 100).round(1)
    overall_df.loc[0, 'Percentage'] = 100.0
    
    if output_path:
        overall_df.to_csv(output_path, index=False)
        print(f"  ✓ Saved: {output_path.name}")
    
    return overall_df


def main():
    """Main function to generate visualizations."""
    print("=" * 60)
    print("Transit Desert Pipeline: Visualization")
    print("=" * 60)
    
    config = load_config()
    setup_plot_style()
    
    city_name = config['study_area'].get('name', '')
    
    maps_dir = PROJECT_ROOT / "outputs" / "maps"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    
    maps_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    gdf = load_results()
    
    print("\n" + "-" * 40)
    print("Creating maps...")
    print("-" * 40)
    
    create_choropleth_map(gdf, 'CPTA_normalized', f'Transit Supply (CPTA Score)\n{city_name}',
                          cmap='RdYlBu', output_path=maps_dir / 'map_cpta.png',
                          legend_title='CPTA Score (0-100)')
    
    create_choropleth_map(gdf, 'TDI_normalized', f'Transit Demand (TDI Score)\n{city_name}',
                          cmap='RdYlBu_r', output_path=maps_dir / 'map_tdi.png',
                          legend_title='TDI Score (0-100)')
    
    create_choropleth_map(gdf, 'transit_gap', f'Transit Gap (Demand - Supply)\n{city_name}',
                          cmap='RdYlBu_r', output_path=maps_dir / 'map_transit_gap.png',
                          legend_title='Transit Gap')
    
    create_classification_map(gdf, output_path=maps_dir / 'map_classification.png', city_name=city_name)
    
    create_lisa_map(gdf, output_path=maps_dir / 'map_lisa_clusters.png', city_name=city_name)
    
    print("\n" + "-" * 40)
    print("Creating figures...")
    print("-" * 40)
    
    create_scatter_plot(gdf, output_path=figures_dir / 'fig_supply_vs_demand.png', city_name=city_name)
    create_distribution_plots(gdf, output_path=figures_dir / 'fig_distributions.png')
    create_equity_comparison(gdf, output_path=figures_dir / 'fig_equity_comparison.png')
    
    print("\n" + "-" * 40)
    print("Creating tables...")
    print("-" * 40)
    
    summary_df = create_summary_table(gdf, output_path=tables_dir / 'summary_statistics.csv')
    
    print("\n  Summary:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"\nOutputs saved to:")
    print(f"  Maps:    {maps_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables:  {tables_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
