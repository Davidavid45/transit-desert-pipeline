"""
04_identify_deserts.py
Identify transit deserts using LISA spatial clustering.

Methodology:
1. Compute Transit Gap = Normalized TDI - Normalized CPTA
2. Apply bivariate LISA to identify spatial clusters
3. Classify tracts: Transit Desert, Transit Stressed, Underserved, Well-Served

Usage:
    python src/04_identify_deserts.py

Outputs:
    - data/processed/transit_deserts.csv
    - data/processed/transit_deserts.gpkg
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import zscore

# PySAL for spatial analysis
try:
    import libpysal
    from esda.moran import Moran, Moran_Local
except ImportError:
    print("PySAL libraries not installed. Run: pip install pysal esda libpysal")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_supply_demand_data():
    """Load supply (CPTA) and demand (TDI) metrics."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    supply_path = processed_dir / "supply_metrics.csv"
    demand_path = processed_dir / "demand_metrics.csv"
    
    if not supply_path.exists():
        raise FileNotFoundError(f"Supply metrics not found: {supply_path}")
    if not demand_path.exists():
        raise FileNotFoundError(f"Demand metrics not found: {demand_path}")
    
    supply_df = pd.read_csv(supply_path, dtype={'GEOID': str})
    demand_df = pd.read_csv(demand_path, dtype={'GEOID': str})
    
    print(f"  ✓ Loaded supply metrics for {len(supply_df)} tracts")
    print(f"  ✓ Loaded demand metrics for {len(demand_df)} tracts")
    
    return supply_df, demand_df

def load_tracts(config):
    """Load Census tract boundaries."""
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    gdf = gdf.to_crs(config['spatial']['crs_projected'])
    
    print(f"  ✓ Loaded {len(gdf)} tract boundaries")
    return gdf

def merge_supply_demand(supply_df, demand_df, tracts_gdf):
    """Merge supply and demand data with tract geometries."""
    print("Merging supply and demand data...")
    
    merged = tracts_gdf[['GEOID', 'geometry']].copy()
    
    # Add supply metrics
    supply_cols = ['GEOID', 'CPTA', 'CPTA_normalized']
    if all(col in supply_df.columns for col in supply_cols):
        merged = merged.merge(supply_df[supply_cols], on='GEOID', how='left')
    else:
        merged['CPTA'] = 0
        merged['CPTA_normalized'] = 50
    
    # Add demand metrics
    demand_cols = ['GEOID', 'TDI', 'TDI_normalized', 
                   'pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']
    available_demand_cols = [col for col in demand_cols if col in demand_df.columns]
    merged = merged.merge(demand_df[available_demand_cols], on='GEOID', how='left')
    
    merged = merged.fillna({'CPTA': 0, 'CPTA_normalized': 0, 'TDI': 0, 'TDI_normalized': 0})
    
    print(f"  ✓ Merged {len(merged)} tracts")
    return merged

def compute_transit_gap(gdf):
    """Compute Transit Gap = Demand - Supply."""
    print("Computing transit gap...")
    
    # Simple difference of normalized scores
    gdf['transit_gap'] = gdf['TDI_normalized'] - gdf['CPTA_normalized']
    
    # Z-score based gap
    gdf['z_TDI'] = zscore(gdf['TDI'].fillna(0))
    gdf['z_CPTA'] = zscore(gdf['CPTA'].fillna(0))
    gdf['transit_gap_z'] = gdf['z_TDI'] - gdf['z_CPTA']
    
    print(f"\n  Transit Gap Statistics:")
    print(f"    Mean: {gdf['transit_gap'].mean():.1f}")
    print(f"    Std:  {gdf['transit_gap'].std():.1f}")
    print(f"    Min:  {gdf['transit_gap'].min():.1f}")
    print(f"    Max:  {gdf['transit_gap'].max():.1f}")
    
    return gdf

def create_spatial_weights(gdf):
    """Create spatial weights matrix for LISA (Queen contiguity)."""
    print("Creating spatial weights matrix...")
    
    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = 'r'  # Row-standardize
    
    islands = w.islands
    if islands:
        print(f"  ⚠ Found {len(islands)} island tracts (no neighbors)")
    
    print(f"  ✓ Created weights for {w.n} tracts")
    print(f"  ✓ Mean neighbors: {w.mean_neighbors:.1f}")
    
    return w

def compute_lisa_univariate(gdf, w, variable='transit_gap'):
    """Compute univariate Local Moran's I for transit gap."""
    print(f"Computing univariate LISA for {variable}...")
    
    y = gdf[variable].values
    lisa = Moran_Local(y, w, permutations=999)
    
    gdf['lisa_I'] = lisa.Is
    gdf['lisa_p'] = lisa.p_sim
    gdf['lisa_q'] = lisa.q  # Quadrant (1=HH, 2=LH, 3=LL, 4=HL)
    
    alpha = 0.05
    gdf['lisa_sig'] = gdf['lisa_p'] < alpha
    
    cluster_labels = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}
    gdf['lisa_cluster'] = gdf.apply(
        lambda row: cluster_labels.get(row['lisa_q'], 'Not Significant') 
        if row['lisa_sig'] else 'Not Significant',
        axis=1
    )
    
    print(f"\n  Global Moran's I: {Moran(y, w).I:.3f}")
    print(f"\n  Cluster counts:")
    for cluster, count in gdf['lisa_cluster'].value_counts().items():
        print(f"    {cluster}: {count}")
    
    return gdf, lisa

def classify_transit_deserts(gdf):
    """Create final transit desert classification."""
    print("Classifying tracts...")
    
    demand_median = gdf['TDI_normalized'].median()
    supply_median = gdf['CPTA_normalized'].median()
    
    def classify_tract(row):
        high_demand = row['TDI_normalized'] >= demand_median
        high_supply = row['CPTA_normalized'] >= supply_median
        
        if high_demand and not high_supply:
            return 'Transit Desert'
        elif high_demand and high_supply:
            return 'Transit Stressed'
        elif not high_demand and not high_supply:
            return 'Underserved'
        else:
            return 'Well-Served'
    
    gdf['classification_simple'] = gdf.apply(classify_tract, axis=1)
    
    def classify_tract_lisa(row):
        if row['lisa_cluster'] == 'High-High':
            return 'Transit Desert (LISA)'
        elif row['lisa_cluster'] == 'High-Low':
            return 'Isolated High Gap'
        elif row['lisa_cluster'] == 'Low-Low':
            return 'Well-Served (LISA)'
        elif row['lisa_cluster'] == 'Low-High':
            return 'Isolated Well-Served'
        else:
            return row['classification_simple']
    
    gdf['classification_lisa'] = gdf.apply(classify_tract_lisa, axis=1)
    
    gdf['transit_desert'] = (
        (gdf['classification_simple'] == 'Transit Desert') | 
        (gdf['lisa_cluster'] == 'High-High')
    )
    
    print(f"\n  Simple Classification:")
    for cat, count in gdf['classification_simple'].value_counts().items():
        print(f"    {cat}: {count}")
    
    print(f"\n  Transit Deserts identified: {gdf['transit_desert'].sum()}")
    
    return gdf

def compute_equity_profile(gdf):
    """Compute demographic profile of transit deserts vs non-deserts."""
    print("\nComputing equity profile of transit deserts...")
    
    deserts = gdf[gdf['transit_desert'] == True]
    non_deserts = gdf[gdf['transit_desert'] == False]
    
    metrics = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']
    
    profile = []
    for metric in metrics:
        if metric in gdf.columns:
            desert_mean = deserts[metric].mean()
            non_desert_mean = non_deserts[metric].mean()
            diff = desert_mean - non_desert_mean
            
            profile.append({
                'Metric': metric,
                'Transit Deserts': f"{desert_mean:.1%}",
                'Non-Deserts': f"{non_desert_mean:.1%}",
                'Difference': f"{diff:+.1%}"
            })
    
    profile_df = pd.DataFrame(profile)
    print("\n  Demographic comparison:")
    print(profile_df.to_string(index=False))
    
    return profile_df

def main():
    """Main function to identify transit deserts."""
    print("=" * 60)
    print("Transit Desert Pipeline: Desert Identification (LISA)")
    print("=" * 60)
    
    config = load_config()
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    supply_df, demand_df = load_supply_demand_data()
    tracts_gdf = load_tracts(config)
    gdf = merge_supply_demand(supply_df, demand_df, tracts_gdf)
    
    # Compute transit gap
    print("\n" + "-" * 40)
    print("Computing transit gap...")
    print("-" * 40)
    gdf = compute_transit_gap(gdf)
    
    # Spatial analysis
    print("\n" + "-" * 40)
    print("Spatial analysis...")
    print("-" * 40)
    w = create_spatial_weights(gdf)
    gdf, lisa_uni = compute_lisa_univariate(gdf, w, variable='transit_gap')
    
    # Classify
    print("\n" + "-" * 40)
    print("Classifying tracts...")
    print("-" * 40)
    gdf = classify_transit_deserts(gdf)
    profile_df = compute_equity_profile(gdf)
    
    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    output_cols = [
        'GEOID', 'geometry',
        'TDI', 'TDI_normalized', 'CPTA', 'CPTA_normalized',
        'transit_gap', 'transit_gap_z',
        'lisa_I', 'lisa_p', 'lisa_cluster', 'lisa_sig',
        'classification_simple', 'classification_lisa', 'transit_desert'
    ]
    
    demo_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']
    output_cols.extend([col for col in demo_cols if col in gdf.columns])
    output_cols = [col for col in output_cols if col in gdf.columns]
    
    output_gdf = gdf[output_cols].copy()
    
    output_df = output_gdf.drop(columns=['geometry'])
    output_df.to_csv(output_dir / "transit_deserts.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'transit_deserts.csv'}")
    
    output_gdf.to_file(output_dir / "transit_deserts.gpkg", driver='GPKG')
    print(f"  ✓ Saved to {output_dir / 'transit_deserts.gpkg'}")
    
    profile_df.to_csv(output_dir / "equity_profile.csv", index=False)
    print(f"  ✓ Saved equity profile to {output_dir / 'equity_profile.csv'}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n  Total tracts analyzed: {len(gdf)}")
    print(f"  Transit Deserts identified: {gdf['transit_desert'].sum()}")
    print(f"  Percentage: {gdf['transit_desert'].mean():.1%}")
    
    if gdf['transit_desert'].sum() > 0:
        desert_tracts = gdf[gdf['transit_desert']]
        print(f"\n  Transit Desert characteristics:")
        print(f"    Average TDI: {desert_tracts['TDI_normalized'].mean():.1f}")
        print(f"    Average CPTA: {desert_tracts['CPTA_normalized'].mean():.1f}")
        print(f"    Average Gap: {desert_tracts['transit_gap'].mean():.1f}")
    
    print("\n" + "=" * 60)
    print("Transit desert identification complete!")
    print("Next step: python src/05_visualize.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
