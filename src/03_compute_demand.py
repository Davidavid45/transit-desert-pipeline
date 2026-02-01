"""
03_compute_demand.py
Compute Transit Dependency Index (TDI) for each Census tract.

TDI Components:
1. Zero-vehicle households (35%)
2. Population below poverty line (25%)
3. Minority population (20%)
4. Elderly population 65+ (20%)

Usage:
    python src/03_compute_demand.py

Outputs:
    - data/processed/demand_metrics.csv
    - data/processed/demand_metrics.gpkg
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_acs_data():
    """
    Load ACS demographic data.
    
    Returns
    -------
    DataFrame
        ACS data with TDI component variables
    """
    acs_path = PROJECT_ROOT / "data" / "raw" / "acs" / "acs_tdi_variables.csv"
    
    if not acs_path.exists():
        raise FileNotFoundError(
            f"ACS data not found: {acs_path}\n"
            "Run 01_download_data.py first, or manually place ACS data in this location."
        )
    
    df = pd.read_csv(acs_path, dtype={'GEOID': str})
    
    print(f"  ✓ Loaded ACS data for {len(df)} tracts")
    
    return df

def load_tracts(config):
    """
    Load Census tract boundaries.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    
    Returns
    -------
    GeoDataFrame
        Census tracts with geometry
    """
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    
    # Ensure GEOID is string
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    
    # Reproject
    crs_projected = config['spatial']['crs_projected']
    gdf = gdf.to_crs(crs_projected)
    
    print(f"  ✓ Loaded {len(gdf)} tract boundaries")
    
    return gdf

def compute_tdi_components(acs_df):
    """
    Compute individual TDI components from ACS data.
    
    Parameters
    ----------
    acs_df : DataFrame
        ACS data with raw variables
    
    Returns
    -------
    DataFrame
        TDI component percentages
    """
    print("Computing TDI components...")
    
    # Check if percentages are already computed
    required_cols = ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']
    
    if all(col in acs_df.columns for col in required_cols):
        print("  ✓ TDI percentages already computed in ACS data")
        return acs_df[['GEOID'] + required_cols].copy()
    
    # Otherwise, compute from raw ACS variables
    tdi = pd.DataFrame()
    tdi['GEOID'] = acs_df['GEOID']
    
    # Zero-vehicle households
    if 'B08201_001E' in acs_df.columns and 'B08201_002E' in acs_df.columns:
        tdi['pct_zero_vehicle'] = np.where(
            acs_df['B08201_001E'] > 0,
            acs_df['B08201_002E'] / acs_df['B08201_001E'],
            0
        )
    else:
        print("  ⚠ Vehicle data not found, using placeholder")
        tdi['pct_zero_vehicle'] = 0
    
    # Poverty rate
    if 'B17001_001E' in acs_df.columns and 'B17001_002E' in acs_df.columns:
        tdi['pct_poverty'] = np.where(
            acs_df['B17001_001E'] > 0,
            acs_df['B17001_002E'] / acs_df['B17001_001E'],
            0
        )
    else:
        print("  ⚠ Poverty data not found, using placeholder")
        tdi['pct_poverty'] = 0
    
    # Minority population (non-white, non-Hispanic)
    if 'B03002_001E' in acs_df.columns and 'B03002_003E' in acs_df.columns:
        tdi['pct_minority'] = np.where(
            acs_df['B03002_001E'] > 0,
            1 - (acs_df['B03002_003E'] / acs_df['B03002_001E']),
            0
        )
    else:
        print("  ⚠ Race/ethnicity data not found, using placeholder")
        tdi['pct_minority'] = 0
    
    # Elderly population (65+)
    elderly_male_cols = ['B01001_020E', 'B01001_021E', 'B01001_022E', 
                         'B01001_023E', 'B01001_024E', 'B01001_025E']
    elderly_female_cols = ['B01001_044E', 'B01001_045E', 'B01001_046E',
                           'B01001_047E', 'B01001_048E', 'B01001_049E']
    
    if 'B01001_001E' in acs_df.columns and all(col in acs_df.columns for col in elderly_male_cols):
        elderly_pop = (
            acs_df[elderly_male_cols].sum(axis=1) + 
            acs_df[elderly_female_cols].sum(axis=1)
        )
        tdi['pct_elderly'] = np.where(
            acs_df['B01001_001E'] > 0,
            elderly_pop / acs_df['B01001_001E'],
            0
        )
    else:
        print("  ⚠ Age data not found, using placeholder")
        tdi['pct_elderly'] = 0
    
    # Cap percentages at 1.0
    for col in required_cols:
        tdi[col] = tdi[col].clip(0, 1)
    
    return tdi

def compute_tdi_score(tdi_components, config):
    """
    Compute final Transit Dependency Index.
    
    TDI = weighted sum of z-scores
    
    Parameters
    ----------
    tdi_components : DataFrame
        Individual TDI component percentages
    config : dict
        Configuration dictionary with weights
    
    Returns
    -------
    DataFrame
        TDI components with final score
    """
    print("Computing TDI scores...")
    
    # Get weights from config
    weights = config['census']['tdi_weights']
    
    print(f"\n  TDI Weights:")
    for component, weight in weights.items():
        print(f"    {component}: {weight}")
    
    # Map column names to weights
    weight_mapping = {
        'pct_zero_vehicle': weights['zero_vehicle'],
        'pct_poverty': weights['poverty'],
        'pct_minority': weights['minority'],
        'pct_elderly': weights['elderly']
    }
    
    # Compute z-scores for each component
    for col in ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']:
        z_col = f'z_{col}'
        values = tdi_components[col].fillna(0)
        if values.std() > 0:
            tdi_components[z_col] = zscore(values)
        else:
            tdi_components[z_col] = 0
    
    # Compute weighted TDI
    tdi_components['TDI'] = (
        weight_mapping['pct_zero_vehicle'] * tdi_components['z_pct_zero_vehicle'] +
        weight_mapping['pct_poverty'] * tdi_components['z_pct_poverty'] +
        weight_mapping['pct_minority'] * tdi_components['z_pct_minority'] +
        weight_mapping['pct_elderly'] * tdi_components['z_pct_elderly']
    )
    
    # Also create normalized 0-100 score
    tdi_min = tdi_components['TDI'].min()
    tdi_max = tdi_components['TDI'].max()
    if tdi_max > tdi_min:
        tdi_components['TDI_normalized'] = (
            (tdi_components['TDI'] - tdi_min) / (tdi_max - tdi_min) * 100
        )
    else:
        tdi_components['TDI_normalized'] = 50
    
    return tdi_components

def create_demand_summary(tdi_df):
    """
    Create summary statistics for demand analysis.
    
    Parameters
    ----------
    tdi_df : DataFrame
        TDI data
    
    Returns
    -------
    dict
        Summary statistics
    """
    summary = {
        'n_tracts': len(tdi_df),
        'tdi_mean': tdi_df['TDI'].mean(),
        'tdi_std': tdi_df['TDI'].std(),
        'tdi_min': tdi_df['TDI'].min(),
        'tdi_max': tdi_df['TDI'].max(),
        'components': {}
    }
    
    for col in ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']:
        summary['components'][col] = {
            'mean': tdi_df[col].mean(),
            'std': tdi_df[col].std(),
            'min': tdi_df[col].min(),
            'max': tdi_df[col].max()
        }
    
    return summary

def main():
    """Main function to compute demand metrics."""
    print("=" * 60)
    print("Transit Desert Pipeline: Demand Metrics (TDI)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Paths
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    acs_df = load_acs_data()
    tracts_gdf = load_tracts(config)
    
    # Compute TDI components
    print("\n" + "-" * 40)
    print("Computing TDI components...")
    print("-" * 40)
    
    tdi_components = compute_tdi_components(acs_df)
    
    # Compute TDI score
    print("\n" + "-" * 40)
    print("Computing TDI score...")
    print("-" * 40)
    
    tdi_df = compute_tdi_score(tdi_components, config)
    
    # Ensure all tracts are included
    all_tracts = tracts_gdf[['GEOID']].copy()
    all_tracts['GEOID'] = all_tracts['GEOID'].astype(str)
    tdi_df['GEOID'] = tdi_df['GEOID'].astype(str)
    tdi_df = all_tracts.merge(tdi_df, on='GEOID', how='left').fillna(0)
    
    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    # CSV
    tdi_df.to_csv(output_dir / "demand_metrics.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'demand_metrics.csv'}")
    
    # GeoPackage (with geometry)
    tdi_gdf = tracts_gdf.merge(tdi_df, on='GEOID')
    tdi_gdf.to_file(output_dir / "demand_metrics.gpkg", driver='GPKG')
    print(f"  ✓ Saved to {output_dir / 'demand_metrics.gpkg'}")
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary Statistics")
    print("-" * 40)
    
    summary = create_demand_summary(tdi_df)
    
    print(f"\n  Tracts analyzed: {summary['n_tracts']}")
    
    print(f"\n  TDI Score:")
    print(f"    Mean: {summary['tdi_mean']:.3f}")
    print(f"    Std:  {summary['tdi_std']:.3f}")
    print(f"    Min:  {summary['tdi_min']:.3f}")
    print(f"    Max:  {summary['tdi_max']:.3f}")
    
    print(f"\n  TDI Normalized (0-100):")
    print(f"    Mean: {tdi_df['TDI_normalized'].mean():.1f}")
    print(f"    Min:  {tdi_df['TDI_normalized'].min():.1f}")
    print(f"    Max:  {tdi_df['TDI_normalized'].max():.1f}")
    
    print(f"\n  Component Statistics:")
    for col, stats in summary['components'].items():
        print(f"\n    {col}:")
        print(f"      Mean: {stats['mean']:.1%}")
        print(f"      Max:  {stats['max']:.1%}")
    
    # Identify high-demand tracts (top quartile)
    q75 = tdi_df['TDI'].quantile(0.75)
    high_demand = tdi_df[tdi_df['TDI'] >= q75]
    print(f"\n  High-demand tracts (TDI >= {q75:.2f}): {len(high_demand)}")
    
    print("\n" + "=" * 60)
    print("Demand metrics complete!")
    print("Next step: python src/04_identify_deserts.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
