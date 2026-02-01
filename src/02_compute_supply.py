"""
02_compute_supply.py
Compute Composite Public Transit Accessibility (CPTA) score for each Census tract.

Metrics computed:
1. Stop Connectivity (Welch method)
2. Route Coverage
3. Frequency - AM Peak
4. Frequency - Midday
5. Frequency - PM Peak
6. Walking Access
7. Span of Service
8. Weekend Ratio
9. Jobs Accessible (optional, requires r5py)

Usage:
    python src/02_compute_supply.py

Outputs:
    - data/processed/supply_metrics.csv
    - data/processed/supply_metrics.gpkg
"""

import os
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.stats import zscore
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import our GTFS utility module
from gtfs_utils import load_gtfs, print_gtfs_summary

def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    
    # Reproject to projected CRS for accurate area/distance calculations
    crs_projected = config['spatial']['crs_projected']
    gdf = gdf.to_crs(crs_projected)
    
    # Calculate tract area in square kilometers
    # Detect CRS units and convert appropriately
    crs_obj = gdf.crs
    if crs_obj.axis_info[0].unit_name == 'metre':
        # Meters - divide by 1e6 to get sq km
        gdf['area_sqkm'] = gdf.geometry.area / 1e6
    elif crs_obj.axis_info[0].unit_name in ['foot', 'US survey foot', 'ft']:
        # Feet - convert to meters then to sq km
        gdf['area_sqkm'] = gdf.geometry.area / (3.28084**2) / 1e6
    else:
        # Assume meters as fallback
        print(f"  ⚠ Unknown CRS units: {crs_obj.axis_info[0].unit_name}, assuming meters")
        gdf['area_sqkm'] = gdf.geometry.area / 1e6
    
    print(f"  ✓ Loaded {len(gdf)} tracts")
    print(f"  ✓ CRS: {gdf.crs}")
    
    return gdf

def get_stops_gdf(feed, config):
    """
    Convert GTFS stops to GeoDataFrame.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    config : dict
        Configuration dictionary
    
    Returns
    -------
    GeoDataFrame
        Stops with geometry
    """
    stops = feed.stops.copy()
    
    # Create geometry
    geometry = [Point(lon, lat) for lon, lat in zip(stops['stop_lon'], stops['stop_lat'])]
    gdf = gpd.GeoDataFrame(stops, geometry=geometry, crs='EPSG:4326')
    
    # Reproject
    crs_projected = config['spatial']['crs_projected']
    gdf = gdf.to_crs(crs_projected)
    
    return gdf

def compute_stop_routes_frequency(feed, config):
    """
    Compute routes and frequency at each stop.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Stop-level metrics: routes_at_stop, frequency by time window
    """
    print("Computing stop-level metrics...")
    
    # Get valid service dates from the feed
    try:
        valid_dates = feed.get_dates()
        if valid_dates is None or len(valid_dates) == 0:
            valid_dates = []
    except:
        valid_dates = []
    
    # Try to use config date, but validate it exists in feed
    analysis_date = None
    config_date = config['gtfs'].get('analysis_date', '')
    
    if config_date:
        # Normalize date format (remove dashes if present)
        config_date_normalized = config_date.replace('-', '')
        
        if config_date_normalized in valid_dates:
            analysis_date = config_date_normalized
            print(f"  Using configured analysis date: {analysis_date}")
        else:
            print(f"  ⚠ Configured date {config_date} not in GTFS calendar")
    
    # If no valid config date, pick a date from the feed
    if analysis_date is None:
        if valid_dates and len(valid_dates) > 0:
            # Pick a date in the middle of the valid range (likely a typical service day)
            mid_idx = len(valid_dates) // 2
            analysis_date = valid_dates[mid_idx]
            print(f"  Auto-selected analysis date: {analysis_date}")
            print(f"    (from {len(valid_dates)} valid dates: {valid_dates[0]} to {valid_dates[-1]})")
        else:
            print("  ⚠ No valid dates found in GTFS calendar, using all stop_times")
            analysis_date = None
    
    # Get stop times for analysis date
    if analysis_date:
        try:
            stop_times = feed.get_stop_times(analysis_date)
            if stop_times is None or len(stop_times) == 0:
                print(f"  ⚠ No stop times for date {analysis_date}, using all stop_times")
                stop_times = feed.stop_times.copy()
        except Exception as e:
            print(f"  ⚠ Error getting stop times for {analysis_date}: {e}")
            stop_times = feed.stop_times.copy()
    else:
        stop_times = feed.stop_times.copy()
    
    print(f"  Stop times loaded: {len(stop_times):,} records")
    
    if len(stop_times) == 0:
        print("  ✗ ERROR: No stop times found! Check GTFS data.")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'stop_id', 'routes_at_stop', 'total_departures', 'destinations_reachable',
            'freq_am_peak', 'freq_midday', 'freq_pm_peak', 'freq_evening'
        ])
    
    # Merge with trips to get route_id
    trips = feed.trips[['trip_id', 'route_id', 'service_id']].copy()
    stop_times = stop_times.merge(trips, on='trip_id', how='left')
    
    # Convert arrival_time to seconds for time window filtering
    def time_to_seconds(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            parts = str(time_str).split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        except:
            return np.nan
    
    stop_times['arrival_seconds'] = stop_times['arrival_time'].apply(time_to_seconds)
    
    # Time windows from config
    time_windows = config['gtfs']['time_windows']
    
    def time_str_to_seconds(t):
        h, m, s = map(int, t.split(':'))
        return h * 3600 + m * 60 + s
    
    # Initialize results
    stop_metrics = []
    
    for stop_id in tqdm(stop_times['stop_id'].unique(), desc="Processing stops"):
        stop_data = stop_times[stop_times['stop_id'] == stop_id]
        
        # Routes at stop
        routes_at_stop = stop_data['route_id'].nunique()
        
        # Frequency by time window (departures per hour)
        freq_metrics = {}
        for window_name, window_times in time_windows.items():
            start_sec = time_str_to_seconds(window_times['start'])
            end_sec = time_str_to_seconds(window_times['end'])
            
            window_trips = stop_data[
                (stop_data['arrival_seconds'] >= start_sec) & 
                (stop_data['arrival_seconds'] < end_sec)
            ]
            
            # Departures per hour
            window_hours = (end_sec - start_sec) / 3600
            departures = len(window_trips)
            freq = departures / window_hours if window_hours > 0 else 0
            freq_metrics[f'freq_{window_name}'] = freq
        
        # Total daily departures
        total_departures = len(stop_data)
        
        # Unique destinations (other stops reachable via routes at this stop)
        route_ids = stop_data['route_id'].unique()
        destinations = stop_times[stop_times['route_id'].isin(route_ids)]['stop_id'].nunique()
        
        stop_metrics.append({
            'stop_id': stop_id,
            'routes_at_stop': routes_at_stop,
            'total_departures': total_departures,
            'destinations_reachable': destinations,
            **freq_metrics
        })
    
    return pd.DataFrame(stop_metrics)

def compute_stop_connectivity_welch(stop_metrics):
    """
    Compute Welch-style stop connectivity score.
    
    Connectivity = routes × frequency × destinations
    
    Parameters
    ----------
    stop_metrics : DataFrame
        Stop-level metrics from compute_stop_routes_frequency
    
    Returns
    -------
    DataFrame
        Stop connectivity scores
    """
    print("Computing Welch connectivity scores...")
    
    # Use AM peak frequency as representative
    # Connectivity = routes × frequency × destinations
    stop_metrics['connectivity'] = (
        stop_metrics['routes_at_stop'] * 
        stop_metrics['freq_am_peak'] * 
        stop_metrics['destinations_reachable']
    )
    
    # Normalize to 0-100 scale
    max_conn = stop_metrics['connectivity'].max()
    if max_conn > 0:
        stop_metrics['connectivity_normalized'] = (stop_metrics['connectivity'] / max_conn) * 100
    else:
        stop_metrics['connectivity_normalized'] = 0
    
    return stop_metrics

def aggregate_stops_to_tracts(stops_gdf, stop_metrics, tracts_gdf, config):
    """
    Aggregate stop-level metrics to tract level.
    
    Parameters
    ----------
    stops_gdf : GeoDataFrame
        Stop locations
    stop_metrics : DataFrame
        Stop-level metrics
    tracts_gdf : GeoDataFrame
        Census tract boundaries
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Tract-level supply metrics
    """
    print("Aggregating stop metrics to tracts...")
    
    # Merge stop metrics with geometry
    stops_gdf = stops_gdf.merge(stop_metrics, on='stop_id')
    
    # Spatial join: stops to tracts
    stops_in_tracts = gpd.sjoin(stops_gdf, tracts_gdf[['GEOID', 'geometry']], how='left', predicate='within')
    
    # Aggregate by tract
    tract_metrics = stops_in_tracts.groupby('GEOID').agg({
        'stop_id': 'count',  # Stop count
        'routes_at_stop': 'sum',  # Total route-stop pairs
        'connectivity': 'sum',  # Sum of connectivity scores
        'freq_am_peak': 'mean',
        'freq_midday': 'mean',
        'freq_pm_peak': 'mean',
        'freq_evening': 'mean',
        'total_departures': 'sum'
    }).reset_index()
    
    tract_metrics.columns = [
        'GEOID', 'stop_count', 'route_stop_pairs', 'connectivity_sum',
        'freq_am_peak', 'freq_midday', 'freq_pm_peak', 'freq_evening',
        'total_departures'
    ]
    
    # Merge with tract areas
    tract_metrics = tract_metrics.merge(
        tracts_gdf[['GEOID', 'area_sqkm']], 
        on='GEOID'
    )
    
    # Calculate density metrics
    tract_metrics['stop_density'] = tract_metrics['stop_count'] / tract_metrics['area_sqkm']
    tract_metrics['connectivity_density'] = tract_metrics['connectivity_sum'] / tract_metrics['area_sqkm']
    
    return tract_metrics

def compute_route_coverage(feed, stops_gdf, tracts_gdf):
    """
    Count unique routes serving each tract.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    stops_gdf : GeoDataFrame
        Stop locations
    tracts_gdf : GeoDataFrame
        Census tracts
    
    Returns
    -------
    DataFrame
        Route coverage by tract
    """
    print("Computing route coverage...")
    
    # Get route-stop relationships
    stop_times = feed.stop_times[['trip_id', 'stop_id']].drop_duplicates()
    trips = feed.trips[['trip_id', 'route_id']]
    route_stops = stop_times.merge(trips, on='trip_id')[['route_id', 'stop_id']].drop_duplicates()
    
    # Add route info to stops
    stops_with_routes = stops_gdf.merge(route_stops, on='stop_id')
    
    # Spatial join to tracts
    stops_in_tracts = gpd.sjoin(
        stops_with_routes, 
        tracts_gdf[['GEOID', 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    # Count unique routes per tract
    route_coverage = stops_in_tracts.groupby('GEOID')['route_id'].nunique().reset_index()
    route_coverage.columns = ['GEOID', 'route_coverage']
    
    return route_coverage

def compute_walking_access(stops_gdf, tracts_gdf, config):
    """
    Compute percentage of tract area within walking distance of transit.
    
    Parameters
    ----------
    stops_gdf : GeoDataFrame
        Stop locations
    tracts_gdf : GeoDataFrame
        Census tracts
    config : dict
        Configuration dictionary with buffer distances
    
    Returns
    -------
    DataFrame
        Walking access percentage by tract
    """
    print("Computing walking access...")
    
    buffer_bus = config['spatial']['buffer_bus']  # meters
    buffer_rail = config['spatial']['buffer_rail']  # meters
    
    # Convert buffer to CRS units (EPSG:2248 is in feet)
    buffer_bus_ft = buffer_bus * 3.28084
    buffer_rail_ft = buffer_rail * 3.28084
    
    # For now, use bus buffer for all stops
    # TODO: Differentiate bus vs rail stops based on route_type
    stops_gdf['buffer'] = stops_gdf.geometry.buffer(buffer_bus_ft)
    
    # Union all buffers
    all_buffers = unary_union(stops_gdf['buffer'].values)
    
    # Calculate intersection area for each tract
    walking_access = []
    
    for idx, tract in tqdm(tracts_gdf.iterrows(), total=len(tracts_gdf), desc="Computing walking access"):
        tract_geom = tract.geometry
        intersection = tract_geom.intersection(all_buffers)
        
        if tract_geom.area > 0:
            pct_covered = (intersection.area / tract_geom.area) * 100
        else:
            pct_covered = 0
        
        walking_access.append({
            'GEOID': tract['GEOID'],
            'walking_access_pct': min(pct_covered, 100)  # Cap at 100%
        })
    
    return pd.DataFrame(walking_access)

def compute_span_of_service(feed, stop_metrics):
    """
    Compute span of service (hours per day with service).
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    stop_metrics : DataFrame
        Stop-level metrics
    
    Returns
    -------
    DataFrame
        Span of service by stop
    """
    print("Computing span of service...")
    
    stop_times = feed.stop_times.copy()
    
    # Convert times to hours
    def time_to_hours(time_str):
        if pd.isna(time_str):
            return np.nan
        parts = str(time_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60
    
    stop_times['arrival_hour'] = stop_times['arrival_time'].apply(time_to_hours)
    
    # Calculate span for each stop
    span = stop_times.groupby('stop_id').agg({
        'arrival_hour': ['min', 'max']
    }).reset_index()
    span.columns = ['stop_id', 'first_departure', 'last_departure']
    span['span_hours'] = span['last_departure'] - span['first_departure']
    
    return span[['stop_id', 'span_hours']]

def compute_weekend_ratio(feed):
    """
    Compute ratio of weekend to weekday service.
    
    Parameters
    ----------
    feed : gtfs_kit.Feed
        GTFS feed
    
    Returns
    -------
    DataFrame
        Weekend ratio by stop
    """
    print("Computing weekend ratio...")
    
    # Get service calendar
    calendar = feed.calendar.copy() if feed.calendar is not None else None
    
    if calendar is None or len(calendar) == 0:
        print("  ⚠ No calendar.txt found, using calendar_dates")
        # Simplified: return 1.0 for all (assume same service)
        return pd.DataFrame({
            'stop_id': feed.stops['stop_id'],
            'weekend_ratio': 1.0
        })
    
    # Identify weekday and weekend service IDs
    weekday_services = calendar[calendar['monday'] == 1]['service_id'].tolist()
    weekend_services = calendar[
        (calendar['saturday'] == 1) | (calendar['sunday'] == 1)
    ]['service_id'].tolist()
    
    # Count trips by service type
    trips = feed.trips.copy()
    
    weekday_trips = trips[trips['service_id'].isin(weekday_services)]
    weekend_trips = trips[trips['service_id'].isin(weekend_services)]
    
    # Get stops served by each
    stop_times = feed.stop_times[['trip_id', 'stop_id']].drop_duplicates()
    
    weekday_stops = stop_times[stop_times['trip_id'].isin(weekday_trips['trip_id'])]
    weekday_count = weekday_stops.groupby('stop_id').size().reset_index(name='weekday_trips')
    
    weekend_stops = stop_times[stop_times['trip_id'].isin(weekend_trips['trip_id'])]
    weekend_count = weekend_stops.groupby('stop_id').size().reset_index(name='weekend_trips')
    
    # Merge and compute ratio
    ratio = weekday_count.merge(weekend_count, on='stop_id', how='outer').fillna(0)
    ratio['weekend_ratio'] = np.where(
        ratio['weekday_trips'] > 0,
        ratio['weekend_trips'] / ratio['weekday_trips'],
        0
    )
    
    return ratio[['stop_id', 'weekend_ratio']]

def compute_cpta_score(tract_metrics):
    """
    Compute final CPTA score as mean of z-scores.
    
    Parameters
    ----------
    tract_metrics : DataFrame
        Tract-level supply metrics
    
    Returns
    -------
    DataFrame
        Tract metrics with CPTA score
    """
    print("Computing CPTA scores...")
    
    # Columns to include in CPTA
    supply_columns = [
        'connectivity_density',  # Welch-style connectivity
        'route_coverage',
        'freq_am_peak',
        'freq_midday',
        'freq_pm_peak',
        'walking_access_pct',
        'span_hours',
        'weekend_ratio'
    ]
    
    # Check which columns exist
    available_cols = [col for col in supply_columns if col in tract_metrics.columns]
    
    print(f"  Using {len(available_cols)} metrics for CPTA:")
    for col in available_cols:
        print(f"    - {col}")
    
    # Compute z-scores for each metric
    for col in available_cols:
        z_col = f'z_{col}'
        values = tract_metrics[col].fillna(0)
        if values.std() > 0:
            tract_metrics[z_col] = zscore(values)
        else:
            tract_metrics[z_col] = 0
    
    # CPTA = mean of z-scores
    z_cols = [f'z_{col}' for col in available_cols]
    tract_metrics['CPTA'] = tract_metrics[z_cols].mean(axis=1)
    
    # Also create normalized 0-100 score
    cpta_min = tract_metrics['CPTA'].min()
    cpta_max = tract_metrics['CPTA'].max()
    if cpta_max > cpta_min:
        tract_metrics['CPTA_normalized'] = (
            (tract_metrics['CPTA'] - cpta_min) / (cpta_max - cpta_min) * 100
        )
    else:
        tract_metrics['CPTA_normalized'] = 50
    
    return tract_metrics

def main():
    """Main function to compute all supply metrics."""
    print("=" * 60)
    print("Transit Desert Pipeline: Supply Metrics (CPTA)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Paths
    gtfs_path = PROJECT_ROOT / "data" / "raw" / "gtfs"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    print("-" * 40)
    
    feed = load_gtfs(gtfs_path)
    print_gtfs_summary(feed)
    
    tracts_gdf = load_tracts(config)
    stops_gdf = get_stops_gdf(feed, config)
    
    print(f"\n  Stops in study area: {len(stops_gdf)}")
    print(f"  Tracts: {len(tracts_gdf)}")
    
    # Compute stop-level metrics
    print("\n" + "-" * 40)
    print("Computing stop-level metrics...")
    print("-" * 40)
    
    stop_metrics = compute_stop_routes_frequency(feed, config)
    stop_metrics = compute_stop_connectivity_welch(stop_metrics)
    
    # Add span of service
    span = compute_span_of_service(feed, stop_metrics)
    stop_metrics = stop_metrics.merge(span, on='stop_id', how='left')
    
    # Add weekend ratio
    weekend = compute_weekend_ratio(feed)
    stop_metrics = stop_metrics.merge(weekend, on='stop_id', how='left')
    
    # Save stop-level metrics
    stop_metrics.to_csv(output_dir / "stop_metrics.csv", index=False)
    print(f"\n  ✓ Saved stop metrics to {output_dir / 'stop_metrics.csv'}")
    
    # Aggregate to tracts
    print("\n" + "-" * 40)
    print("Aggregating to tract level...")
    print("-" * 40)
    
    tract_metrics = aggregate_stops_to_tracts(stops_gdf, stop_metrics, tracts_gdf, config)
    
    # Add route coverage
    route_coverage = compute_route_coverage(feed, stops_gdf, tracts_gdf)
    tract_metrics = tract_metrics.merge(route_coverage, on='GEOID', how='left')
    
    # Add walking access
    walking_access = compute_walking_access(stops_gdf, tracts_gdf, config)
    tract_metrics = tract_metrics.merge(walking_access, on='GEOID', how='left')
    
    # Aggregate span and weekend ratio to tract level
    stops_gdf_metrics = stops_gdf.merge(stop_metrics[['stop_id', 'span_hours', 'weekend_ratio']], on='stop_id')
    stops_in_tracts = gpd.sjoin(stops_gdf_metrics, tracts_gdf[['GEOID', 'geometry']], how='left', predicate='within')
    
    tract_span = stops_in_tracts.groupby('GEOID').agg({
        'span_hours': 'mean',
        'weekend_ratio': 'mean'
    }).reset_index()
    
    tract_metrics = tract_metrics.merge(tract_span, on='GEOID', how='left')
    
    # Fill NaN for tracts with no stops
    tract_metrics = tract_metrics.fillna(0)
    
    # Ensure all tracts are included
    all_tracts = tracts_gdf[['GEOID']].copy()
    tract_metrics = all_tracts.merge(tract_metrics, on='GEOID', how='left').fillna(0)
    
    # Compute CPTA score
    print("\n" + "-" * 40)
    print("Computing CPTA score...")
    print("-" * 40)
    
    tract_metrics = compute_cpta_score(tract_metrics)
    
    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    # CSV
    tract_metrics.to_csv(output_dir / "supply_metrics.csv", index=False)
    print(f"  ✓ Saved to {output_dir / 'supply_metrics.csv'}")
    
    # GeoPackage (with geometry)
    tract_metrics_gdf = tracts_gdf.merge(tract_metrics, on='GEOID')
    tract_metrics_gdf.to_file(output_dir / "supply_metrics.gpkg", driver='GPKG')
    print(f"  ✓ Saved to {output_dir / 'supply_metrics.gpkg'}")
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary Statistics")
    print("-" * 40)
    
    print(f"\n  Tracts analyzed: {len(tract_metrics)}")
    print(f"\n  CPTA Score:")
    print(f"    Mean: {tract_metrics['CPTA'].mean():.2f}")
    print(f"    Std:  {tract_metrics['CPTA'].std():.2f}")
    print(f"    Min:  {tract_metrics['CPTA'].min():.2f}")
    print(f"    Max:  {tract_metrics['CPTA'].max():.2f}")
    
    print(f"\n  CPTA Normalized (0-100):")
    print(f"    Mean: {tract_metrics['CPTA_normalized'].mean():.1f}")
    print(f"    Min:  {tract_metrics['CPTA_normalized'].min():.1f}")
    print(f"    Max:  {tract_metrics['CPTA_normalized'].max():.1f}")
    
    print("\n" + "=" * 60)
    print("Supply metrics complete!")
    print("Next step: python src/03_compute_demand.py")
    print("=" * 60)

if __name__ == "__main__":
    main()