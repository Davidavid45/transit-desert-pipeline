"""
02b_compute_jobs_accessibility.py
Compute job accessibility metric using r5py travel time matrix.

This script calculates the number of jobs reachable within a specified
travel time threshold (default: 45 minutes) by transit from each Census tract.

Prerequisites:
- r5py installed: pip install r5py
- Java 11+ installed
- GTFS data downloaded (run 01_download_data.py first)
- OpenStreetMap PBF file for the region

Usage:
    python src/02b_compute_jobs_accessibility.py

Outputs:
    - data/processed/jobs_accessibility.csv
    - Updates supply_metrics.csv with jobs_accessible column
"""

import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, date, time, timedelta
import requests
import zipfile
from io import BytesIO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for r5py
try:
    import r5py
    R5PY_AVAILABLE = True
except ImportError:
    R5PY_AVAILABLE = False
    print("⚠ r5py not installed. Run: pip install r5py")
    print("  Also requires Java 11+")


def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_osm_pbf(config, output_dir):
    """
    Download OpenStreetMap PBF file for Maryland.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Directory to save PBF file
    
    Returns
    -------
    Path
        Path to PBF file
    """
    osm_dir = output_dir / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    
    pbf_path = osm_dir / "maryland-latest.osm.pbf"
    
    if pbf_path.exists():
        print(f"  ✓ OSM PBF already exists: {pbf_path}")
        return pbf_path
    
    # Download from Geofabrik
    url = "https://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf"
    print(f"  Downloading OSM data from: {url}")
    print("  (This may take a few minutes...)")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(pbf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {pct:.1f}%", end='', flush=True)
        
        print(f"\n  ✓ Downloaded OSM PBF to {pbf_path}")
        return pbf_path
        
    except Exception as e:
        print(f"  ✗ Error downloading OSM: {e}")
        print("  → Please download manually from:")
        print(f"    {url}")
        print(f"    Save to: {pbf_path}")
        return None


def download_lodes_data(config, output_dir):
    """
    Download LEHD LODES Workplace Area Characteristics (WAC) data.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Directory to save data
    
    Returns
    -------
    DataFrame
        Jobs by Census block
    """
    lodes_dir = output_dir / "lodes"
    lodes_dir.mkdir(parents=True, exist_ok=True)
    
    state_fips = config['study_area']['state_fips']
    lodes_year = config['accessibility']['lodes_year']
    
    # State abbreviation from FIPS
    state_abbrev = 'md'  # Maryland
    
    # WAC file URL
    # Format: state_wac_S000_JT00_YYYY.csv.gz
    filename = f"{state_abbrev}_wac_S000_JT00_{lodes_year}.csv.gz"
    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state_abbrev}/wac/{filename}"
    
    csv_path = lodes_dir / filename.replace('.gz', '')
    
    if csv_path.exists():
        print(f"  ✓ LODES data already exists: {csv_path}")
        return pd.read_csv(csv_path)
    
    print(f"  Downloading LODES WAC data from: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Decompress and save
        import gzip
        with gzip.open(BytesIO(response.content), 'rt') as gz:
            df = pd.read_csv(gz)
        
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Downloaded LODES data: {len(df)} blocks")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error downloading LODES: {e}")
        return None


def aggregate_jobs_to_tracts(lodes_df, config):
    """
    Aggregate block-level jobs to tract level.
    
    Parameters
    ----------
    lodes_df : DataFrame
        LODES WAC data with w_geocode (block FIPS)
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Jobs by tract
    """
    # Extract tract GEOID from block GEOID (first 11 digits)
    lodes_df['tract_geoid'] = lodes_df['w_geocode'].astype(str).str[:11]
    
    # Filter to study area
    county_fips = config['study_area']['full_fips']  # e.g., "24510"
    lodes_df = lodes_df[lodes_df['tract_geoid'].str.startswith(county_fips)]
    
    # Aggregate total jobs (C000) to tract
    tract_jobs = lodes_df.groupby('tract_geoid').agg({
        'C000': 'sum'  # Total jobs
    }).reset_index()
    
    tract_jobs.columns = ['GEOID', 'total_jobs']
    
    print(f"  ✓ Aggregated jobs for {len(tract_jobs)} tracts")
    print(f"  ✓ Total jobs in study area: {tract_jobs['total_jobs'].sum():,}")
    
    return tract_jobs


def load_tract_centroids(config):
    """
    Load tract centroids for travel time calculation.
    
    Returns
    -------
    GeoDataFrame
        Tract centroids
    """
    tracts_path = PROJECT_ROOT / "data" / "raw" / "census" / "study_area_tracts.shp"
    
    if not tracts_path.exists():
        raise FileNotFoundError(f"Tract shapefile not found: {tracts_path}")
    
    gdf = gpd.read_file(tracts_path)
    
    # Calculate centroids
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    centroids = gdf_wgs84.copy()
    centroids['geometry'] = gdf_wgs84.geometry.centroid
    
    # Add lat/lon columns
    centroids['lat'] = centroids.geometry.y
    centroids['lon'] = centroids.geometry.x
    
    print(f"  ✓ Loaded {len(centroids)} tract centroids")
    
    return centroids[['GEOID', 'geometry', 'lat', 'lon']]


def compute_travel_time_matrix(origins, destinations, gtfs_path, osm_path, config):
    """
    Compute travel time matrix using r5py.
    
    Parameters
    ----------
    origins : GeoDataFrame
        Origin points (tract centroids)
    destinations : GeoDataFrame
        Destination points (tract centroids with jobs)
    gtfs_path : Path
        Path to GTFS zip file
    osm_path : Path
        Path to OSM PBF file
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Travel time matrix
    """
    if not R5PY_AVAILABLE:
        raise ImportError("r5py is required for travel time calculation")
    
    print("  Building transport network (this may take several minutes)...")
    
    # Create transport network
    transport_network = r5py.TransportNetwork(
        osm_pbf=str(osm_path),
        gtfs=[str(gtfs_path)]
    )
    
    # Set departure time (AM peak)
    departure_time = datetime(2024, 10, 15, 8, 0)  # Tuesday 8:00 AM
    
    # Travel time threshold from config
    max_time = timedelta(minutes=config['accessibility']['travel_time_threshold'])
    
    print(f"  Computing travel times (max {config['accessibility']['travel_time_threshold']} min)...")
    
    # Compute travel time matrix
    travel_time_matrix = r5py.TravelTimeMatrixComputer(
        transport_network,
        origins=origins,
        destinations=destinations,
        departure=departure_time,
        departure_time_window=timedelta(hours=1),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        max_time=max_time
    ).compute_travel_times()
    
    print(f"  ✓ Computed {len(travel_time_matrix)} origin-destination pairs")
    
    return travel_time_matrix


def calculate_jobs_accessible(travel_times, tract_jobs, config):
    """
    Calculate jobs accessible from each tract within time threshold.
    
    Parameters
    ----------
    travel_times : DataFrame
        Travel time matrix with from_id, to_id, travel_time
    tract_jobs : DataFrame
        Jobs per tract
    config : dict
        Configuration dictionary
    
    Returns
    -------
    DataFrame
        Jobs accessible from each tract
    """
    threshold = config['accessibility']['travel_time_threshold']
    
    # Filter to trips within threshold
    reachable = travel_times[travel_times['travel_time'] <= threshold].copy()
    
    # Merge with jobs
    reachable = reachable.merge(
        tract_jobs,
        left_on='to_id',
        right_on='GEOID',
        how='left'
    ).fillna(0)
    
    # Sum jobs reachable from each origin
    jobs_accessible = reachable.groupby('from_id').agg({
        'total_jobs': 'sum'
    }).reset_index()
    
    jobs_accessible.columns = ['GEOID', 'jobs_accessible']
    
    print(f"\n  Jobs Accessibility Statistics:")
    print(f"    Mean jobs accessible: {jobs_accessible['jobs_accessible'].mean():,.0f}")
    print(f"    Median: {jobs_accessible['jobs_accessible'].median():,.0f}")
    print(f"    Max: {jobs_accessible['jobs_accessible'].max():,.0f}")
    
    return jobs_accessible


def update_supply_metrics(jobs_accessible):
    """
    Update supply_metrics.csv with jobs accessibility.
    
    Parameters
    ----------
    jobs_accessible : DataFrame
        Jobs accessible from each tract
    """
    supply_path = PROJECT_ROOT / "data" / "processed" / "supply_metrics.csv"
    
    if not supply_path.exists():
        print("  ⚠ supply_metrics.csv not found. Run 02_compute_supply.py first.")
        return
    
    supply_df = pd.read_csv(supply_path, dtype={'GEOID': str})
    
    # Merge jobs accessibility
    supply_df = supply_df.merge(
        jobs_accessible,
        on='GEOID',
        how='left'
    ).fillna(0)
    
    # Recalculate CPTA with jobs accessibility
    # ... (This would require updating the z-score calculation)
    
    supply_df.to_csv(supply_path, index=False)
    print(f"  ✓ Updated supply_metrics.csv with jobs_accessible column")


def main():
    """Main function to compute job accessibility."""
    print("=" * 60)
    print("Transit Desert Pipeline: Job Accessibility (r5py)")
    print("=" * 60)
    
    if not R5PY_AVAILABLE:
        print("\n⚠ r5py is not installed.")
        print("  To install: pip install r5py")
        print("  Also requires Java 11+")
        print("\n  Alternatively, job accessibility can be computed using:")
        print("  - OpenTripPlanner")
        print("  - Conveyal Analysis")
        sys.exit(1)
    
    config = load_config()
    
    # Paths
    raw_dir = PROJECT_ROOT / "data" / "raw"
    external_dir = PROJECT_ROOT / "data" / "external"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    external_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download OSM data
    print("\n" + "-" * 40)
    print("Step 1: OpenStreetMap Data")
    print("-" * 40)
    
    osm_path = download_osm_pbf(config, external_dir)
    if osm_path is None:
        print("  ✗ Cannot proceed without OSM data")
        sys.exit(1)
    
    # Step 2: Download LODES data
    print("\n" + "-" * 40)
    print("Step 2: LEHD LODES Job Data")
    print("-" * 40)
    
    lodes_df = download_lodes_data(config, external_dir)
    if lodes_df is None:
        print("  ✗ Cannot proceed without LODES data")
        sys.exit(1)
    
    tract_jobs = aggregate_jobs_to_tracts(lodes_df, config)
    
    # Step 3: Load tract centroids
    print("\n" + "-" * 40)
    print("Step 3: Load Tract Centroids")
    print("-" * 40)
    
    centroids = load_tract_centroids(config)
    
    # Add jobs to centroids for destination weighting
    centroids_with_jobs = centroids.merge(tract_jobs, on='GEOID', how='left').fillna(0)
    
    # Step 4: Compute travel time matrix
    print("\n" + "-" * 40)
    print("Step 4: Compute Travel Time Matrix")
    print("-" * 40)
    
    gtfs_path = raw_dir / "gtfs" / "gtfs.zip"
    if not gtfs_path.exists():
        print(f"  ✗ GTFS not found: {gtfs_path}")
        print("    Run 01_download_data.py first")
        sys.exit(1)
    
    try:
        travel_times = compute_travel_time_matrix(
            origins=centroids,
            destinations=centroids_with_jobs,
            gtfs_path=gtfs_path,
            osm_path=osm_path,
            config=config
        )
    except Exception as e:
        print(f"  ✗ Error computing travel times: {e}")
        print("\n  Common issues:")
        print("  - Java not installed or wrong version (needs Java 11+)")
        print("  - Insufficient memory (try increasing Java heap)")
        print("  - GTFS or OSM data issues")
        sys.exit(1)
    
    # Step 5: Calculate jobs accessible
    print("\n" + "-" * 40)
    print("Step 5: Calculate Jobs Accessible")
    print("-" * 40)
    
    jobs_accessible = calculate_jobs_accessible(travel_times, tract_jobs, config)
    
    # Save results
    jobs_accessible.to_csv(processed_dir / "jobs_accessibility.csv", index=False)
    print(f"  ✓ Saved to {processed_dir / 'jobs_accessibility.csv'}")
    
    # Update supply metrics
    print("\n" + "-" * 40)
    print("Step 6: Update Supply Metrics")
    print("-" * 40)
    
    update_supply_metrics(jobs_accessible)
    
    print("\n" + "=" * 60)
    print("Job accessibility calculation complete!")
    print("\nNote: Re-run 02_compute_supply.py to recalculate CPTA with jobs metric")
    print("      Then re-run 04_identify_deserts.py to update classifications")
    print("=" * 60)


if __name__ == "__main__":
    main()