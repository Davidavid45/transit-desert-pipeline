"""
01_download_data.py
Download GTFS, Census ACS, and Census tract shapefiles for transit desert analysis.

Usage:
    python src/01_download_data.py

Outputs:
    - data/raw/gtfs/          : GTFS feed files
    - data/raw/census/        : Census tract shapefile
    - data/raw/acs/           : ACS demographic data
"""

import os
import sys
import yaml
import zipfile
import requests
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import geopandas as gpd
from census import Census
from us import states

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_config():
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_gtfs(config, output_dir):
    """
    Download GTFS feed from URL.
    
    Handles various GTFS packaging formats:
    - Single zip with txt files
    - Zip containing multiple mode-specific zips
    - Multiple separate URLs for different modes
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with GTFS URL(s)
    output_dir : Path
        Directory to save GTFS files
    """
    gtfs_dir = output_dir / "gtfs"
    gtfs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if URL is a single string or list of URLs
    gtfs_config = config['gtfs']
    
    if isinstance(gtfs_config.get('url'), list):
        # Multiple URLs (multi-modal)
        urls = gtfs_config['url']
        print(f"Downloading {len(urls)} GTFS feeds (multi-modal)...")
    else:
        # Single URL
        urls = [gtfs_config['url']]
        print(f"Downloading GTFS from: {urls[0]}")
    
    downloaded_files = []
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Determine filename
            if len(urls) == 1:
                filename = "gtfs.zip"
            else:
                # Try to get name from URL or use index
                url_parts = url.rstrip('/').split('/')
                filename = url_parts[-1] if url_parts[-1].endswith('.zip') else f"gtfs_{i+1}.zip"
            
            zip_path = gtfs_dir / filename
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_files.append(zip_path)
            print(f"  ✓ Downloaded: {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error downloading {url}: {e}")
    
    if not downloaded_files:
        print("  ✗ No GTFS files downloaded")
        print("  → Please download manually and place in data/raw/gtfs/")
        return None
    
    # Analyze what was downloaded
    print(f"\n  Analyzing downloaded GTFS data...")
    
    for zip_path in downloaded_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                names = zf.namelist()
                txt_files = [n for n in names if n.endswith('.txt') and '/' not in n]
                nested_zips = [n for n in names if n.endswith('.zip')]
                
                if txt_files:
                    print(f"    {zip_path.name}: Standard GTFS ({len(txt_files)} txt files)")
                    # Extract txt files
                    zf.extractall(gtfs_dir)
                elif nested_zips:
                    print(f"    {zip_path.name}: Multi-modal package ({len(nested_zips)} zip files)")
                    # Extract nested zips
                    zf.extractall(gtfs_dir)
                else:
                    # Check for nested folder structure
                    nested_txt = [n for n in names if n.endswith('.txt')]
                    if nested_txt:
                        print(f"    {zip_path.name}: Nested folder structure")
                        zf.extractall(gtfs_dir)
                    else:
                        print(f"    {zip_path.name}: Unknown format")
        except zipfile.BadZipFile:
            print(f"    ⚠ {zip_path.name}: Not a valid zip file")
    
    # List what's in the directory now
    all_files = list(gtfs_dir.glob("*"))
    txt_files = [f for f in all_files if f.suffix == '.txt']
    zip_files = [f for f in all_files if f.suffix == '.zip']
    
    print(f"\n  GTFS directory contents:")
    print(f"    txt files: {len(txt_files)}")
    print(f"    zip files: {len(zip_files)}")
    
    if txt_files:
        for f in txt_files[:5]:
            print(f"      - {f.name}")
        if len(txt_files) > 5:
            print(f"      ... and {len(txt_files) - 5} more")
    
    if zip_files:
        for f in zip_files[:5]:
            print(f"      - {f.name}")
        if len(zip_files) > 5:
            print(f"      ... and {len(zip_files) - 5} more")
    
    return gtfs_dir

def download_census_tracts(config, output_dir):
    """
    Download Census tract boundaries from TIGER/Line.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with FIPS codes
    output_dir : Path
        Directory to save shapefile
    """
    census_dir = output_dir / "census"
    census_dir.mkdir(parents=True, exist_ok=True)
    
    state_fips = config['study_area']['state_fips']
    county_fips = config['study_area']['county_fips']
    
    # TIGER/Line URL for tracts
    year = config['census']['acs_year']
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    
    print(f"Downloading Census tracts from: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Extract shapefile
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(census_dir / "tracts_state")
        
        # Read and filter to county
        shp_files = list((census_dir / "tracts_state").glob("*.shp"))
        if shp_files:
            gdf = gpd.read_file(shp_files[0])
            
            # Filter to study area county
            gdf_county = gdf[gdf['COUNTYFP'] == county_fips].copy()
            
            # Save filtered shapefile
            output_path = census_dir / "study_area_tracts.shp"
            gdf_county.to_file(output_path)
            
            print(f"  ✓ Downloaded {len(gdf_county)} tracts for county {county_fips}")
            print(f"  ✓ Saved to {output_path}")
            
            return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading tracts: {e}")
        return None

def download_acs_data(config, output_dir):
    """
    Download ACS 5-year estimates for TDI calculation.
    
    Variables needed:
    - B08201: Household Size by Vehicles Available
    - B17001: Poverty Status
    - B03002: Hispanic or Latino Origin by Race
    - B01001: Sex by Age (for 65+)
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_dir : Path
        Directory to save CSV
    """
    acs_dir = output_dir / "acs"
    acs_dir.mkdir(parents=True, exist_ok=True)
    
    state_fips = config['study_area']['state_fips']
    county_fips = config['study_area']['county_fips']
    year = config['census']['acs_year']
    
    print(f"Downloading ACS {year} 5-year estimates...")
    print("  Note: Requires Census API key. Set CENSUS_API_KEY environment variable.")
    
    api_key = os.environ.get('CENSUS_API_KEY')
    if not api_key:
        print("  ⚠ No API key found. Creating placeholder file with variable list.")
        print("  → Get a free key at: https://api.census.gov/data/key_signup.html")
        
        # Create placeholder with variable definitions
        variables_info = {
            'Variable': [
                'B08201_001E', 'B08201_002E',  # Vehicles available
                'B17001_001E', 'B17001_002E',  # Poverty status
                'B03002_001E', 'B03002_003E',  # Race/ethnicity
                'B01001_001E',  # Total population
                # Age 65+ variables (males + females)
                'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E', 'B01001_024E', 'B01001_025E',
                'B01001_044E', 'B01001_045E', 'B01001_046E', 'B01001_047E', 'B01001_048E', 'B01001_049E'
            ],
            'Description': [
                'Total households', 'Households with 0 vehicles',
                'Total population for poverty', 'Below poverty level',
                'Total population', 'White alone, not Hispanic',
                'Total population for age',
                'Male 65-66', 'Male 67-69', 'Male 70-74', 'Male 75-79', 'Male 80-84', 'Male 85+',
                'Female 65-66', 'Female 67-69', 'Female 70-74', 'Female 75-79', 'Female 80-84', 'Female 85+'
            ],
            'Used_For': [
                'Zero-vehicle rate', 'Zero-vehicle rate',
                'Poverty rate', 'Poverty rate',
                'Minority %', 'Minority %',
                'Elderly %',
                'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %',
                'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %', 'Elderly %'
            ]
        }
        pd.DataFrame(variables_info).to_csv(acs_dir / "acs_variables_needed.csv", index=False)
        print(f"  ✓ Variable list saved to {acs_dir / 'acs_variables_needed.csv'}")
        return None
    
    try:
        c = Census(api_key)
        
        # Define variables to download
        variables = [
            'NAME',
            # Vehicles available (B08201)
            'B08201_001E',  # Total households
            'B08201_002E',  # No vehicle available
            # Poverty (B17001)
            'B17001_001E',  # Total population for poverty determination
            'B17001_002E',  # Below poverty level
            # Race/Ethnicity (B03002)
            'B03002_001E',  # Total population
            'B03002_003E',  # White alone, not Hispanic
            # Age (B01001) - need 65+ for elderly
            'B01001_001E',  # Total population
            # Males 65+
            'B01001_020E', 'B01001_021E', 'B01001_022E', 
            'B01001_023E', 'B01001_024E', 'B01001_025E',
            # Females 65+
            'B01001_044E', 'B01001_045E', 'B01001_046E',
            'B01001_047E', 'B01001_048E', 'B01001_049E'
        ]
        
        # Download tract-level data
        data = c.acs5.state_county_tract(
            fields=variables,
            state_fips=state_fips,
            county_fips=county_fips,
            tract='*',
            year=year
        )
        
        df = pd.DataFrame(data)
        
        # Create GEOID
        df['GEOID'] = df['state'] + df['county'] + df['tract']
        
        # Calculate derived variables
        df['pct_zero_vehicle'] = df['B08201_002E'] / df['B08201_001E']
        df['pct_poverty'] = df['B17001_002E'] / df['B17001_001E']
        df['pct_minority'] = 1 - (df['B03002_003E'] / df['B03002_001E'])
        
        # Elderly (sum all 65+ age groups)
        elderly_male = df[['B01001_020E', 'B01001_021E', 'B01001_022E', 
                           'B01001_023E', 'B01001_024E', 'B01001_025E']].sum(axis=1)
        elderly_female = df[['B01001_044E', 'B01001_045E', 'B01001_046E',
                             'B01001_047E', 'B01001_048E', 'B01001_049E']].sum(axis=1)
        df['pct_elderly'] = (elderly_male + elderly_female) / df['B01001_001E']
        
        # Handle division by zero
        for col in ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']:
            df[col] = df[col].fillna(0)
        
        # Save
        output_path = acs_dir / "acs_tdi_variables.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  ✓ Downloaded ACS data for {len(df)} tracts")
        print(f"  ✓ Saved to {output_path}")
        
        # Summary stats
        print("\n  Summary of TDI variables:")
        for col in ['pct_zero_vehicle', 'pct_poverty', 'pct_minority', 'pct_elderly']:
            print(f"    {col}: mean={df[col].mean():.2%}, max={df[col].max():.2%}")
        
        return output_path
        
    except Exception as e:
        print(f"  ✗ Error downloading ACS data: {e}")
        return None

def main():
    """Main function to download all required data."""
    print("=" * 60)
    print("Transit Desert Pipeline: Data Download")
    print("=" * 60)
    
    # Load config
    config = load_config()
    study_area = config['study_area']['name']
    print(f"\nStudy Area: {study_area}")
    print(f"FIPS: {config['study_area']['full_fips']}")
    
    # Set up output directory
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-" * 40)
    print("Step 1: Download GTFS Feed")
    print("-" * 40)
    download_gtfs(config, output_dir)
    
    print("\n" + "-" * 40)
    print("Step 2: Download Census Tract Boundaries")
    print("-" * 40)
    download_census_tracts(config, output_dir)
    
    print("\n" + "-" * 40)
    print("Step 3: Download ACS Demographic Data")
    print("-" * 40)
    download_acs_data(config, output_dir)
    
    print("\n" + "=" * 60)
    print("Data download complete!")
    print("Next step: python src/02_compute_supply.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
