"""
gtfs_utils.py
Utility functions for loading GTFS data in various formats.

GTFS data can come in several packaging formats:
1. Single zip file with txt files directly inside (gtfs.zip)
2. Zip file containing multiple mode-specific zip files (e.g., bus.zip, rail.zip)
3. Directory with txt files
4. Directory with multiple zip files
5. Nested zip (zip containing a folder containing txt files)

This module handles all these cases and merges multi-modal feeds.
"""

import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Union, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import gtfs_kit as gk
    import pandas as pd
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install gtfs_kit pandas")
    sys.exit(1)


def find_gtfs_files(directory: Path) -> dict:
    """
    Scan a directory for GTFS data in various formats.
    
    Parameters
    ----------
    directory : Path
        Directory to scan
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'txt_files': list of .txt files found
        - 'zip_files': list of .zip files found
        - 'type': 'txt', 'single_zip', 'multi_zip', or 'unknown'
    """
    result = {
        'txt_files': [],
        'zip_files': [],
        'type': 'unknown'
    }
    
    if not directory.exists():
        return result
    
    # Look for txt files (direct GTFS format)
    txt_files = list(directory.glob('*.txt'))
    result['txt_files'] = [f.name for f in txt_files]
    
    # Look for zip files
    zip_files = list(directory.glob('*.zip'))
    result['zip_files'] = [f.name for f in zip_files]
    
    # Determine type
    required_gtfs_files = {'stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt'}
    
    if required_gtfs_files.issubset(set(result['txt_files'])):
        result['type'] = 'txt'
    elif len(result['zip_files']) == 1:
        result['type'] = 'single_zip'
    elif len(result['zip_files']) > 1:
        result['type'] = 'multi_zip'
    
    return result


def is_valid_gtfs_zip(zip_path: Path) -> bool:
    """
    Check if a zip file contains valid GTFS data.
    
    Parameters
    ----------
    zip_path : Path
        Path to zip file
    
    Returns
    -------
    bool
        True if zip contains required GTFS files
    """
    required = {'stops.txt', 'routes.txt', 'trips.txt', 'stop_times.txt'}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = set(zf.namelist())
            # Check for files at root level
            if required.issubset(names):
                return True
            # Check for files in a subdirectory
            for name in names:
                base = os.path.basename(name)
                if base in required:
                    return True
        return False
    except:
        return False


def extract_nested_zip(zip_path: Path, output_dir: Path) -> Path:
    """
    Extract a zip file, handling nested structures.
    
    Some agencies package GTFS as:
    - zip > txt files (normal)
    - zip > folder > txt files (nested)
    - zip > zip files (multi-modal)
    
    Parameters
    ----------
    zip_path : Path
        Path to zip file
    output_dir : Path
        Directory to extract to
    
    Returns
    -------
    Path
        Path to directory containing txt files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    
    # Check if txt files are at root
    txt_files = list(output_dir.glob('*.txt'))
    if txt_files:
        return output_dir
    
    # Check for nested folder
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        txt_files = list(subdir.glob('*.txt'))
        if txt_files:
            return subdir
    
    # Check for nested zip files
    nested_zips = list(output_dir.glob('*.zip'))
    if nested_zips:
        # This is a multi-modal package
        return output_dir
    
    return output_dir


def merge_gtfs_feeds(feeds: List) -> object:
    """
    Merge multiple GTFS feeds into one.
    
    Parameters
    ----------
    feeds : List[gtfs_kit.Feed]
        List of GTFS feeds to merge
    
    Returns
    -------
    gtfs_kit.Feed
        Merged feed
    """
    if len(feeds) == 1:
        return feeds[0]
    
    print(f"  Merging {len(feeds)} GTFS feeds...")
    
    # Start with first feed
    merged = feeds[0]
    
    # Concatenate dataframes from other feeds
    for feed in feeds[1:]:
        # Merge each table
        for table in ['agency', 'stops', 'routes', 'trips', 'stop_times', 
                      'calendar', 'calendar_dates', 'shapes', 'frequencies',
                      'transfers', 'pathways', 'levels', 'feed_info']:
            merged_df = getattr(merged, table, None)
            feed_df = getattr(feed, table, None)
            
            if merged_df is not None and feed_df is not None:
                # Concatenate and remove duplicates
                combined = pd.concat([merged_df, feed_df], ignore_index=True)
                # Remove duplicates based on primary key columns
                if table == 'stops':
                    combined = combined.drop_duplicates(subset=['stop_id'], keep='first')
                elif table == 'routes':
                    combined = combined.drop_duplicates(subset=['route_id'], keep='first')
                elif table == 'trips':
                    combined = combined.drop_duplicates(subset=['trip_id'], keep='first')
                elif table == 'stop_times':
                    combined = combined.drop_duplicates(subset=['trip_id', 'stop_sequence'], keep='first')
                setattr(merged, table, combined)
            elif feed_df is not None:
                setattr(merged, table, feed_df)
    
    return merged


def load_gtfs(gtfs_path: Union[str, Path], dist_units: str = 'mi') -> object:
    """
    Load GTFS data from various formats.
    
    Handles:
    1. Single zip file with txt files
    2. Directory with txt files
    3. Directory with single zip file
    4. Directory with multiple zip files (multi-modal)
    5. Zip file with nested folder structure
    6. Zip file containing multiple zip files
    
    Parameters
    ----------
    gtfs_path : str or Path
        Path to GTFS data (zip file or directory)
    dist_units : str
        Distance units ('mi' or 'km')
    
    Returns
    -------
    gtfs_kit.Feed
        Loaded (and possibly merged) GTFS feed
    """
    gtfs_path = Path(gtfs_path)
    
    print(f"Loading GTFS from: {gtfs_path}")
    
    # Case 1: Direct zip file provided
    if gtfs_path.is_file() and gtfs_path.suffix == '.zip':
        return _load_gtfs_zip(gtfs_path, dist_units)
    
    # Case 2: Directory provided
    if gtfs_path.is_dir():
        return _load_gtfs_directory(gtfs_path, dist_units)
    
    raise FileNotFoundError(f"GTFS path not found: {gtfs_path}")


def _load_gtfs_zip(zip_path: Path, dist_units: str) -> object:
    """Load GTFS from a zip file."""
    
    # First, check what's inside the zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        
        # Check for txt files at root
        txt_files = [n for n in names if n.endswith('.txt') and '/' not in n]
        
        # Check for nested zip files
        nested_zips = [n for n in names if n.endswith('.zip')]
        
        # Check for nested folder with txt files
        nested_txt = [n for n in names if n.endswith('.txt') and '/' in n]
    
    # Case: Normal zip with txt files at root
    if txt_files:
        print(f"  Format: Standard GTFS zip")
        return gk.read_feed(zip_path, dist_units=dist_units)
    
    # Case: Zip contains other zip files (multi-modal)
    if nested_zips:
        print(f"  Format: Multi-modal zip (contains {len(nested_zips)} zip files)")
        return _load_multi_modal_zip(zip_path, dist_units)
    
    # Case: Nested folder structure
    if nested_txt:
        print(f"  Format: Nested folder structure")
        # Extract to temp directory and load
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            extracted = extract_nested_zip(zip_path, tmpdir)
            return gk.read_feed(extracted, dist_units=dist_units)
    
    # Try loading directly (let gtfs_kit handle it)
    return gk.read_feed(zip_path, dist_units=dist_units)


def _load_multi_modal_zip(zip_path: Path, dist_units: str) -> object:
    """Load and merge multiple GTFS feeds from a zip containing zip files."""
    
    feeds = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Extract outer zip
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        
        # Find and load each inner zip
        inner_zips = list(tmpdir.glob('**/*.zip'))
        
        for inner_zip in inner_zips:
            if is_valid_gtfs_zip(inner_zip):
                mode_name = inner_zip.stem
                print(f"    Loading {mode_name}...")
                try:
                    feed = gk.read_feed(inner_zip, dist_units=dist_units)
                    feeds.append(feed)
                    print(f"      ✓ {len(feed.stops)} stops, {len(feed.routes)} routes")
                except Exception as e:
                    print(f"      ⚠ Error loading {mode_name}: {e}")
    
    if not feeds:
        raise ValueError("No valid GTFS feeds found in zip file")
    
    # Merge feeds
    merged = merge_gtfs_feeds(feeds)
    
    return merged


def _load_gtfs_directory(directory: Path, dist_units: str) -> object:
    """Load GTFS from a directory."""
    
    scan = find_gtfs_files(directory)
    
    # Case: Directory contains txt files directly
    if scan['type'] == 'txt':
        print(f"  Format: Directory with txt files")
        return gk.read_feed(directory, dist_units=dist_units)
    
    # Case: Directory contains single zip
    if scan['type'] == 'single_zip':
        zip_path = directory / scan['zip_files'][0]
        print(f"  Format: Directory with single zip ({scan['zip_files'][0]})")
        return _load_gtfs_zip(zip_path, dist_units)
    
    # Case: Directory contains multiple zips (multi-modal)
    if scan['type'] == 'multi_zip':
        print(f"  Format: Directory with {len(scan['zip_files'])} zip files (multi-modal)")
        return _load_multi_modal_directory(directory, scan['zip_files'], dist_units)
    
    # Try common filenames
    common_names = ['gtfs.zip', 'google_transit.zip', 'transit.zip', 'feed.zip']
    for name in common_names:
        if (directory / name).exists():
            print(f"  Format: Found {name}")
            return _load_gtfs_zip(directory / name, dist_units)
    
    raise FileNotFoundError(
        f"Could not find valid GTFS data in {directory}\n"
        f"  Found txt files: {scan['txt_files'][:5]}\n"
        f"  Found zip files: {scan['zip_files'][:5]}"
    )


def _load_multi_modal_directory(directory: Path, zip_files: List[str], dist_units: str) -> object:
    """Load and merge multiple GTFS feeds from a directory with multiple zips."""
    
    feeds = []
    
    for zip_name in zip_files:
        zip_path = directory / zip_name
        if is_valid_gtfs_zip(zip_path):
            mode_name = zip_path.stem
            print(f"    Loading {mode_name}...")
            try:
                feed = gk.read_feed(zip_path, dist_units=dist_units)
                feeds.append(feed)
                print(f"      ✓ {len(feed.stops)} stops, {len(feed.routes)} routes")
            except Exception as e:
                print(f"      ⚠ Error loading {mode_name}: {e}")
    
    if not feeds:
        raise ValueError("No valid GTFS feeds found in directory")
    
    # Merge feeds
    merged = merge_gtfs_feeds(feeds)
    
    return merged


def print_gtfs_summary(feed) -> None:
    """Print summary of loaded GTFS feed."""
    print(f"\n  GTFS Summary:")
    print(f"    Stops:  {len(feed.stops):,}")
    print(f"    Routes: {len(feed.routes):,}")
    print(f"    Trips:  {len(feed.trips):,}")
    
    if feed.agency is not None and len(feed.agency) > 0:
        agencies = feed.agency['agency_name'].tolist()
        print(f"    Agencies: {', '.join(agencies[:3])}")
        if len(agencies) > 3:
            print(f"              ... and {len(agencies) - 3} more")
    
    if feed.routes is not None and 'route_type' in feed.routes.columns:
        route_types = {
            0: 'Tram/Light Rail',
            1: 'Subway/Metro',
            2: 'Rail',
            3: 'Bus',
            4: 'Ferry',
            5: 'Cable Tram',
            6: 'Aerial Lift',
            7: 'Funicular',
            11: 'Trolleybus',
            12: 'Monorail'
        }
        type_counts = feed.routes['route_type'].value_counts()
        print(f"    Route types:")
        for rt, count in type_counts.items():
            rt_name = route_types.get(rt, f'Type {rt}')
            print(f"      {rt_name}: {count}")


# Common GTFS feed URLs for major US cities
GTFS_FEEDS = {
    'baltimore': 'https://feeds.mta.maryland.gov/gtfs/local-bus',
    'philadelphia': 'https://www3.septa.org/developer/gtfs/google_transit.zip',  # Note: SEPTA has multiple files
    'washington_dc': 'https://api.wmata.com/gtfs/bus-gtfs.zip',  # Requires API key
    'boston': 'https://cdn.mbta.com/MBTA_GTFS.zip',
    'new_york': 'http://web.mta.info/developers/data/nyct/subway/google_transit.zip',  # Subway only
    'chicago': 'https://www.transitchicago.com/downloads/sch_data/google_transit.zip',
    'los_angeles': 'https://gitlab.com/LACMTA/gtfs_bus/-/raw/master/gtfs_bus.zip',
    'san_francisco': 'https://gtfs.sfmta.com/transitdata/google_transit.zip',
    'seattle': 'https://metro.kingcounty.gov/GTFS/google_transit.zip',
    'denver': 'https://www.rtd-denver.com/files/gtfs/google_transit.zip',
    'atlanta': 'https://itsmarta.com/MARTA-GTFS.zip',
    'miami': 'https://www.miamidade.gov/transit/googletransit/google_transit.zip',
    'portland': 'https://developer.trimet.org/gtfs/gtfs.zip',
    'minneapolis': 'https://svc.metrotransit.org/mtgtfs/gtfs.zip',
    'phoenix': 'https://www.valleymetro.org/GTFS/valley_metro_gtfs.zip',
    'detroit': 'https://transitfeeds.com/p/smart-suburban-mobility-authority-for-regional-transportation/301/latest/download',
    'cleveland': 'https://transitfeeds.com/p/greater-cleveland-regional-transit-authority/219/latest/download',
    'pittsburgh': 'https://transitfeeds.com/p/port-authority-of-allegheny-county/77/latest/download',
}


if __name__ == "__main__":
    # Test with command line argument
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        feed = load_gtfs(path)
        print_gtfs_summary(feed)
    else:
        print("Usage: python gtfs_utils.py <path_to_gtfs>")
        print("\nSupported formats:")
        print("  - Single zip file with txt files")
        print("  - Directory with txt files")
        print("  - Directory with one or more zip files")
        print("  - Zip containing multiple mode-specific zips")
