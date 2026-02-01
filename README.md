# Transit Desert Identification Pipeline

A reproducible geospatial framework for equity-focused transit service gap analysis.

## Overview

This pipeline implements a **Composite Public Transit Accessibility (CPTA)** framework combined with a **Transit Dependency Index (TDI)** to identify transit deserts—areas where transit supply fails to meet transit-dependent demand.

### Key Features

- **9 transit supply metrics** including stop connectivity, service frequency by time period, and walking access
- **Equity-weighted demand** using zero-vehicle households, poverty, minority population, and elderly population
- **LISA spatial clustering** to identify statistically significant clusters of transit deserts
- **Fully reproducible** using open data sources (GTFS, Census ACS, OpenStreetMap)
- **Configurable** for any US city with available GTFS data

## Methodology

### Transit Supply (CPTA Score)

| Metric | Description |
|--------|-------------|
| Stop Connectivity | Welch method: routes × frequency × destinations at each stop |
| Route Coverage | Unique routes serving each tract |
| Frequency (AM Peak) | Average headway 6-9 AM |
| Frequency (Midday) | Average headway 10 AM - 3 PM |
| Frequency (PM Peak) | Average headway 4-7 PM |
| Walking Access | % of tract within 400m (bus) or 800m (rail) of stops |
| Span of Service | Hours per day with transit service |
| Weekend Ratio | Weekend trips ÷ Weekday trips |
| Jobs Accessible | Jobs reachable within 45 minutes by transit |

### Transit Demand (TDI Score)

| Component | Weight | Source |
|-----------|--------|--------|
| Zero-vehicle households | 0.35 | ACS B08201 |
| Below poverty line | 0.25 | ACS B17001 |
| Minority population | 0.20 | ACS B03002 |
| Elderly (65+) | 0.20 | ACS B01001 |

### Transit Desert Identification

Transit Gap = Normalized TDI - Normalized CPTA

Classification:
- **Transit Desert**: High demand, low supply (LISA High-High cluster)
- **Transit Stressed**: High demand, high supply
- **Underserved**: Low demand, low supply
- **Well-Served**: Low demand, high supply

## Installation

### Requirements
- Python 3.10+
- Java 11+ (optional, only for jobs accessibility calculation)

```bash
# Clone the repository
git clone https://github.com/Davidavid45/transit-desert-pipeline.git
cd transit-desert-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Jobs accessibility calculation (step 2b) requires Java 11+ and r5py. If you skip this step, the CPTA score will be calculated without the jobs accessibility metric.

## Usage

### Quick Start (Option 1: Run entire pipeline)

```bash
# Set api key
export CENSUS_API_KEY="your_actual_key_here"

# Run all steps at once
python run_pipeline.py

# Or run specific steps
python run_pipeline.py --steps 1,2,3,4,5

# Include jobs accessibility calculation (requires Java 11+)
python run_pipeline.py --include-jobs
```

### Manual Step-by-Step (Option 2)

```bash
# 1. Download data
python src/01_download_data.py

# 2. Compute supply metrics (CPTA)
python src/02_compute_supply.py

# 2b. (Optional) Compute jobs accessibility - requires Java 11+ and r5py
python src/02b_compute_jobs_accessibility.py

# 3. Compute demand metrics (TDI)
python src/03_compute_demand.py

# 4. Identify transit deserts (LISA clustering)
python src/04_identify_deserts.py

# 5. Generate visualizations
python src/05_visualize.py
```

### Analyzing a Different City

1. Edit `config/Config.yaml` with your city's:
   - State and county FIPS codes
   - GTFS feed URL
   - Projected coordinate system
   - Analysis date

2. Run the pipeline: `python run_pipeline.py`

## Project Structure

```
transit-desert-pipeline/
├── config/
│   └── Config.yaml              # City-specific parameters
├── data/
│   ├── raw/                     # Downloaded data (GTFS, Census)
│   │   ├── gtfs/
│   │   ├── census/
│   │   └── acs/
│   ├── processed/               # Intermediate files
│   └── external/                # Additional data (LODES, OSM)
├── src/
│   ├── 01_download_data.py
│   ├── 02_compute_supply.py
│   ├── 02b_compute_jobs_accessibility.py  # Optional: requires r5py
│   ├── 03_compute_demand.py
│   ├── 04_identify_deserts.py
│   ├── 05_visualize.py
│   └── gtfs_utils.py            # Helper functions
├── notebooks/
│   └── baltimore_analysis.ipynb
├── outputs/
│   ├── maps/
│   ├── tables/
│   └── figures/
├── logs/                         # Pipeline execution logs
├── run_pipeline.py              # Automated pipeline runner
├── requirements.txt
└── README.md
```

## Data Sources

All data sources are publicly available and downloaded automatically by the pipeline:

| Data | Source | Access Method |
|------|--------|---------------|
| GTFS | Transit Agency | Configured in Config.yaml |
| Census ACS | US Census Bureau | Census API (5-year estimates) |
| LODES Jobs | Census LEHD | Automatic download |
| Street Network | OpenStreetMap | Via OSMnx/r5py |
| Census Tracts | TIGER/Line | Census API |

## Citation

If you use this pipeline in your research, please cite:

```bibtex
Adegoke, O. (2026). Identifying Transit Deserts: A Reproducible Geospatial 
Framework for Equity-Focused Service Gap Analysis. 
[Journal details pending publication]
```

## License

MIT License

## Acknowledgments

This methodology draws from:
- Aman & Smith-Colin (2020) - CPTA framework
- Welch & Mishra (2013) - Stop connectivity
- Jiao & Dillivan (2013) - Transit desert concept
- Guo & Brakewood (2024) - Spatiotemporal accessibility