# Safe Drinking Water Act Violations in New York State

An analysis of drinking water quality violations across New York State, examining whether smaller or lower-income communities are disproportionately affected by Safe Drinking Water Act (SDWA) noncompliance.

## Purpose

The Safe Drinking Water Act establishes federal standards for drinking water quality in the United States. Public water systems must comply with maximum contaminant levels, treatment techniques, and monitoring and reporting requirements. When systems violate these standards, the violations are recorded in EPA's Safe Drinking Water Information System (SDWIS).

This project asks: **where are drinking water violations concentrated in New York State, and are smaller or economically disadvantaged communities disproportionately affected?**

The analysis proceeds in four stages:

1. **Data acquisition** — pulling violation records, water system characteristics, geocoordinates, and Census demographics from federal APIs
2. **Data integration** — performing a point-in-polygon spatial join to place each geocoded water system inside its enclosing Census tract, linking system-level violation data to tract-level demographics
3. **Statistical analysis** — modeling violation rates as a function of system size, source water type, ownership, and community income/demographics
4. **Visualization** — mapping violation hotspots and demographic patterns across New York State

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [EPA SDWIS via Envirofacts API](https://www.epa.gov/enviro/envirofacts-data-service-api-v1) | Water system inventory and violation records | REST API (no key required) |
| [EPA ECHO](https://echo.epa.gov/) | Geocoded facility coordinates for water systems | REST API |
| [U.S. Census ACS 5-Year Estimates](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Tract-level demographics (income, poverty, race) | REST API (key required — free) |
| [Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Tract boundary shapefiles for spatial joins | Direct download |

## Reproduction Instructions

### Prerequisites

- Python 3.13+
- A free Census API key ([request one here](https://api.census.gov/data/key_signup.html))

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/drinking-water-violations-nys.git
cd drinking-water-violations-nys
pip install -r requirements.txt
```

Save your Census API key to a file called `apikey.txt` in the project root (one line, no whitespace):

```bash
echo "your_key_here" > apikey.txt
```

### Running the Analysis

Run the scripts in order:

| Script | Description | Outputs |
|--------|-------------|---------|
| `scripts/01_fetch_sdwis.py` | Downloads water system and violation data from EPA | `data/raw/water_system_ny.csv`, `data/raw/violation_ny.csv` |
| `scripts/02_fetch_census.py` | Downloads tract-level ACS demographics + TIGER shapefile | `data/raw/census_tracts_ny.csv`, `data/raw/tl_2022_36_tract.zip` |
| `scripts/03_geocode_systems.py` | Retrieves lat/lon coordinates for each water system | `data/raw/system_locations_ny.csv` |
| `scripts/04_merge_and_clean.py` | Spatial join of systems to Census tracts; builds analytical file | `output/analytical_dataset.csv` |
| `scripts/05_analysis.py` | Runs statistical models and generates summary tables | `output/model_results.txt` |
| `scripts/06_visualize.py` | Produces maps and figures | `output/figures/` |

```bash
python scripts/01_fetch_sdwis.py
python scripts/02_fetch_census.py
python scripts/03_geocode_systems.py
python scripts/04_merge_and_clean.py
python scripts/05_analysis.py
python scripts/06_visualize.py
```

## Results

*TODO: Results discussion will be added once the analysis is complete.*

## Repository Structure

```
drinking-water-violations-nys/
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── apikey.txt                 ← Your Census API key (not tracked in git)
├── .gitignore
├── data/
│   └── raw/                   ← Downloaded data (not tracked in git)
├── output/
│   ├── figures/               ← Maps and charts
│   └── analytical_dataset.csv ← Merged analysis-ready dataset
└── scripts/
    ├── 01_fetch_sdwis.py      ← Download EPA SDWIS data
    ├── 02_fetch_census.py     ← Download Census ACS tract data + TIGER shapefile
    ├── 03_geocode_systems.py  ← Geocode water system locations via ECHO API
    ├── 04_merge_and_clean.py  ← Spatial join to Census tracts; build analytical dataset
    ├── 05_analysis.py         ← Statistical analysis
    └── 06_visualize.py        ← Maps and figures
```

## Author

Lucas — MPA Candidate, Syracuse University (PAI 600: Advanced Policy Analysis)
