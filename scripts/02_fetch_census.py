"""
02_fetch_census.py
==================
Fetches American Community Survey (ACS) 5-Year demographic data for
New York State at the Census tract level from the Census Bureau API,
and downloads the matching TIGER/Line tract shapefile for spatial joins.

Variables pulled:
    - Median household income (B19013_001E)
    - Total population (B01003_001E)
    - Population below poverty level (B17001_002E)
    - Total pop for poverty denominator (B17001_001E)
    - White alone population (B02001_002E)
    - Black alone population (B02001_003E)
    - Hispanic/Latino population (B03003_003E)
    - Total pop for race denominator (B02001_001E)

Outputs:
    data/raw/census_tracts_ny.csv       — demographic data by tract
    data/raw/tl_2022_36_tract.zip       — TIGER/Line tract shapefile for NY

Prerequisites:
    - A free Census API key: https://api.census.gov/data/key_signup.html
    - Save it to apikey.txt in the project root (one line, no whitespace)

Usage:
    python scripts/02_fetch_census.py
"""

import os
import sys
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
STATE_FIPS = "36"  # New York
ACS_YEAR = 2022    # Most recent 5-year ACS with complete data
ACS_DATASET = "acs/acs5"

# TIGER/Line shapefile for NY tracts
TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/"
    "tl_2022_36_tract.zip"
)

# Variables to request (Census code: readable name)
VARIABLES = {
    "B01003_001E": "total_population",
    "B19013_001E": "median_household_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "population_below_poverty",
    "B02001_001E": "race_total",
    "B02001_002E": "white_alone",
    "B02001_003E": "black_alone",
    "B03003_003E": "hispanic_latino",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_tiger_shapefile():
    """Download the TIGER/Line tract shapefile for New York State."""
    outpath = os.path.join(OUTPUT_DIR, "tl_2022_36_tract.zip")
    if os.path.exists(outpath):
        print(f"  TIGER shapefile already exists: {outpath}")
        return outpath

    print(f"  Downloading TIGER/Line tract boundaries ...")
    response = requests.get(TIGER_URL, timeout=120, stream=True)
    response.raise_for_status()

    with open(outpath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(outpath) / 1_000_000
    print(f"  ✓ Saved ({size_mb:.1f} MB) → {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Read Census API key from apikey.txt in project root
    apikey_path = os.path.join(os.path.dirname(__file__), "..", "apikey.txt")
    if not os.path.exists(apikey_path):
        print(f"ERROR: Census API key file not found at {apikey_path}")
        print("Get a free key at: https://api.census.gov/data/key_signup.html")
        print("Save it to apikey.txt in the project root directory.")
        sys.exit(1)

    with open(apikey_path, "r") as f:
        api_key = f.read().strip()

    if not api_key:
        print("ERROR: apikey.txt is empty.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Fetch tract-level ACS data
    # -----------------------------------------------------------------------
    # The Census API allows querying all tracts in a state at once
    print(f"Fetching ACS {ACS_YEAR} 5-year tract-level data for NY ...")

    var_list = ",".join(["NAME"] + list(VARIABLES.keys()))
    url = (
        f"https://api.census.gov/data/{ACS_YEAR}/{ACS_DATASET}"
        f"?get={var_list}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}"
        f"&key={api_key}"
    )

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data = response.json()

    header = data[0]
    rows = data[1:]
    print(f"  Retrieved {len(rows):,} tracts")

    df = pd.DataFrame(rows, columns=header)

    # Rename coded variable columns to readable names
    df = df.rename(columns=VARIABLES)

    # Build a full GEOID (state + county + tract) for spatial joining
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["county_fips"] = df["state"] + df["county"]

    # Convert numeric columns
    numeric_cols = list(VARIABLES.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute derived measures
    df["poverty_rate"] = (
        df["population_below_poverty"] / df["poverty_universe"] * 100
    ).round(2)
    df["pct_white"] = (df["white_alone"] / df["race_total"] * 100).round(2)
    df["pct_black"] = (df["black_alone"] / df["race_total"] * 100).round(2)
    df["pct_hispanic"] = (df["hispanic_latino"] / df["race_total"] * 100).round(2)
    df["pct_nonwhite"] = (100 - df["pct_white"]).round(2)

    # Keep useful columns
    keep_cols = [
        "GEOID", "county_fips", "NAME",
        "total_population", "median_household_income",
        "poverty_rate", "pct_white", "pct_black",
        "pct_hispanic", "pct_nonwhite",
    ]
    df = df[keep_cols].copy()
    df = df.rename(columns={"NAME": "tract_name"})

    # Save tract-level demographics
    outpath = os.path.join(OUTPUT_DIR, "census_tracts_ny.csv")
    df.to_csv(outpath, index=False)
    print(f"  ✓ Saved {len(df):,} tracts → {outpath}")

    # -----------------------------------------------------------------------
    # 2. Download TIGER/Line shapefile for spatial joins
    # -----------------------------------------------------------------------
    print()
    download_tiger_shapefile()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\nSample rows:")
    print(df.head(5).to_string(index=False))

    print(f"\nSummary statistics:")
    print(df[["total_population", "median_household_income",
              "poverty_rate", "pct_nonwhite"]].describe().round(2))

    # Flag tracts with missing income (common for low-pop tracts)
    missing_income = df["median_household_income"].isna().sum()
    print(f"\nTracts with missing median income: {missing_income} "
          f"({missing_income/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
