"""
02_fetch_census.py
==================
Fetches Census demographic data for New York State at the tract level
from TWO eras to match the violation comparison periods:

    1. Census 2000 Summary File 3 (SF3) — demographics for 2000–2005 period
    2. ACS 2022 5-Year estimates         — demographics for 2020–2025 period

Also downloads the matching TIGER/Line tract shapefiles for spatial joins
in each era (tract boundaries change between decennials).

Variables (matched across eras):
    - Median household income
    - Total population
    - Poverty rate
    - Racial composition (% white, Black, Hispanic, non-white)

Outputs:
    data/raw/census_tracts_ny_2000.csv    — Census 2000 tract demographics
    data/raw/census_tracts_ny_2022.csv    — ACS 2022 tract demographics
    data/raw/tl_2000_36_tract.zip         — TIGER 2000 tract shapefile
    data/raw/tl_2022_36_tract.zip         — TIGER 2022 tract shapefile

Prerequisites:
    - A free Census API key: https://api.census.gov/data/key_signup.html
    - Save it to apikey.txt in the project root

Usage:
    python scripts/02_fetch_census.py
"""

import os
import sys
import requests
import pandas as pd
from time import sleep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 300) -> requests.Response:
    """Fetch a URL with retries and longer timeout for flaky Census API."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt}/{max_retries} (timeout={timeout}s) ...")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"    Timed out, retrying in {wait}s ...")
                sleep(wait)
            else:
                raise

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
STATE_FIPS = "36"  # New York

# TIGER/Line shapefiles
TIGER_2022_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/"
    "tl_2022_36_tract.zip"
)
# Census 2000 tracts via TIGER2010 (which includes 2000 boundaries)
TIGER_2000_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2000/"
    "tl_2010_36_tract00.zip"
)

# ---------------------------------------------------------------------------
# ACS 2022 5-Year variables
# ---------------------------------------------------------------------------

ACS_VARIABLES = {
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
# Census 2000 SF3 variables (different table IDs than ACS)
# ---------------------------------------------------------------------------
# Census 2000 SF3 API: https://api.census.gov/data/2000/dec/sf3
# Note: variable names differ from ACS

SF3_VARIABLES = {
    "P001001": "total_population",          # Total population
    "P053001": "median_household_income",   # Median household income
    "P087001": "poverty_universe",          # Poverty status determined
    "P087002": "population_below_poverty",  # Income below poverty level
    "P003001": "race_total",               # Total for race
    "P003003": "white_alone",              # White alone
    "P003004": "black_alone",              # Black or African American alone
    "P004002": "hispanic_latino",          # Hispanic or Latino
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url: str, filename: str) -> str:
    """Download a file if it doesn't already exist."""
    outpath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(outpath):
        print(f"  Already exists: {outpath}")
        return outpath

    print(f"  Downloading {filename} ...")
    response = requests.get(url, timeout=180, stream=True)
    response.raise_for_status()

    with open(outpath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(outpath) / 1_000_000
    print(f"  ✓ Saved ({size_mb:.1f} MB) → {outpath}")
    return outpath


def compute_derived_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute poverty rate and racial composition percentages.
    Also cleans Census sentinel values (negative numbers = missing data).
    The Census API returns codes like -666666666 for suppressed/missing data.
    ACS top-codes income at $250,001.
    """
    # Clean sentinel values: any negative value in numeric columns → NaN
    numeric_cols = ["total_population", "median_household_income",
                    "poverty_universe", "population_below_poverty",
                    "race_total", "white_alone", "black_alone", "hispanic_latino"]
    for col in numeric_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # Also cap income at reasonable maximum (ACS top-codes at 250001)
    if "median_household_income" in df.columns:
        bad_income = df["median_household_income"] < 0
        n_bad = bad_income.sum()
        if n_bad > 0:
            print(f"    Cleaned {n_bad} sentinel income values → NaN")

    df["poverty_rate"] = (
        df["population_below_poverty"] / df["poverty_universe"] * 100
    ).round(2)
    df["pct_white"] = (df["white_alone"] / df["race_total"] * 100).round(2)
    df["pct_black"] = (df["black_alone"] / df["race_total"] * 100).round(2)
    df["pct_hispanic"] = (df["hispanic_latino"] / df["race_total"] * 100).round(2)
    df["pct_nonwhite"] = (100 - df["pct_white"]).round(2)
    return df


# ---------------------------------------------------------------------------
# Fetch ACS 2022
# ---------------------------------------------------------------------------

def fetch_acs_2022(api_key: str) -> pd.DataFrame:
    """Fetch ACS 2022 5-Year tract-level demographics for NY."""
    print("\n" + "=" * 60)
    print("FETCHING ACS 2022 5-YEAR (for 2020–2025 period)")
    print("=" * 60)

    var_list = ",".join(["NAME"] + list(ACS_VARIABLES.keys()))
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get={var_list}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}"
        f"&key={api_key}"
    )

    response = fetch_with_retry(url)
    data = response.json()

    header = data[0]
    rows = data[1:]
    print(f"  Retrieved {len(rows):,} tracts")

    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns=ACS_VARIABLES)

    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["county_fips"] = df["state"] + df["county"]

    for col in ACS_VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = compute_derived_measures(df)

    keep_cols = [
        "GEOID", "county_fips", "NAME",
        "total_population", "median_household_income",
        "poverty_rate", "pct_white", "pct_black",
        "pct_hispanic", "pct_nonwhite",
    ]
    df = df[keep_cols].copy()
    df = df.rename(columns={"NAME": "tract_name"})

    return df


# ---------------------------------------------------------------------------
# Fetch Census 2000 SF3
# ---------------------------------------------------------------------------

def fetch_census_2000(api_key: str) -> pd.DataFrame:
    """
    Fetch Census 2000 SF3 tract-level demographics for NY.

    The 2000 Census API uses different variable names and GEOID formats.
    Tract GEOIDs in 2000 are 11 digits: state(2) + county(3) + tract(6).
    """
    print("\n" + "=" * 60)
    print("FETCHING CENSUS 2000 SF3 (for 2000–2005 period)")
    print("=" * 60)

    var_list = ",".join(["NAME"] + list(SF3_VARIABLES.keys()))
    url = (
        f"https://api.census.gov/data/2000/dec/sf3"
        f"?get={var_list}"
        f"&for=tract:*"
        f"&in=state:{STATE_FIPS}"
        f"&key={api_key}"
    )

    print(f"  Requesting Census 2000 SF3 data ...")
    response = fetch_with_retry(url)
    data = response.json()

    header = data[0]
    rows = data[1:]
    print(f"  Retrieved {len(rows):,} tracts")

    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns=SF3_VARIABLES)

    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["county_fips"] = df["state"] + df["county"]

    for col in SF3_VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = compute_derived_measures(df)

    keep_cols = [
        "GEOID", "county_fips", "NAME",
        "total_population", "median_household_income",
        "poverty_rate", "pct_white", "pct_black",
        "pct_hispanic", "pct_nonwhite",
    ]
    df = df[keep_cols].copy()
    df = df.rename(columns={"NAME": "tract_name"})

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    # 1. Fetch ACS 2022 (for recent period)
    # -----------------------------------------------------------------------
    df_2022 = fetch_acs_2022(api_key)
    outpath = os.path.join(OUTPUT_DIR, "census_tracts_ny_2022.csv")
    df_2022.to_csv(outpath, index=False)
    print(f"  ✓ Saved {len(df_2022):,} tracts → {outpath}")

    # Also save as the default name for backward compatibility
    df_2022.to_csv(os.path.join(OUTPUT_DIR, "census_tracts_ny.csv"), index=False)

    # -----------------------------------------------------------------------
    # 2. Fetch Census 2000 SF3 (for historical period)
    # -----------------------------------------------------------------------
    try:
        df_2000 = fetch_census_2000(api_key)
        outpath = os.path.join(OUTPUT_DIR, "census_tracts_ny_2000.csv")
        df_2000.to_csv(outpath, index=False)
        print(f"  ✓ Saved {len(df_2000):,} tracts → {outpath}")
    except Exception as e:
        print(f"\n  ⚠ Census 2000 SF3 API failed: {e}")
        print("  The 2000 decennial API can be unreliable.")
        print("  Fallback: the analysis will use 2022 demographics for both periods.")
        print("  For better results, download Census 2000 data manually from:")
        print("  https://data.census.gov/table/DECENNIALSF32000.P053")

    # -----------------------------------------------------------------------
    # 3. Download TIGER/Line shapefiles
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DOWNLOADING TIGER/LINE TRACT SHAPEFILES")
    print("=" * 60)

    download_file(TIGER_2022_URL, "tl_2022_36_tract.zip")

    try:
        download_file(TIGER_2000_URL, "tl_2000_36_tract.zip")
    except Exception as e:
        print(f"  ⚠ 2000 TIGER download failed: {e}")
        print("  Will use 2022 boundaries for both periods (minor boundary shifts).")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nACS 2022 (recent period demographics):")
    print(df_2022[["total_population", "median_household_income",
                    "poverty_rate", "pct_nonwhite"]].describe().round(2))

    if 'df_2000' in dir():
        print(f"\nCensus 2000 (historical period demographics):")
        print(df_2000[["total_population", "median_household_income",
                        "poverty_rate", "pct_nonwhite"]].describe().round(2))

        print(f"\nDemographic shift (median values):")
        for col in ["median_household_income", "poverty_rate", "pct_nonwhite"]:
            v2000 = df_2000[col].median()
            v2022 = df_2022[col].median()
            print(f"  {col}: {v2000:.1f} → {v2022:.1f} (Δ {v2022-v2000:+.1f})")


if __name__ == "__main__":
    main()
