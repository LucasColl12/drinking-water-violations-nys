"""
01_fetch_sdwis.py
=================
Fetches Safe Drinking Water Information System (SDWIS) data for New York State
from EPA's Envirofacts REST API.

Downloads two tables:
    1. WATER_SYSTEM  — inventory of all public water systems (PWS) in NY
    2. VIOLATION      — all SDWA violations for NY water systems

Data is saved as CSV files in data/raw/.

API docs: https://www.epa.gov/enviro/envirofacts-data-service-api-v1
SDWIS data dictionary: https://echo.epa.gov/tools/data-downloads/sdwa-download-summary

Usage:
    python scripts/01_fetch_sdwis.py
"""

import os
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://data.epa.gov/efservice"
STATE_CODE = "NY"  # PRIMACY_AGENCY_CODE for New York
PAGE_SIZE = 10000  # Envirofacts default max rows per request

# Tables to download and the filter column for state
TABLES = {
    "WATER_SYSTEM": "PRIMACY_AGENCY_CODE",
    "VIOLATION": "PRIMACY_AGENCY_CODE",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch_table(table_name: str, filter_col: str, filter_val: str) -> pd.DataFrame:
    """
    Paginate through the Envirofacts API and return all rows for a table
    matching the given filter as a DataFrame.

    The API returns up to 10,000 rows per request. We keep requesting
    successive pages until a request returns fewer rows than PAGE_SIZE,
    indicating we've reached the end.

    Parameters
    ----------
    table_name : str
        Name of the SDWIS table (e.g., 'WATER_SYSTEM', 'VIOLATION').
    filter_col : str
        Column to filter on (e.g., 'PRIMACY_AGENCY_CODE').
    filter_val : str
        Value to filter for (e.g., 'NY').

    Returns
    -------
    pd.DataFrame
        All matching rows concatenated into a single DataFrame.
    """
    all_rows = []
    start = 0

    while True:
        end = start + PAGE_SIZE - 1
        url = (
            f"{BASE_URL}/{table_name}/{filter_col}/{filter_val}"
            f"/JSON/rows/{start}:{end}"
        )

        print(f"  Fetching {table_name} rows {start:,}–{end:,} ...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        data = response.json()

        if not data:
            break

        all_rows.extend(data)
        print(f"    → received {len(data):,} rows (total so far: {len(all_rows):,})")

        # If we got fewer than a full page, we've reached the end
        if len(data) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(1)  # Be polite to the API

    df = pd.DataFrame(all_rows)
    return df


def main():
    """Download SDWIS water system and violation data for New York State."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for table_name, filter_col in TABLES.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {table_name} (state={STATE_CODE})")
        print(f"{'='*60}")

        df = fetch_table(table_name, filter_col, STATE_CODE)

        if df.empty:
            print(f"  WARNING: No data returned for {table_name}.")
            continue

        # Standardize column names to lowercase for consistency
        df.columns = [c.lower() for c in df.columns]

        outpath = os.path.join(OUTPUT_DIR, f"{table_name.lower()}_{STATE_CODE.lower()}.csv")
        df.to_csv(outpath, index=False)
        print(f"\n  ✓ Saved {len(df):,} rows → {outpath}")

        # Print a quick summary
        print(f"\n  Columns ({len(df.columns)}):")
        for col in df.columns:
            non_null = df[col].notna().sum()
            print(f"    {col:40s}  {non_null:>6,} non-null")

    print("\n" + "="*60)
    print("Done! Raw SDWIS data saved to data/raw/")
    print("="*60)


if __name__ == "__main__":
    main()
