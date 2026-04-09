"""
01_fetch_sdwis.py
=================
Fetches Safe Drinking Water Information System (SDWIS) data for New York State
from EPA's Envirofacts REST API.

Downloads two tables:
    1. WATER_SYSTEM  — inventory of all public water systems (PWS) in NY
    2. VIOLATION      — all SDWA violations for NY water systems

Data is saved as CSV files in data/raw/.

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
STATE_CODE = "NY"
PAGE_SIZE = 10000

TABLES = {
    "WATER_SYSTEM": "PRIMACY_AGENCY_CODE",
    "VIOLATION": "PRIMACY_AGENCY_CODE",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def fetch_table(table_name: str, filter_col: str, filter_val: str) -> pd.DataFrame:
    """Paginate through the Envirofacts API and return all matching rows."""
    all_rows = []
    start = 0

    while True:
        end = start + PAGE_SIZE - 1
        url = (
            f"{BASE_URL}/{table_name}/{filter_col}/{filter_val}"
            f"/rows/{start}:{end}/JSON"
        )

        print(f"  Fetching {table_name} rows {start:,}–{end:,} ...")
        response = requests.get(url, timeout=120)

        if response.status_code != 200 or not response.text.strip().startswith("["):
            print(f"    ✗ Bad response (status {response.status_code}). Stopping.")
            break

        data = response.json()
        if not data:
            break

        all_rows.extend(data)
        print(f"    → {len(data):,} rows (total: {len(all_rows):,})")

        if len(data) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(1)

    return pd.DataFrame(all_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for table_name, filter_col in TABLES.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {table_name} (state={STATE_CODE})")
        print(f"{'='*60}")

        df = fetch_table(table_name, filter_col, STATE_CODE)

        if df.empty:
            print(f"  WARNING: No data returned for {table_name}.")
            continue

        df.columns = [c.lower() for c in df.columns]

        outpath = os.path.join(OUTPUT_DIR, f"{table_name.lower()}_{STATE_CODE.lower()}.csv")
        df.to_csv(outpath, index=False)
        print(f"\n  ✓ Saved {len(df):,} rows → {outpath}")
        print(f"  Columns: {list(df.columns)}")

    print("\n" + "="*60)
    print("Done! Raw SDWIS data saved to data/raw/")
    print("="*60)


if __name__ == "__main__":
    main()
