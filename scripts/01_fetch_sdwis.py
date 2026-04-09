"""
01_fetch_sdwis.py
=================
Fetches Safe Drinking Water Information System (SDWIS) data for New York State.

Strategy:
    1. Try the EPA Envirofacts REST API (both V2 and V1 URL formats)
    2. If the API fails, fall back to the ECHO SDWA bulk download (ZIP of CSVs)

Downloads two tables:
    1. WATER_SYSTEM / pub_water_systems — inventory of all public water systems
    2. VIOLATION / violations            — all SDWA violations for NY systems

Data is saved as CSV files in data/raw/.

API docs:
    - Envirofacts: https://www.epa.gov/enviro/envirofacts-data-service-api
    - ECHO bulk:   https://echo.epa.gov/tools/data-downloads/sdwa-download-summary

Usage:
    python scripts/01_fetch_sdwis.py
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_CODE = "NY"
PAGE_SIZE = 10000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ECHO SDWA bulk download (fallback)
ECHO_SDWA_ZIP_URL = "https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip"

# ---------------------------------------------------------------------------
# Approach 1: Envirofacts REST API
# ---------------------------------------------------------------------------

def try_envirofacts_url(table_name: str, filter_col: str, filter_val: str,
                        start: int, end: int) -> requests.Response | None:
    """
    Try multiple URL formats for the Envirofacts API and return the first
    one that returns valid JSON.
    """
    base = "https://data.epa.gov/efservice"

    # URL patterns to try (V2 with program prefix, V1 without, different cases)
    url_patterns = [
        f"{base}/sdwis.{table_name.lower()}/{filter_col}/{filter_val}/JSON/rows/{start}:{end}",
        f"{base}/{table_name}/{filter_col}/{filter_val}/JSON/rows/{start}:{end}",
        f"{base}/{table_name.lower()}/{filter_col.lower()}/{filter_val}/JSON/rows/{start}:{end}",
        f"{base}/{table_name}/{filter_col}/{filter_val}/rows/{start}:{end}/JSON",
    ]

    for url in url_patterns:
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and resp.text.strip().startswith("["):
                return resp
        except requests.RequestException:
            continue

    return None


def fetch_table_envirofacts(table_name: str, filter_col: str,
                            filter_val: str) -> pd.DataFrame | None:
    """
    Paginate through the Envirofacts API. Returns DataFrame or None if
    the API doesn't work.
    """
    # First, test if any URL pattern works
    print(f"  Testing Envirofacts API for {table_name} ...")
    test_resp = try_envirofacts_url(table_name, filter_col, filter_val, 0, 4)

    if test_resp is None:
        print(f"  ✗ Envirofacts API did not return valid JSON for {table_name}.")
        return None

    # Found a working URL format — extract it
    working_url = test_resp.url
    base_pattern = working_url.rsplit("/rows/", 1)[0]
    print(f"  ✓ Working URL pattern found: {base_pattern}/rows/...")

    data = test_resp.json()
    if not data:
        return None

    print(f"    Test returned {len(data)} rows with columns: {list(data[0].keys())[:8]}...")

    # Now paginate through all results
    all_rows = []
    start = 0
    while True:
        end = start + PAGE_SIZE - 1
        url = f"{base_pattern}/rows/{start}:{end}"

        if start > 0:
            print(f"  Fetching rows {start:,}–{end:,} ...")
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200 or not resp.text.strip().startswith("["):
                break
            data = resp.json()
        # else: already have first page from test

        if not data:
            break

        all_rows.extend(data)
        print(f"    → {len(data):,} rows (total: {len(all_rows):,})")

        if len(data) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(1)

        # Reset data for next iteration
        data = None

    return pd.DataFrame(all_rows) if all_rows else None


# ---------------------------------------------------------------------------
# Approach 2: ECHO SDWA bulk download (fallback)
# ---------------------------------------------------------------------------

def fetch_echo_bulk_download() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Download the ECHO SDWA ZIP file and extract water systems and violations
    for New York State.

    The ZIP contains several CSVs including:
        - SDWA_PUB_WATER_SYSTEMS.csv
        - SDWA_VIOLATIONS.csv
    """
    print(f"\n  Downloading ECHO SDWA bulk file ...")
    print(f"  URL: {ECHO_SDWA_ZIP_URL}")
    print(f"  (This is a large file — may take a few minutes)")

    try:
        resp = requests.get(ECHO_SDWA_ZIP_URL, timeout=600, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ ECHO download failed: {e}")
        return None

    # Read ZIP into memory and extract CSVs
    print(f"  Download complete ({len(resp.content) / 1_000_000:.1f} MB). Extracting ...")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = zf.namelist()
        print(f"  Files in ZIP: {csv_names}")

        # Find the water systems and violations files
        pws_file = next((f for f in csv_names
                         if "PUB_WATER_SYSTEM" in f.upper()
                         or "WATER_SYSTEM" in f.upper()), None)
        viol_file = next((f for f in csv_names
                          if "VIOLATION" in f.upper()
                          and "ENFORCEMENT" not in f.upper()), None)

        if not pws_file or not viol_file:
            print(f"  ✗ Could not find expected CSVs in ZIP.")
            print(f"  Available files: {csv_names}")
            return None

        print(f"  Reading {pws_file} ...")
        systems = pd.read_csv(zf.open(pws_file), dtype=str, low_memory=False)

        print(f"  Reading {viol_file} ...")
        violations = pd.read_csv(zf.open(viol_file), dtype=str, low_memory=False)

    # Filter to New York
    # Try multiple possible state column names
    for col in ["PWSID", "pwsid"]:
        if col in systems.columns:
            systems = systems[systems[col].str.startswith("NY", na=False)]
            break
    for col in ["STATE_CODE", "state_code", "PRIMACY_AGENCY_CODE"]:
        if col in systems.columns:
            systems = systems[systems[col].str.upper() == "NY"]
            break

    for col in ["PWSID", "pwsid"]:
        if col in violations.columns:
            violations = violations[violations[col].str.startswith("NY", na=False)]
            break

    print(f"  NY water systems: {len(systems):,}")
    print(f"  NY violations:    {len(violations):,}")

    return systems, violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    systems_df = None
    violations_df = None

    # -----------------------------------------------------------------------
    # Try Approach 1: Envirofacts API
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("APPROACH 1: Envirofacts REST API")
    print("=" * 60)

    systems_df = fetch_table_envirofacts(
        "WATER_SYSTEM", "PRIMACY_AGENCY_CODE", STATE_CODE
    )
    if systems_df is not None:
        print(f"  ✓ Water systems: {len(systems_df):,} rows")

    violations_df = fetch_table_envirofacts(
        "VIOLATION", "PRIMACY_AGENCY_CODE", STATE_CODE
    )
    if violations_df is not None:
        print(f"  ✓ Violations: {len(violations_df):,} rows")

    # -----------------------------------------------------------------------
    # Try Approach 2: ECHO bulk download (if API failed)
    # -----------------------------------------------------------------------
    if systems_df is None or violations_df is None:
        print("\n" + "=" * 60)
        print("APPROACH 2: ECHO SDWA bulk download (fallback)")
        print("=" * 60)

        result = fetch_echo_bulk_download()
        if result is not None:
            if systems_df is None:
                systems_df = result[0]
            if violations_df is None:
                violations_df = result[1]

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    if systems_df is not None and not systems_df.empty:
        systems_df.columns = [c.lower() for c in systems_df.columns]
        outpath = os.path.join(OUTPUT_DIR, "water_system_ny.csv")
        systems_df.to_csv(outpath, index=False)
        print(f"\n✓ Saved {len(systems_df):,} water systems → {outpath}")
        print(f"  Columns: {list(systems_df.columns)}")
    else:
        print("\n✗ ERROR: Could not retrieve water system data.")
        print("  Try downloading manually from:")
        print("  https://echo.epa.gov/tools/data-downloads#drinking-water")
        return

    if violations_df is not None and not violations_df.empty:
        violations_df.columns = [c.lower() for c in violations_df.columns]
        outpath = os.path.join(OUTPUT_DIR, "violation_ny.csv")
        violations_df.to_csv(outpath, index=False)
        print(f"✓ Saved {len(violations_df):,} violations → {outpath}")
        print(f"  Columns: {list(violations_df.columns)}")
    else:
        print("\n✗ ERROR: Could not retrieve violation data.")
        return

    print("\n" + "=" * 60)
    print("Done! Raw SDWIS data saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
