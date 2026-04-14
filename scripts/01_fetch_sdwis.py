"""
01_fetch_sdwis.py
=================
Fetches Safe Drinking Water Information System (SDWIS) data for New York State.

Strategy (revised — ECHO first):
    1. Try the ECHO SDWA bulk download (ZIP of CSVs) — this is the most
       reliable source and includes the full violation history back to the
       1990s, which we need for the 2000–2005 comparison period.
    2. If ECHO fails, fall back to the Envirofacts REST API with a small
       page size (500 rows) to avoid timeouts.

Downloads two tables:
    1. WATER_SYSTEM — inventory of all public water systems
    2. VIOLATION    — all SDWA violations for NY systems

Data is saved as CSV files in data/raw/.

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
PAGE_SIZE = 500  # Small pages to avoid Envirofacts timeouts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

ECHO_SDWA_ZIP_URL = "https://echo.epa.gov/files/echodownloads/SDWA_latest_downloads.zip"

MIN_EXPECTED_SYSTEMS = 100
MIN_EXPECTED_VIOLATIONS = 100

# ---------------------------------------------------------------------------
# Approach 1 (preferred): ECHO SDWA bulk download
# ---------------------------------------------------------------------------

def fetch_echo_bulk_download() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Download the ECHO SDWA ZIP file and extract NY water systems + violations.
    This is the preferred source because:
        - Full historical data (back to 1990s)
        - All violation metadata columns including dates and categories
        - No pagination/timeout issues
    The ZIP is ~200–400 MB and contains national data; we filter to NY.
    """
    print("  Downloading ECHO SDWA bulk file ...")
    print(f"  URL: {ECHO_SDWA_ZIP_URL}")
    print("  (This is a large file — typically takes 2–5 minutes)\n")

    try:
        resp = requests.get(ECHO_SDWA_ZIP_URL, timeout=600, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ ECHO download failed: {e}")
        return None

    content = resp.content
    print(f"  Download complete ({len(content) / 1_000_000:.1f} MB). Extracting ...\n")

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = zf.namelist()
        print(f"  Files in ZIP: {csv_names}\n")

        # Find water systems file
        pws_file = next((f for f in csv_names
                         if "PUB_WATER_SYSTEM" in f.upper()
                         or "WATER_SYSTEM" in f.upper()), None)

        # Find violations file (exclude enforcement actions)
        viol_file = next((f for f in csv_names
                          if "VIOLATION" in f.upper()
                          and "ENFORCEMENT" not in f.upper()), None)

        if not pws_file or not viol_file:
            print(f"  ✗ Could not find expected CSVs in ZIP.")
            print(f"  Available files: {csv_names}")
            return None

        print(f"  Reading {pws_file} ...")
        systems = pd.read_csv(zf.open(pws_file), dtype=str, low_memory=False)
        print(f"    → {len(systems):,} total rows, columns: {list(systems.columns)[:10]}...")

        print(f"  Reading {viol_file} ...")
        violations = pd.read_csv(zf.open(viol_file), dtype=str, low_memory=False)
        print(f"    → {len(violations):,} total rows, columns: {list(violations.columns)[:10]}...")

    # Filter to New York using PWSID prefix (most reliable)
    pwsid_col_sys = next((c for c in systems.columns if c.upper() == "PWSID"), None)
    pwsid_col_viol = next((c for c in violations.columns if c.upper() == "PWSID"), None)

    if pwsid_col_sys:
        systems = systems[systems[pwsid_col_sys].str.startswith("NY", na=False)]
    if pwsid_col_viol:
        violations = violations[violations[pwsid_col_viol].str.startswith("NY", na=False)]

    print(f"\n  NY water systems: {len(systems):,}")
    print(f"  NY violations:    {len(violations):,}")

    return systems, violations


# ---------------------------------------------------------------------------
# Approach 2 (fallback): Envirofacts REST API
# ---------------------------------------------------------------------------

def try_envirofacts_url(table_name: str, filter_col: str, filter_val: str,
                        start: int, end: int) -> requests.Response | None:
    """Try multiple URL formats and return the first that returns valid JSON."""
    base = "https://data.epa.gov/efservice"
    url_patterns = [
        f"{base}/sdwis.{table_name.lower()}/{filter_col}/{filter_val}/JSON/rows/{start}:{end}",
        f"{base}/{table_name}/{filter_col}/{filter_val}/JSON/rows/{start}:{end}",
        f"{base}/{table_name.lower()}/{filter_col.lower()}/{filter_val}/JSON/rows/{start}:{end}",
        f"{base}/{table_name}/{filter_col}/{filter_val}/rows/{start}:{end}/JSON",
    ]
    for url in url_patterns:
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200 and resp.text.strip().startswith("["):
                return resp
        except requests.RequestException:
            continue
    return None


def fetch_table_envirofacts(table_name: str, filter_col: str,
                            filter_val: str) -> pd.DataFrame | None:
    """
    Paginate through the Envirofacts API with small page sizes.
    Returns DataFrame or None if the API doesn't work.
    """
    print(f"  Testing Envirofacts API for {table_name} ...")
    test_resp = try_envirofacts_url(table_name, filter_col, filter_val, 0, 4)

    if test_resp is None:
        print(f"  ✗ Envirofacts API did not return valid JSON for {table_name}.")
        return None

    working_url = test_resp.url
    base_pattern = working_url.rsplit("/rows/", 1)[0]
    print(f"  ✓ Working URL pattern: {base_pattern}/rows/...")

    test_data = test_resp.json()
    if not test_data:
        return None
    print(f"    Columns: {list(test_data[0].keys())[:8]}...")

    # Paginate with small pages to avoid timeouts
    all_rows = []
    start = 0
    consecutive_failures = 0

    while True:
        end = start + PAGE_SIZE - 1
        print(f"  Fetching rows {start:,}–{end:,} ...", end=" ", flush=True)

        url = f"{base_pattern}/rows/{start}:{end}"
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code != 200 or not resp.text.strip().startswith("["):
                print(f"failed (status {resp.status_code})")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("  Too many consecutive failures, stopping.")
                    break
                start += PAGE_SIZE
                time.sleep(2)
                continue
            data = resp.json()
        except requests.Timeout:
            print("timeout!")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("  Too many timeouts, stopping.")
                break
            start += PAGE_SIZE
            time.sleep(2)
            continue
        except Exception as e:
            print(f"error: {e}")
            break

        if not data:
            print("empty, done.")
            break

        consecutive_failures = 0
        all_rows.extend(data)
        print(f"→ {len(data):,} rows (total: {len(all_rows):,})")

        if len(data) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(0.5)

    if not all_rows:
        return None

    print(f"  ✓ Total rows fetched: {len(all_rows):,}")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    systems_df = None
    violations_df = None

    # --- Approach 1: ECHO bulk download (preferred) ---
    print("=" * 60)
    print("APPROACH 1: ECHO SDWA bulk download (preferred)")
    print("=" * 60)

    result = fetch_echo_bulk_download()
    if result is not None:
        systems_df, violations_df = result

    echo_ok = (
        systems_df is not None and len(systems_df) >= MIN_EXPECTED_SYSTEMS
        and violations_df is not None and len(violations_df) >= MIN_EXPECTED_VIOLATIONS
    )

    # --- Approach 2: Envirofacts API (fallback) ---
    if not echo_ok:
        print("\n" + "=" * 60)
        print("APPROACH 2: Envirofacts REST API (fallback)")
        print(f"  (page size: {PAGE_SIZE} to avoid timeouts)")
        print("=" * 60)

        if systems_df is None or len(systems_df) < MIN_EXPECTED_SYSTEMS:
            systems_df = fetch_table_envirofacts(
                "WATER_SYSTEM", "PRIMACY_AGENCY_CODE", STATE_CODE
            )
        if violations_df is None or len(violations_df) < MIN_EXPECTED_VIOLATIONS:
            violations_df = fetch_table_envirofacts(
                "VIOLATION", "PRIMACY_AGENCY_CODE", STATE_CODE
            )

    # --- Save ---
    if systems_df is not None and not systems_df.empty:
        systems_df.columns = [c.lower() for c in systems_df.columns]
        outpath = os.path.join(OUTPUT_DIR, "water_system_ny.csv")
        systems_df.to_csv(outpath, index=False)
        print(f"\n✓ Saved {len(systems_df):,} water systems → {outpath}")
        print(f"  Columns: {list(systems_df.columns)[:15]}...")
    else:
        print("\n✗ ERROR: Could not retrieve water system data.")
        print("  Try downloading manually from:")
        print("  https://echo.epa.gov/tools/data-downloads#drinking-water")
        return

    if violations_df is not None and not violations_df.empty:
        violations_df.columns = [c.lower() for c in violations_df.columns]

        date_cols = [c for c in violations_df.columns
                     if "date" in c or "begin" in c or "end" in c]
        cat_cols = [c for c in violations_df.columns
                    if "category" in c or "health" in c or "violation_c" in c]
        print(f"  Date columns: {date_cols}")
        print(f"  Category columns: {cat_cols}")

        outpath = os.path.join(OUTPUT_DIR, "violation_ny.csv")
        violations_df.to_csv(outpath, index=False)
        print(f"✓ Saved {len(violations_df):,} violations → {outpath}")
        print(f"  Columns: {list(violations_df.columns)[:15]}...")

        # Quick date range diagnostic
        for dc in date_cols:
            try:
                dates = pd.to_datetime(violations_df[dc], errors="coerce")
                valid = dates.notna().sum()
                if valid > 0:
                    print(f"  {dc}: {dates.min()} → {dates.max()} ({valid:,} valid)")
            except Exception:
                pass
    else:
        print("\n✗ ERROR: Could not retrieve violation data.")
        return

    print("\n" + "=" * 60)
    print("Done! Raw SDWIS data saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
