"""
01_load_sdwis.py
================
Loads manually downloaded SDWA data from EPA's ECHO bulk download and
filters to New York State.

Download the data yourself from:
    https://echo.epa.gov/tools/data-downloads#drinking-water

Place the downloaded ZIP or extracted CSVs in data/raw/. This script
will find and process them automatically.

Data is saved as CSV files in data/raw/.
Expected files (the script searches for these patterns):
    - *PUB_WATER_SYSTEM* or *WATER_SYSTEM* → water system inventory
    - *VIOLATION* (not *ENFORCEMENT*)       → violation records

Outputs:
    data/raw/water_system_ny.csv
    data/raw/violation_ny.csv

Usage:
    python scripts/01_load_sdwis.py
"""

import os
import glob
import zipfile
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_file(directory: str, patterns: list[str],
              exclude: list[str] | None = None) -> str | None:
    """
    Search a directory for a CSV matching any of the given patterns
    (case-insensitive). Optionally exclude files matching certain terms.
    """
    exclude = [e.upper() for e in (exclude or [])]
    for f in os.listdir(directory):
        f_upper = f.upper()
        if not f_upper.endswith(".CSV"):
            continue
        if any(ex in f_upper for ex in exclude):
            continue
        if any(pat.upper() in f_upper for pat in patterns):
            return os.path.join(directory, f)
    return None


def extract_zips(directory: str):
    """Extract any ZIP files in the directory that haven't been extracted yet."""
    for f in glob.glob(os.path.join(directory, "*.zip")):
        # Skip TIGER shapefiles
        if "tract" in f.lower() or "tiger" in f.lower():
            continue
        print(f"  Extracting {os.path.basename(f)} ...")
        try:
            with zipfile.ZipFile(f) as zf:
                zf.extractall(directory)
                print(f"    → Extracted: {zf.namelist()}")
        except Exception as e:
            print(f"    → Failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading SDWA data from local files")
    print("=" * 60)
    print(f"  Looking in: {os.path.abspath(RAW_DIR)}\n")

    # Step 1: Extract any ZIP files
    extract_zips(RAW_DIR)

    # Step 2: Find the water systems CSV
    print("\nSearching for water system file ...")
    pws_path = find_file(RAW_DIR,
                         ["PUB_WATER_SYSTEM", "WATER_SYSTEM"],
                         exclude=["VIOLATION", "ENFORCEMENT"])

    if pws_path is None:
        # Check if our filtered output already exists
        filtered_pws = os.path.join(RAW_DIR, "water_system_ny.csv")
        if os.path.exists(filtered_pws):
            print(f"  Found existing filtered file: {filtered_pws}")
            pws_path = filtered_pws
        else:
            print("  ✗ Could not find a water systems CSV.")
            print("  Expected a file containing 'PUB_WATER_SYSTEM' or 'WATER_SYSTEM'")
            print("  in its filename. Download from:")
            print("  https://echo.epa.gov/tools/data-downloads#drinking-water")
            return
    else:
        print(f"  Found: {os.path.basename(pws_path)}")

    # Step 3: Find the violations CSV
    # Prefer SDWA_VIOLATIONS_ENFORCEMENT.csv — this is the main violations
    # table with violation_category_code and full date fields.
    # Exclude PN_VIOLATION (public notification associations) and SITE_VISIT.
    print("\nSearching for violations file ...")
    viol_path = find_file(RAW_DIR,
                          ["VIOLATIONS_ENFORCEMENT"],
                          exclude=["SITE_VISIT"])

    # Fallback: any file with VIOLATION in the name
    if viol_path is None:
        viol_path = find_file(RAW_DIR,
                              ["VIOLATION"],
                              exclude=["PN_VIOLATION", "SITE_VISIT",
                                       "EVENTS", "MILESTONE"])

    if viol_path is None:
        filtered_viol = os.path.join(RAW_DIR, "violation_ny.csv")
        if os.path.exists(filtered_viol):
            print(f"  Found existing filtered file: {filtered_viol}")
            viol_path = filtered_viol
        else:
            print("  ✗ Could not find a violations CSV.")
            print("  Expected a file containing 'VIOLATION' in its filename.")
            return
    else:
        print(f"  Found: {os.path.basename(viol_path)}")

    # Step 4: Load and filter to NY
    print(f"\nLoading water systems from {os.path.basename(pws_path)} ...")
    systems = pd.read_csv(pws_path, dtype=str, low_memory=False)
    print(f"  Total rows: {len(systems):,}")
    print(f"  Columns: {list(systems.columns)[:12]}...")

    # Standardize column names to lowercase
    systems.columns = [c.lower() for c in systems.columns]

    # Filter to NY using PWSID prefix
    if "pwsid" in systems.columns:
        systems = systems[systems["pwsid"].str.startswith("NY", na=False)]
    print(f"  NY systems: {len(systems):,}")

    print(f"\nLoading violations from {os.path.basename(viol_path)} ...")
    violations = pd.read_csv(viol_path, dtype=str, low_memory=False)
    print(f"  Total rows: {len(violations):,}")
    print(f"  Columns: {list(violations.columns)[:12]}...")

    violations.columns = [c.lower() for c in violations.columns]

    if "pwsid" in violations.columns:
        violations = violations[violations["pwsid"].str.startswith("NY", na=False)]
    print(f"  NY violations: {len(violations):,}")

    # Step 5: Save filtered files
    out_sys = os.path.join(RAW_DIR, "water_system_ny.csv")
    systems.to_csv(out_sys, index=False)
    print(f"\n✓ Saved {len(systems):,} water systems → {out_sys}")

    out_viol = os.path.join(RAW_DIR, "violation_ny.csv")
    violations.to_csv(out_viol, index=False)
    print(f"✓ Saved {len(violations):,} violations → {out_viol}")

    # Step 6: Diagnostics
    print(f"\n{'='*60}")
    print("DIAGNOSTICS")
    print(f"{'='*60}")

    # Date columns
    date_cols = [c for c in violations.columns
                 if "date" in c or "begin" in c or "end" in c]
    print(f"\n  Date columns: {date_cols}")
    for dc in date_cols:
        try:
            dates = pd.to_datetime(violations[dc], errors="coerce")
            valid = dates.notna().sum()
            if valid > 0:
                print(f"    {dc}: {dates.min().date()} → {dates.max().date()} "
                      f"({valid:,} valid)")
        except Exception:
            pass

    # Category columns
    cat_cols = [c for c in violations.columns
                if "category" in c or "health" in c]
    print(f"\n  Category/health columns: {cat_cols}")
    for cc in cat_cols:
        print(f"    {cc}: {violations[cc].value_counts().head(10).to_dict()}")

    print(f"\n{'='*60}")
    print("Done! Filtered SDWIS data ready in data/raw/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
