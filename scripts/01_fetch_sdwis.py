"""
01_load_sdwis.py
================
Loads manually downloaded SDWA data from EPA's ECHO bulk download and
filters to New York State.

Download from: https://echo.epa.gov/tools/data-downloads#drinking-water
Place the downloaded ZIP or extracted folder in data/raw/.

Expected files:
    - SDWA_PUB_WATER_SYSTEMS.csv     → water system inventory
    - SDWA_VIOLATIONS_ENFORCEMENT.csv → violation records (preferred)

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

def find_csv(directory: str, must_contain: str,
             must_not_contain: list[str] | None = None) -> str | None:
    """
    Recursively search directory for a CSV whose filename contains
    must_contain (case-insensitive), excluding files whose name
    contains any string in must_not_contain.
    """
    must_not_contain = [s.upper() for s in (must_not_contain or [])]

    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            f_upper = f.upper()
            if not f_upper.endswith(".CSV"):
                continue
            if any(ex in f_upper for ex in must_not_contain):
                continue
            if must_contain.upper() in f_upper:
                return os.path.join(root, f)
    return None


def extract_zips(directory: str):
    """Extract any non-TIGER ZIP files in the directory."""
    for f in glob.glob(os.path.join(directory, "*.zip")):
        if "tract" in f.lower() or "tiger" in f.lower():
            continue
        print(f"  Extracting {os.path.basename(f)} ...")
        try:
            with zipfile.ZipFile(f) as zf:
                zf.extractall(directory)
                print(f"    → Extracted {len(zf.namelist())} files")
        except Exception as e:
            print(f"    → Failed: {e}")


def _load_and_filter_ny(filepath: str, chunk_size: int = 50_000) -> pd.DataFrame:
    """
    Read a large national CSV in chunks and keep only NY rows (PWSID
    starting with 'NY'). Much faster than loading the whole file.
    """
    chunks = []
    total_read = 0

    for chunk in pd.read_csv(filepath, dtype=str, low_memory=False,
                             chunksize=chunk_size):
        chunk.columns = [c.lower() for c in chunk.columns]
        total_read += len(chunk)

        if "pwsid" in chunk.columns:
            ny_rows = chunk[chunk["pwsid"].str.startswith("NY", na=False)]
        else:
            ny_rows = chunk

        if len(ny_rows) > 0:
            chunks.append(ny_rows)

        print(f"    Read {total_read:,} rows, kept {sum(len(c) for c in chunks):,} NY ...",
              end="\r", flush=True)

    print()  # newline after progress
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Loading SDWA data from local files")
    print("=" * 60)
    print(f"  Looking in: {os.path.abspath(RAW_DIR)}\n")

    # Extract any ZIP files first
    extract_zips(RAW_DIR)

    # ------------------------------------------------------------------
    # Find water systems file
    # ------------------------------------------------------------------
    print("Searching for water system file ...")

    # Prefer SDWA_PUB_WATER_SYSTEMS.csv (full national file from ECHO)
    pws_path = find_csv(RAW_DIR, "PUB_WATER_SYSTEM",
                        must_not_contain=["VIOLATION", "ENFORCEMENT",
                                          "FACILITY", "GEOGRAPHIC"])

    if pws_path:
        print(f"  ✓ Found: {os.path.relpath(pws_path, RAW_DIR)}")
    else:
        print("  ✗ Could not find SDWA_PUB_WATER_SYSTEMS.csv")
        print("  Download from: https://echo.epa.gov/tools/data-downloads#drinking-water")
        return

    # ------------------------------------------------------------------
    # Find violations file
    # ------------------------------------------------------------------
    print("\nSearching for violations file ...")

    # STRONGLY prefer SDWA_VIOLATIONS_ENFORCEMENT.csv — this has
    # violation_category_code, is_health_based_ind, and full date fields.
    viol_path = find_csv(RAW_DIR, "VIOLATIONS_ENFORCEMENT",
                         must_not_contain=["PN_VIOLATION"])

    if viol_path:
        print(f"  ✓ Found: {os.path.relpath(viol_path, RAW_DIR)}")
    else:
        # Fallback: any VIOLATION file that isn't PN or SITE_VISIT
        viol_path = find_csv(RAW_DIR, "VIOLATION",
                             must_not_contain=["PN_VIOLATION", "SITE_VISIT",
                                               "EVENTS", "MILESTONE",
                                               "violation_ny"])
        if viol_path:
            print(f"  Found (fallback): {os.path.relpath(viol_path, RAW_DIR)}")
        else:
            print("  ✗ Could not find violations CSV.")
            return

    # ------------------------------------------------------------------
    # Load and filter to NY
    # ------------------------------------------------------------------
    # These are NATIONAL files (millions of rows). We read in chunks
    # and filter to NY on the fly to avoid loading everything into memory.

    print(f"\nLoading water systems (chunked, filtering to NY) ...")
    systems = _load_and_filter_ny(pws_path)
    print(f"  NY systems: {len(systems):,}")

    print(f"\nLoading violations (chunked, filtering to NY) ...")
    violations = _load_and_filter_ny(viol_path)
    print(f"  NY violations: {len(violations):,}")

    # ------------------------------------------------------------------
    # Save (overwrite any previous filtered files)
    # ------------------------------------------------------------------
    out_sys = os.path.join(RAW_DIR, "water_system_ny.csv")
    systems.to_csv(out_sys, index=False)
    print(f"\n✓ Saved {len(systems):,} water systems → {out_sys}")

    out_viol = os.path.join(RAW_DIR, "violation_ny.csv")
    violations.to_csv(out_viol, index=False)
    print(f"✓ Saved {len(violations):,} violations → {out_viol}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DIAGNOSTICS")
    print(f"{'='*60}")

    print(f"\n  Columns: {list(violations.columns)}")

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

    # Category / health columns
    cat_cols = [c for c in violations.columns
                if "category" in c or "health" in c or "violation_c" in c]
    print(f"\n  Category/health columns: {cat_cols}")
    for cc in cat_cols:
        vc = violations[cc].value_counts()
        print(f"    {cc}:")
        for val, count in vc.head(10).items():
            print(f"      {val}: {count:,}")

    print(f"\n{'='*60}")
    print("Done! Filtered SDWIS data ready in data/raw/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()