"""
03_geocode_systems.py
=====================
Retrieves latitude/longitude coordinates for New York State community
water systems.

Strategy:
    1. Check if user has downloaded the ECHO Exporter file (which contains
       lat/lon for all EPA-regulated facilities). This is the most reliable
       source. Download from: https://echo.epa.gov/tools/data-downloads
       Look for "ECHO Exporter" → ECHO_EXPORTER.csv

    2. If no Exporter file, use the ECHO SDW get_download endpoint, which
       returns a CSV with coordinates (unlike get_qid which omits them).

    3. If the API fails, fall back to county-level centroid geocoding
       using the county_served field from the water systems data.

Requires: data/raw/water_system_ny.csv (from 01_load_sdwis.py)
Outputs:  data/raw/system_locations_ny.csv

Usage:
    python scripts/03_geocode_systems.py
"""

import os
import io
import time
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ECHO SDW REST endpoints
ECHO_GET_SYSTEMS = "https://echodata.epa.gov/echo/sdw_rest_services.get_systems"
ECHO_GET_DOWNLOAD = "https://echodata.epa.gov/echo/sdw_rest_services.get_download"

# Approximate county centroids for NY counties (fallback geocoding)
# These are rough centroids — good enough for tract-level spatial join
NY_COUNTY_CENTROIDS = {
    "ALBANY": (42.65, -73.75), "ALLEGANY": (42.25, -78.03),
    "BRONX": (40.85, -73.87), "BROOME": (42.16, -75.82),
    "CATTARAUGUS": (42.25, -78.68), "CAYUGA": (42.93, -76.57),
    "CHAUTAUQUA": (42.30, -79.42), "CHEMUNG": (42.14, -76.77),
    "CHENANGO": (42.49, -75.61), "CLINTON": (44.75, -73.68),
    "COLUMBIA": (42.25, -73.63), "CORTLAND": (42.60, -76.07),
    "DELAWARE": (42.20, -74.97), "DUTCHESS": (41.77, -73.74),
    "ERIE": (42.76, -78.78), "ESSEX": (44.12, -73.77),
    "FRANKLIN": (44.59, -74.30), "FULTON": (43.11, -74.42),
    "GENESEE": (43.00, -78.19), "GREENE": (42.28, -74.12),
    "HAMILTON": (43.66, -74.50), "HERKIMER": (43.42, -74.96),
    "JEFFERSON": (44.00, -75.85), "KINGS": (40.63, -73.95),
    "LEWIS": (43.78, -75.45), "LIVINGSTON": (42.73, -77.76),
    "MADISON": (42.91, -75.67), "MONROE": (43.08, -77.68),
    "MONTGOMERY": (42.90, -74.44), "NASSAU": (40.73, -73.59),
    "NEW YORK": (40.78, -73.97), "NIAGARA": (43.17, -78.82),
    "ONEIDA": (43.24, -75.46), "ONONDAGA": (43.01, -76.19),
    "ONTARIO": (42.85, -77.30), "ORANGE": (41.40, -74.31),
    "ORLEANS": (43.25, -78.23), "OSWEGO": (43.46, -76.21),
    "OTSEGO": (42.63, -75.03), "PUTNAM": (41.43, -73.75),
    "QUEENS": (40.71, -73.83), "RENSSELAER": (42.71, -73.51),
    "RICHMOND": (40.58, -74.15), "ROCKLAND": (41.15, -74.02),
    "SARATOGA": (43.10, -73.86), "SCHENECTADY": (42.82, -74.06),
    "SCHOHARIE": (42.59, -74.44), "SCHUYLER": (42.39, -76.87),
    "SENECA": (42.78, -76.82), "ST. LAWRENCE": (44.50, -75.09),
    "STEUBEN": (42.27, -77.38), "SUFFOLK": (40.88, -72.80),
    "SULLIVAN": (41.72, -74.78), "TIOGA": (42.17, -76.31),
    "TOMPKINS": (42.45, -76.47), "ULSTER": (41.89, -74.26),
    "WARREN": (43.56, -73.84), "WASHINGTON": (43.32, -73.43),
    "WAYNE": (43.22, -77.06), "WESTCHESTER": (41.12, -73.76),
    "WYOMING": (42.70, -78.23), "YATES": (42.63, -77.11),
    # Variant names
    "SAINT LAWRENCE": (44.50, -75.09), "ST LAWRENCE": (44.50, -75.09),
}


# ---------------------------------------------------------------------------
# Approach 1: ECHO Exporter file (manual download)
# ---------------------------------------------------------------------------

def try_echo_exporter() -> pd.DataFrame | None:
    """
    Look for a manually downloaded ECHO Exporter CSV which contains
    lat/lon for all EPA-regulated facilities.
    """
    print("  Checking for ECHO Exporter file ...")

    for f in os.listdir(RAW_DIR):
        if "EXPORTER" in f.upper() and f.upper().endswith(".CSV"):
            path = os.path.join(RAW_DIR, f)
            print(f"  Found: {f}")

            # This file is huge — read only the columns we need
            # and filter to SDWA facilities in NY
            try:
                cols_to_read = None
                # First peek at columns
                peek = pd.read_csv(path, nrows=0, dtype=str, low_memory=False)
                all_cols = [c.upper() for c in peek.columns]
                print(f"    Columns available: {len(all_cols)}")

                # Map column names (case-insensitive)
                col_map = {}
                for target, candidates in {
                    "registry_id": ["REGISTRYID", "REGISTRY_ID", "FRS_ID"],
                    "pwsid": ["SDWAIDS", "SDWA_IDS", "PWSID"],
                    "fac_name": ["FACNAME", "FAC_NAME"],
                    "latitude": ["FACLAT", "FAC_LAT", "LATITUDE"],
                    "longitude": ["FACLONG", "FAC_LONG", "LONGITUDE"],
                    "fac_county": ["FACSTDCOUNTYNAME", "FAC_COUNTY", "COUNTY"],
                    "fac_state": ["FACSTATE", "FAC_STATE", "STATE"],
                }.items():
                    for cand in candidates:
                        if cand in all_cols:
                            # Get the original-case column name
                            idx = all_cols.index(cand)
                            col_map[target] = peek.columns[idx]
                            break

                if "latitude" not in col_map or "pwsid" not in col_map:
                    print(f"    Missing required columns. Found: {col_map}")
                    continue

                usecols = list(col_map.values())
                print(f"    Reading columns: {usecols}")

                df = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
                df = df.rename(columns={v: k for k, v in col_map.items()})

                # Filter to NY SDWA facilities
                if "fac_state" in df.columns:
                    df = df[df["fac_state"].str.upper() == "NY"]
                if "pwsid" in df.columns:
                    df = df[df["pwsid"].str.startswith("NY", na=False)]

                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

                geocoded = df["latitude"].notna().sum()
                print(f"    NY facilities: {len(df):,}, geocoded: {geocoded:,}")

                if geocoded > 0:
                    return df

            except Exception as e:
                print(f"    Error reading: {e}")
                continue

    print("  No ECHO Exporter file found.")
    return None


# ---------------------------------------------------------------------------
# Approach 2: ECHO SDW get_download CSV endpoint
# ---------------------------------------------------------------------------

def try_echo_download_csv() -> pd.DataFrame | None:
    """
    Use the ECHO SDW get_download endpoint which returns a CSV
    (unlike get_qid which returns JSON without coordinates).
    """
    print("\n  Trying ECHO SDW get_download endpoint ...")

    # Step 1: Submit query to get a QID
    query_params = {
        "output": "JSON",
        "p_st": "NY",
        "p_act": "Y",
        "p_pws_type": "CWS",
        "responseset": 1,
    }

    try:
        resp = requests.get(ECHO_GET_SYSTEMS, params=query_params, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        qid = result.get("Results", {}).get("QueryID")
        total = result.get("Results", {}).get("QueryRows", 0)
        print(f"    QID: {qid} | Total systems: {total}")
    except Exception as e:
        print(f"    Query failed: {e}")
        return None

    if not qid:
        print("    No QID returned.")
        return None

    # Step 2: Use get_download to get CSV with all columns including lat/lon
    download_params = {
        "qid": qid,
        "output": "CSV",
    }

    try:
        print(f"    Requesting CSV download ...")
        resp = requests.get(ECHO_GET_DOWNLOAD, params=download_params, timeout=300)
        resp.raise_for_status()

        # Parse CSV from response
        df = pd.read_csv(io.StringIO(resp.text), dtype=str, low_memory=False)
        print(f"    Got {len(df):,} rows, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns)[:15]}...")

        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Find lat/lon columns
        lat_col = None
        lon_col = None
        for c in df.columns:
            if "lat" in c and lat_col is None:
                lat_col = c
            if ("long" in c or "lng" in c or "lon" in c) and lon_col is None:
                lon_col = c

        if lat_col and lon_col:
            print(f"    Lat column: {lat_col}, Lon column: {lon_col}")
            df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
            df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
            geocoded = df["latitude"].notna().sum()
            print(f"    Geocoded: {geocoded:,}/{len(df):,}")
            if geocoded > 0:
                return df

        print("    No coordinate columns found in download.")
        print(f"    Available columns: {list(df.columns)}")

    except Exception as e:
        print(f"    Download failed: {e}")

    return None


# ---------------------------------------------------------------------------
# Approach 3: County centroid fallback
# ---------------------------------------------------------------------------

def geocode_by_county(systems: pd.DataFrame) -> pd.DataFrame:
    """
    Assign approximate lat/lon based on county centroids.
    This is a rough fallback but sufficient for tract-level spatial joins
    (the point will land in *some* tract in the county).
    """
    print("\n  Applying county centroid fallback geocoding ...")

    df = systems.copy()

    # Find county column
    county_col = None
    for c in df.columns:
        if "county" in c.lower():
            county_col = c
            break

    if county_col is None:
        print("    No county column found!")
        return df

    df["_county_upper"] = df[county_col].astype(str).str.upper().str.strip()

    matched = 0
    lats = []
    lons = []

    for _, row in df.iterrows():
        county = row["_county_upper"]
        if county in NY_COUNTY_CENTROIDS:
            lat, lon = NY_COUNTY_CENTROIDS[county]
            lats.append(lat)
            lons.append(lon)
            matched += 1
        else:
            lats.append(np.nan)
            lons.append(np.nan)

    df["latitude"] = lats
    df["longitude"] = lons
    df = df.drop(columns=["_county_upper"])

    print(f"    Matched {matched:,}/{len(df):,} systems to county centroids")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Geocoding NY water systems")
    print("=" * 60)

    # Load the water systems file for reference
    systems_path = os.path.join(RAW_DIR, "water_system_ny.csv")
    if not os.path.exists(systems_path):
        print(f"  ERROR: {systems_path} not found. Run 01_load_sdwis.py first.")
        return

    systems = pd.read_csv(systems_path, dtype=str)
    systems.columns = [c.lower() for c in systems.columns]
    print(f"  Loaded {len(systems):,} water systems\n")

    # --- Try approaches in order ---
    locations = None

    # Approach 1: ECHO Exporter
    exporter_df = try_echo_exporter()
    if exporter_df is not None and len(exporter_df) > 0:
        locations = exporter_df
        print(f"\n  ✓ Using ECHO Exporter data ({len(locations):,} facilities)")

    # Approach 2: ECHO get_download CSV
    if locations is None:
        download_df = try_echo_download_csv()
        if download_df is not None and len(download_df) > 0:
            locations = download_df
            print(f"\n  ✓ Using ECHO download CSV ({len(locations):,} systems)")

    # Approach 3: County centroid fallback
    if locations is None:
        print("\n  ⚠ API approaches failed. Using county centroid fallback.")
        locations = geocode_by_county(systems)
        print(f"  ✓ Using county centroids ({len(locations):,} systems)")

    # Standardize columns
    locations.columns = [c.lower() for c in locations.columns]

    # Ensure we have the key columns
    for col in ["latitude", "longitude"]:
        if col not in locations.columns:
            locations[col] = np.nan
        else:
            locations[col] = pd.to_numeric(locations[col], errors="coerce")

    if "population_served" not in locations.columns:
        # Try alternate names
        for cand in ["population_served_count", "popserved", "currsvdpop",
                      "populationservedcount"]:
            if cand in locations.columns:
                locations = locations.rename(columns={cand: "population_served"})
                break

    if "population_served" in locations.columns:
        locations["population_served"] = pd.to_numeric(
            locations["population_served"], errors="coerce"
        )

    # Report
    geocoded = locations["latitude"].notna().sum()
    total = len(locations)
    print(f"\n  Final geocoding: {geocoded:,}/{total:,} systems "
          f"({geocoded/total*100:.1f}%)")

    # Save
    outpath = os.path.join(RAW_DIR, "system_locations_ny.csv")
    locations.to_csv(outpath, index=False)
    print(f"  ✓ Saved {len(locations):,} systems → {outpath}")

    if geocoded == 0:
        print("\n  ═══════════════════════════════════════════════════")
        print("  STILL 0% GEOCODED — RECOMMENDED FIX:")
        print("  Download the ECHO Exporter file from:")
        print("  https://echo.epa.gov/tools/data-downloads")
        print("  Place ECHO_EXPORTER.csv in data/raw/ and re-run.")
        print("  ═══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()