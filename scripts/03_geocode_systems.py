"""
03_geocode_systems.py
=====================
Retrieves latitude/longitude coordinates for New York State public water
systems using EPA's ECHO Drinking Water REST API.

If the ECHO API doesn't provide coordinates, falls back to zip code
centroids from the SDWIS water system data.

Requires: data/raw/water_system_ny.csv (from 01_fetch_sdwis.py)
Outputs:  data/raw/system_locations_ny.csv

Usage:
    python scripts/03_geocode_systems.py
"""

import os
import json
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ECHO_BASE = "https://echodata.epa.gov/echo/sdw_rest_services.get_systems"
ECHO_QID = "https://echodata.epa.gov/echo/sdw_rest_services.get_qid"

# ---------------------------------------------------------------------------
# ECHO API
# ---------------------------------------------------------------------------

def fetch_echo_systems(state: str = "NY", page_size: int = 1000) -> pd.DataFrame:
    """
    Query ECHO's Safe Drinking Water REST service for all community water
    systems in a state. Dynamically detects field names from the response.
    """

    # Step 1: Submit query
    print("  Submitting query to ECHO ...")
    query_params = {
        "output": "JSON",
        "p_st": state,
        "p_act": "Y",
        "responseset": page_size,
    }

    response = requests.get(ECHO_BASE, params=query_params, timeout=120)
    response.raise_for_status()
    result = response.json()

    results = result.get("Results", {})
    qid = results.get("QueryID")
    total_count = int(results.get("QueryRows", 0))
    print(f"  Query ID: {qid} | Total systems: {total_count:,}")

    # Step 2: Get first page via QID to see actual field names
    print("  Fetching page 1 via QID ...")
    page_params = {
        "output": "JSON",
        "qid": qid,
        "pageno": 1,
        "responseset": page_size,
    }
    resp = requests.get(ECHO_QID, params=page_params, timeout=120)
    resp.raise_for_status()
    page_result = resp.json().get("Results", {})

    # Discover which key holds the facility list
    facility_key = None
    for key in page_result:
        if isinstance(page_result[key], list) and len(page_result[key]) > 0:
            if isinstance(page_result[key][0], dict):
                facility_key = key
                break

    if not facility_key:
        print("  ✗ Could not find facility list in ECHO response.")
        print(f"  Response keys: {list(page_result.keys())}")
        return pd.DataFrame()

    print(f"  Found facility data under key: '{facility_key}'")
    sample = page_result[facility_key][0]
    print(f"  Sample record keys: {list(sample.keys())}")

    # Discover lat/lon field names
    lat_field = None
    lon_field = None
    for key in sample.keys():
        kl = key.lower()
        if kl in ("latitude", "faclat", "lat", "registrylat", "facilitylatitude"):
            lat_field = key
        if kl in ("longitude", "faclong", "lng", "lon", "registrylong",
                   "facilitylongitude", "faclon"):
            lon_field = key

    if lat_field and lon_field:
        print(f"  Lat/Lon fields: {lat_field}, {lon_field}")
    else:
        print(f"  ⚠ Could not auto-detect lat/lon fields.")
        print(f"  All keys: {list(sample.keys())}")

    # Discover PWSID field
    pwsid_field = None
    for key in sample.keys():
        if key.lower() in ("pwsid", "pwsid0", "pws_id"):
            pwsid_field = key
            break
    if not pwsid_field:
        # Take the first field that looks like a PWSID (starts with 2-letter code)
        for key, val in sample.items():
            if isinstance(val, str) and len(val) == 9 and val[:2].isalpha():
                pwsid_field = key
                break

    print(f"  PWSID field: {pwsid_field}")

    # Step 3: Parse all pages
    all_records = _parse_page(page_result[facility_key], pwsid_field,
                              lat_field, lon_field)
    print(f"    → Page 1: {len(all_records):,} records")

    page = 2
    while len(all_records) < total_count:
        print(f"    → Fetching page {page} ...")
        page_params["pageno"] = page
        resp = requests.get(ECHO_QID, params=page_params, timeout=120)
        resp.raise_for_status()
        pr = resp.json().get("Results", {})
        facilities = pr.get(facility_key, [])

        if not facilities:
            break

        records = _parse_page(facilities, pwsid_field, lat_field, lon_field)
        all_records.extend(records)
        print(f"      got {len(records):,} (total: {len(all_records):,})")
        page += 1
        time.sleep(0.5)

    return pd.DataFrame(all_records)


def _parse_page(facilities: list, pwsid_field, lat_field, lon_field) -> list[dict]:
    """Extract records from a list of ECHO facility dicts."""
    records = []
    for fac in facilities:
        rec = {
            "pwsid": fac.get(pwsid_field, "") if pwsid_field else "",
        }

        # Grab all potentially useful fields dynamically
        for key, val in fac.items():
            clean_key = key.lower().replace(" ", "_")
            if clean_key not in rec:
                rec[clean_key] = val

        # Ensure we have canonical lat/lon columns
        if lat_field:
            rec["latitude"] = fac.get(lat_field)
        if lon_field:
            rec["longitude"] = fac.get(lon_field)

        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Geocoding NY water systems via ECHO API")
    print("=" * 60)

    df = fetch_echo_systems(state="NY")

    if df.empty:
        print("✗ No data from ECHO. Cannot proceed with geocoding.")
        return

    # Clean coordinate columns
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Report geocoding coverage
    if "latitude" in df.columns:
        geocoded = df["latitude"].notna().sum()
    else:
        geocoded = 0
    total = len(df)
    print(f"\n  Geocoding coverage: {geocoded:,}/{total:,} ({geocoded/total*100:.1f}%)")

    if geocoded == 0:
        print("\n  ⚠ No lat/lon from ECHO. Attempting zip code fallback ...")
        df = _fallback_zip_geocode(df)

    outpath = os.path.join(RAW_DIR, "system_locations_ny.csv")
    df.to_csv(outpath, index=False)
    print(f"\n  ✓ Saved {len(df):,} systems → {outpath}")


def _fallback_zip_geocode(df: pd.DataFrame) -> pd.DataFrame:
    """
    If ECHO didn't give us coordinates, try to pull zip codes from the
    SDWIS water_system data and geocode via zip centroid lookup.
    """
    sdwis_path = os.path.join(RAW_DIR, "water_system_ny.csv")
    if not os.path.exists(sdwis_path):
        print("    No SDWIS data available for zip fallback.")
        return df

    sdwis = pd.read_csv(sdwis_path, dtype=str)

    # Merge zip codes from SDWIS onto ECHO data
    if "zip_code" in sdwis.columns and "pwsid" in sdwis.columns:
        zip_lookup = sdwis[["pwsid", "zip_code", "city_name", "state_code"]].drop_duplicates("pwsid")
        df = df.merge(zip_lookup, on="pwsid", how="left", suffixes=("", "_sdwis"))
        zips_found = df["zip_code"].notna().sum()
        print(f"    Matched {zips_found:,}/{len(df):,} systems to zip codes from SDWIS.")
        print("    (Zip codes will be used for county-level analysis as fallback)")
    else:
        print(f"    SDWIS columns: {list(sdwis.columns)}")
        print("    Could not find zip_code column.")

    return df


if __name__ == "__main__":
    main()
