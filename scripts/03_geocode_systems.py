"""
03_geocode_systems.py
=====================
Retrieves latitude/longitude coordinates for New York State public water
systems using EPA's ECHO (Enforcement and Compliance History Online) API.

BUG FIXES from original:
    - Page 1 of ECHO response has a different structure (metadata) —
      system records may be absent or keyed differently. Now handled.
    - Tries multiple field names for lat/lon (ECHO has changed these
      across API versions).
    - Adds diagnostic output to catch 0% geocoding issues.

API docs: https://echo.epa.gov/tools/web-services

Requires: data/raw/water_system_ny.csv (from 01_fetch_sdwis.py)
Outputs:  data/raw/system_locations_ny.csv

Usage:
    python scripts/03_geocode_systems.py
"""

import os
import time
import json
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ECHO_BASE = "https://echodata.epa.gov/echo/sdw_rest_services.get_systems"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fetch_echo_systems(state: str = "NY", page_size: int = 1000) -> pd.DataFrame:
    """
    Query ECHO's Safe Drinking Water REST service for all CWS in a state.

    The ECHO API has a two-step process:
        1. Submit query → get a QID (query ID)
        2. Page through results using the QID

    The first response page often contains metadata but may have records
    in a different location than subsequent pages.
    """
    print("  Submitting query to ECHO ...")
    query_params = {
        "output": "JSON",
        "p_st": state,
        "p_act": "Y",
        "p_pws_type": "CWS",
        "responseset": page_size,
    }

    response = requests.get(ECHO_BASE, params=query_params, timeout=120)
    response.raise_for_status()
    result = response.json()

    results = result.get("Results", {})
    qid = results.get("QueryID")
    total_count = int(results.get("QueryRows", 0))
    print(f"  Query ID: {qid} | Total systems: {total_count:,}")

    # --- Diagnostic: show what keys are in the first response ---
    print(f"  First response keys: {list(results.keys())[:15]}")

    # Parse first page — try multiple possible record locations
    all_records = _parse_echo_page(results, verbose=True)
    print(f"    → Page 1: {len(all_records)} records")

    # Page through remaining results
    if total_count > 0:
        page = 2
        while len(all_records) < total_count:
            print(f"    → Fetching page {page} ...")
            page_url = "https://echodata.epa.gov/echo/sdw_rest_services.get_qid"
            page_params = {
                "output": "JSON",
                "qid": qid,
                "pageno": page,
                "responseset": page_size,
            }
            try:
                resp = requests.get(page_url, params=page_params, timeout=120)
                resp.raise_for_status()
                page_result = resp.json().get("Results", {})
                page_records = _parse_echo_page(page_result)
            except Exception as e:
                print(f"      Error: {e}")
                break

            if not page_records:
                print(f"      No records returned, stopping.")
                break

            all_records.extend(page_records)
            print(f"      got {len(page_records)} (total: {len(all_records):,})")
            page += 1
            time.sleep(0.5)

            # Safety limit
            if page > 50:
                print("      Hit page limit (50), stopping.")
                break

    df = pd.DataFrame(all_records)
    return df


def _parse_echo_page(results: dict, verbose: bool = False) -> list[dict]:
    """
    Extract water system records from an ECHO API response page.

    ECHO has used different JSON structures over time. This function
    tries multiple possible locations for the records and field names.
    """
    # Try multiple possible keys for the facility list
    facilities = None
    for key in ["WaterSystems", "Facilities", "MapData", "CWAFacilities",
                "SDWAFacilities", "SdwSystems"]:
        if key in results and isinstance(results[key], list):
            facilities = results[key]
            if verbose:
                print(f"    Found records under key '{key}': {len(facilities)} items")
            break

    if facilities is None:
        # Maybe the records are directly in a list at the top level
        if verbose:
            print(f"    No recognized facility key found.")
            print(f"    Available keys: {list(results.keys())}")
            # Show a sample of any list-type values
            for k, v in results.items():
                if isinstance(v, list) and len(v) > 0:
                    print(f"    Key '{k}' is a list with {len(v)} items")
                    if isinstance(v[0], dict):
                        print(f"      Sample keys: {list(v[0].keys())[:10]}")
                        facilities = v
                        break
        if facilities is None:
            return []

    records = []
    for i, fac in enumerate(facilities):
        if not isinstance(fac, dict):
            continue

        # Show field names from first record for debugging
        if verbose and i == 0:
            print(f"    Sample record keys: {list(fac.keys())[:15]}")
            # Show lat/lon candidates
            for k, v in fac.items():
                k_lower = k.lower()
                if any(term in k_lower for term in ["lat", "lon", "lng", "coord"]):
                    print(f"      Coordinate field: {k} = {v}")

        # Try multiple field name patterns for each attribute
        lat = _get_first(fac, ["RegistryLatitude", "Latitude", "Lat",
                                "FacLat", "latitude", "lat",
                                "CentLat", "FacDerivedLat"])
        lon = _get_first(fac, ["RegistryLongitude", "Longitude", "Lng",
                                "FacLong", "longitude", "lng", "lon",
                                "CentLon", "FacDerivedLong"])

        records.append({
            "pwsid": _get_first(fac, ["PWSId", "PwsId", "pwsid",
                                       "SourceID", "FacId"]) or "",
            "pws_name": _get_first(fac, ["PWSName", "PwsName", "Name",
                                          "FacName", "pws_name"]) or "",
            "latitude": lat,
            "longitude": lon,
            "county_served": _get_first(fac, ["CountyServed", "County",
                                               "FacCounty"]) or "",
            "city_served": _get_first(fac, ["CityServed", "City",
                                             "FacCity"]) or "",
            "population_served": _get_first(fac, ["PopulationServedCount",
                                                    "PopulationServed",
                                                    "CurrSvdPop"]) or "",
            "source_water_type": _get_first(fac, ["SourceWaterType",
                                                    "SrcWaterType"]) or "",
            "pws_type": _get_first(fac, ["PWSTypeCode", "PwsType"]) or "",
            "owner_type": _get_first(fac, ["OwnerTypeDesc", "OwnerType"]) or "",
        })

    return records


def _get_first(d: dict, keys: list) -> str | None:
    """Return the value of the first key found in dict, or None."""
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return d[k]
    return None


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Geocoding NY water systems via ECHO API")
    print("=" * 60)

    df = fetch_echo_systems(state="NY")

    if df.empty:
        print("WARNING: No data returned from ECHO API.")
        return

    # Clean coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["population_served"] = pd.to_numeric(df["population_served"], errors="coerce")

    # Geocoding diagnostics
    geocoded = df["latitude"].notna().sum()
    total = len(df)
    print(f"\n  Geocoding coverage: {geocoded:,}/{total:,} systems "
          f"({geocoded/total*100:.1f}%)")

    if geocoded == 0:
        print("\n  ⚠ WARNING: 0% geocoding rate!")
        print("  Showing sample lat/lon values from raw data:")
        print(df[["pwsid", "pws_name", "latitude", "longitude"]].head(10).to_string())
        print("\n  This likely means the ECHO API response format has changed.")
        print("  Check the 'Sample record keys' output above for coordinate field names.")

    outpath = os.path.join(RAW_DIR, "system_locations_ny.csv")
    df.to_csv(outpath, index=False)
    print(f"  ✓ Saved {len(df):,} systems → {outpath}")


if __name__ == "__main__":
    main()