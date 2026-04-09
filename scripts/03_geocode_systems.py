"""
03_geocode_systems.py
=====================
Retrieves latitude/longitude coordinates for New York State public water
systems using EPA's ECHO (Enforcement and Compliance History Online) API.

The ECHO Drinking Water facility search returns geocoded locations from
EPA's Facility Registry Service (FRS). We query in batches by county
to stay within API limits.

API docs: https://echo.epa.gov/tools/web-services

Requires: data/raw/water_system_ny.csv (from 01_fetch_sdwis.py)
Outputs:  data/raw/system_locations_ny.csv

Usage:
    python scripts/03_geocode_systems.py
"""

import os
import time
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
    Query ECHO's Safe Drinking Water REST service for all water systems
    in a state, returning PWSID, name, lat, lon, county, pop served.

    ECHO paginates differently than Envirofacts — it uses a QID (query ID)
    that you get from the first request, then page through results.
    """

    # Step 1: Submit the query to get a QID
    print("  Submitting query to ECHO ...")
    query_params = {
        "output": "JSON",
        "p_st": state,
        "p_act": "Y",             # Active systems only
        "p_pws_type": "CWS",      # Community water systems (residential)
        "responseset": page_size,
    }

    response = requests.get(ECHO_BASE, params=query_params, timeout=120)
    response.raise_for_status()
    result = response.json()

    # The ECHO API nests data under 'Results'
    results = result.get("Results", {})
    qid = results.get("QueryID")
    total_count = int(results.get("QueryRows", 0))
    print(f"  Query ID: {qid} | Total systems: {total_count:,}")

    # Parse first page of results
    all_records = _parse_echo_page(results)
    print(f"    → Page 1: {len(all_records)} records")

    # Step 2: Page through remaining results
    if total_count > page_size:
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
            resp = requests.get(page_url, params=page_params, timeout=120)
            resp.raise_for_status()
            page_result = resp.json().get("Results", {})
            page_records = _parse_echo_page(page_result)

            if not page_records:
                break

            all_records.extend(page_records)
            print(f"      got {len(page_records)} (total: {len(all_records):,})")
            page += 1
            time.sleep(0.5)

    df = pd.DataFrame(all_records)
    return df


def _parse_echo_page(results: dict) -> list[dict]:
    """Extract water system records from an ECHO API response page."""
    facilities = results.get("WaterSystems", results.get("Facilities", []))
    records = []
    for fac in facilities:
        records.append({
            "pwsid": fac.get("PWSId", ""),
            "pws_name": fac.get("PWSName", fac.get("Name", "")),
            "latitude": fac.get("Latitude", fac.get("Lat", None)),
            "longitude": fac.get("Longitude", fac.get("Lng", None)),
            "county_served": fac.get("CountyServed", ""),
            "city_served": fac.get("CityServed", ""),
            "population_served": fac.get("PopulationServedCount", ""),
            "source_water_type": fac.get("SourceWaterType", ""),
            "pws_type": fac.get("PWSTypeCode", ""),
            "owner_type": fac.get("OwnerTypeDesc", ""),
        })
    return records


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("Geocoding NY water systems via ECHO API")
    print("=" * 60)

    df = fetch_echo_systems(state="NY")

    if df.empty:
        print("WARNING: No data returned from ECHO API.")
        print("Falling back to county-level matching from SDWIS data.")
        print("You can still proceed — spatial join will use county names.")
        return

    # Clean up coordinate columns
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["population_served"] = pd.to_numeric(df["population_served"], errors="coerce")

    # Report geocoding coverage
    geocoded = df["latitude"].notna().sum()
    total = len(df)
    print(f"\n  Geocoding coverage: {geocoded:,}/{total:,} systems "
          f"({geocoded/total*100:.1f}%)")

    outpath = os.path.join(RAW_DIR, "system_locations_ny.csv")
    df.to_csv(outpath, index=False)
    print(f"  ✓ Saved {len(df):,} systems → {outpath}")


if __name__ == "__main__":
    main()
