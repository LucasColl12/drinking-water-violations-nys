"""
03_geocode_systems.py
=====================
Builds a geocoded water system file by combining:
    1. System details from ECHO (county, owner type, pop served, etc.)
    2. Zip codes from SDWIS water system data
    3. Approximate lat/lon from Census ZCTA (Zip Code) centroid gazetteer

The ECHO Drinking Water API does not return lat/lon, so we use zip code
centroids as the best available proxy for system locations.

Requires:
    data/raw/water_system_ny.csv (from 01_fetch_sdwis.py)
Outputs:
    data/raw/system_locations_ny.csv

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
ECHO_QID = "https://echodata.epa.gov/echo/sdw_rest_services.get_qid"

# ---------------------------------------------------------------------------
# Step 1: Fetch system details from ECHO
# ---------------------------------------------------------------------------

def fetch_echo_systems(state: str = "NY", page_size: int = 1000) -> pd.DataFrame:
    """Fetch community water system records from ECHO."""

    print("  Submitting query to ECHO ...")
    params = {
        "output": "JSON",
        "p_st": state,
        "p_act": "Y",
        "responseset": page_size,
    }
    resp = requests.get(ECHO_BASE, params=params, timeout=120)
    resp.raise_for_status()
    results = resp.json().get("Results", {})
    qid = results.get("QueryID")
    total = int(results.get("QueryRows", 0))
    print(f"  Query ID: {qid} | Total systems: {total:,}")

    all_records = []
    page = 1
    while len(all_records) < total:
        print(f"    → Fetching page {page} ...")
        qid_params = {
            "output": "JSON", "qid": qid,
            "pageno": page, "responseset": page_size,
        }
        r = requests.get(ECHO_QID, params=qid_params, timeout=120)
        r.raise_for_status()
        facilities = r.json().get("Results", {}).get("WaterSystems", [])
        if not facilities:
            break

        for fac in facilities:
            all_records.append({
                "pwsid": fac.get("PWSId", ""),
                "pws_name": fac.get("PWSName", ""),
                "county_served": fac.get("CountiesServed", ""),
                "city_served": fac.get("CitiesServed", ""),
                "zip_served": fac.get("ZipCodesServed", ""),
                "fips_codes": fac.get("FIPSCodes", ""),
                "population_served": fac.get("PopulationServedCount", ""),
                "source_water_type": fac.get("PrimarySourceDesc", ""),
                "pws_type": fac.get("PWSTypeCode", ""),
                "owner_type": fac.get("OwnerDesc", ""),
                "serious_violator": fac.get("SeriousViolator", ""),
                "registry_id": fac.get("RegistryID", ""),
            })

        print(f"      got {len(facilities):,} (total: {len(all_records):,})")
        page += 1
        time.sleep(0.5)

    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Step 2: Download ZCTA gazetteer for zip code centroids
# ---------------------------------------------------------------------------

def download_zcta_centroids() -> pd.DataFrame:
    """Download ZCTA shapefile from TIGER/Line and compute centroids."""

    outpath = os.path.join(RAW_DIR, "zcta_centroids.csv")
    if os.path.exists(outpath):
        print(f"  ZCTA centroids already cached: {outpath}")
        return pd.read_csv(outpath, dtype={"zip_code": str})

    import geopandas as gpd

    zcta_url = (
        "https://www2.census.gov/geo/tiger/TIGER2022/"
        "ZCTA520/tl_2022_us_zcta520.zip"
    )
    print(f"  Downloading ZCTA shapefile (this is ~800MB, may take a few minutes) ...")
    gdf = gpd.read_file(zcta_url).to_crs(epsg=4326)

    # Compute centroid lat/lon for each ZCTA
    gdf["latitude"] = gdf.geometry.centroid.y
    gdf["longitude"] = gdf.geometry.centroid.x
    gdf = gdf.rename(columns={"ZCTA5CE20": "zip_code"})

    result = gdf[["zip_code", "latitude", "longitude"]].copy()
    result.to_csv(outpath, index=False)
    print(f"  ✓ Computed centroids for {len(result):,} zip codes → {outpath}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # --- ECHO system data ---
    print("=" * 60)
    print("Step 1: Fetch system details from ECHO")
    print("=" * 60)
    echo_df = fetch_echo_systems(state="NY")
    print(f"  ECHO systems: {len(echo_df):,}")

    # --- Zip codes from SDWIS ---
    print(f"\n{'='*60}")
    print("Step 2: Merge zip codes from SDWIS")
    print("=" * 60)
    sdwis_path = os.path.join(RAW_DIR, "water_system_ny.csv")
    sdwis = pd.read_csv(sdwis_path, dtype=str)

    # Get the primary zip code per system from SDWIS
    zip_lookup = sdwis[["pwsid", "zip_code"]].drop_duplicates("pwsid")
    # Clean zip to 5 digits
    zip_lookup["zip_code"] = zip_lookup["zip_code"].str.strip().str[:5]

    echo_df = echo_df.merge(zip_lookup, on="pwsid", how="left")
    zip_match = echo_df["zip_code"].notna().sum()
    print(f"  Matched zip codes: {zip_match:,}/{len(echo_df):,}")

    # --- ZCTA centroids ---
    print(f"\n{'='*60}")
    print("Step 3: Geocode via ZCTA centroids")
    print("=" * 60)
    centroids = download_zcta_centroids()

    echo_df = echo_df.merge(centroids, on="zip_code", how="left")

    geocoded = echo_df["latitude"].notna().sum()
    total = len(echo_df)
    print(f"\n  Geocoding coverage: {geocoded:,}/{total:,} ({geocoded/total*100:.1f}%)")

    # --- Clean up ---
    echo_df["population_served"] = pd.to_numeric(
        echo_df["population_served"], errors="coerce"
    )

    outpath = os.path.join(RAW_DIR, "system_locations_ny.csv")
    echo_df.to_csv(outpath, index=False)
    print(f"  ✓ Saved {len(echo_df):,} systems → {outpath}")


if __name__ == "__main__":
    main()