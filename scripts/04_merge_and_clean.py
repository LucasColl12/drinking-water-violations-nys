"""
04_merge_and_clean.py
=====================
Merges SDWIS violations, ECHO system details (with zip-code-derived
coordinates), and Census ACS tract demographics into one analytical dataset.

Each geocoded water system is spatially joined to the Census tract it falls
within, linking system-level violation data to tract-level demographics.

Requires:
    data/raw/water_system_ny.csv       (from 01_fetch_sdwis.py)
    data/raw/violation_ny.csv          (from 01_fetch_sdwis.py)
    data/raw/system_locations_ny.csv   (from 03_geocode_systems.py)
    data/raw/census_tracts_ny.csv      (from 02_fetch_census.py)
    data/raw/tl_2022_36_tract.zip      (from 02_fetch_census.py)

Outputs:
    output/analytical_dataset.csv

Usage:
    python scripts/04_merge_and_clean.py
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def classify_system_size(pop: float) -> str:
    """EPA standard size categories."""
    if pd.isna(pop) or pop <= 0:
        return "Unknown"
    elif pop <= 500:
        return "Very Small"
    elif pop <= 3300:
        return "Small"
    elif pop <= 10000:
        return "Medium"
    elif pop <= 100000:
        return "Large"
    else:
        return "Very Large"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load data ---
    print("Loading raw datasets ...")
    violations = pd.read_csv(
        os.path.join(RAW_DIR, "violation_ny.csv"), dtype={"pwsid": str}
    )
    locations = pd.read_csv(
        os.path.join(RAW_DIR, "system_locations_ny.csv"), dtype={"pwsid": str}
    )
    census = pd.read_csv(
        os.path.join(RAW_DIR, "census_tracts_ny.csv"),
        dtype={"GEOID": str, "county_fips": str},
    )

    print(f"  Violations:     {len(violations):,}")
    print(f"  Systems:        {len(locations):,}")
    print(f"  Census tracts:  {len(census):,}")

    # --- Load tract shapefile ---
    print("\nLoading TIGER/Line tract boundaries ...")
    tiger_path = os.path.join(RAW_DIR, "tl_2022_36_tract.zip")
    tracts_gdf = gpd.read_file(f"zip://{tiger_path}").to_crs(epsg=4326)
    print(f"  Loaded {len(tracts_gdf):,} tract polygons")

    # --- Aggregate violations per system ---
    print("\nAggregating violations by system ...")

    # Use actual column names from the API
    viol_counts = (
        violations.groupby("pwsid")
        .agg(
            total_violations=("pwsid", "size"),
            health_violations=(
                "is_health_based_ind",
                lambda x: (x.astype(str).str.upper() == "Y").sum()
            ),
            monitoring_violations=(
                "violation_category_code",
                lambda x: x.isin(["MR", "MON", "Other"]).sum()
            ),
        )
        .reset_index()
    )
    print(f"  Systems with violations: {len(viol_counts):,}")
    print(f"  Total violations: {viol_counts['total_violations'].sum():,}")

    # --- Merge violations onto system locations ---
    print("\nMerging violations with system data ...")
    df = locations.merge(viol_counts, on="pwsid", how="left")

    for col in ["total_violations", "health_violations", "monitoring_violations"]:
        df[col] = df[col].fillna(0).astype(int)

    df["population_served"] = pd.to_numeric(df["population_served"], errors="coerce")
    df["system_size_category"] = df["population_served"].apply(classify_system_size)

    df["violations_per_1k_pop"] = np.where(
        df["population_served"] > 0,
        (df["total_violations"] / df["population_served"] * 1000).round(3),
        np.nan,
    )

    # --- Spatial join to Census tracts ---
    print("\nSpatial join (point-in-polygon) ...")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    geocoded = df.dropna(subset=["latitude", "longitude"]).copy()
    not_geocoded = df[df["latitude"].isna() | df["longitude"].isna()].copy()
    print(f"  Geocoded:     {len(geocoded):,}")
    print(f"  Not geocoded: {len(not_geocoded):,}")

    if not geocoded.empty:
        geometry = [
            Point(xy) for xy in zip(geocoded["longitude"], geocoded["latitude"])
        ]
        systems_gdf = gpd.GeoDataFrame(geocoded, geometry=geometry, crs="EPSG:4326")

        joined = gpd.sjoin(
            systems_gdf,
            tracts_gdf[["GEOID", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = pd.DataFrame(
            joined.drop(columns=["geometry", "index_right"], errors="ignore")
        )

        # Attach Census demographics
        joined = joined.merge(census, on="GEOID", how="left")

        matched = joined["median_household_income"].notna().sum()
        print(f"  Matched to tract: {matched:,}/{len(joined):,} "
              f"({matched / len(joined) * 100:.1f}%)")
    else:
        joined = pd.DataFrame()

    # Add back non-geocoded systems
    for col in ["GEOID", "county_fips", "tract_name",
                "total_population", "median_household_income",
                "poverty_rate", "pct_white", "pct_black",
                "pct_hispanic", "pct_nonwhite"]:
        if col not in not_geocoded.columns:
            not_geocoded[col] = np.nan

    df_final = pd.concat([joined, not_geocoded], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["pwsid"], keep="first")

    # --- Select final columns ---
    final_cols = [
        "pwsid", "pws_name", "county_served", "city_served",
        "latitude", "longitude", "zip_code",
        "population_served", "system_size_category",
        "source_water_type", "owner_type",
        "total_violations", "health_violations", "monitoring_violations",
        "violations_per_1k_pop",
        "GEOID", "county_fips",
        "median_household_income", "poverty_rate",
        "pct_white", "pct_black", "pct_hispanic", "pct_nonwhite",
    ]
    final_cols = [c for c in final_cols if c in df_final.columns]
    df_final = df_final[final_cols].copy()

    # --- Save ---
    outpath = os.path.join(OUTPUT_DIR, "analytical_dataset.csv")
    df_final.to_csv(outpath, index=False)
    print(f"\n  ✓ Saved: {len(df_final):,} rows → {outpath}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total systems:             {len(df_final):,}")
    print(f"  With ≥1 violation:         {(df_final['total_violations'] > 0).sum():,}")
    print(f"  Total violations:          {df_final['total_violations'].sum():,}")
    print(f"  Geocoded:                  {df_final['latitude'].notna().sum():,}")
    print(f"  Matched to Census tract:   {df_final['median_household_income'].notna().sum():,}")
    print(f"\nViolations by system size:")
    print(
        df_final.groupby("system_size_category")["total_violations"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )


if __name__ == "__main__":
    main()
