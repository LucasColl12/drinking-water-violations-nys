"""
04_merge_and_clean.py
=====================
Merges SDWIS water system data, violation records, geocoordinates, and
Census ACS tract-level demographics into a single analytical dataset.

The key step is a spatial join: each geocoded water system point is placed
inside the Census tract polygon that contains it, linking the system to
that tract's demographic characteristics. This gives a much finer-grained
equity picture than county-level matching.

The final dataset has one row per community water system with:
    - System characteristics (size, source type, owner type)
    - Violation counts and rates
    - Location (lat, lon, county, Census tract GEOID)
    - Community demographics (income, poverty, race) for the enclosing tract

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
# Helper: classify system size
# ---------------------------------------------------------------------------

def classify_system_size(pop: float) -> str:
    """
    Classify water system size using EPA's standard size categories.

    Categories per EPA:
        Very Small:  serves 25–500
        Small:       serves 501–3,300
        Medium:      serves 3,301–10,000
        Large:       serves 10,001–100,000
        Very Large:  serves > 100,000
    """
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

    # -----------------------------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------------------------
    print("Loading raw datasets ...")

    systems = pd.read_csv(os.path.join(RAW_DIR, "water_system_ny.csv"),
                          dtype={"pwsid": str})
    violations = pd.read_csv(os.path.join(RAW_DIR, "violation_ny.csv"),
                             dtype={"pwsid": str})
    locations = pd.read_csv(os.path.join(RAW_DIR, "system_locations_ny.csv"),
                            dtype={"pwsid": str})
    census = pd.read_csv(os.path.join(RAW_DIR, "census_tracts_ny.csv"),
                         dtype={"GEOID": str, "county_fips": str})

    print(f"  Water systems:  {len(systems):,}")
    print(f"  Violations:     {len(violations):,}")
    print(f"  Locations:      {len(locations):,}")
    print(f"  Census tracts:  {len(census):,}")

    # -----------------------------------------------------------------------
    # 2. Load TIGER/Line tract shapefile for spatial join
    # -----------------------------------------------------------------------
    print("\nLoading TIGER/Line tract boundaries ...")
    tiger_path = os.path.join(RAW_DIR, "tl_2022_36_tract.zip")
    tracts_gdf = gpd.read_file(f"zip://{tiger_path}")

    # Ensure CRS is geographic (EPSG:4326) to match lat/lon coordinates
    tracts_gdf = tracts_gdf.to_crs(epsg=4326)
    print(f"  Loaded {len(tracts_gdf):,} tract polygons")

    # -----------------------------------------------------------------------
    # 3. Aggregate violations per system
    # -----------------------------------------------------------------------
    print("\nAggregating violations by system ...")

    # Check what violation category column is actually called
    # (API may return different casing or names)
    viol_cat_col = None
    for candidate in ["violation_category_code", "VIOLATION_CATEGORY_CODE",
                      "contaminant_code", "CONTAMINANT_CODE",
                      "violation_code", "VIOLATION_CODE"]:
        if candidate in violations.columns:
            viol_cat_col = candidate
            break

    if viol_cat_col:
        print(f"  Using violation category column: {viol_cat_col}")
        print(f"  Unique values: {violations[viol_cat_col].unique()[:20]}")

        viol_counts = (
            violations.groupby("pwsid")
            .agg(
                total_violations=("pwsid", "size"),
                health_violations=(
                    viol_cat_col,
                    lambda x: x.isin(["MCL", "TT", "MRDL",
                                       "mcl", "tt", "mrdl"]).sum()
                ),
                monitoring_violations=(
                    viol_cat_col,
                    lambda x: x.isin(["MON", "MR", "RPT",
                                       "mon", "mr", "rpt"]).sum()
                ),
            )
            .reset_index()
        )
    else:
        print("  WARNING: No violation category column found.")
        print(f"  Available columns: {list(violations.columns)}")
        print("  Counting total violations only.")
        viol_counts = (
            violations.groupby("pwsid")
            .size()
            .reset_index(name="total_violations")
        )
        viol_counts["health_violations"] = 0
        viol_counts["monitoring_violations"] = 0

    print(f"  Systems with violations: {len(viol_counts):,}")

    # -----------------------------------------------------------------------
    # 4. Build system-level dataset with locations
    # -----------------------------------------------------------------------
    print("\nMerging system locations with violation counts ...")

    df = locations.merge(viol_counts, on="pwsid", how="left")

    # Fill NaN violation counts with 0 (systems with no violations)
    for col in ["total_violations", "health_violations", "monitoring_violations"]:
        df[col] = df[col].fillna(0).astype(int)

    # Add system size categories
    df["population_served"] = pd.to_numeric(df["population_served"], errors="coerce")
    df["system_size_category"] = df["population_served"].apply(classify_system_size)

    # Violation rate per 1,000 people served
    df["violations_per_1k_pop"] = np.where(
        df["population_served"] > 0,
        (df["total_violations"] / df["population_served"] * 1000).round(3),
        np.nan
    )

    # -----------------------------------------------------------------------
    # 5. Spatial join: place each water system inside its Census tract
    # -----------------------------------------------------------------------
    print("\nPerforming spatial join (point-in-polygon) ...")

    # Create GeoDataFrame from system lat/lon
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    geocoded = df.dropna(subset=["latitude", "longitude"]).copy()
    not_geocoded = df[df["latitude"].isna() | df["longitude"].isna()].copy()

    print(f"  Geocoded systems:     {len(geocoded):,}")
    print(f"  Not geocoded:         {len(not_geocoded):,}")

    geometry = [Point(xy) for xy in zip(geocoded["longitude"], geocoded["latitude"])]
    systems_gdf = gpd.GeoDataFrame(geocoded, geometry=geometry, crs="EPSG:4326")

    # Spatial join: find which tract polygon each system point falls within
    joined = gpd.sjoin(
        systems_gdf,
        tracts_gdf[["GEOID", "geometry"]],
        how="left",
        predicate="within",
    )

    # Drop the geometry and spatial index columns
    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"],
                                       errors="ignore"))

    # Merge Census demographics onto the GEOID from the spatial join
    joined = joined.merge(census, on="GEOID", how="left")

    matched = joined["median_household_income"].notna().sum()
    print(f"  Matched to Census tract: {matched:,}/{len(joined):,} "
          f"({matched/len(joined)*100:.1f}%)")

    # Add back the non-geocoded systems (they'll have NaN demographics)
    for col in ["GEOID", "county_fips", "tract_name",
                "total_population", "median_household_income",
                "poverty_rate", "pct_white", "pct_black",
                "pct_hispanic", "pct_nonwhite"]:
        if col not in not_geocoded.columns:
            not_geocoded[col] = np.nan

    df_final = pd.concat([joined, not_geocoded], ignore_index=True)

    # -----------------------------------------------------------------------
    # 6. Final cleanup
    # -----------------------------------------------------------------------
    final_cols = [
        "pwsid", "pws_name", "county_served", "city_served",
        "latitude", "longitude",
        "population_served", "system_size_category",
        "source_water_type", "owner_type",
        "total_violations", "health_violations", "monitoring_violations",
        "violations_per_1k_pop",
        "GEOID", "county_fips",
        "median_household_income", "poverty_rate",
        "pct_white", "pct_black", "pct_hispanic", "pct_nonwhite",
    ]

    # Only keep columns that actually exist
    final_cols = [c for c in final_cols if c in df_final.columns]
    df_final = df_final[final_cols].copy()

    # Drop duplicate rows that can arise from spatial join edge cases
    # (a point on a boundary might match two tracts)
    df_final = df_final.drop_duplicates(subset=["pwsid"], keep="first")

    # -----------------------------------------------------------------------
    # 7. Save
    # -----------------------------------------------------------------------
    outpath = os.path.join(OUTPUT_DIR, "analytical_dataset.csv")
    df_final.to_csv(outpath, index=False)
    print(f"\n  ✓ Saved analytical dataset: {len(df_final):,} rows → {outpath}")

    # Summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total water systems:           {len(df_final):,}")
    print(f"  Systems with ≥1 violation:     {(df_final['total_violations'] > 0).sum():,}")
    print(f"  Total violations:              {df_final['total_violations'].sum():,}")
    print(f"  Geocoded (lat/lon):            {df_final['latitude'].notna().sum():,}")
    print(f"  Matched to Census tract:       {df_final['median_household_income'].notna().sum():,}")

    print(f"\nViolations by system size:")
    size_summary = (
        df_final.groupby("system_size_category")["total_violations"]
        .agg(["count", "sum", "mean"])
        .round(2)
    )
    print(size_summary)

    print(f"\nDemographic summary of matched systems:")
    demo_cols = ["median_household_income", "poverty_rate", "pct_nonwhite"]
    demo_cols = [c for c in demo_cols if c in df_final.columns]
    print(df_final[demo_cols].describe().round(2))


if __name__ == "__main__":
    main()
