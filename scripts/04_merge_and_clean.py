"""
04_merge_and_clean.py
=====================
Merges SDWIS water system data, violation records, geocoordinates, and
Census demographics into analytical datasets comparing two periods.

KEY DESIGN DECISIONS:
    - Health-based violations only (MCL, TT, MRDL)
    - Two comparison periods: 2000–2005 (historical) vs. 2020–2025 (recent)
    - Era-appropriate demographics: Census 2000 for historical period,
      ACS 2022 for recent period. Falls back to 2022 for both if 2000
      data is unavailable.
    - Spatial join places each geocoded system inside its Census tract
      for fine-grained equity analysis.

Outputs:
    output/analytical_dataset.csv       — one row per system-period (long)
    output/analytical_dataset_wide.csv  — one row per system, period columns

Usage:
    python scripts/04_merge_and_clean.py
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PERIOD_HISTORICAL = ("2000-01-01", "2005-12-31", "2000–2005")
PERIOD_RECENT = ("2020-01-01", "2025-12-31", "2020–2025")

HEALTH_CODES = {"MCL", "TT", "MRDL"}

# ---------------------------------------------------------------------------
# Helpers
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


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive)."""
    for c in candidates:
        matches = [col for col in df.columns if col.lower() == c.lower()]
        if matches:
            return matches[0]
    return None


def spatial_join_tracts(systems_df: pd.DataFrame, tiger_path: str,
                        census_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform point-in-polygon spatial join of geocoded systems into
    Census tract polygons, then merge tract demographics.

    Returns the systems DataFrame with Census columns appended.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    tracts_gdf = gpd.read_file(f"zip://{tiger_path}")
    tracts_gdf = tracts_gdf.to_crs(epsg=4326)

    # Standardize GEOID column name (2000 shapefiles use CTIDFP00)
    geoid_col = None
    for candidate in ["GEOID", "GEOID10", "CTIDFP00", "CTIDFP", "TRACTCE"]:
        if candidate in tracts_gdf.columns:
            geoid_col = candidate
            break
    if geoid_col and geoid_col != "GEOID":
        tracts_gdf = tracts_gdf.rename(columns={geoid_col: "GEOID"})

    print(f"    Loaded {len(tracts_gdf):,} tract polygons")

    geocoded = systems_df.dropna(subset=["latitude", "longitude"]).copy()
    not_geocoded = systems_df[
        systems_df["latitude"].isna() | systems_df["longitude"].isna()
    ].copy()

    print(f"    Geocoded systems:  {len(geocoded):,}")
    print(f"    Not geocoded:      {len(not_geocoded):,}")

    if len(geocoded) == 0:
        print("    ⚠ No geocoded systems — skipping spatial join.")
        # Still add Census columns as NaN
        for col in census_df.columns:
            if col not in systems_df.columns:
                systems_df[col] = np.nan
        return systems_df

    geometry = [Point(xy) for xy in zip(geocoded["longitude"], geocoded["latitude"])]
    systems_gdf = gpd.GeoDataFrame(geocoded, geometry=geometry, crs="EPSG:4326")

    joined = gpd.sjoin(
        systems_gdf,
        tracts_gdf[["GEOID", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"],
                                       errors="ignore"))

    # Merge Census demographics onto GEOID
    joined = joined.merge(census_df, on="GEOID", how="left")

    matched = joined["median_household_income"].notna().sum()
    total = len(joined)
    pct = matched / total * 100 if total > 0 else 0
    print(f"    Matched to Census tract: {matched:,}/{total:,} ({pct:.1f}%)")

    # Add non-geocoded systems back with NaN demographics
    for col in census_df.columns:
        if col not in not_geocoded.columns:
            not_geocoded[col] = np.nan

    result = pd.concat([joined, not_geocoded], ignore_index=True)
    result = result.drop_duplicates(subset=["pwsid"], keep="first")

    return result


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

    print(f"  Water systems:  {len(systems):,}")
    print(f"  Violations:     {len(violations):,}")
    print(f"  Locations:      {len(locations):,}")

    # Load Census data for both eras
    census_2022_path = os.path.join(RAW_DIR, "census_tracts_ny_2022.csv")
    census_2000_path = os.path.join(RAW_DIR, "census_tracts_ny_2000.csv")
    census_default_path = os.path.join(RAW_DIR, "census_tracts_ny.csv")

    # Recent period demographics
    if os.path.exists(census_2022_path):
        census_recent = pd.read_csv(census_2022_path, dtype={"GEOID": str, "county_fips": str})
    elif os.path.exists(census_default_path):
        census_recent = pd.read_csv(census_default_path, dtype={"GEOID": str, "county_fips": str})
    else:
        print("  ERROR: No Census tract data found!")
        return

    # Historical period demographics
    has_2000_census = os.path.exists(census_2000_path)
    if has_2000_census:
        census_historical = pd.read_csv(census_2000_path, dtype={"GEOID": str, "county_fips": str})
        print(f"  Census 2000 tracts: {len(census_historical):,}")
    else:
        print("  ⚠ No Census 2000 data — using 2022 demographics for both periods.")
        census_historical = census_recent

    print(f"  Census 2022 tracts: {len(census_recent):,}")

    # -----------------------------------------------------------------------
    # 2. Filter violations to HEALTH-BASED only and assign periods
    # -----------------------------------------------------------------------
    print("\nFiltering to health-based violations ...")

    cat_col = find_column(violations, [
        "violation_category_code", "contaminant_code",
        "violation_code", "viol_category_code",
    ])
    date_col = find_column(violations, [
        "compl_per_begin_date", "compliance_begin_date",
        "violation_date", "viol_begin_date", "begin_date",
    ])

    # Also check for is_health_based_ind (some SDWIS sources have this)
    health_ind_col = find_column(violations, ["is_health_based_ind"])

    print(f"  Category column:     {cat_col}")
    print(f"  Date column:         {date_col}")
    print(f"  Health indicator:    {health_ind_col}")

    # Filter to health-based violations
    if cat_col:
        violations["_cat"] = violations[cat_col].astype(str).str.upper().str.strip()
        health_mask = violations["_cat"].isin(HEALTH_CODES)
        # Also include if is_health_based_ind == 'Y'
        if health_ind_col:
            health_mask = health_mask | (
                violations[health_ind_col].astype(str).str.upper() == "Y"
            )
        print(f"  Total violations:        {len(violations):,}")
        print(f"  Health-based violations: {health_mask.sum():,}")
        violations = violations[health_mask].copy()
    elif health_ind_col:
        health_mask = violations[health_ind_col].astype(str).str.upper() == "Y"
        print(f"  Using health indicator column")
        print(f"  Health-based violations: {health_mask.sum():,}")
        violations = violations[health_mask].copy()
    else:
        print("  ⚠ No category or health indicator column found.")
        print("  Proceeding with ALL violations.")

    # Parse dates and assign periods
    if date_col:
        violations["_viol_date"] = pd.to_datetime(violations[date_col], errors="coerce")
        n_parsed = violations["_viol_date"].notna().sum()
        print(f"\n  Parsed dates: {n_parsed:,}/{len(violations):,}")

        if n_parsed > 0:
            print(f"  Date range: {violations['_viol_date'].min()} → "
                  f"{violations['_viol_date'].max()}")

        hist_start, hist_end, hist_label = PERIOD_HISTORICAL
        rec_start, rec_end, rec_label = PERIOD_RECENT

        violations["period"] = np.where(
            violations["_viol_date"].between(hist_start, hist_end), hist_label,
            np.where(
                violations["_viol_date"].between(rec_start, rec_end), rec_label,
                "other"
            )
        )

        print(f"\n  Violations by period:")
        print(violations["period"].value_counts().to_string(header=False))

        violations = violations[violations["period"].isin([hist_label, rec_label])].copy()
        print(f"  Kept {len(violations):,} violations in comparison periods")
    else:
        print("  ⚠ No date column found — all violations treated as single period.")
        violations["period"] = "all"

    # -----------------------------------------------------------------------
    # 3. Aggregate violations per system per period
    # -----------------------------------------------------------------------
    print("\nAggregating health violations by system and period ...")

    _, _, hist_label = PERIOD_HISTORICAL
    _, _, rec_label = PERIOD_RECENT

    viol_counts = (
        violations.groupby(["pwsid", "period"])
        .size()
        .reset_index(name="health_violations")
    )
    print(f"  System-period combinations with violations: {len(viol_counts):,}")

    # -----------------------------------------------------------------------
    # 4. Build system base from locations
    # -----------------------------------------------------------------------
    print("\nBuilding system base ...")

    base = locations.copy()
    base["population_served"] = pd.to_numeric(base["population_served"], errors="coerce")
    base["system_size_category"] = base["population_served"].apply(classify_system_size)
    base["latitude"] = pd.to_numeric(base["latitude"], errors="coerce")
    base["longitude"] = pd.to_numeric(base["longitude"], errors="coerce")

    print(f"  Systems: {len(base):,}")
    print(f"  With coordinates: {base['latitude'].notna().sum():,}")

    # -----------------------------------------------------------------------
    # 5. Spatial join for EACH period's Census data
    # -----------------------------------------------------------------------
    tiger_2022 = os.path.join(RAW_DIR, "tl_2022_36_tract.zip")
    tiger_2000 = os.path.join(RAW_DIR, "tl_2000_36_tract.zip")

    # Use 2022 TIGER as fallback if 2000 doesn't exist
    if not os.path.exists(tiger_2000):
        tiger_2000 = tiger_2022

    demo_cols = ["GEOID", "county_fips", "tract_name",
                 "total_population", "median_household_income",
                 "poverty_rate", "pct_white", "pct_black",
                 "pct_hispanic", "pct_nonwhite"]

    has_geopandas = True
    try:
        import geopandas  # noqa
    except ImportError:
        has_geopandas = False
        print("  ⚠ GeoPandas not installed — skipping spatial join.")
        print("  Demographics will not be linked to systems.")

    # --- Historical period: spatial join with 2000 Census ---
    print(f"\nSpatial join for HISTORICAL period ({hist_label}) ...")
    if has_geopandas and os.path.exists(tiger_2000):
        base_hist = spatial_join_tracts(base, tiger_2000, census_historical)
    else:
        base_hist = base.copy()
        for col in demo_cols:
            if col not in base_hist.columns:
                base_hist[col] = np.nan

    base_hist["period"] = hist_label

    # --- Recent period: spatial join with 2022 Census ---
    print(f"\nSpatial join for RECENT period ({rec_label}) ...")
    if has_geopandas and os.path.exists(tiger_2022):
        base_rec = spatial_join_tracts(base, tiger_2022, census_recent)
    else:
        base_rec = base.copy()
        for col in demo_cols:
            if col not in base_rec.columns:
                base_rec[col] = np.nan

    base_rec["period"] = rec_label

    # -----------------------------------------------------------------------
    # 6. Combine into LONG dataset
    # -----------------------------------------------------------------------
    print("\nCreating long-format dataset ...")

    long = pd.concat([base_hist, base_rec], ignore_index=True)

    # Merge violation counts
    long = long.merge(viol_counts, on=["pwsid", "period"], how="left")
    long["health_violations"] = long["health_violations"].fillna(0).astype(int)

    long["violations_per_1k_pop"] = np.where(
        long["population_served"] > 0,
        (long["health_violations"] / long["population_served"] * 1000).round(3),
        np.nan
    )

    # -----------------------------------------------------------------------
    # 7. Create WIDE dataset
    # -----------------------------------------------------------------------
    print("Creating wide-format dataset ...")

    wide_viols = viol_counts.pivot_table(
        index="pwsid", columns="period",
        values="health_violations", fill_value=0,
    ).reset_index()

    wide_viols.columns = [
        c if c == "pwsid" else f"health_violations_{c.replace('–', '_').replace(' ', '')}"
        for c in wide_viols.columns
    ]

    # Use the recent-period base for the wide dataset (most current system info)
    wide = base_rec.drop(columns=["period"], errors="ignore").copy()
    wide = wide.merge(wide_viols, on="pwsid", how="left")

    for col in wide.columns:
        if col.startswith("health_violations_"):
            wide[col] = wide[col].fillna(0).astype(int)

    hist_col = f"health_violations_{hist_label.replace('–', '_').replace(' ', '')}"
    rec_col = f"health_violations_{rec_label.replace('–', '_').replace(' ', '')}"

    if hist_col in wide.columns and rec_col in wide.columns:
        wide["violation_change"] = wide[rec_col] - wide[hist_col]
        wide["had_violations_historical"] = (wide[hist_col] > 0).astype(int)
        wide["had_violations_recent"] = (wide[rec_col] > 0).astype(int)

    # -----------------------------------------------------------------------
    # 8. Select columns and save
    # -----------------------------------------------------------------------
    long_cols = [
        "pwsid", "pws_name", "county_served", "city_served",
        "latitude", "longitude",
        "population_served", "system_size_category",
        "source_water_type", "owner_type",
        "period", "health_violations", "violations_per_1k_pop",
        "GEOID", "county_fips",
        "median_household_income", "poverty_rate",
        "pct_white", "pct_black", "pct_hispanic", "pct_nonwhite",
    ]
    long_cols = [c for c in long_cols if c in long.columns]
    long = long[long_cols].copy()

    outpath_long = os.path.join(OUTPUT_DIR, "analytical_dataset.csv")
    long.to_csv(outpath_long, index=False)
    print(f"\n  ✓ Long dataset:  {len(long):,} rows → {outpath_long}")

    outpath_wide = os.path.join(OUTPUT_DIR, "analytical_dataset_wide.csv")
    wide.to_csv(outpath_wide, index=False)
    print(f"  ✓ Wide dataset:  {len(wide):,} rows → {outpath_wide}")

    # -----------------------------------------------------------------------
    # 9. Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total water systems:     {len(base):,}")
    print(f"  Geocoded (lat/lon):      {base['latitude'].notna().sum():,}")

    print(f"\nHealth violations by period:")
    period_summary = (
        long.groupby("period")["health_violations"]
        .agg(["sum", "mean", lambda x: (x > 0).sum()])
    )
    period_summary.columns = ["total", "mean_per_system", "systems_with_violations"]
    print(period_summary)

    if has_2000_census:
        print(f"\nDemographics by period (median values for matched systems):")
        for period_label in [hist_label, rec_label]:
            pdf = long[long["period"] == period_label]
            matched = pdf["median_household_income"].notna().sum()
            if matched > 0:
                print(f"\n  {period_label} (matched: {matched:,}):")
                print(f"    Median HH income:  ${pdf['median_household_income'].median():,.0f}")
                print(f"    Median poverty:    {pdf['poverty_rate'].median():.1f}%")
                print(f"    Median % nonwhite: {pdf['pct_nonwhite'].median():.1f}%")


if __name__ == "__main__":
    main()
