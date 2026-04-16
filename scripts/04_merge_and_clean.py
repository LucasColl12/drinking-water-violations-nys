"""
04_merge_and_clean.py
=====================
Merges SDWIS water system data, violation records, geocoordinates, and
Census demographics into analytical datasets comparing two periods.

KEY FEATURES:
    - Health-based violations only — identified via violation_category_code,
      is_health_based_ind, OR violation_code prefix patterns
    - Two comparison periods: 2000–2005 (historical) vs. 2020–2025 (recent)
    - Era-appropriate demographics where available (Census 2000 vs. ACS 2022)

Violation code identification:
    The ECHO bulk download may not include violation_category_code. In that
    case, we identify health-based violations using:
      - violation_code starting with "01" through "09" = MCL violations
      - violation_code starting with "1" or "2" (two-digit) = treatment technique
      - Or via is_health_based_ind = 'Y' if present
    See: https://echo.epa.gov/tools/data-downloads/sdwa-download-summary

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

# Health-based violation category codes
# MCL = Maximum Contaminant Level, TT = Treatment Technique, MRDL = Max Residual Disinfectant Level
# Excludes MR (Monitoring/Reporting) and MON (Monitoring) which are procedural, not health-based
HEALTH_CATEGORIES = {"MCL", "TT", "MRDL"}

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


def identify_health_violations(violations: pd.DataFrame) -> pd.DataFrame:
    """
    Flag health-based violations using whatever columns are available.

    Priority:
        Method 1: violation_category_code IN (MCL, TT, MRDL) — most reliable
        Method 2: is_health_based_ind = 'Y' — EPA's own classification
        Method 3: violation_code prefixes — ONLY used as last resort if
                  Methods 1 and 2 found nothing (i.e., the columns don't exist)

    The violation_code prefix approach was previously too aggressive —
    code "03" alone captured ~290K violations that include monitoring
    violations grouped under rule family 03. Restricting to Methods 1+2
    gives the cleanest health-only count.
    """
    n_total = len(violations)
    violations["is_health"] = False
    methods_found = 0

    # Method 1: violation_category_code (most reliable)
    cat_col = find_column(violations, [
        "violation_category_code", "viol_category_code",
    ])
    if cat_col:
        cat_upper = violations[cat_col].astype(str).str.upper().str.strip()
        method1 = cat_upper.isin(HEALTH_CATEGORIES)
        violations.loc[method1, "is_health"] = True
        print(f"    Method 1 (category code): {method1.sum():,} health violations")
        print(f"      Category distribution:")
        print(cat_upper.value_counts().head(10).to_string(header=False))
        methods_found += 1

    # Method 2: is_health_based_ind
    health_ind = find_column(violations, ["is_health_based_ind"])
    if health_ind:
        method2 = violations[health_ind].astype(str).str.upper() == "Y"
        # Only add violations not already flagged by Method 1
        new_from_m2 = method2 & ~violations["is_health"]
        violations.loc[method2, "is_health"] = True
        print(f"    Method 2 (health indicator): {method2.sum():,} total, "
              f"{new_from_m2.sum():,} additional beyond Method 1")
        methods_found += 1

    # Method 3: violation_code prefixes — LAST RESORT ONLY
    # Only used if neither Method 1 nor Method 2 columns exist
    if methods_found == 0:
        print("    ⚠ No category or health indicator columns found.")
        print("    Falling back to violation_code prefix matching.")
        viol_code_col = find_column(violations, ["violation_code", "viol_code"])
        if viol_code_col:
            code_str = violations[viol_code_col].astype(str).str.strip()

            # Restrict to specific known MCL/TT codes only
            # 01 = MCL (inorganic), 02 = MCL (organic), 04 = MCL (SOC)
            # 07 = MCL (rads), 10 = TT (surface water), 21-27 = TT variants
            strict_prefixes = ("01", "02", "04", "07")
            strict_codes = {"10", "21", "22", "23", "24", "25", "26", "27"}

            method3 = (
                code_str.str[:2].isin(strict_prefixes) |
                code_str.isin(strict_codes)
            )
            violations.loc[method3, "is_health"] = True
            print(f"    Method 3 (violation code, strict): {method3.sum():,}")
    else:
        # Show violation codes for the health violations we found (diagnostic)
        viol_code_col = find_column(violations, ["violation_code", "viol_code"])
        if viol_code_col:
            health_codes = violations.loc[
                violations["is_health"], viol_code_col
            ].value_counts().head(10)
            print(f"    Top violation codes among health violations:")
            for code, count in health_codes.items():
                print(f"      {code}: {count:,}")

    total_health = violations["is_health"].sum()
    print(f"\n    TOTAL health-based violations: {total_health:,}/{n_total:,} "
          f"({total_health/n_total*100:.1f}%)")

    return violations


def try_spatial_join(systems_df: pd.DataFrame, tiger_path: str,
                     census_df: pd.DataFrame) -> pd.DataFrame:
    """Spatial join systems to Census tracts. Returns None on failure."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("    GeoPandas not installed — skipping spatial join.")
        return None

    if not os.path.exists(tiger_path):
        print(f"    TIGER file not found: {tiger_path}")
        return None

    tracts_gdf = gpd.read_file(f"zip://{tiger_path}")
    tracts_gdf = tracts_gdf.to_crs(epsg=4326)

    # Standardize GEOID
    for candidate in ["GEOID", "GEOID10", "CTIDFP00", "CTIDFP"]:
        if candidate in tracts_gdf.columns and candidate != "GEOID":
            tracts_gdf = tracts_gdf.rename(columns={candidate: "GEOID"})
            break

    print(f"    Loaded {len(tracts_gdf):,} tract polygons")

    geocoded = systems_df.dropna(subset=["latitude", "longitude"]).copy()
    if len(geocoded) == 0:
        print("    No geocoded systems for spatial join.")
        return None

    geometry = [Point(xy) for xy in zip(geocoded["longitude"], geocoded["latitude"])]
    systems_gdf = gpd.GeoDataFrame(geocoded, geometry=geometry, crs="EPSG:4326")

    joined = gpd.sjoin(
        systems_gdf, tracts_gdf[["GEOID", "geometry"]],
        how="left", predicate="within",
    )
    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"],
                                       errors="ignore"))
    joined = joined.merge(census_df, on="GEOID", how="left")

    matched = joined["median_household_income"].notna().sum()
    print(f"    Matched to tract: {matched:,}/{len(joined):,}")

    return joined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("Loading raw datasets ...")

    violations = pd.read_csv(os.path.join(RAW_DIR, "violation_ny.csv"), dtype=str)
    locations = pd.read_csv(os.path.join(RAW_DIR, "system_locations_ny.csv"), dtype=str)

    violations.columns = [c.lower() for c in violations.columns]
    locations.columns = [c.lower() for c in locations.columns]

    print(f"  Violations:     {len(violations):,}")
    print(f"  Locations:      {len(locations):,}")
    print(f"  Location cols:  {list(locations.columns)[:15]}")
    print(f"  Violation cols: {list(violations.columns)}")

    # Census data
    census_2022_path = os.path.join(RAW_DIR, "census_tracts_ny_2022.csv")
    census_2000_path = os.path.join(RAW_DIR, "census_tracts_ny_2000.csv")
    census_default = os.path.join(RAW_DIR, "census_tracts_ny.csv")

    for path in [census_2022_path, census_default]:
        if os.path.exists(path):
            census_recent = pd.read_csv(path, dtype={"GEOID": str, "county_fips": str})
            print(f"  Census (recent): {len(census_recent):,} tracts from {os.path.basename(path)}")
            break
    else:
        print("  ERROR: No Census data found!")
        return

    has_2000 = os.path.exists(census_2000_path)
    if has_2000:
        census_hist = pd.read_csv(census_2000_path, dtype={"GEOID": str, "county_fips": str})
        print(f"  Census (historical): {len(census_hist):,} tracts")
    else:
        print("  Using 2022 demographics for both periods (no Census 2000 data).")
        census_hist = census_recent

    # Clean Census sentinel values (Census API returns e.g. -666666666 for missing)
    for census_df in [census_recent, census_hist]:
        if "median_household_income" in census_df.columns:
            bad = census_df["median_household_income"] < 0
            if bad.any():
                census_df.loc[bad, "median_household_income"] = np.nan
                print(f"    Cleaned {bad.sum()} sentinel income values → NaN")

    # -----------------------------------------------------------------------
    # 2. Identify health-based violations
    # -----------------------------------------------------------------------
    print("\nIdentifying health-based violations ...")
    violations = identify_health_violations(violations)
    violations = violations[violations["is_health"]].copy()
    print(f"  Keeping {len(violations):,} health-based violations")

    # -----------------------------------------------------------------------
    # 3. Parse dates and assign periods
    # -----------------------------------------------------------------------
    print("\nParsing violation dates ...")

    date_col = find_column(violations, [
        "compl_per_begin_date", "compliance_begin_date",
        "non_compl_per_begin_date", "violation_date",
    ])

    if date_col:
        violations["_viol_date"] = pd.to_datetime(violations[date_col], errors="coerce")
        n_parsed = violations["_viol_date"].notna().sum()
        print(f"  Using: {date_col}")
        print(f"  Parsed: {n_parsed:,}/{len(violations):,}")

        if n_parsed > 0:
            print(f"  Range: {violations['_viol_date'].min().date()} → "
                  f"{violations['_viol_date'].max().date()}")

        hist_start, hist_end, hist_label = PERIOD_HISTORICAL
        rec_start, rec_end, rec_label = PERIOD_RECENT

        violations["period"] = np.where(
            violations["_viol_date"].between(hist_start, hist_end), hist_label,
            np.where(
                violations["_viol_date"].between(rec_start, rec_end), rec_label,
                "other"
            )
        )

        print(f"\n  By period:")
        print(violations["period"].value_counts().to_string(header=False))

        violations = violations[violations["period"].isin([hist_label, rec_label])].copy()
        print(f"  Kept: {len(violations):,}")
    else:
        print("  ⚠ No date column found!")
        violations["period"] = "all"

    # -----------------------------------------------------------------------
    # 4. Aggregate violations per system per period
    # -----------------------------------------------------------------------
    _, _, hist_label = PERIOD_HISTORICAL
    _, _, rec_label = PERIOD_RECENT

    viol_counts = (
        violations.groupby(["pwsid", "period"])
        .size()
        .reset_index(name="health_violations")
    )
    print(f"\n  System-period violation records: {len(viol_counts):,}")

    # -----------------------------------------------------------------------
    # 5. Build system base
    # -----------------------------------------------------------------------
    print("\nBuilding system base ...")

    base = locations.copy()

    # Standardize column names — ECHO Exporter uses different names than SDWIS
    rename_map = {}
    for canonical, variants in {
        "county_served": ["fac_county", "countyserved", "facstdcountyname", "county"],
        "city_served": ["fac_city", "cityserved", "city_name"],
        "pws_name": ["fac_name", "pwsname", "facname"],
        "source_water_type": ["sourcewatertype", "srcwatertype", "water_type_code"],
        "owner_type": ["ownertypedesc", "ownertype", "owner_type_code"],
    }.items():
        if canonical not in base.columns:
            found = find_column(base, variants)
            if found:
                rename_map[found] = canonical

    if rename_map:
        base = base.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")

    # Also pull county/city/name from water systems file if still missing
    missing_cols = [c for c in ["county_served", "pws_name", "source_water_type", "owner_type"]
                    if c not in base.columns]
    if missing_cols:
        systems_path = os.path.join(RAW_DIR, "water_system_ny.csv")
        if os.path.exists(systems_path):
            sys_df = pd.read_csv(systems_path, dtype=str, low_memory=False)
            sys_df.columns = [c.lower() for c in sys_df.columns]

            # Map SDWIS column names
            sys_rename = {}
            for canonical, variants in {
                "county_served": ["county_served"],
                "pws_name": ["pws_name"],
                "source_water_type": ["primary_source_code", "gw_sw_code"],
                "owner_type": ["owner_type_code"],
            }.items():
                if canonical in missing_cols:
                    found = find_column(sys_df, variants)
                    if found:
                        sys_rename[found] = canonical

            if sys_rename:
                merge_cols = ["pwsid"] + list(sys_rename.keys())
                merge_cols = [c for c in merge_cols if c in sys_df.columns]
                supplement = sys_df[merge_cols].drop_duplicates(subset=["pwsid"])
                supplement = supplement.rename(columns=sys_rename)
                base = base.merge(supplement, on="pwsid", how="left")
                print(f"  Merged from water systems: {list(sys_rename.values())}")

    # Normalize population column — ECHO and SDWIS use different names
    pop_col = find_column(base, [
        "population_served", "population_served_count",
        "populationservedcount", "popserved", "currsvdpop",
        "pop_cat_5_code",
    ])
    if pop_col and pop_col != "population_served":
        base = base.rename(columns={pop_col: "population_served"})

    # If locations file has no population data, pull it from water systems file
    if "population_served" not in base.columns:
        print("  ⚠ No population column in locations file, checking water systems ...")
        systems_path = os.path.join(RAW_DIR, "water_system_ny.csv")
        if os.path.exists(systems_path):
            sys_df = pd.read_csv(systems_path, dtype=str, low_memory=False)
            sys_df.columns = [c.lower() for c in sys_df.columns]
            sys_pop_col = find_column(sys_df, [
                "population_served_count", "population_served",
                "populationservedcount",
            ])
            if sys_pop_col:
                pop_lookup = sys_df[["pwsid", sys_pop_col]].drop_duplicates(subset=["pwsid"])
                pop_lookup = pop_lookup.rename(columns={sys_pop_col: "population_served"})
                base = base.merge(pop_lookup, on="pwsid", how="left")
                print(f"    Merged population data from water systems file")

    if "population_served" in base.columns:
        base["population_served"] = pd.to_numeric(base["population_served"], errors="coerce")
    else:
        print("  ⚠ No population data found — system size categories will be 'Unknown'")
        base["population_served"] = np.nan

    base["system_size_category"] = base.get("population_served", pd.Series()).apply(classify_system_size)
    base["latitude"] = pd.to_numeric(base.get("latitude"), errors="coerce")
    base["longitude"] = pd.to_numeric(base.get("longitude"), errors="coerce")

    geocoded_n = base["latitude"].notna().sum()
    print(f"  Systems: {len(base):,}, geocoded: {geocoded_n:,}")

    # -----------------------------------------------------------------------
    # 6. Spatial joins for each period
    # -----------------------------------------------------------------------
    tiger_2022 = os.path.join(RAW_DIR, "tl_2022_36_tract.zip")
    tiger_2000 = os.path.join(RAW_DIR, "tl_2000_36_tract.zip")
    if not os.path.exists(tiger_2000):
        tiger_2000 = tiger_2022

    demo_cols = ["GEOID", "county_fips", "tract_name",
                 "total_population", "median_household_income",
                 "poverty_rate", "pct_white", "pct_black",
                 "pct_hispanic", "pct_nonwhite"]

    print(f"\nSpatial join — historical ({hist_label}) ...")
    base_hist = try_spatial_join(base, tiger_2000, census_hist)
    if base_hist is None:
        base_hist = base.copy()
        for col in demo_cols:
            if col not in base_hist.columns:
                base_hist[col] = np.nan
    base_hist = base_hist.drop_duplicates(subset=["pwsid"], keep="first")
    base_hist["period"] = hist_label

    print(f"\nSpatial join — recent ({rec_label}) ...")
    base_rec = try_spatial_join(base, tiger_2022, census_recent)
    if base_rec is None:
        base_rec = base.copy()
        for col in demo_cols:
            if col not in base_rec.columns:
                base_rec[col] = np.nan
    base_rec = base_rec.drop_duplicates(subset=["pwsid"], keep="first")
    base_rec["period"] = rec_label

    # -----------------------------------------------------------------------
    # 7. Long dataset
    # -----------------------------------------------------------------------
    print("\nCreating long-format dataset ...")

    long = pd.concat([base_hist, base_rec], ignore_index=True)
    long = long.merge(viol_counts, on=["pwsid", "period"], how="left")
    long["health_violations"] = long["health_violations"].fillna(0).astype(int)

    long["violations_per_1k_pop"] = np.where(
        long["population_served"] > 0,
        (long["health_violations"] / long["population_served"] * 1000).round(3),
        np.nan
    )

    # -----------------------------------------------------------------------
    # 8. Wide dataset
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

    wide = base_rec.drop(columns=["period"], errors="ignore").copy()
    wide = wide.merge(wide_viols, on="pwsid", how="left")

    for col in wide.columns:
        if col.startswith("health_violations_"):
            wide[col] = wide[col].fillna(0).astype(int)

    hist_col = f"health_violations_{hist_label.replace('–', '_')}"
    rec_col = f"health_violations_{rec_label.replace('–', '_')}"

    if hist_col in wide.columns and rec_col in wide.columns:
        wide["violation_change"] = wide[rec_col] - wide[hist_col]
        wide["had_violations_historical"] = (wide[hist_col] > 0).astype(int)
        wide["had_violations_recent"] = (wide[rec_col] > 0).astype(int)

    # -----------------------------------------------------------------------
    # 9. Select columns and save
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
    long = long[long_cols]

    outpath_long = os.path.join(OUTPUT_DIR, "analytical_dataset.csv")
    long.to_csv(outpath_long, index=False)
    print(f"\n  ✓ Long:  {len(long):,} rows → {outpath_long}")

    outpath_wide = os.path.join(OUTPUT_DIR, "analytical_dataset_wide.csv")
    wide.to_csv(outpath_wide, index=False)
    print(f"  ✓ Wide:  {len(wide):,} rows → {outpath_wide}")

    # -----------------------------------------------------------------------
    # 10. Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")

    for p in [hist_label, rec_label]:
        pdf = long[long["period"] == p]
        n = len(pdf)
        v = (pdf["health_violations"] > 0).sum()
        t = pdf["health_violations"].sum()
        matched = pdf["median_household_income"].notna().sum()
        print(f"\n  {p}:")
        print(f"    Systems: {n:,}")
        print(f"    With ≥1 health violation: {v:,} ({v/n*100:.1f}%)")
        print(f"    Total health violations: {t:,}")
        print(f"    Matched to Census tract: {matched:,}")

        if matched > 0:
            print(f"    Median HH income: ${pdf['median_household_income'].median():,.0f}")
            print(f"    Median poverty rate: {pdf['poverty_rate'].median():.1f}%")
            print(f"    Median % nonwhite: {pdf['pct_nonwhite'].median():.1f}%")


if __name__ == "__main__":
    main()
