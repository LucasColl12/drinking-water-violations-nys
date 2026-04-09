"""
05_analysis.py
==============
Statistical analysis of SDWA violation patterns across New York State.

Analyses:
    1. Descriptive statistics — violation rates by system size, source type,
       and ownership
    2. Correlation analysis — violations vs. community demographics
    3. Regression model — negative binomial regression of violation counts
       on system and community characteristics

Requires: output/analytical_dataset.csv (from 04_merge_and_clean.py)
Outputs:  output/model_results.txt

Usage:
    python scripts/05_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_dataset.csv"))
    print(f"Loaded {len(df):,} water systems\n")

    results = []  # Collect text output

    # ===================================================================
    # 1. DESCRIPTIVE STATISTICS
    # ===================================================================
    results.append("=" * 70)
    results.append("SDWA VIOLATION ANALYSIS — NEW YORK STATE")
    results.append("=" * 70)

    results.append("\n1. OVERVIEW\n")
    total_sys = len(df)
    with_viol = (df["total_violations"] > 0).sum()
    results.append(f"   Total community water systems:  {total_sys:,}")
    results.append(f"   Systems with ≥1 violation:      {with_viol:,} "
                   f"({with_viol/total_sys*100:.1f}%)")
    results.append(f"   Total violations:               {df['total_violations'].sum():,}")
    results.append(f"   Health-based violations:        {df['health_violations'].sum():,}")
    results.append(f"   Monitoring/reporting violations: {df['monitoring_violations'].sum():,}")

    # --- By system size ---
    results.append("\n2. VIOLATIONS BY SYSTEM SIZE\n")
    size_order = ["Very Small", "Small", "Medium", "Large", "Very Large"]
    size_table = (
        df.groupby("system_size_category")
        .agg(
            n_systems=("pwsid", "count"),
            pct_with_violation=("total_violations", lambda x: (x > 0).mean() * 100),
            mean_violations=("total_violations", "mean"),
            median_violations=("total_violations", "median"),
            total_violations=("total_violations", "sum"),
        )
        .reindex(size_order)
        .round(2)
    )
    results.append(size_table.to_string())

    # --- By source water type ---
    results.append("\n\n3. VIOLATIONS BY SOURCE WATER TYPE\n")
    if "source_water_type" in df.columns:
        source_table = (
            df.groupby("source_water_type")
            .agg(
                n_systems=("pwsid", "count"),
                pct_with_violation=("total_violations", lambda x: (x > 0).mean() * 100),
                mean_violations=("total_violations", "mean"),
            )
            .round(2)
            .sort_values("n_systems", ascending=False)
        )
        results.append(source_table.to_string())

    # --- By owner type ---
    results.append("\n\n4. VIOLATIONS BY OWNERSHIP TYPE\n")
    if "owner_type" in df.columns:
        owner_table = (
            df.groupby("owner_type")
            .agg(
                n_systems=("pwsid", "count"),
                pct_with_violation=("total_violations", lambda x: (x > 0).mean() * 100),
                mean_violations=("total_violations", "mean"),
            )
            .round(2)
            .sort_values("n_systems", ascending=False)
        )
        results.append(owner_table.to_string())

    # ===================================================================
    # 2. CORRELATION ANALYSIS
    # ===================================================================
    results.append("\n\n5. CORRELATIONS: VIOLATIONS vs. DEMOGRAPHICS\n")

    demo_vars = [
        ("median_household_income", "Median household income"),
        ("poverty_rate", "Poverty rate (%)"),
        ("pct_nonwhite", "Percent non-white (%)"),
    ]

    # Drop rows missing demographics
    df_demo = df.dropna(subset=["median_household_income", "total_violations"])

    for var, label in demo_vars:
        if var in df_demo.columns:
            r, p = stats.pearsonr(df_demo[var], df_demo["total_violations"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            results.append(f"   {label:40s}  r = {r:+.3f}  (p = {p:.4f}) {sig}")

    # ===================================================================
    # 3. REGRESSION MODEL
    # ===================================================================
    results.append("\n\n6. NEGATIVE BINOMIAL REGRESSION\n")
    results.append("   DV: total violation count")
    results.append("   IVs: log(population served), poverty rate, pct nonwhite,")
    results.append("         source water type (ground vs. surface)\n")

    # Prepare regression data
    df_reg = df.dropna(subset=[
        "total_violations", "population_served",
        "poverty_rate", "pct_nonwhite"
    ]).copy()

    # Log-transform population (it's heavily right-skewed)
    df_reg = df_reg[df_reg["population_served"] > 0].copy()
    df_reg["log_pop"] = np.log(df_reg["population_served"])

    # Create groundwater indicator if source_water_type exists
    if "source_water_type" in df_reg.columns:
        df_reg["is_groundwater"] = (
            df_reg["source_water_type"].str.upper().str.contains("GROUND", na=False)
        ).astype(int)
    else:
        df_reg["is_groundwater"] = 0

    try:
        model = smf.negativebinomial(
            "total_violations ~ log_pop + poverty_rate + pct_nonwhite + is_groundwater",
            data=df_reg
        ).fit(disp=False)

        results.append(model.summary().as_text())

    except Exception as e:
        results.append(f"   Model failed to converge: {e}")
        results.append("   Falling back to OLS for reference ...")

        try:
            ols_model = smf.ols(
                "total_violations ~ log_pop + poverty_rate + pct_nonwhite + is_groundwater",
                data=df_reg
            ).fit()
            results.append(ols_model.summary().as_text())
        except Exception as e2:
            results.append(f"   OLS also failed: {e2}")

    # ===================================================================
    # Save results
    # ===================================================================
    results_text = "\n".join(results)
    print(results_text)

    outpath = os.path.join(OUTPUT_DIR, "model_results.txt")
    with open(outpath, "w") as f:
        f.write(results_text)
    print(f"\n✓ Results saved → {outpath}")


if __name__ == "__main__":
    main()
