"""
05_analysis.py
==============
Statistical analysis of health-based SDWA violation patterns across
New York State, comparing 2000–2005 (historical) vs. 2020–2025 (recent).
 
Analyses:
    1. Descriptive statistics — violation rates by period, system size,
       source type, and ownership
    2. Period comparison — paired tests for changes in violation rates
    3. Correlation analysis — demographics vs. violations in each period
    4. Regression models — negative binomial for each period, plus a
       pooled model with period interaction terms
    5. Equity gap analysis — are demographic disparities widening or
       narrowing over time?
 
Requires:
    output/analytical_dataset.csv       (long format, from 04)
    output/analytical_dataset_wide.csv  (wide format, from 04)
 
Outputs:
    output/model_results.txt
 
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
 
HIST_LABEL = "2000–2005"
REC_LABEL = "2020–2025"
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    long = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_dataset.csv"))
    wide = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_dataset_wide.csv"))
 
    print(f"Loaded long dataset:  {len(long):,} rows")
    print(f"Loaded wide dataset:  {len(wide):,} rows\n")
 
    results = []
 
    # ===================================================================
    # 1. OVERVIEW
    # ===================================================================
    results.append("=" * 70)
    results.append("HEALTH-BASED SDWA VIOLATION ANALYSIS — NEW YORK STATE")
    results.append("Comparing 2000–2005 (Historical) vs. 2020–2025 (Recent)")
    results.append("=" * 70)
 
    results.append("\n1. OVERVIEW BY PERIOD\n")
 
    for period_label in [HIST_LABEL, REC_LABEL]:
        pdf = long[long["period"] == period_label]
        total_sys = len(pdf)
        with_viol = (pdf["health_violations"] > 0).sum()
        total_viols = pdf["health_violations"].sum()
 
        results.append(f"   --- {period_label} ---")
        results.append(f"   Total community water systems:  {total_sys:,}")
        results.append(f"   Systems with ≥1 health violation: {with_viol:,} "
                       f"({with_viol/total_sys*100:.1f}%)")
        results.append(f"   Total health violations:        {total_viols:,}")
        results.append(f"   Mean violations per system:     {pdf['health_violations'].mean():.3f}")
        results.append(f"   Median:                         {pdf['health_violations'].median():.1f}")
        results.append("")
 
    # ===================================================================
    # 2. VIOLATIONS BY SYSTEM SIZE — PERIOD COMPARISON
    # ===================================================================
    results.append("\n2. HEALTH VIOLATIONS BY SYSTEM SIZE (Period Comparison)\n")
 
    size_order = ["Very Small", "Small", "Medium", "Large", "Very Large"]
 
    for period_label in [HIST_LABEL, REC_LABEL]:
        results.append(f"   --- {period_label} ---")
        pdf = long[long["period"] == period_label]
        size_table = (
            pdf.groupby("system_size_category")
            .agg(
                n_systems=("pwsid", "count"),
                pct_with_violation=("health_violations",
                                    lambda x: (x > 0).mean() * 100),
                mean_violations=("health_violations", "mean"),
                total_violations=("health_violations", "sum"),
            )
            .reindex(size_order)
            .round(2)
        )
        results.append(size_table.to_string())
        results.append("")
 
    # ===================================================================
    # 3. VIOLATIONS BY SOURCE WATER TYPE
    # ===================================================================
    results.append("\n3. HEALTH VIOLATIONS BY SOURCE WATER TYPE\n")
 
    if "source_water_type" in long.columns:
        for period_label in [HIST_LABEL, REC_LABEL]:
            results.append(f"   --- {period_label} ---")
            pdf = long[long["period"] == period_label]
            source_table = (
                pdf.groupby("source_water_type")
                .agg(
                    n_systems=("pwsid", "count"),
                    pct_with_violation=("health_violations",
                                        lambda x: (x > 0).mean() * 100),
                    mean_violations=("health_violations", "mean"),
                )
                .round(2)
                .sort_values("n_systems", ascending=False)
            )
            results.append(source_table.to_string())
            results.append("")
 
    # ===================================================================
    # 4. VIOLATIONS BY OWNERSHIP TYPE
    # ===================================================================
    results.append("\n4. HEALTH VIOLATIONS BY OWNERSHIP TYPE\n")
 
    if "owner_type" in long.columns:
        for period_label in [HIST_LABEL, REC_LABEL]:
            results.append(f"   --- {period_label} ---")
            pdf = long[long["period"] == period_label]
            owner_table = (
                pdf.groupby("owner_type")
                .agg(
                    n_systems=("pwsid", "count"),
                    pct_with_violation=("health_violations",
                                        lambda x: (x > 0).mean() * 100),
                    mean_violations=("health_violations", "mean"),
                )
                .round(2)
                .sort_values("n_systems", ascending=False)
            )
            results.append(owner_table.to_string())
            results.append("")
 
    # ===================================================================
    # 5. PAIRED PERIOD COMPARISON (system-level changes)
    # ===================================================================
    results.append("\n5. SYSTEM-LEVEL CHANGE ANALYSIS (Historical → Recent)\n")
 
    hist_col = "health_violations_2000_2005"
    rec_col = "health_violations_2020_2025"
 
    if hist_col in wide.columns and rec_col in wide.columns:
        # Overall change
        had_hist = (wide[hist_col] > 0).sum()
        had_rec = (wide[rec_col] > 0).sum()
        results.append(f"   Systems with ≥1 health violation:")
        results.append(f"     Historical (2000–2005): {had_hist:,}")
        results.append(f"     Recent (2020–2025):     {had_rec:,}")
        results.append(f"     Change:                 {had_rec - had_hist:+,}")
 
        # Mean violations
        mean_hist = wide[hist_col].mean()
        mean_rec = wide[rec_col].mean()
        results.append(f"\n   Mean health violations per system:")
        results.append(f"     Historical: {mean_hist:.3f}")
        results.append(f"     Recent:     {mean_rec:.3f}")
        results.append(f"     Change:     {mean_rec - mean_hist:+.3f}")
 
        # Wilcoxon signed-rank test (paired, non-parametric)
        paired = wide[[hist_col, rec_col]].dropna()
        if len(paired) > 20:
            try:
                stat, p = stats.wilcoxon(paired[hist_col], paired[rec_col],
                                         alternative="two-sided",
                                         zero_method="zsplit")
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                results.append(f"\n   Wilcoxon signed-rank test:")
                results.append(f"     statistic = {stat:,.1f}, p = {p:.4f} {sig}")
            except Exception as e:
                results.append(f"\n   Wilcoxon test failed: {e}")
 
        # McNemar test: did systems gain/lose violation status?
        a = ((wide[hist_col] == 0) & (wide[rec_col] == 0)).sum()  # clean both
        b = ((wide[hist_col] == 0) & (wide[rec_col] > 0)).sum()   # gained
        c = ((wide[hist_col] > 0) & (wide[rec_col] == 0)).sum()   # lost
        d = ((wide[hist_col] > 0) & (wide[rec_col] > 0)).sum()    # both
 
        results.append(f"\n   Transition matrix (violation status):")
        results.append(f"     Clean → Clean:      {a:,}")
        results.append(f"     Clean → Violated:   {b:,}  (newly non-compliant)")
        results.append(f"     Violated → Clean:   {c:,}  (improved)")
        results.append(f"     Violated → Violated: {d:,}  (persistent)")
 
        if b + c > 0:
            try:
                mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
                mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
                sig = "***" if mcnemar_p < 0.001 else "**" if mcnemar_p < 0.01 else "*" if mcnemar_p < 0.05 else ""
                results.append(f"\n   McNemar's test (asymmetry in transitions):")
                results.append(f"     χ² = {mcnemar_stat:.2f}, p = {mcnemar_p:.4f} {sig}")
            except Exception as e:
                results.append(f"\n   McNemar test failed: {e}")
 
        # Change by system size
        results.append(f"\n   Change by system size:")
        for size in ["Very Small", "Small", "Medium", "Large", "Very Large"]:
            subset = wide[wide["system_size_category"] == size]
            if len(subset) > 0:
                h = subset[hist_col].mean()
                r = subset[rec_col].mean()
                results.append(f"     {size:12s}  {h:.3f} → {r:.3f}  ({r-h:+.3f})")
 
    # ===================================================================
    # 6. CORRELATION ANALYSIS — BY PERIOD
    # ===================================================================
    results.append("\n\n6. CORRELATIONS: HEALTH VIOLATIONS vs. DEMOGRAPHICS\n")
 
    demo_vars = [
        ("median_household_income", "Median household income"),
        ("poverty_rate", "Poverty rate (%)"),
        ("pct_nonwhite", "Percent non-white (%)"),
    ]
 
    for period_label in [HIST_LABEL, REC_LABEL]:
        results.append(f"   --- {period_label} ---")
        pdf = long[(long["period"] == period_label)].dropna(
            subset=["median_household_income", "health_violations"]
        )
 
        for var, label in demo_vars:
            if var in pdf.columns and pdf[var].notna().sum() > 10:
                r, p = stats.pearsonr(pdf[var], pdf["health_violations"])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                results.append(
                    f"   {label:40s}  r = {r:+.3f}  (p = {p:.4f}) {sig}"
                )
        results.append("")
 
    # ===================================================================
    # 7. REGRESSION MODELS — PERIOD-SPECIFIC
    # ===================================================================
    results.append("\n7. NEGATIVE BINOMIAL REGRESSION (by period)\n")
    results.append("   DV: health violation count")
    results.append("   IVs: log(population served), poverty rate, pct nonwhite,")
    results.append("         source water type (ground vs. surface)\n")
 
    for period_label in [HIST_LABEL, REC_LABEL]:
        results.append(f"   {'='*60}")
        results.append(f"   {period_label}")
        results.append(f"   {'='*60}")
 
        pdf = long[long["period"] == period_label].copy()
        df_reg = pdf.dropna(subset=[
            "health_violations", "population_served",
            "poverty_rate", "pct_nonwhite"
        ]).copy()
 
        df_reg = df_reg[df_reg["population_served"] > 0].copy()
        df_reg["log_pop"] = np.log(df_reg["population_served"])
 
        if "source_water_type" in df_reg.columns:
            df_reg["is_groundwater"] = (
                df_reg["source_water_type"].str.upper()
                .str.contains("GROUND", na=False)
            ).astype(int)
        else:
            df_reg["is_groundwater"] = 0
 
        n_obs = len(df_reg)
        results.append(f"   N = {n_obs:,}")
 
        if n_obs < 30:
            results.append("   Too few observations for regression.\n")
            continue
 
        try:
            model = smf.negativebinomial(
                "health_violations ~ log_pop + poverty_rate + pct_nonwhite + is_groundwater",
                data=df_reg
            ).fit(disp=False)
            results.append(model.summary().as_text())
 
        except Exception as e:
            results.append(f"   NB model failed: {e}")
            results.append("   Falling back to OLS ...")
            try:
                ols_model = smf.ols(
                    "health_violations ~ log_pop + poverty_rate + pct_nonwhite + is_groundwater",
                    data=df_reg
                ).fit()
                results.append(ols_model.summary().as_text())
            except Exception as e2:
                results.append(f"   OLS also failed: {e2}")
 
        results.append("")
 
    # ===================================================================
    # 8. POOLED MODEL WITH PERIOD INTERACTION
    # ===================================================================
    results.append("\n8. POOLED MODEL WITH PERIOD INTERACTION\n")
    results.append("   Tests whether the relationship between demographics")
    results.append("   and violations has changed over time.\n")
 
    df_pool = long.dropna(subset=[
        "health_violations", "population_served",
        "poverty_rate", "pct_nonwhite"
    ]).copy()
    df_pool = df_pool[df_pool["population_served"] > 0].copy()
    df_pool["log_pop"] = np.log(df_pool["population_served"])
 
    if "source_water_type" in df_pool.columns:
        df_pool["is_groundwater"] = (
            df_pool["source_water_type"].str.upper()
            .str.contains("GROUND", na=False)
        ).astype(int)
    else:
        df_pool["is_groundwater"] = 0
 
    # Create binary period indicator (1 = recent)
    df_pool["is_recent"] = (df_pool["period"] == REC_LABEL).astype(int)
 
    results.append(f"   N = {len(df_pool):,}")
 
    try:
        pooled = smf.negativebinomial(
            "health_violations ~ log_pop + poverty_rate + pct_nonwhite "
            "+ is_groundwater + is_recent "
            "+ is_recent:poverty_rate + is_recent:pct_nonwhite",
            data=df_pool
        ).fit(disp=False)
        results.append(pooled.summary().as_text())
 
        results.append("\n   Key interaction terms:")
        results.append("   - is_recent:poverty_rate  → Has the poverty–violation")
        results.append("     relationship changed between periods?")
        results.append("   - is_recent:pct_nonwhite  → Has the race–violation")
        results.append("     relationship changed between periods?")
 
    except Exception as e:
        results.append(f"   Pooled model failed: {e}")
 
    # ===================================================================
    # 9. EQUITY GAP ANALYSIS
    # ===================================================================
    results.append("\n\n9. EQUITY GAP ANALYSIS\n")
    results.append("   Comparing violation rates in high-poverty vs. low-poverty tracts,")
    results.append("   and majority-nonwhite vs. majority-white tracts, across periods.\n")
 
    for period_label in [HIST_LABEL, REC_LABEL]:
        results.append(f"   --- {period_label} ---")
        pdf = long[(long["period"] == period_label)].dropna(
            subset=["poverty_rate", "health_violations"]
        )
 
        if len(pdf) < 20:
            results.append("   Too few observations.\n")
            continue
 
        # High poverty (above median) vs low poverty
        pov_median = pdf["poverty_rate"].median()
        high_pov = pdf[pdf["poverty_rate"] >= pov_median]["health_violations"]
        low_pov = pdf[pdf["poverty_rate"] < pov_median]["health_violations"]
 
        results.append(f"   Poverty threshold (median): {pov_median:.1f}%")
        results.append(f"     High-poverty tracts: mean = {high_pov.mean():.3f} "
                       f"(n={len(high_pov):,})")
        results.append(f"     Low-poverty tracts:  mean = {low_pov.mean():.3f} "
                       f"(n={len(low_pov):,})")
 
        if len(high_pov) > 5 and len(low_pov) > 5:
            u_stat, u_p = stats.mannwhitneyu(high_pov, low_pov, alternative="two-sided")
            sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else ""
            results.append(f"     Mann-Whitney U: p = {u_p:.4f} {sig}")
 
        # Majority nonwhite vs white
        if "pct_nonwhite" in pdf.columns:
            maj_nw = pdf[pdf["pct_nonwhite"] >= 50]["health_violations"]
            maj_w = pdf[pdf["pct_nonwhite"] < 50]["health_violations"]
 
            results.append(f"\n   Majority non-white (≥50%) vs. majority white:")
            results.append(f"     Majority non-white: mean = {maj_nw.mean():.3f} "
                           f"(n={len(maj_nw):,})")
            results.append(f"     Majority white:     mean = {maj_w.mean():.3f} "
                           f"(n={len(maj_w):,})")
 
            if len(maj_nw) > 5 and len(maj_w) > 5:
                u_stat2, u_p2 = stats.mannwhitneyu(maj_nw, maj_w,
                                                     alternative="two-sided")
                sig2 = "***" if u_p2 < 0.001 else "**" if u_p2 < 0.01 else "*" if u_p2 < 0.05 else ""
                results.append(f"     Mann-Whitney U: p = {u_p2:.4f} {sig2}")
 
        results.append("")
 
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
