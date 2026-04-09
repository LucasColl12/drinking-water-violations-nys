"""
06_visualize.py
===============
Generates maps and figures for the SDWA violation analysis.

Outputs (saved to output/figures/):
    1. fig1_violations_by_county.png    — Choropleth of violation rates by county
    2. fig2_violations_by_size.png      — Bar chart of violation rates by system size
    3. fig3_income_vs_violations.png    — Scatter: median income vs. violation rate
    4. fig4_system_map.html             — Interactive Folium map of all systems
    5. fig5_violation_types.png         — Stacked bar of violation categories

Requires:
    output/analytical_dataset.csv (from 04_merge_and_clean.py)

Usage:
    python scripts/06_visualize.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 5)

# ---------------------------------------------------------------------------
# Figure 1: Violations by county (choropleth)
# ---------------------------------------------------------------------------

def fig1_violations_by_county(df: pd.DataFrame):
    """
    County-level choropleth of violation rates.
    Uses GeoPandas with Census TIGER/Line county boundaries.
    Falls back to a horizontal bar chart if shapefiles aren't available.
    """
    print("  [1/5] County violation rates ...")

    county_stats = (
        df.groupby("county_served")
        .agg(
            n_systems=("pwsid", "count"),
            total_violations=("total_violations", "sum"),
            pop_served=("population_served", "sum"),
        )
        .reset_index()
    )
    county_stats["viol_per_1k"] = (
        county_stats["total_violations"] / county_stats["pop_served"] * 1000
    ).replace([np.inf, -np.inf], np.nan)

    # Try GeoPandas choropleth
    try:
        import geopandas as gpd

        # Download NY county boundaries from Census TIGER
        tiger_url = (
            "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/"
            "tl_2022_us_county.zip"
        )
        counties_gdf = gpd.read_file(tiger_url)
        ny_counties = counties_gdf[counties_gdf["STATEFP"] == "36"].copy()
        ny_counties["county_match"] = ny_counties["NAME"].str.upper()

        county_stats["county_match"] = county_stats["county_served"].str.upper().str.strip()
        merged = ny_counties.merge(county_stats, on="county_match", how="left")

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        merged.plot(
            column="viol_per_1k",
            cmap="YlOrRd",
            linewidth=0.5,
            edgecolor="gray",
            legend=True,
            legend_kwds={"label": "Violations per 1,000 people served", "shrink": 0.6},
            missing_kwds={"color": "lightgray", "label": "No data"},
            ax=ax,
        )
        ax.set_title("Drinking Water Violations by County\nNew York State", fontsize=16)
        ax.axis("off")

    except Exception as e:
        print(f"    GeoPandas choropleth failed ({e}), using bar chart fallback")

        # Fallback: top 20 counties bar chart
        top20 = county_stats.dropna(subset=["viol_per_1k"]).nlargest(20, "viol_per_1k")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top20["county_served"], top20["viol_per_1k"], color=PALETTE[3])
        ax.set_xlabel("Violations per 1,000 people served")
        ax.set_title("Top 20 Counties by Drinking Water Violation Rate\nNew York State",
                      fontsize=14)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_violations_by_county.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Violations by system size
# ---------------------------------------------------------------------------

def fig2_violations_by_size(df: pd.DataFrame):
    """Bar chart comparing violation rates across EPA system size categories."""
    print("  [2/5] Violations by system size ...")

    size_order = ["Very Small", "Small", "Medium", "Large", "Very Large"]
    size_stats = (
        df.groupby("system_size_category")
        .agg(
            pct_with_violation=("total_violations", lambda x: (x > 0).mean() * 100),
            mean_violations=("total_violations", "mean"),
        )
        .reindex(size_order)
        .dropna()
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: % of systems with at least one violation
    ax1.bar(size_stats["system_size_category"],
            size_stats["pct_with_violation"],
            color=PALETTE)
    ax1.set_ylabel("% of systems with ≥1 violation")
    ax1.set_title("Share of Systems with Violations")
    ax1.tick_params(axis="x", rotation=30)

    # Right: mean violation count
    ax2.bar(size_stats["system_size_category"],
            size_stats["mean_violations"],
            color=PALETTE)
    ax2.set_ylabel("Mean violation count per system")
    ax2.set_title("Average Violations per System")
    ax2.tick_params(axis="x", rotation=30)

    fig.suptitle("SDWA Violations by Water System Size — New York State",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_violations_by_size.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Income vs. violations scatter
# ---------------------------------------------------------------------------

def fig3_income_scatter(df: pd.DataFrame):
    """
    County-level scatter plot: median household income vs. mean violations.
    Each point is one county, sized by number of systems.
    """
    print("  [3/5] Income vs. violations scatter ...")

    county_agg = (
        df.dropna(subset=["median_household_income"])
        .groupby("county_served")
        .agg(
            mean_violations=("total_violations", "mean"),
            median_income=("median_household_income", "first"),
            poverty_rate=("poverty_rate", "first"),
            n_systems=("pwsid", "count"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        county_agg["median_income"],
        county_agg["mean_violations"],
        s=county_agg["n_systems"] * 10,
        c=county_agg["poverty_rate"],
        cmap="YlOrRd",
        alpha=0.7,
        edgecolors="gray",
        linewidth=0.5,
    )

    # Trend line
    mask = county_agg[["median_income", "mean_violations"]].dropna().index
    if len(mask) > 2:
        from scipy import stats as sp_stats
        slope, intercept, r, p, se = sp_stats.linregress(
            county_agg.loc[mask, "median_income"],
            county_agg.loc[mask, "mean_violations"],
        )
        x_line = np.linspace(county_agg["median_income"].min(),
                             county_agg["median_income"].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "k--", alpha=0.5,
                label=f"r = {r:.3f}, p = {p:.3f}")
        ax.legend(fontsize=11)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Poverty rate (%)")

    ax.set_xlabel("Median Household Income ($)")
    ax.set_ylabel("Mean Violations per System")
    ax.set_title("County Income vs. Drinking Water Violations\nNew York State",
                 fontsize=14)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_income_vs_violations.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Interactive Folium map
# ---------------------------------------------------------------------------

def fig4_interactive_map(df: pd.DataFrame):
    """Interactive map of water systems colored by violation status."""
    print("  [4/5] Interactive Folium map ...")

    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        print("    Folium not installed, skipping interactive map.")
        return

    # Filter to geocoded systems
    mapped = df.dropna(subset=["latitude", "longitude"]).copy()
    if mapped.empty:
        print("    No geocoded systems available, skipping map.")
        return

    # Center on NY
    center_lat = mapped["latitude"].median()
    center_lon = mapped["longitude"].median()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                   tiles="cartodbpositron")

    # Color by violation status
    for _, row in mapped.iterrows():
        color = "red" if row["total_violations"] > 0 else "green"
        popup_text = (
            f"<b>{row.get('pws_name', 'Unknown')}</b><br>"
            f"PWSID: {row['pwsid']}<br>"
            f"Pop. served: {row.get('population_served', 'N/A'):,.0f}<br>"
            f"Violations: {row['total_violations']}<br>"
            f"County: {row.get('county_served', 'N/A')}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3 + min(row["total_violations"] * 0.5, 15),
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=250),
        ).add_to(m)

    outpath = os.path.join(FIG_DIR, "fig4_system_map.html")
    m.save(outpath)
    print(f"    Saved interactive map → {outpath}")


# ---------------------------------------------------------------------------
# Figure 5: Violation type breakdown
# ---------------------------------------------------------------------------

def fig5_violation_types(df: pd.DataFrame):
    """Stacked bar chart showing health vs. monitoring violations by size."""
    print("  [5/5] Violation type breakdown ...")

    size_order = ["Very Small", "Small", "Medium", "Large", "Very Large"]
    type_data = (
        df.groupby("system_size_category")
        .agg(
            health=("health_violations", "sum"),
            monitoring=("monitoring_violations", "sum"),
        )
        .reindex(size_order)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    type_data.plot(kind="bar", stacked=True, ax=ax,
                   color=[PALETTE[4], PALETTE[1]])
    ax.set_ylabel("Total violations")
    ax.set_xlabel("System size category")
    ax.set_title("Health-Based vs. Monitoring Violations by System Size\nNew York State",
                 fontsize=14)
    ax.legend(["Health-based (MCL/TT)", "Monitoring/Reporting"], loc="upper right")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_violation_types.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_dataset.csv"))
    print(f"Loaded {len(df):,} water systems\n")

    fig1_violations_by_county(df)
    fig2_violations_by_size(df)
    fig3_income_scatter(df)
    fig4_interactive_map(df)
    fig5_violation_types(df)

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
