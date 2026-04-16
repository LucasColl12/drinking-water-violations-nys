"""
06_visualize.py
===============
Generates maps and figures for the health-based SDWA violation analysis,
comparing 2000–2005 (historical) vs. 2020–2025 (recent).

Outputs (saved to output/figures/):
    1. fig1_county_comparison.png     — Side-by-side county violation rates
    2. fig2_size_comparison.png       — Grouped bars: violation rates by size × period
    3. fig3_income_scatter.png        — Scatter: income vs. violations, faceted by period
    4. fig4_system_map.html           — Interactive Folium map (color = change status)
    5. fig5_slope_chart.png           — Slope chart: top counties then vs. now
    6. fig6_equity_gaps.png           — Equity gap comparison across periods

Requires:
    output/analytical_dataset.csv       (long format, from 04)
    output/analytical_dataset_wide.csv  (wide format, from 04)

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
# Paths and config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

HIST_LABEL = "2000–2005"
REC_LABEL = "2020–2025"
PERIOD_COLORS = {HIST_LABEL: "#2c7bb6", REC_LABEL: "#d7191c"}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 5)

# ---------------------------------------------------------------------------
# Figure 1: County violation rate comparison (side-by-side bars)
# ---------------------------------------------------------------------------

def fig1_county_comparison(long: pd.DataFrame):
    """
    Top 20 counties by health violation rate, comparing periods.
    Uses horizontal grouped bar chart.
    """
    print("  [1/6] County comparison ...")

    county_period = (
        long.groupby(["county_served", "period"])
        .agg(
            total_viols=("health_violations", "sum"),
            pop_served=("population_served", "sum"),
            n_systems=("pwsid", "count"),
        )
        .reset_index()
    )
    county_period["rate_per_1k"] = (
        county_period["total_viols"] / county_period["pop_served"] * 1000
    ).replace([np.inf, -np.inf], np.nan)

    # Get top 20 counties by combined violation rate across both periods
    county_total = (
        county_period.groupby("county_served")["total_viols"]
        .sum()
        .nlargest(20)
        .index
    )
    plot_data = county_period[county_period["county_served"].isin(county_total)]

    fig, ax = plt.subplots(figsize=(12, 9))

    pivot = plot_data.pivot_table(
        index="county_served", columns="period",
        values="rate_per_1k", fill_value=0
    )
    # Sort by recent period rate
    if REC_LABEL in pivot.columns:
        pivot = pivot.sort_values(REC_LABEL, ascending=True)

    pivot.plot(kind="barh", ax=ax, color=[PERIOD_COLORS.get(c, "gray")
                                           for c in pivot.columns],
               width=0.7)

    ax.set_xlabel("Health Violations per 1,000 People Served")
    ax.set_title("Top 20 Counties: Health-Based Violation Rates\n"
                 "2000–2005 vs. 2020–2025", fontsize=14)
    ax.legend(title="Period")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_county_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Violations by system size (grouped bars)
# ---------------------------------------------------------------------------

def fig2_size_comparison(long: pd.DataFrame):
    """Grouped bar chart: violation rates by system size and period."""
    print("  [2/6] System size comparison ...")

    size_order = ["Very Small", "Small", "Medium", "Large", "Very Large"]

    size_period = (
        long.groupby(["system_size_category", "period"])
        .agg(
            pct_with_viol=("health_violations", lambda x: (x > 0).mean() * 100),
            mean_viols=("health_violations", "mean"),
        )
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: % with at least one violation
    pivot_pct = size_period.pivot_table(
        index="system_size_category", columns="period",
        values="pct_with_viol", fill_value=0
    ).reindex(size_order).dropna(how="all")

    pivot_pct.plot(kind="bar", ax=ax1,
                   color=[PERIOD_COLORS.get(c, "gray") for c in pivot_pct.columns],
                   width=0.7)
    ax1.set_ylabel("% of systems with ≥1 health violation")
    ax1.set_title("Share of Systems with Health Violations")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(title="Period")

    # Right: mean violations
    pivot_mean = size_period.pivot_table(
        index="system_size_category", columns="period",
        values="mean_viols", fill_value=0
    ).reindex(size_order).dropna(how="all")

    pivot_mean.plot(kind="bar", ax=ax2,
                    color=[PERIOD_COLORS.get(c, "gray") for c in pivot_mean.columns],
                    width=0.7)
    ax2.set_ylabel("Mean health violations per system")
    ax2.set_title("Average Health Violations per System")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(title="Period")

    fig.suptitle("Health-Based SDWA Violations by System Size — NY State",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_size_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Income vs. violations scatter (faceted by period)
# ---------------------------------------------------------------------------

def fig3_income_scatter(long: pd.DataFrame):
    """
    County-level scatter: median income vs. mean health violations,
    one panel per period.
    """
    print("  [3/6] Income scatter by period ...")

    county_agg = (
        long.dropna(subset=["median_household_income"])
        .groupby(["county_served", "period"])
        .agg(
            mean_violations=("health_violations", "mean"),
            median_income=("median_household_income", "first"),
            poverty_rate=("poverty_rate", "first"),
            n_systems=("pwsid", "count"),
        )
        .reset_index()
    )

    periods = [HIST_LABEL, REC_LABEL]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, period_label in zip(axes, periods):
        pdf = county_agg[county_agg["period"] == period_label]

        scatter = ax.scatter(
            pdf["median_income"],
            pdf["mean_violations"],
            s=pdf["n_systems"] * 10,
            c=pdf["poverty_rate"],
            cmap="YlOrRd",
            alpha=0.7,
            edgecolors="gray",
            linewidth=0.5,
        )

        # Trend line
        mask = pdf[["median_income", "mean_violations"]].dropna().index
        if len(mask) > 2:
            from scipy import stats as sp_stats
            slope, intercept, r, p, se = sp_stats.linregress(
                pdf.loc[mask, "median_income"],
                pdf.loc[mask, "mean_violations"],
            )
            x_line = np.linspace(pdf["median_income"].min(),
                                 pdf["median_income"].max(), 100)
            ax.plot(x_line, intercept + slope * x_line, "k--", alpha=0.5,
                    label=f"r = {r:.3f}, p = {p:.3f}")
            ax.legend(fontsize=10)

        ax.set_xlabel("Median Household Income ($)")
        ax.set_title(period_label, fontsize=13, fontweight="bold",
                     color=PERIOD_COLORS[period_label])
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

    axes[0].set_ylabel("Mean Health Violations per System")
    cbar = plt.colorbar(scatter, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Poverty rate (%)")

    fig.suptitle("County Income vs. Health Violations — NY State\n"
                 "Each point = one county, sized by # of systems",
                 fontsize=14, y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_income_scatter.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Interactive map (color by change status)
# ---------------------------------------------------------------------------

def fig4_interactive_map(wide: pd.DataFrame):
    """
    Interactive Folium map of water systems colored by violation change:
    green = improved, red = worsened, yellow = persistent, gray = clean both.
    """
    print("  [4/6] Interactive map ...")

    try:
        import folium
    except ImportError:
        print("    Folium not installed, skipping interactive map.")
        return

    mapped = wide.dropna(subset=["latitude", "longitude"]).copy()
    if mapped.empty:
        print("    No geocoded systems available, skipping map.")
        return

    hist_col = "health_violations_2000_2005"
    rec_col = "health_violations_2020_2025"

    if hist_col not in mapped.columns or rec_col not in mapped.columns:
        print("    Period columns not found, skipping map.")
        return

    # Classify change status
    def classify_change(row):
        h, r = row[hist_col], row[rec_col]
        if h == 0 and r == 0:
            return ("clean", "gray", "No violations in either period")
        elif h > 0 and r == 0:
            return ("improved", "green", "Had violations → now clean")
        elif h == 0 and r > 0:
            return ("worsened", "red", "Was clean → now has violations")
        elif r > h:
            return ("worsened", "red", f"Violations increased ({h} → {r})")
        elif r < h:
            return ("improved", "green", f"Violations decreased ({h} → {r})")
        else:
            return ("persistent", "orange", f"Same violation count ({h})")

    mapped[["change_status", "color", "change_desc"]] = (
        mapped.apply(classify_change, axis=1, result_type="expand")
    )

    center_lat = mapped["latitude"].median()
    center_lon = mapped["longitude"].median()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                   tiles="cartodbpositron")

    for _, row in mapped.iterrows():
        radius = 3 + min(max(row[hist_col], row[rec_col]) * 0.5, 15)
        popup_text = (
            f"<b>{row.get('pws_name', 'Unknown')}</b><br>"
            f"PWSID: {row['pwsid']}<br>"
            f"Pop. served: {row.get('population_served', 'N/A'):,.0f}<br>"
            f"County: {row.get('county_served', 'N/A')}<br>"
            f"<hr>"
            f"Health violations (2000–2005): {row[hist_col]}<br>"
            f"Health violations (2020–2025): {row[rec_col]}<br>"
            f"<b>Status: {row['change_desc']}</b>"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=row["color"],
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=280),
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:5px;
                border:2px solid gray; font-size:13px;">
        <b>Violation Change</b><br>
        <i style="color:green">●</i> Improved<br>
        <i style="color:red">●</i> Worsened<br>
        <i style="color:orange">●</i> Persistent<br>
        <i style="color:gray">●</i> Clean (both periods)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    outpath = os.path.join(FIG_DIR, "fig4_system_map.html")
    m.save(outpath)
    print(f"    Saved interactive map → {outpath}")


# ---------------------------------------------------------------------------
# Figure 5: Slope chart — county-level change over time
# ---------------------------------------------------------------------------

def fig5_slope_chart(long: pd.DataFrame):
    """
    Slope chart showing the top 15 counties' health violation rates
    in the historical vs. recent period.
    """
    print("  [5/6] Slope chart ...")

    county_period = (
        long.groupby(["county_served", "period"])
        .agg(
            total_viols=("health_violations", "sum"),
            pop_served=("population_served", "sum"),
        )
        .reset_index()
    )
    county_period["rate_per_1k"] = (
        county_period["total_viols"] / county_period["pop_served"] * 1000
    ).replace([np.inf, -np.inf], np.nan)

    # Pivot to wide for slope chart
    pivot = county_period.pivot_table(
        index="county_served", columns="period",
        values="rate_per_1k", fill_value=0
    )

    if HIST_LABEL not in pivot.columns or REC_LABEL not in pivot.columns:
        print("    Missing period columns, skipping slope chart.")
        return

    # Top 15 counties by max rate in either period
    pivot["max_rate"] = pivot[[HIST_LABEL, REC_LABEL]].max(axis=1)
    top15 = pivot.nlargest(15, "max_rate")

    fig, ax = plt.subplots(figsize=(10, 8))

    for county in top15.index:
        hist_val = top15.loc[county, HIST_LABEL]
        rec_val = top15.loc[county, REC_LABEL]

        # Color by direction of change
        color = "#2ca02c" if rec_val < hist_val else "#d62728" if rec_val > hist_val else "#999999"
        linewidth = 2 if abs(rec_val - hist_val) > 0.5 else 1

        ax.plot([0, 1], [hist_val, rec_val], color=color,
                linewidth=linewidth, alpha=0.7)
        ax.scatter([0, 1], [hist_val, rec_val], color=color, s=40, zorder=5)

        # Labels
        ax.annotate(county, xy=(0, hist_val), xytext=(-0.05, hist_val),
                    ha="right", va="center", fontsize=8)
        ax.annotate(county, xy=(1, rec_val), xytext=(1.05, rec_val),
                    ha="left", va="center", fontsize=8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([HIST_LABEL, REC_LABEL], fontsize=12, fontweight="bold")
    ax.set_ylabel("Health Violations per 1,000 People Served")
    ax.set_title("County-Level Health Violation Rate Changes\n"
                 "Top 15 Counties — New York State", fontsize=14)
    ax.set_xlim(-0.3, 1.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ca02c", lw=2, label="Improved"),
        Line2D([0], [0], color="#d62728", lw=2, label="Worsened"),
        Line2D([0], [0], color="#999999", lw=2, label="No change"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_slope_chart.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: Equity gap comparison
# ---------------------------------------------------------------------------

def fig6_equity_gaps(long: pd.DataFrame):
    """
    Bar chart comparing the violation rate gap between
    high-poverty vs. low-poverty tracts across periods.
    """
    print("  [6/6] Equity gap comparison ...")

    gaps = []

    for period_label in [HIST_LABEL, REC_LABEL]:
        pdf = long[(long["period"] == period_label)].dropna(
            subset=["poverty_rate", "health_violations"]
        )
        if len(pdf) < 20:
            continue

        pov_med = pdf["poverty_rate"].median()
        high_pov_mean = pdf[pdf["poverty_rate"] >= pov_med]["health_violations"].mean()
        low_pov_mean = pdf[pdf["poverty_rate"] < pov_med]["health_violations"].mean()

        gaps.append({
            "period": period_label,
            "metric": "Poverty gap",
            "high_group": high_pov_mean,
            "low_group": low_pov_mean,
            "gap": high_pov_mean - low_pov_mean,
        })

        if "pct_nonwhite" in pdf.columns:
            maj_nw_mean = pdf[pdf["pct_nonwhite"] >= 50]["health_violations"].mean()
            maj_w_mean = pdf[pdf["pct_nonwhite"] < 50]["health_violations"].mean()

            gaps.append({
                "period": period_label,
                "metric": "Race gap",
                "high_group": maj_nw_mean,
                "low_group": maj_w_mean,
                "gap": maj_nw_mean - maj_w_mean,
            })

    if not gaps:
        print("    Not enough data for equity gap chart.")
        return

    gap_df = pd.DataFrame(gaps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute mean violations by group and period
    for i, metric in enumerate(["Poverty gap", "Race gap"]):
        ax = ax1 if i == 0 else ax2
        mdf = gap_df[gap_df["metric"] == metric]

        if mdf.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        x = np.arange(len(mdf))
        width = 0.35

        if metric == "Poverty gap":
            labels = ("High-poverty tracts", "Low-poverty tracts")
        else:
            labels = ("Majority non-white", "Majority white")

        bars1 = ax.bar(x - width / 2, mdf["high_group"], width,
                       label=labels[0], color="#d7191c", alpha=0.8)
        bars2 = ax.bar(x + width / 2, mdf["low_group"], width,
                       label=labels[1], color="#2c7bb6", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(mdf["period"])
        ax.set_ylabel("Mean Health Violations per System")
        ax.set_title(f"{metric.replace('gap', 'Gap')}: "
                     f"{'Poverty' if i == 0 else 'Racial'} Disparity",
                     fontsize=13)
        ax.legend()

        # Annotate the gap
        for j, (_, row) in enumerate(mdf.iterrows()):
            gap_val = row["gap"]
            ax.annotate(f"Gap: {gap_val:.3f}",
                        xy=(j, max(row["high_group"], row["low_group"])),
                        xytext=(j, max(row["high_group"], row["low_group"]) * 1.1),
                        ha="center", fontsize=10, fontweight="bold",
                        color="darkred" if gap_val > 0 else "darkgreen")

    fig.suptitle("Environmental Justice: Health Violation Disparities\n"
                 "2000–2005 vs. 2020–2025 — NY State", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_equity_gaps.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    long = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_dataset.csv"))
    wide = pd.read_csv(os.path.join(OUTPUT_DIR, "analytical_dataset_wide.csv"))

    print(f"Loaded long dataset:  {len(long):,} rows")
    print(f"Loaded wide dataset:  {len(wide):,} rows")

    # Standardize column names (handle ECHO Exporter vs. SDWIS naming)
    for df in [long, wide]:
        renames = {}
        for canonical, variants in {
            "county_served": ["fac_county"],
            "pws_name": ["fac_name"],
            "city_served": ["fac_city"],
        }.items():
            if canonical not in df.columns:
                for v in variants:
                    if v in df.columns:
                        renames[v] = canonical
                        break
        if renames:
            df.rename(columns=renames, inplace=True)

    print(f"  Columns: {list(long.columns)[:12]}...\n")

    fig1_county_comparison(long)
    fig2_size_comparison(long)
    fig3_income_scatter(long)
    fig4_interactive_map(wide)
    fig5_slope_chart(long)
    fig6_equity_gaps(long)

    print(f"\n✓ All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()