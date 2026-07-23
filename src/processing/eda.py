"""EDA visualisations for screening hit rate data.

Produces 5 publication-quality plots saved to outputs/figures/.
Run standalone: python -m src.viz.eda
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def figures_dir(process: str) -> Path:
    """Figures folder for a pipeline stage, e.g. outputs/detection/figures/."""
    d = OUTPUTS_DIR / process / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir(process: str) -> Path:
    """CSV/results folder for a pipeline stage, e.g. outputs/detection/tables/."""
    d = OUTPUTS_DIR / process / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Palette ───────────────────────────────────────────────────────────────────
GREY = "#8a9bb0"
LIGHT_GREY = "#e8ecf0"

LIST_COLORS = {
    "ListA": "#2c5f8a",
    "ListB": "#e67e22",
    "ListC": "#27ae60",
    "ListD": "#8e44ad",
    "ListE": "#c0392b",
}

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _list_color(col: str) -> str:
    for prefix, color in LIST_COLORS.items():
        if col.startswith(prefix):
            return color
    return GREY


def _save(fig: plt.Figure, name: str, process: str, dpi: int = 150) -> None:
    path = figures_dir(process) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Plot 1: Time series small multiples ───────────────────────────────────────
def plot_time_series_overview(df: pd.DataFrame, series_cols: list[str]) -> None:
    """5×3 grid: one panel per series, full time range."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(5, 3, figsize=(16, 14), sharex=True)
        fig.suptitle(
            "Screening Hit Rate — All Series (2023-12-18 to 2024-01-12 UTC)",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        # day boundaries: one tick per calendar day at midnight UTC
        day_starts = pd.date_range(
            start=df.index[0].normalize(), end=df.index[-1], freq="D", tz="UTC"
        )
        # skip days that fall in weekend gaps (no data)
        valid_day_starts = [d for d in day_starts if df.index[0] <= d <= df.index[-1]]

        for ax, col in zip(axes.flat, series_cols):
            color = _list_color(col)
            s = df[col]
            ax.plot(s.index, s.values, linewidth=0.8, color=color, alpha=0.85)
            ax.fill_between(s.index, 0, s.values, alpha=0.12, color=color)
            ax.set_title(col, fontsize=9, fontweight="bold", color=color)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

            # vertical day lines + date labels at top of each panel
            for d in valid_day_starts:
                ax.axvline(d, color="#cccccc", linewidth=0.5, zorder=0)
            ax.set_xticks(valid_day_starts)
            ax.set_xticklabels(
                [d.strftime("%b %d") for d in valid_day_starts],
                rotation=60,
                ha="right",
                fontsize=6.5,
            )
            ax.tick_params(axis="x", which="both", labelbottom=True, length=3)

        # weekend shade bands — find gaps > 24h
        timestamps = df.index
        for i in range(1, len(timestamps)):
            gap_h = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
            if gap_h > 24:
                for ax in axes.flat:
                    ax.axvspan(timestamps[i - 1], timestamps[i], color=LIGHT_GREY, alpha=0.6, linewidth=0)

        fig.tight_layout()
        _save(fig, "01_time_series_overview", "processing")


# ── Plot 2: Diurnal profile heatmap ───────────────────────────────────────────
def plot_diurnal_heatmap(df: pd.DataFrame, series_cols: list[str]) -> None:
    """Heatmap: hour-of-day (rows) × series (cols), colour = median hit rate."""
    active_cols = [c for c in series_cols if c != "ListB_field5"]

    hourly_median = (
        df[active_cols + ["hour_of_day"]]
        .groupby("hour_of_day")[active_cols]
        .median()
    )

    # normalise per-series to [0,1] so sparse/dense series are comparable
    norm = hourly_median.div(hourly_median.max().replace(0, 1))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1]})

        # Left: raw median values
        sns.heatmap(
            hourly_median,
            ax=axes[0],
            cmap="YlOrBr",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Median hit rate", "shrink": 0.8},
            fmt=".3f",
            annot=False,
        )
        axes[0].set_title("Median Hit Rate by Hour × Series (raw)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Hour of day (UTC)", fontsize=10)
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)

        # Right: series-normalised (shows WHEN each series is active)
        sns.heatmap(
            norm,
            ax=axes[1],
            cmap="Blues",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Normalised activity (0–1 per series)", "shrink": 0.8},
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Normalised Activity by Hour × Series", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)

        fig.suptitle("Diurnal Patterns — When Are Series Active?", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, "02_diurnal_heatmap", "processing")


# ── Plot 3: Missing & zero data map ───────────────────────────────────────────
def plot_data_quality_map(df: pd.DataFrame, series_cols: list[str]) -> None:
    """Binary heatmap showing missing (red) and zero (orange) per hour × series."""
    n = len(df)
    cols = series_cols

    # 0 = has data, 1 = zero, 2 = missing
    quality = pd.DataFrame(0, index=range(n), columns=cols)
    for col in cols:
        quality.loc[df[col].isna().values, col] = 2
        quality.loc[(df[col] == 0).values, col] = 1

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 8))

        cmap = matplotlib.colormaps["RdYlGn_r"].resampled(3)
        img = ax.imshow(
            quality.T.values,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )

        # x-axis: date labels every 24h (start of each day)
        day_ticks = [i for i in range(0, n, 24)]
        day_labels = [df.index[i].strftime("%b %d") for i in day_ticks]
        ax.set_xticks(day_ticks)
        ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=8)

        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=9)
        ax.set_xlabel("Date (UTC)", fontsize=10)
        ax.set_title(
            "Data Quality Map — Missing (red) vs Zero (yellow) vs Present (green)",
            fontsize=12,
            fontweight="bold",
        )

        cbar = fig.colorbar(img, ax=ax, ticks=[0.33, 1.0, 1.67])
        cbar.set_ticklabels(["Present", "Zero", "Missing"])
        cbar.ax.tick_params(labelsize=9)

        fig.tight_layout()
        _save(fig, "03_data_quality_map", "processing")


# ── Plot 4: Hourly profiles per series (small multiples) ─────────────────────
def plot_hourly_profiles(df: pd.DataFrame, series_cols: list[str]) -> None:
    """For each series: median and IQR of hit rate by hour-of-day."""
    active_cols = [c for c in series_cols if c != "ListB_field5"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(5, 3, figsize=(16, 13), sharey=False)
        fig.suptitle(
            "Diurnal Profiles — Median ± IQR by Hour of Day",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        hours = np.arange(24)

        for ax, col in zip(axes.flat, active_cols):
            color = _list_color(col)
            grp = df.groupby("hour_of_day")[col]
            med = grp.median()
            q25 = grp.quantile(0.25)
            q75 = grp.quantile(0.75)

            ax.fill_between(hours, q25, q75, alpha=0.25, color=color, label="IQR")
            ax.plot(hours, med, color=color, linewidth=2, label="Median")
            ax.set_title(col, fontsize=9, fontweight="bold", color=color)
            ax.set_xlim(0, 23)
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.tick_params(labelsize=7)

        # hide unused panel (ListB_field5 excluded → 14 plots in 15-panel grid)
        if len(active_cols) < len(axes.flat):
            for ax in axes.flat[len(active_cols):]:
                ax.set_visible(False)

        fig.tight_layout()
        _save(fig, "04_hourly_profiles", "processing")


# ── Plot 5: Value distributions (non-zero only) ───────────────────────────────
def plot_distributions(df: pd.DataFrame, series_cols: list[str]) -> None:
    """Box-and-whisker plots of non-zero hit rate values per series."""
    active_cols = [c for c in series_cols if c != "ListB_field5"]

    records = []
    for col in active_cols:
        vals = df[col].dropna()
        vals = vals[vals > 0]
        for v in vals:
            list_name = col.split("_")[0]
            records.append({"series": col, "hit_rate": v, "list": list_name})

    plot_df = pd.DataFrame(records)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 6))

        palette = {col: _list_color(col) for col in active_cols}
        sns.boxplot(
            data=plot_df,
            x="series",
            y="hit_rate",
            palette=palette,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            linewidth=0.8,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Hit rate (log scale, non-zero values)", fontsize=10)
        ax.set_title(
            "Hit Rate Distributions by Series (non-zero values, log scale)",
            fontsize=12,
            fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        # List colour legend
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=l) for l, c in LIST_COLORS.items()]
        ax.legend(handles=handles, title="Sanctions List", fontsize=8, title_fontsize=8, loc="upper right")

        fig.tight_layout()
        _save(fig, "05_distributions", "processing")


# ── Plot 5b: Skew — why median over mean ──────────────────────────────────────
def plot_skew_mean_vs_median(df: pd.DataFrame, series_cols: list[str]) -> None:
    """Per-field histograms with mean vs median marked — the robustness argument.

    Hit-rate distributions are right-skewed and zero-inflated, so the mean is
    dragged upward by a few large values while the median sits on the bulk of
    the data. Marking both on each field's histogram shows visually why the
    baseline uses median + MAD rather than mean + std: the mean/median gap is
    exactly the leverage an outlier or holiday spike would have on the baseline.
    """
    from scipy.stats import skew

    active_cols = [c for c in series_cols if c != "ListB_field5"]
    n = len(active_cols)
    ncol = 5
    nrow = int(np.ceil(n / ncol))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(28, 5.2 * nrow))

        for ax, col in zip(axes.flat, active_cols):
            color = _list_color(col)
            vals = df[col].dropna()
            vals = vals[vals > 0]  # non-zero: the active-hour distribution
            if len(vals) < 2:
                ax.set_visible(False)
                continue

            mean_v = float(vals.mean())
            med_v = float(vals.median())
            sk = float(skew(vals))

            # clip x-view at p99 so the tail doesn't flatten the bulk
            hi = float(np.quantile(vals, 0.99))
            ax.hist(vals, bins=40, range=(0, hi), color=color, alpha=0.55,
                    edgecolor="white", linewidth=0.3)

            ax.axvline(med_v, color="#145a32", linewidth=3.0,
                       label=f"median = {med_v:.3f}")
            ax.axvline(mean_v, color="#c0392b", linewidth=3.0, linestyle="--",
                       label=f"mean = {mean_v:.3f}")

            ax.set_title(col, fontsize=20, fontweight="bold", color=color)
            ax.tick_params(labelsize=14)
            ax.text(0.96, 0.96,
                    f"skew = {sk:.1f}\nmean/median = {mean_v / med_v:.2f}x",
                    transform=ax.transAxes, ha="right", va="top", fontsize=15,
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8))
            ax.legend(fontsize=14, loc="center right")

        for ax in axes.flat[n:]:
            ax.set_visible(False)

        fig.supxlabel("Hit rate (non-zero values, x clipped at 99th pct)", fontsize=20)
        fig.supylabel("Frequency", fontsize=20)
        fig.tight_layout()
        _save(fig, "05b_skew_mean_vs_median", "processing")


# ── Plot 6: Correlation heatmap ───────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame, series_cols: list[str]) -> None:
    """15×15 Spearman correlation heatmap between all series."""
    corr = df[series_cols].corr(method="spearman")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 9))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle only

        sns.heatmap(
            corr,
            ax=ax,
            mask=mask,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        )
        ax.set_title(
            "Spearman Correlation Between Series\n(within-list pairs expected to correlate)",
            fontsize=12,
            fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

        fig.tight_layout()
        _save(fig, "06_correlation_heatmap", "processing")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.processing.load import load_data, get_series_cols, data_quality_report

    print("Loading data...")
    df = load_data()
    cols = get_series_cols(df)

    print("\nData quality report:")
    print(data_quality_report(df).to_string())

    print("\nGenerating plots...")
    plot_time_series_overview(df, cols)
    plot_diurnal_heatmap(df, cols)
    plot_data_quality_map(df, cols)
    plot_hourly_profiles(df, cols)
    plot_distributions(df, cols)
    plot_skew_mean_vs_median(df, cols)
    plot_correlation(df, cols)

    print("\nDone. All figures saved to outputs/processing/figures/")
