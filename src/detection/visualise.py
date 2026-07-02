"""Anomaly visualisations with three-tier flag taxonomy."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.processing.eda import FIGURES_DIR, STYLE, LIGHT_GREY, _list_color, _save

BAND_MULTIPLIER = 2.5

# Marker styles per flag type
FLAG_STYLES = {
    "drop_zscore":             {"color": "#c0392b", "marker": "v", "s": 55,  "label": "Drop anomaly (z-score)"},
    "contextual_zero":         {"color": "#e67e22", "marker": "x", "s": 55,  "label": "Contextual zero (active hour, no hits)"},
    "contextual_zero_holiday": {"color": "#95a5a6", "marker": "x", "s": 40,  "label": "Zero on holiday (volume effect)"},
}


def _add_legend(ax: plt.Axes, present_types: set[str]) -> None:
    handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for k, v in FLAG_STYLES.items()
        if k in present_types
    ]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper right")


def _draw_background(ax: plt.Axes, df: pd.DataFrame) -> None:
    timestamps = df.index
    for i in range(1, len(timestamps)):
        if (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600 > 24:
            ax.axvspan(timestamps[i - 1], timestamps[i],
                       color=LIGHT_GREY, alpha=0.6, linewidth=0)
    day_starts = pd.date_range(
        start=df.index[0].normalize(), end=df.index[-1], freq="D", tz="UTC"
    )
    for d in day_starts:
        ax.axvline(d, color="#dddddd", linewidth=0.5, zorder=0)


def plot_series_with_anomalies(
    df: pd.DataFrame,
    series: str,
    baseline: dict[str, pd.DataFrame],
    anomalies: pd.DataFrame,
) -> None:
    """Individual series plot: baseline band + flag markers by type."""
    color = _list_color(series)
    hours = df["hour_of_day"].values
    exp_med = baseline["median"].loc[hours, series].values
    exp_mad = baseline["mad"].loc[hours, series].values
    lower = np.maximum(exp_med - BAND_MULTIPLIER * exp_mad, 0)

    flags = anomalies[anomalies["series"] == series]
    present_types = set(flags["flag_type"].unique())

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(df.index, lower, exp_med, alpha=0.15, color=color,
                        label=f"Expected band (median ± {BAND_MULTIPLIER} MAD)")
        ax.plot(df.index, exp_med, color=color, linewidth=1.0,
                linestyle="--", alpha=0.5, label="Hourly median baseline")
        ax.plot(df.index, df[series].values, linewidth=0.9,
                color=color, alpha=0.9, label="Observed hit rate")

        for ftype, style in FLAG_STYLES.items():
            subset = flags[flags["flag_type"] == ftype]
            if not subset.empty:
                ax.scatter(
                    subset["timestamp"], subset["observed"],
                    color=style["color"], marker=style["marker"],
                    s=style["s"], zorder=5,
                )

        # worst z-score annotation
        zscore_flags = flags[flags["flag_type"] == "drop_zscore"]
        if not zscore_flags.empty:
            worst_idx = zscore_flags["z_score"].idxmin()
            worst = zscore_flags.loc[worst_idx]
            ax.annotate(
                f"Worst: z={worst['z_score']:.2f}\n{pd.Timestamp(worst['timestamp']).strftime('%b %d %H:%M')}",
                xy=(worst["timestamp"], worst["observed"]),
                xytext=(15, 20), textcoords="offset points",
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8),
            )

        _draw_background(ax, df)

        n_critical = len(flags[flags["flag_type"].isin(["drop_zscore", "contextual_zero"])])
        n_holiday = len(flags[flags["flag_type"] == "contextual_zero_holiday"])
        subtitle = f"{n_critical} critical flag(s)  |  {n_holiday} holiday zero(s)"
        ax.set_title(f"{series} — Drop Anomaly Detection  |  {subtitle}",
                     fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Date (UTC)", fontsize=10)
        ax.set_ylabel("Hit rate", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        _add_legend(ax, present_types)
        fig.tight_layout()
        _save(fig, f"anomaly_{series.lower()}")


def plot_anomaly_overview(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    anomalies: pd.DataFrame,
) -> None:
    """Small-multiple overview: all series with colour-coded flag markers."""
    from src.processing.load import INACTIVE_SERIES
    active_cols = [c for c in series_cols if c not in INACTIVE_SERIES]

    ncols = 3
    nrows = (len(active_cols) + ncols - 1) // ncols

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3), sharex=True)
        fig.suptitle("Anomaly Detection Overview — All Series",
                     fontsize=13, fontweight="bold", y=1.01)

        for ax, col in zip(axes.flat, active_cols):
            color = _list_color(col)
            hours = df["hour_of_day"].values
            exp_med = baseline["median"].loc[hours, col].values
            exp_mad = baseline["mad"].loc[hours, col].values
            lower = np.maximum(exp_med - BAND_MULTIPLIER * exp_mad, 0)

            ax.fill_between(df.index, lower, exp_med, alpha=0.15, color=color)
            ax.plot(df.index, exp_med, color=color, linewidth=0.6,
                    linestyle="--", alpha=0.5)
            ax.plot(df.index, df[col].values, linewidth=0.7,
                    color=color, alpha=0.85)

            flags = anomalies[anomalies["series"] == col]
            for ftype, style in FLAG_STYLES.items():
                subset = flags[flags["flag_type"] == ftype]
                if not subset.empty:
                    ax.scatter(subset["timestamp"], subset["observed"],
                               color=style["color"], marker=style["marker"],
                               s=18, zorder=5)

            n_critical = len(flags[flags["flag_type"].isin(["drop_zscore", "contextual_zero"])])
            n_holiday = len(flags[flags["flag_type"] == "contextual_zero_holiday"])
            title = f"{col}  ({n_critical} critical"
            if n_holiday:
                title += f" + {n_holiday} holiday"
            title += ")"
            ax.set_title(title, fontsize=8, fontweight="bold", color=color)
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

            timestamps = df.index
            for i in range(1, len(timestamps)):
                if (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600 > 24:
                    ax.axvspan(timestamps[i - 1], timestamps[i],
                               color=LIGHT_GREY, alpha=0.5, linewidth=0)

        for ax in axes.flat[len(active_cols):]:
            ax.set_visible(False)

        # global legend
        handles = [mpatches.Patch(color=v["color"], label=v["label"])
                   for v in FLAG_STYLES.values()]
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout()
        _save(fig, "anomaly_overview")


def plot_threshold_sensitivity(sensitivity_df: "pd.DataFrame") -> None:
    """Bar chart showing flag counts at multiple z-score thresholds."""
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        thresholds = [str(t) for t in sensitivity_df.index]
        x = np.arange(len(thresholds))
        width = 0.22

        cols = {
            "n_drop_zscore_non_holiday": ("#c0392b", "Z-score drop (non-holiday)"),
            "n_drop_zscore_holiday":     ("#e8a89c", "Z-score drop (holiday)"),
            "n_contextual_zero":         ("#e67e22", "Contextual zero (active hour)"),
            "n_holiday_zero":            ("#bdc3c7", "Contextual zero (holiday)"),
        }

        for i, (col, (color, label)) in enumerate(cols.items()):
            ax.bar(x + (i - 1.5) * width, sensitivity_df[col],
                   width=width, color=color, label=label, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([f"threshold = {t}" for t in thresholds], fontsize=10)
        ax.set_ylabel("Number of flags", fontsize=10)
        ax.set_title(
            "Threshold Sensitivity — Flag Counts at z = -2.0 / -2.5 / -3.0",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")

        for i, (col, (color, _)) in enumerate(cols.items()):
            for j, val in enumerate(sensitivity_df[col]):
                ax.text(j + (i - 1.5) * width, val + 0.3, str(int(val)),
                        ha="center", va="bottom", fontsize=8, color="#444444")

        fig.tight_layout()
        _save(fig, "07_threshold_sensitivity")
