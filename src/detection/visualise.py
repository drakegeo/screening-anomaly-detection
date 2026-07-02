"""Anomaly visualisations with three-tier flag taxonomy."""

from pathlib import Path

import matplotlib
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
    """Fixed two-entry legend shown on every plot for consistency."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor="#555555",
               markeredgewidth=1.5, markersize=7, linestyle="none",
               label="Anomaly flag (drop or zero during active hour)"),
        Line2D([0], [0], marker="x", color="black",
               markersize=7, markeredgewidth=2.0, linestyle="none",
               label="Holiday zero — likely volume effect, not list failure"),
    ]
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
    """Two-panel plot per series.

    Top: time series with flag markers labelled by hour-of-day.
    Bottom: 24h diurnal baseline profile with flagged hours highlighted.
    """
    color = _list_color(series)
    hours_arr = df["hour_of_day"].values
    exp_med = baseline["median"].loc[hours_arr, series].values
    exp_mad = baseline["mad"].loc[hours_arr, series].values
    lower = np.maximum(exp_med - BAND_MULTIPLIER * exp_mad, 0)

    flags = anomalies[anomalies["series"] == series]
    present_types = set(flags["flag_type"].unique())

    critical_flags = flags[flags["flag_type"].isin(["drop_zscore", "contextual_zero"])]
    holiday_flags  = flags[flags["flag_type"] == "contextual_zero_holiday"]

    critical_hours = sorted(set(pd.to_datetime(critical_flags["timestamp"]).dt.hour.unique()))
    holiday_hours  = sorted(set(pd.to_datetime(holiday_flags["timestamp"]).dt.hour.unique()))
    flagged_hours  = critical_hours  # bottom panel lines = critical only

    # one distinct color per critical hour — offset +5 to avoid series line colors
    _cmap = matplotlib.colormaps["tab10"]
    hour_colors = {h: _cmap((i + 5) % 10) for i, h in enumerate(critical_hours)}

    with plt.rc_context(STYLE):
        fig, (ax_ts, ax_hr) = plt.subplots(
            2, 1, figsize=(22, 7),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
        )

        # ── TOP: time series ──────────────────────────────────────────────────
        # red zone = alarm (below threshold); grey zone = expected range
        ax_ts.fill_between(df.index, 0, lower,
                           alpha=0.12, color="#c0392b", label="Alarm zone (below threshold)")
        ax_ts.fill_between(df.index, lower, exp_med,
                           alpha=0.15, color="#95a5a6", label="Expected range (median ± 2.5 MAD)")
        ax_ts.plot(df.index, exp_med, color="#7f8c8d", linewidth=0.8,
                   alpha=0.7, label="Hourly median baseline")
        ax_ts.plot(df.index, lower, color="#c0392b", linewidth=0.6,
                   linestyle=":", alpha=0.5)
        ax_ts.plot(df.index, df[series].values, linewidth=1.8,
                   color=color, alpha=0.9, label="Observed hit rate")

        # pass 1: x black (holiday zeros, smaller) — drawn first so o sits on top
        for _, flag in flags[flags["flag_type"] == "contextual_zero_holiday"].iterrows():
            ax_ts.scatter(flag["timestamp"], flag["observed"],
                          color="black", marker="x", s=30, zorder=5, linewidths=1.5)

        # pass 2: o hollow (critical anomalies, hour-colored) — drawn on top
        for _, flag in flags[flags["flag_type"] != "contextual_zero_holiday"].iterrows():
            hour = pd.Timestamp(flag["timestamp"]).hour
            hc = hour_colors[hour]
            ax_ts.scatter(flag["timestamp"], flag["observed"],
                          facecolors="none", edgecolors=hc,
                          marker="o", s=60, zorder=6, linewidths=1.8)

        _draw_background(ax_ts, df)

        n_critical = len(flags[flags["flag_type"].isin(["drop_zscore", "contextual_zero"])])
        n_holiday  = len(flags[flags["flag_type"] == "contextual_zero_holiday"])
        ax_ts.set_title(
            f"{series}  |  {n_critical} critical flag(s)  +  {n_holiday} holiday zero(s)\n"
            f"Marker colour = hour of day — trace each colour to the diurnal profile below",
            fontsize=11, fontweight="bold", color="black",
        )
        ax_ts.set_xlabel("Date (UTC)", fontsize=9)
        ax_ts.set_ylabel("Hit rate", fontsize=9)
        # small negative lower bound so y=0 markers sit above the axis spine
        _ymax = ax_ts.get_ylim()[1]
        ax_ts.set_ylim(bottom=-_ymax * 0.03)
        ax_ts.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        _add_legend(ax_ts, present_types)

        # ── BOTTOM: 24h diurnal baseline profile ──────────────────────────────
        hour_range = np.arange(24)
        med_profile = baseline["median"][series].values
        mad_profile = baseline["mad"][series].values
        lo_profile  = np.maximum(med_profile - BAND_MULTIPLIER * mad_profile, 0)

        # red = alarm zone (below threshold), grey = expected range
        ax_hr.fill_between(hour_range, 0, lo_profile,
                           alpha=0.18, color="#c0392b", label="Alarm zone")
        ax_hr.fill_between(hour_range, lo_profile, med_profile,
                           alpha=0.20, color="#95a5a6", label="Expected range")
        ax_hr.plot(hour_range, med_profile, color="#2c5f8a",
                   linewidth=1.5, label="Median baseline")
        ax_hr.plot(hour_range, lo_profile, color="#c0392b",
                   linewidth=1.2, linestyle="--", label="Alarm threshold")

        # flagged hours: vertical lines colored per hour (matches top-panel markers)
        for h in flagged_hours:
            hc = hour_colors[h]
            ax_hr.axvline(h, color=hc, linewidth=1.8, alpha=0.9, zorder=5)
            ax_hr.text(h + 0.2, med_profile.max() * 0.85,
                       f"{h:02d}h", fontsize=7, color=hc, fontweight="bold")

        ax_hr.set_xlim(-0.8, 23.8)
        ax_hr.set_xticks(range(0, 24, 2))
        ax_hr.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=7)
        ax_hr.set_xlabel("Hour of day (UTC)", fontsize=8)
        ax_hr.set_ylabel("Hit rate", fontsize=8)
        ax_hr.set_title(
            "24h Baseline Profile — grey = expected range, red = alarm zone, "
            "vertical lines = hours where flag triggered",
            fontsize=8, color="#555555",
        )
        ax_hr.legend(fontsize=7, loc="upper right")
        ax_hr.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

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
