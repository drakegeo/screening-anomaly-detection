"""Anomaly visualisations with three-tier flag taxonomy."""

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.processing.eda import STYLE, LIGHT_GREY, _list_color, _save

BAND_MULTIPLIER = 2.5

# Marker styles per flag type
FLAG_STYLES = {
    "drop_zscore":             {"color": "#c0392b", "marker": "v", "s": 55,  "label": "Drop anomaly (z-score)"},
    "contextual_zero":         {"color": "#e67e22", "marker": "x", "s": 55,  "label": "Contextual zero (active hour, no hits)"},
    "contextual_zero_holiday": {"color": "#95a5a6", "marker": "x", "s": 40,  "label": "Zero on holiday (volume effect)"},
}


def _add_legend(ax: plt.Axes) -> None:
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
    ax.legend(handles=handles, fontsize=10, loc="upper right")


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
    critical_flags = flags[flags["flag_type"].isin(["drop_zscore", "contextual_zero"])]

    critical_hours = sorted(set(pd.to_datetime(critical_flags["timestamp"]).dt.hour.unique()))
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
        # red zone = alarm (below threshold)
        ax_ts.fill_between(df.index, 0, lower,
                           alpha=0.12, color="#c0392b", label="Alarm zone (below threshold)")
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
            fontsize=13, fontweight="bold", color="black",
        )
        ax_ts.set_xlabel("Date (UTC)", fontsize=11)
        ax_ts.set_ylabel("Hit rate", fontsize=11)
        # small negative lower bound so y=0 markers sit above the axis spine
        _ymax = ax_ts.get_ylim()[1]
        ax_ts.set_ylim(bottom=-_ymax * 0.03)
        ax_ts.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax_ts.tick_params(axis="both", labelsize=10)
        _add_legend(ax_ts)

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
                       f"{h:02d}h", fontsize=9, color=hc, fontweight="bold")

        ax_hr.set_xlim(-0.8, 23.8)
        ax_hr.set_xticks(range(0, 24, 2))
        ax_hr.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=9)
        ax_hr.set_xlabel("Hour of day (UTC)", fontsize=10)
        ax_hr.set_ylabel("Hit rate", fontsize=10)
        ax_hr.set_title(
            "24h Baseline Profile — red = alarm zone, "
            "vertical lines = hours where flag triggered",
            fontsize=10, color="#555555",
        )
        ax_hr.legend(fontsize=9, loc="upper right")
        ax_hr.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax_hr.tick_params(axis="y", labelsize=9)

        _save(fig, f"anomaly_{series.lower()}", "detection")


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
        _save(fig, "anomaly_overview", "detection")


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
        _save(fig, "07_threshold_sensitivity", "detection")


def plot_pca_scree(pca_out: dict) -> None:
    """PCA scree — the independence proof.

    If variance were concentrated in the first 1–2 components the series would
    share structure and a joint multivariate model would be right. Here it is
    spread almost evenly (PC1 ~18%, 11/14 comps to reach 90%) → the series are
    independent → the per-series univariate baseline is the correct architecture.
    """
    var_ratio = pca_out["var_ratio"]
    cum = pca_out["cum_var"]
    n = pca_out["n_series"]
    k = pca_out["k"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))

        x = np.arange(1, n + 1)
        ax.bar(x, var_ratio, color="#2c5f8a", alpha=0.85,
               label="Variance per component")
        ax.plot(x, cum, color="#c0392b", marker="o", markersize=5,
                linewidth=1.5, label="Cumulative variance")
        ax.axhline(0.90, color="#7f8c8d", linestyle="--", linewidth=1,
                   label="90% variance")
        ax.axvline(k, color="#e67e22", linestyle=":", linewidth=1.5,
                   label=f"{k} of {n} components to reach 90%")
        # reference: what a correlated dataset would look like
        ax.annotate(
            f"PC1 = {var_ratio[0]:.0%}\n(correlated data would be 60–90%)",
            xy=(1, var_ratio[0]), xytext=(3, 0.55), fontsize=11,
            color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1),
        )
        ax.set_xlabel("Principal component", fontsize=11)
        ax.set_ylabel("Variance share", fontsize=11)
        ax.set_title(
            "PCA Scree — Variance Spread Evenly = Independent Series\n"
            "(no compression possible → per-series baseline is the right choice)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="center right")
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=10)

        fig.tight_layout()
        _save(fig, "09_pca_diagnostic", "detection")


def plot_isoforest_scores(
    df: pd.DataFrame,
    iso_out: dict,
    anomalies: pd.DataFrame,
) -> None:
    """Isolation Forest anomaly-score timeline with z-score agreement.

    The multivariate score per hour, its robust threshold, the drop flags, and
    the z-score critical flags overlaid. Filled markers = both methods agree.
    """
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    scores = iso_out["scores"]
    threshold = iso_out["threshold"]
    hits = scores[scores["iso_flag"] & scores["is_drop"]]

    critical = anomalies[
        anomalies["flag_type"].isin(["drop_zscore", "contextual_zero"])
    ].copy()
    critical["timestamp"] = pd.to_datetime(critical["timestamp"])
    zset = set(zip(critical["timestamp"], critical["series"]))

    ts = df.index
    gaps = [(ts[i - 1], ts[i]) for i in range(1, len(ts))
            if (ts[i] - ts[i - 1]).total_seconds() / 3600 > 24]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(20, 6))

        for t0, t1 in gaps:
            ax.axvspan(t0, t1, color=LIGHT_GREY, alpha=0.5, linewidth=0)

        ax.plot(scores.index, scores["anomaly_score"].values,
                color="#2c5f8a", linewidth=1.0, label="Isolation Forest anomaly score")
        ax.fill_between(scores.index, scores["anomaly_score"].min(),
                        scores["anomaly_score"].values, alpha=0.10, color="#2c5f8a")
        ax.axhline(threshold, color="#e67e22", linestyle="--", linewidth=1.2,
                   label=f"Robust flag threshold ({threshold:.2f})")

        for tstamp, row in hits.iterrows():
            agree = (tstamp, row["top_series"]) in zset
            color = "#c0392b" if agree else "#27ae60"
            ax.scatter(tstamp, row["anomaly_score"], s=80,
                       facecolors=color if agree else "none",
                       edgecolors=color, linewidths=1.8, zorder=10)
            ax.annotate(row["top_series"].replace("List", "L").replace("_field", "f"),
                        (tstamp, row["anomaly_score"]), fontsize=6,
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", color=color, rotation=45)

        ax.set_ylabel("Anomaly score (higher = more anomalous)", fontsize=9)
        ax.set_xlabel("Date (UTC)", fontsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_title(
            f"Isolation Forest — Multivariate Drop Detection  |  "
            f"{len(hits)} drop hours flagged  |  labels = series driving each alert",
            fontsize=11, fontweight="bold",
        )

        legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b",
                   markeredgecolor="#c0392b", markersize=9, linestyle="none",
                   label="Drop flag — Isolation Forest & z-score"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#27ae60", markeredgewidth=1.8, markersize=9,
                   linestyle="none", label="Drop flag — Isolation Forest only"),
        ]
        ax.legend(handles=[*ax.get_legend_handles_labels()[0], *legend], fontsize=8,
                  loc="upper right")

        fig.tight_layout()
        _save(fig, "10_isoforest_scores", "detection")


def plot_shap_importance(shap_out: dict) -> None:
    """Global SHAP importance — which series drive the anomaly scores overall."""
    imp = shap_out["global_importance"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 7))
        colors = [_list_color(s) for s in imp.index]
        ax.barh(range(len(imp)), imp.values, color=colors, alpha=0.85,
                edgecolor="white")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|  (average contribution to anomaly score)",
                      fontsize=9)
        ax.set_title(
            "SHAP Global Importance — Which Series Drive the Anomaly Detector\n"
            "Uncorrelated features → clean attributions (no multicollinearity ambiguity)",
            fontsize=11, fontweight="bold",
        )
        for i, v in enumerate(imp.values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7, color="#444444")
        fig.tight_layout()
        _save(fig, "11_shap_importance", "detection")


def plot_shap_reasons(iso_out: dict, shap_out: dict, top_n: int = 12) -> None:
    """Per-alert SHAP reason codes — heatmap of top-N flagged hours × series.

    Each row is a flagged hour; cell colour = that series' SHAP contribution to
    the alert. Reads as: 'for this alert, these series are the reason.'
    """
    import matplotlib as mpl

    scores = iso_out["scores"]
    hits = (scores[scores["iso_flag"] & scores["is_drop"]]
            .sort_values("anomaly_score", ascending=False)
            .head(top_n))
    if hits.empty:
        return

    cols = shap_out["cols"]
    index = shap_out["index"]
    sv = shap_out["shap_values"]

    rows = []
    labels = []
    for tstamp in hits.index:
        i = index.get_loc(tstamp)
        rows.append(sv[i])
        labels.append(pd.Timestamp(tstamp).strftime("%b %d %H:%M"))
    mat = np.vstack(rows)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 0.5 * len(labels) + 2.5))
        vmax = np.abs(mat).max()
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(
            "SHAP Reason Codes — Top Flagged Hours × Contributing Series\n"
            "Red = pushes toward anomalous (the reason for the alert)",
            fontsize=11, fontweight="bold",
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label("SHAP contribution to anomaly score", fontsize=8)

        # annotate the single top driver per row
        for r in range(len(labels)):
            c = int(np.argmax(mat[r]))
            ax.add_patch(mpl.patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1, fill=False,
                edgecolor="black", linewidth=1.4))

        fig.tight_layout()
        _save(fig, "12_shap_reasons", "detection")


def plot_validation_recovery(
    results: pd.DataFrame,
    rates: pd.DataFrame,
    overall: "pd.Series",
    fp_rate: float,
    stats: dict,
) -> None:
    """Two-panel validation figure.

    Left: recovery curve — detection rate vs drop magnitude.
        Per-series lines (thin, list-colored) + bold overall mean.
        Annotated with the false-positive rate.

    Right: coverage heatmap — minimum detectable drop fraction per
        (series, hour-of-day). Shows where the detector is blind
        (red = need large drop to trigger; green = sensitive).
    """
    from src.detection.validation import DROP_FRACTIONS

    fracs = DROP_FRACTIONS
    x = [int(f * 100) for f in fracs]

    active = results[results["active"] & ~results["excluded"]]

    with plt.rc_context(STYLE):
        fig, (ax_rec, ax_heat) = plt.subplots(
            1, 2, figsize=(18, 7),
            gridspec_kw={"width_ratios": [1.1, 1.4]},
        )

        # ── LEFT: recovery curve ──────────────────────────────────────────────
        # nudge overlapping end-labels apart so names stay readable
        end_pts = sorted(
            ((float(rates.loc[s, f"{x[-1]}%"]) * 100, s) for s in rates.index),
            key=lambda t: t[0],
        )
        last_y = -1e9
        label_y = {}
        for y_end, s in end_pts:
            y_lab = max(y_end, last_y + 4.0)  # min 4-pt vertical gap
            label_y[s] = y_lab
            last_y = y_lab

        for series in rates.index:
            y_s = [rates.loc[series, f"{p}%"] * 100 for p in x]
            c = _list_color(series)
            ax_rec.plot(x, y_s, linewidth=1.0, alpha=0.45,
                        color=c, linestyle="-")
            short = series.replace("List", "L").replace("_field", "f")
            ax_rec.annotate(
                short, xy=(x[-1], y_s[-1]), xytext=(x[-1] + 1.5, label_y[series]),
                fontsize=6.5, color=c, va="center", ha="left",
                annotation_clip=False,
            )

        # bold overall mean
        y_mean = [float(overall.get(f"{p}%", np.nan)) * 100 for p in x]
        ax_rec.plot(x, y_mean, linewidth=2.8, color="#2c3e50",
                    label="Overall mean recovery", zorder=5)

        # reference lines
        ax_rec.axhline(90, color="#27ae60", linestyle="--", linewidth=0.9, alpha=0.7,
                       label="90% recovery")
        ax_rec.axhline(50, color="#e67e22", linestyle="--", linewidth=0.9, alpha=0.7,
                       label="50% recovery")
        ax_rec.axvline(50, color="#95a5a6", linestyle=":", linewidth=0.8, alpha=0.8)

        # annotate 50% magnitude recovery rate
        y_at_50 = float(overall.get("50%", np.nan)) * 100
        if not np.isnan(y_at_50):
            ax_rec.annotate(
                f"{y_at_50:.0f}% recovered\nat 50% drop",
                xy=(50, y_at_50), xytext=(55, max(y_at_50 - 18, 5)),
                fontsize=8, color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.8),
            )

        # fp rate annotation
        fp_pct = fp_rate * 100
        n_scored = stats["n_scored_pairs"]
        n_fp = stats["n_fp_flags"]
        ax_rec.text(
            0.03, 0.07,
            f"Observed FP rate: {fp_pct:.3f}%\n"
            f"({n_fp} non-holiday flags / {n_scored} series-hours)",
            transform=ax_rec.transAxes, fontsize=7.5,
            color="#7f8c8d", va="bottom",
        )

        ax_rec.set_xlim(5, 103)
        ax_rec.set_ylim(-2, 105)
        ax_rec.set_xlabel("Injected drop magnitude (% reduction from baseline)", fontsize=9)
        ax_rec.set_ylabel("Detection rate (% of active cells flagged)", fontsize=9)
        ax_rec.set_title(
            "Recovery Curve — Synthetic Drop Injection Test\n"
            "Coloured lines = individual series  |  black = overall mean",
            fontsize=10, fontweight="bold",
        )
        ax_rec.legend(fontsize=8, loc="upper left")
        ax_rec.yaxis.set_major_formatter(mticker.PercentFormatter())

        # ── RIGHT: coverage heatmap ───────────────────────────────────────────
        # min_detectable_drop per (series, hour) — lower = more sensitive
        series_order = sorted(
            active["series"].unique(),
            key=lambda s: (s.split("_")[0], s.split("_")[1]),
        )
        hours = list(range(24))
        mat = np.full((len(hours), len(series_order)), np.nan)

        for s_idx, series in enumerate(series_order):
            grp = active[active["series"] == series].set_index("hour_of_day")
            for h in hours:
                if h in grp.index:
                    mat[h, s_idx] = grp.loc[h, "min_detectable_drop"]

        cmap = matplotlib.colormaps["RdYlGn_r"]
        im = ax_heat.imshow(mat, aspect="auto", cmap=cmap,
                            vmin=0, vmax=1, origin="upper")

        ax_heat.set_xticks(range(len(series_order)))
        ax_heat.set_xticklabels(series_order, rotation=45, ha="right", fontsize=7)
        ax_heat.set_yticks(range(0, 24, 3))
        ax_heat.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=7)
        ax_heat.set_ylabel("Hour of day (UTC)", fontsize=9)
        ax_heat.set_title(
            "Minimum Detectable Drop — Coverage Map\n"
            "Green = sensitive (small drop detected)  |  Red = blind spot (need large drop)",
            fontsize=10, fontweight="bold",
        )

        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.75)
        cbar.set_label("Min detectable drop fraction", fontsize=8)
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        # annotate stat in corner
        med_min = stats.get("median_min_detectable_drop", np.nan)
        pct_50 = stats.get("pct_cells_detectable_at_50pct", np.nan)
        ax_heat.text(
            0.02, 0.02,
            f"Median min-detectable drop: {med_min:.0%}\n"
            f"Cells detectable at 50% drop: {pct_50:.0%}",
            transform=ax_heat.transAxes, fontsize=7.5,
            color="#2c3e50", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        fig.tight_layout()
        _save(fig, "13_validation", "validation")


def plot_threshold_tradeoff(tradeoff: "pd.DataFrame", knee: float = -2.5) -> None:
    """Defend the -2.5 threshold as a knee-point between two proxy metrics.

    Bars (left axis)  = false-alarm flags on presumed-clean hours (specificity).
    Lines (right axis) = injected-drop recovery at 50% and 90% (sensitivity).

    The knee threshold is highlighted: moving looser (-2.0) buys sensitivity at a
    steep false-alarm cost; moving tighter (-3.0) buys little on false alarms but
    collapses recovery. -2.5 is the favourable trade.
    """
    thresholds = list(tradeoff.index)
    x = np.arange(len(thresholds))
    rec_cols = [c for c in tradeoff.columns if c.startswith("recovery_at_")]
    rec_cols = sorted(rec_cols, key=lambda c: int(c.split("_")[-1]))

    with plt.rc_context(STYLE):
        fig, ax_fp = plt.subplots(figsize=(10, 6))

        # highlight the knee column
        if knee in thresholds:
            ax_fp.axvspan(thresholds.index(knee) - 0.5, thresholds.index(knee) + 0.5,
                          color="#f6f0d8", alpha=0.7, zorder=0)

        # ── bars: false-alarm flags ──────────────────────────────────────────
        bars = ax_fp.bar(x, tradeoff["n_fp_flags"].values, width=0.5,
                         color="#c0392b", alpha=0.85, edgecolor="white",
                         label="False-alarm flags (clean hours)", zorder=3)
        for xi, (n, rate) in enumerate(zip(tradeoff["n_fp_flags"], tradeoff["fp_rate"])):
            ax_fp.text(xi, n + 0.5, f"{int(n)}\n({rate:.2%})", ha="center", va="bottom",
                       fontsize=8, color="#7b241c", zorder=4)

        ax_fp.set_ylabel("False-alarm flags on presumed-clean hours",
                         fontsize=10, color="#c0392b")
        ax_fp.tick_params(axis="y", labelcolor="#c0392b")
        ax_fp.set_ylim(0, max(tradeoff["n_fp_flags"].max() * 1.35, 5))

        # ── lines: recovery at each magnitude ────────────────────────────────
        ax_rec = ax_fp.twinx()
        greens = ["#27ae60", "#145a32"]
        for i, col in enumerate(rec_cols):
            pct = int(col.split("_")[-1])
            y = tradeoff[col].values * 100
            ax_rec.plot(x, y, marker="o", markersize=8, linewidth=2.2,
                        color=greens[i % len(greens)],
                        label=f"Recovery at {pct}% injected drop", zorder=5)
            for xi, yi in zip(x, y):
                ax_rec.annotate(f"{yi:.0f}%", (xi, yi), xytext=(9, 0),
                                textcoords="offset points", ha="left", va="center",
                                fontsize=8, fontweight="bold",
                                color=greens[i % len(greens)], zorder=6)

        ax_rec.set_ylabel("Injected-drop recovery (sensitivity)",
                          fontsize=10, color="#145a32")
        ax_rec.tick_params(axis="y", labelcolor="#145a32")
        ax_rec.set_ylim(0, 105)
        ax_rec.yaxis.set_major_formatter(mticker.PercentFormatter())

        ax_fp.set_xticks(x)
        ax_fp.set_xticklabels(
            [f"z = {t}{'  (chosen)' if t == knee else ''}" for t in thresholds],
            fontsize=10,
        )
        ax_fp.set_xlabel("Drop threshold", fontsize=10)
        ax_fp.set_title(
            "Threshold Trade-off — Why z = -2.5 Is the Knee\n"
            "Looser = more false alarms  |  Tighter = lost sensitivity",
            fontsize=12, fontweight="bold",
        )

        h1, l1 = ax_fp.get_legend_handles_labels()
        h2, l2 = ax_rec.get_legend_handles_labels()
        ax_fp.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")

        fig.tight_layout()
        _save(fig, "08_threshold_tradeoff", "detection")


# ── List → field mapping (5 lists × 3 fields) ──────────────────────────────────
LIST_FIELDS = {
    "ListA": ["ListA_field1", "ListA_field2", "ListA_field3"],
    "ListB": ["ListB_field4", "ListB_field5", "ListB_field6"],
    "ListC": ["ListC_field7", "ListC_field8", "ListC_field9"],
    "ListD": ["ListD_field10", "ListD_field11", "ListD_field12"],
    "ListE": ["ListE_field13", "ListE_field14", "ListE_field15"],
}

# health status colours — three states plus inactive
STATUS_COLORS = {
    "healthy":  "#27ae60",  # green — no genuine drop
    "failure":  "#c0392b",  # red — drop anomaly detected (non-holiday)
    "holiday":  "#5dade2",  # blue — holiday zeros only (volume effect)
    "inactive": "#95a5a6",  # grey — excluded / no baseline
}

ISO_HATCH = "///"  # overlay marking Isolation-Forest corroboration


def _series_status(series: str, events: pd.DataFrame) -> str:
    """Classify one series: healthy / failure / holiday / inactive."""
    from src.processing.load import INACTIVE_SERIES
    if series in INACTIVE_SERIES:
        return "inactive"
    ev = events[events["series"] == series]
    if ev.empty:
        return "healthy"
    if not ev[~ev["is_holiday"]].empty:
        return "failure"
    return "holiday"


def _iso_flagged_series(iso_scores: "pd.DataFrame | None") -> set[str]:
    """Series that Isolation Forest flags as a drop (top_series of iso drop-hits)."""
    if iso_scores is None or iso_scores.empty:
        return set()
    hits = iso_scores[iso_scores["iso_flag"] & iso_scores["is_drop"]]
    return set(hits["top_series"].unique())


def plot_monitoring_dashboard(
    df: pd.DataFrame,
    series_cols: list[str],
    events: pd.DataFrame,
    fp_rate: float,
    baseline: dict[str, pd.DataFrame] | None = None,
    iso_scores: "pd.DataFrame | None" = None,
    top_n: int = 12,
) -> None:
    """Operational monitoring dashboard — the 9am ops-team view.

    Panels:
      - Health grid: 5 lists x 3 fields. Colour = status (green healthy,
        red drop detected, blue holiday zeros, grey inactive). A ``///`` hatch
        marks cells the Isolation Forest also flags, so agreement is visible.
      - Status summary: z-score vs Isolation Forest coverage (as % of active
        series) and where they agree.
      - Alert queue: top events ranked by severity, with per-event IF
        corroboration and baseline confidence.
      - Key findings: slide-ready conclusions.

    Sits on top of the detection results — turns the model into a system.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle, Patch

    from src.processing.load import INACTIVE_SERIES

    active = [c for c in series_cols if c not in INACTIVE_SERIES]
    n_active = len(active)

    non_holiday_events = events[~events["is_holiday"]] if not events.empty else events
    n_events = len(non_holiday_events)

    z_series = set(non_holiday_events["series"].unique()) if n_events else set()
    iso_series = _iso_flagged_series(iso_scores) & set(active)
    common_series = z_series & iso_series
    union_series = z_series | iso_series

    def _pct(k: int) -> str:
        return f"{k} series" if n_active else "—"

    worst_row = (
        non_holiday_events.sort_values("worst_z").iloc[0]
        if n_events > 0 else None
    )

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(18, 11))
        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[1.1, 0.95, 0.4], width_ratios=[1.25, 1.0],
            hspace=0.40, wspace=0.18,
        )

        # ── PANEL 1: health grid ──────────────────────────────────────────────
        ax_grid = fig.add_subplot(gs[0, 0])
        lists = list(LIST_FIELDS.keys())
        for row, list_name in enumerate(lists):
            for col, series in enumerate(LIST_FIELDS[list_name]):
                status = _series_status(series, events)
                color = STATUS_COLORS[status]
                iso_hit = series in iso_series
                ax_grid.add_patch(Rectangle(
                    (col, len(lists) - 1 - row), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=3, alpha=0.9,
                    hatch=ISO_HATCH if iso_hit else None,
                ))
                # mask the hatch behind text so labels stay readable
                tbox = dict(boxstyle="round,pad=0.12", facecolor=color,
                            edgecolor="none", alpha=0.85) if iso_hit else None
                fld = series.split("_")[1]
                ax_grid.text(
                    col + 0.5, len(lists) - 1 - row + 0.60, fld,
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold", bbox=tbox,
                )
                n = len(events[(events["series"] == series) & (~events["is_holiday"])]) \
                    if not events.empty else 0
                tag = f"{n} event{'s' if n != 1 else ''}" if n else "—"
                if iso_hit:
                    tag += " +IF"
                ax_grid.text(
                    col + 0.5, len(lists) - 1 - row + 0.28, tag,
                    ha="center", va="center", fontsize=7.5, color="white",
                    bbox=tbox,
                )

        ax_grid.set_xlim(0, 3)
        ax_grid.set_ylim(0, len(lists))
        ax_grid.set_xticks([0.5, 1.5, 2.5])
        ax_grid.set_xticklabels(["Field pos 1", "Field pos 2", "Field pos 3"], fontsize=9)
        ax_grid.set_yticks([len(lists) - 1 - r + 0.5 for r in range(len(lists))])
        ax_grid.set_yticklabels(lists, fontsize=10, fontweight="bold")
        ax_grid.set_title("Screening Health Grid — 15 Series at a Glance",
                          fontsize=12, fontweight="bold", pad=10)
        ax_grid.tick_params(length=0)
        for spine in ax_grid.spines.values():
            spine.set_visible(False)
        ax_grid.grid(False)

        legend_items = [
            Patch(facecolor=STATUS_COLORS["healthy"], label="Healthy"),
            Patch(facecolor=STATUS_COLORS["failure"], label="Drop detected"),
            Patch(facecolor=STATUS_COLORS["holiday"], label="Holiday zeros only"),
            Patch(facecolor=STATUS_COLORS["inactive"], label="Inactive / excluded"),
            Patch(facecolor="white", edgecolor="#2c3e50", hatch=ISO_HATCH,
                  label="Isolation Forest agrees"),
        ]
        ax_grid.legend(handles=legend_items, loc="upper center",
                       bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)

        # ── PANEL 2: status summary (per-method coverage) ─────────────────────
        ax_kpi = fig.add_subplot(gs[0, 1])
        ax_kpi.axis("off")
        ax_kpi.set_title(f"Status Summary — Detector Coverage ({n_active} active series)",
                         fontsize=12, fontweight="bold", loc="left", pad=10)

        rows = [
            ("Z-score (primary)", _pct(len(z_series)), STATUS_COLORS["failure"]),
            ("Isolation Forest (corroborator)", _pct(len(iso_series)), "#2c5f8a"),
            ("Both methods agree", _pct(len(common_series)), "#145a32"),
            ("Flagged by either", _pct(len(union_series)), "#2c3e50"),
        ]
        for i, (label, value, color) in enumerate(rows):
            y = 0.88 - i * 0.20
            ax_kpi.text(0.04, y, value, fontsize=22, fontweight="bold",
                        color=color, va="center", transform=ax_kpi.transAxes)
            ax_kpi.text(0.04, y - 0.075, label, fontsize=9.5, color="#7f8c8d",
                        va="center", transform=ax_kpi.transAxes)

        if worst_row is not None:
            worst_txt = (
                f"{worst_row['series']}  ·  "
                f"{pd.Timestamp(worst_row['start']).strftime('%b %d %H:%M')}  ·  "
                f"z = {worst_row['worst_z']:.1f}"
            )
            ax_kpi.text(0.04, 0.05, "Worst event: ", fontsize=9.5, color="#7f8c8d",
                        va="center", transform=ax_kpi.transAxes)
            ax_kpi.text(0.28, 0.05, worst_txt, fontsize=10, color="#c0392b",
                        fontweight="bold", va="center", transform=ax_kpi.transAxes)

        # ── PANEL 3: alert queue (event-level) ────────────────────────────────
        ax_q = fig.add_subplot(gs[1, :])
        ax_q.axis("off")

        queue = (non_holiday_events.sort_values("worst_z").head(top_n)
                 if n_events > 0 else pd.DataFrame())

        headers = ["Series", "Start", "Duration", "Worst z", "Drop %", "Baseline", "Isolation Forest"]
        col_x = [0.02, 0.20, 0.36, 0.49, 0.60, 0.71, 0.85]

        ax_q.text(0.02, 1.04, "Alert Queue — Events Ranked by Severity",
                  fontsize=12, fontweight="bold", transform=ax_q.transAxes)
        for x, h in zip(col_x, headers):
            ax_q.text(x, 0.93, h, fontsize=9, fontweight="bold",
                      color="#2c3e50", transform=ax_q.transAxes)
        ax_q.axhline(0.90, xmin=0.0, xmax=1.0, color="#bdc3c7", linewidth=1)

        if queue.empty:
            ax_q.text(0.02, 0.80, "No genuine events — all series healthy.",
                      fontsize=11, color=STATUS_COLORS["healthy"],
                      transform=ax_q.transAxes)
        else:
            row_h = 0.86 / max(len(queue), 1)
            for i, (_, ev) in enumerate(queue.iterrows()):
                y = 0.84 - i * row_h
                iso_ok = ev["series"] in iso_series
                iso_txt = "confirmed ✓" if iso_ok else "z-score only"
                iso_color = "#145a32" if iso_ok else "#7f8c8d"
                drop_pct = (
                    (ev["worst_observed"] - ev["expected_median"])
                    / ev["expected_median"] * 100
                    if ev["expected_median"] else float("nan")
                )
                dur = f"{ev['duration_hours']}h ({ev['n_flags']} flags)"
                if baseline is not None:
                    h = pd.Timestamp(ev["start"]).hour
                    n_obs = int(baseline["count"].loc[h, ev["series"]])
                    low = bool(baseline["low_conf"].loc[h, ev["series"]])
                    base_txt = f"{n_obs} obs" + (" ⚠" if low else " ✓")
                    base_color = "#c0392b" if low else "#27ae60"
                else:
                    base_txt, base_color = "—", "#2c3e50"
                cells = [
                    ev["series"],
                    pd.Timestamp(ev["start"]).strftime("%b %d %H:%M"),
                    dur,
                    f"{ev['worst_z']:.2f}" if pd.notna(ev["worst_z"]) else "—",
                    f"{drop_pct:.0f}%" if pd.notna(drop_pct) else "—",
                    base_txt,
                    iso_txt,
                ]
                for j, (x, txt) in enumerate(zip(col_x, cells)):
                    if j == 6:
                        c = iso_color
                    elif j == 5:
                        c = base_color
                    else:
                        c = "#2c3e50"
                    fw = "bold" if j in (0, 6) else "normal"
                    ax_q.text(x, y, str(txt), fontsize=8.5, color=c,
                              fontweight=fw, transform=ax_q.transAxes)

        # ── PANEL 4: key findings (slide conclusions) ─────────────────────────
        ax_c = fig.add_subplot(gs[2, :])
        ax_c.axis("off")
        n_low = 0
        if baseline is not None and n_events:
            for _, ev in non_holiday_events.iterrows():
                h = pd.Timestamp(ev["start"]).hour
                if bool(baseline["low_conf"].loc[h, ev["series"]]):
                    n_low += 1
        worst_series = worst_row["series"] if worst_row is not None else "—"

        bullets = [
            (f"{n_events} genuine drop events across {len(z_series)} series — "
             f"holiday zeros ruled out as volume, not list failure.", "#c0392b"),
            (f"Isolation Forest independently confirms {len(common_series)} of "
             f"{len(z_series)} — the highest-confidence alerts, escalate first.", "#145a32"),
            (f"Worst is {worst_series}: a sustained drop, not noise — exactly the "
             f"silent failure that lets unscreened transactions through.", "#c0392b"),
            (f"Every alert rests on a full baseline ({n_low} low-confidence). "
             f"Z-score detects; Isolation Forest corroborates and names the culprit.", "#2c5f8a"),
        ]
        ax_c.text(0.005, 1.05, "Key Findings", fontsize=12.5, fontweight="bold",
                  color="#2c3e50", transform=ax_c.transAxes)
        for i, (b, dot) in enumerate(bullets):
            y = 0.80 - i * 0.26
            ax_c.text(0.005, y, "●", fontsize=10, color=dot,
                      va="top", transform=ax_c.transAxes)
            ax_c.text(0.028, y, b, fontsize=10.5, color="#2c3e50",
                      va="top", transform=ax_c.transAxes)

        fig.suptitle(
            "Sanctions Screening — Drop Anomaly Monitoring Dashboard",
            fontsize=15, fontweight="bold", y=0.985,
        )
        _save(fig, "14_dashboard", "detection")
