"""Anomaly scoring and flagging with three-tier flag taxonomy."""

import numpy as np
import pandas as pd

from src.processing.load import INACTIVE_SERIES, PROBLEMATIC_SERIES, KNOWN_HOLIDAYS

DROP_THRESHOLD = -2.5
DROP_THRESHOLD_WIDE = -2.0          # ListB_field6: high missingness, wider band
CONTEXTUAL_ZERO_MIN_MEDIAN = 0.005  # hourly median floor for contextual zero to fire

# Series excluded from contextual zero (too unstable to produce reliable signal)
CONTEXTUAL_ZERO_EXCLUDED = set(INACTIVE_SERIES) | set(PROBLEMATIC_SERIES)

FLAG_TYPES = {
    "drop_zscore":             "Robust z-score drop (statistically significant fall below baseline)",
    "contextual_zero":         "Zero during normally-active hour (potential list failure)",
    "contextual_zero_holiday": "Zero on known holiday — likely volume-driven, not list failure",
}


def _is_holiday(ts: pd.Timestamp) -> bool:
    return ts.normalize().tz_convert("UTC") in KNOWN_HOLIDAYS


def score_anomalies(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    drop_threshold: float = DROP_THRESHOLD,
) -> pd.DataFrame:
    """Score every observation and return flat table of flagged anomalies.

    Output columns:
        timestamp, series, observed, expected_median, z_score,
        flag_type, flag_label, is_holiday, low_confidence
    """
    active_cols = [c for c in series_cols if c not in INACTIVE_SERIES]
    median_df = baseline["median"]
    mad_df = baseline["mad"]
    low_conf_df = baseline["low_conf"]

    records = []

    for col in active_cols:
        # ListB_field6 gets wider z-score threshold; excluded from contextual zero
        threshold = DROP_THRESHOLD_WIDE if col in PROBLEMATIC_SERIES else drop_threshold

        for idx, row in df.iterrows():
            hour = int(row["hour_of_day"])
            val = row[col]
            holiday = bool(row["is_holiday"])

            if pd.isna(val):
                continue

            exp_median = median_df.loc[hour, col]
            exp_mad = mad_df.loc[hour, col]
            low_conf = bool(low_conf_df.loc[hour, col])

            if pd.isna(exp_median) or pd.isna(exp_mad):
                continue

            # ── z-score drop ──────────────────────────────────────────────────
            if exp_mad > 0 and exp_median > 0:
                z = (val - exp_median) / exp_mad
                if z < threshold:
                    records.append({
                        "timestamp": idx,
                        "series": col,
                        "observed": val,
                        "expected_median": exp_median,
                        "z_score": round(z, 4),
                        "flag_type": "drop_zscore",
                        "flag_label": FLAG_TYPES["drop_zscore"],
                        "is_holiday": holiday,
                        "low_confidence": low_conf,
                    })
                    continue

            # ── contextual zero (excluded for broken series) ──────────────────
            if col in CONTEXTUAL_ZERO_EXCLUDED:
                continue

            if val == 0 and exp_median >= CONTEXTUAL_ZERO_MIN_MEDIAN:
                ftype = "contextual_zero_holiday" if holiday else "contextual_zero"
                records.append({
                    "timestamp": idx,
                    "series": col,
                    "observed": val,
                    "expected_median": exp_median,
                    "z_score": np.nan,
                    "flag_type": ftype,
                    "flag_label": FLAG_TYPES[ftype],
                    "is_holiday": holiday,
                    "low_confidence": low_conf,
                })

    if not records:
        return pd.DataFrame(columns=[
            "timestamp", "series", "observed", "expected_median",
            "z_score", "flag_type", "flag_label", "is_holiday", "low_confidence",
        ])

    out = (
        pd.DataFrame(records)
        .sort_values(["series", "timestamp"])
        .reset_index(drop=True)
    )
    out["observed"] = out["observed"].round(6)
    out["expected_median"] = out["expected_median"].round(6)
    return out


def anomaly_summary(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Per-series summary table with severity ranking."""
    if anomalies.empty:
        return pd.DataFrame()

    grp = anomalies.groupby("series")
    summary = pd.DataFrame({
        "n_total": grp.size(),
        "n_drop_zscore": grp.apply(lambda x: (x["flag_type"] == "drop_zscore").sum()),
        "n_drop_zscore_holiday": grp.apply(
            lambda x: ((x["flag_type"] == "drop_zscore") & x["is_holiday"]).sum()
        ),
        "n_contextual_zero": grp.apply(lambda x: (x["flag_type"] == "contextual_zero").sum()),
        "n_holiday_zero": grp.apply(lambda x: (x["flag_type"] == "contextual_zero_holiday").sum()),
        "worst_z": grp["z_score"].min().round(3),
        "worst_drop_pct": grp.apply(
            lambda x: (
                (x["observed"] - x["expected_median"])
                / x["expected_median"].replace(0, np.nan)
                * 100
            ).min()
        ).round(1),
        "first_flag": grp["timestamp"].min(),
        "last_flag": grp["timestamp"].max(),
    })
    return summary.sort_values("worst_z")


def group_into_events(
    anomalies: pd.DataFrame,
    max_gap_hours: int = 2,
) -> pd.DataFrame:
    """Collapse consecutive-hour flags of the same series into single events.

    A 6-hour sustained drop is ONE operational event, not six independent alerts.
    Counting hour-level flags overstates severity and inflates alert volume. This
    groups flags of the same series that are within ``max_gap_hours`` of each other
    into a single event, so the results reflect what an analyst actually responds to.

    ``max_gap_hours=2`` tolerates a one-hour recovery blip inside a sustained drop.

    Returns one row per event:
        series, flag_type, start, end, duration_hours, n_flags,
        worst_z, worst_observed, expected_median, is_holiday
    """
    if anomalies.empty:
        return pd.DataFrame(columns=[
            "series", "flag_type", "start", "end", "duration_hours",
            "n_flags", "worst_z", "worst_observed", "expected_median", "is_holiday",
        ])

    a = anomalies.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"])
    a = a.sort_values(["series", "timestamp"])

    events = []
    for series, grp in a.groupby("series"):
        grp = grp.sort_values("timestamp")
        # new event when gap from previous flag exceeds max_gap_hours
        gap = grp["timestamp"].diff().dt.total_seconds().div(3600)
        event_id = (gap > max_gap_hours).fillna(True).cumsum()

        for _, ev in grp.groupby(event_id):
            z_vals = ev["z_score"].dropna()
            events.append({
                "series": series,
                "flag_type": ev["flag_type"].mode().iloc[0],
                "start": ev["timestamp"].min(),
                "end": ev["timestamp"].max(),
                "duration_hours": int(
                    (ev["timestamp"].max() - ev["timestamp"].min()).total_seconds() / 3600
                ) + 1,
                "n_flags": len(ev),
                "worst_z": round(z_vals.min(), 3) if not z_vals.empty else np.nan,
                "worst_observed": round(ev["observed"].min(), 6),
                "expected_median": round(ev["expected_median"].max(), 6),
                "is_holiday": bool(ev["is_holiday"].any()),
            })

    out = pd.DataFrame(events).sort_values(["worst_z", "start"]).reset_index(drop=True)
    return out


def threshold_sensitivity(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    thresholds: list[float] = [-2.0, -2.5, -3.0],
) -> pd.DataFrame:
    """Run scoring at multiple thresholds. Returns comparison table."""
    rows = []
    for t in thresholds:
        a = score_anomalies(df, series_cols, baseline, drop_threshold=t)
        rows.append({
            "threshold": t,
            "n_drop_zscore": (a["flag_type"] == "drop_zscore").sum(),
            "n_drop_zscore_non_holiday": ((a["flag_type"] == "drop_zscore") & ~a["is_holiday"]).sum(),
            "n_drop_zscore_holiday": ((a["flag_type"] == "drop_zscore") & a["is_holiday"]).sum(),
            "n_contextual_zero": (a["flag_type"] == "contextual_zero").sum(),
            "n_holiday_zero": (a["flag_type"] == "contextual_zero_holiday").sum(),
            "n_series_flagged": a[a["flag_type"] == "drop_zscore"]["series"].nunique(),
        })
    return pd.DataFrame(rows).set_index("threshold")
