"""Defensive data checks — the questions an interviewer will ask.

Two checks that validate design choices rather than detect anomalies:

1. day_of_week_check
   We baseline on hour-of-day only. Would hour x day-of-week be better?
   Adding day-of-week splits each baseline cell 5 ways (20 obs -> ~4 obs),
   so it only earns its keep if it materially reduces residual spread. This
   quantifies the trade-off so the choice is defensible, not assumed.

2. alert_confidence_audit
   A "drop" from a sparse, few-observation baseline can be noise, not signal.
   This looks up the baseline observation count behind each top alert, proving
   whether the headline anomalies rest on solid baselines or thin ones.
"""

import numpy as np
import pandas as pd

from src.processing.load import INACTIVE_SERIES
from src.detection.anomaly import CONTEXTUAL_ZERO_MIN_MEDIAN


def day_of_week_check(
    df: pd.DataFrame,
    series_cols: list[str],
) -> pd.DataFrame:
    """Quantify how much day-of-week would reduce residual spread vs hour-only.

    For each series, compares the residual MAD under two baselines:
        - hour-of-day only        (our production choice, 20 obs/cell)
        - hour-of-day x day-of-week (finer, but ~4 obs/cell)

    A large reduction means day-of-week carries real signal; a small reduction
    means the extra granularity just buys sparsity and overfitting.

    Returns DataFrame indexed by series:
        hour_mad, hour_dow_mad, reduction_pct
    """
    active = [c for c in series_cols if c not in INACTIVE_SERIES]
    rows = []
    for c in active:
        med_h = df.groupby("hour_of_day")[c].transform("median")

        mad_h = (df[c] - med_h).abs().median()
        med_hd = df.groupby(["hour_of_day", "day_of_week"])[c].transform("median")

        mad_hd = (df[c] - med_hd).abs().median()
        red = (1 - mad_hd / mad_h) * 100 if mad_h > 0 else np.nan

        rows.append({
            "series": c,
            "hour_mad": round(float(mad_h), 6),
            "hour_dow_mad": round(float(mad_hd), 6),
            "reduction_pct": round(float(red), 1) if not np.isnan(red) else np.nan,
        })

    out = pd.DataFrame(rows).set_index("series")
    return out


def alert_confidence_audit(
    events: pd.DataFrame,
    baseline: dict[str, pd.DataFrame],
    top_n: int = 12,
) -> pd.DataFrame:
    """Attach baseline observation count + confidence to each top alert.

    Answers: is our #1 alert real, or an artifact of a thin baseline?

    Returns the top-N non-holiday events with added columns:
        hour, n_obs, low_conf, baseline_median
    """
    if events.empty:
        return events

    ev = events[~events["is_holiday"]].sort_values("worst_z").head(top_n).copy()
    cnt = baseline["count"]
    low = baseline["low_conf"]
    med = baseline["median"]

    hours, n_obs, low_conf, base_med = [], [], [], []
    for _, e in ev.iterrows():
        s = e["series"]
        h = pd.Timestamp(e["start"]).hour
        hours.append(h)
        n_obs.append(int(cnt.loc[h, s]) if s in cnt.columns else np.nan)
        low_conf.append(bool(low.loc[h, s]) if s in low.columns else True)
        base_med.append(round(float(med.loc[h, s]), 6) if s in med.columns else np.nan)

    ev["hour"] = hours
    ev["n_obs"] = n_obs
    ev["low_conf"] = low_conf
    ev["baseline_median"] = base_med
    return ev


def baseline_confidence_summary(
    baseline: dict[str, pd.DataFrame],
    series_cols: list[str],
) -> pd.DataFrame:
    """Per-series count of active-hour cells that are low-confidence (<3 obs).

    A clean sheet (0 low-confidence active cells) means every flag the detector
    can raise rests on a full baseline.
    """
    active = [c for c in series_cols if c not in INACTIVE_SERIES]
    med = baseline["median"]
    low = baseline["low_conf"]
    rows = []
    for c in active:
        is_active = med[c] > CONTEXTUAL_ZERO_MIN_MEDIAN
        rows.append({
            "series": c,
            "n_active_hours": int(is_active.sum()),
            "n_low_conf_active": int((low[c] & is_active).sum()),
        })
    return pd.DataFrame(rows).set_index("series")
