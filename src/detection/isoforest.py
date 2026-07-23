"""Third detector: Isolation Forest + SHAP (non-linear multivariate).

Why this method, and why *after* PCA
-------------------------------------
PCA failed on this data precisely because the 15 series are independent — a linear
model needs correlation to compress, and there is none. Isolation Forest is the
opposite: it isolates anomalies by *randomly splitting a single feature at a time*.
Independent, sharply-separable features are isolated near the root of the trees, so
independence is a strength, not a weakness. This makes Isolation Forest the natural
multivariate method for exactly the regime where PCA cannot help.

The model scores each *hour* (a 14-dim vector of standardised residuals) for how
easily it is isolated from the rest — a single multivariate anomaly score per hour.

Explainability (SHAP)
---------------------
Compliance teams will not act on a black-box score. SHAP (TreeExplainer) attributes
each hour's anomaly score to individual series — a plain reason code such as
"ListA_field3 contributed the most to this alert". Because the features are
uncorrelated, SHAP attributions are clean (no multicollinearity ambiguity).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detection.pca import _standardised_residual_matrix

RANDOM_STATE = 42
N_ESTIMATORS = 300
SCORE_MAD_K = 3   # robust flag threshold on the anomaly score


def fit_isolation_forest(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    n_estimators: int = N_ESTIMATORS,
    score_k: float = SCORE_MAD_K,
) -> dict:
    """Fit an Isolation Forest on the standardised-residual matrix.

    Returns dict:
        scores    : DataFrame [anomaly_score, iso_flag, top_series, top_resid, is_drop]
        model     : fitted IsolationForest
        X         : residual matrix (np.ndarray)
        cols      : list[str]
        index     : DatetimeIndex
        threshold : float — robust flag threshold on anomaly_score
    """
    X_full, cols, index = _standardised_residual_matrix(df, series_cols, baseline)

    # Drops-only encoding: clip positive residuals (spikes) to 0 so the forest
    # isolates hours that are extreme in the *downward* direction. A normal or
    # elevated hour maps to ~0 (dense region); a genuine drop stands out. This
    # aligns the multivariate detector with the drop-only business goal, exactly
    # like the primary z-score test.
    X = np.minimum(X_full, 0.0)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=RANDOM_STATE,
    )
    model.fit(X)

    # score_samples: higher = more normal. Invert so higher = more anomalous.
    raw = model.score_samples(X)

    anomaly_score = -raw

    med = np.median(anomaly_score)
    mad = np.median(np.abs(anomaly_score - med)) * 1.4826
    threshold = med + score_k * (mad if mad > 0 else anomaly_score.std())

    # dominant series per hour = most negative residual (drives the drop)
    top_idx = np.argmin(X, axis=1)
    top_series = [cols[i] for i in top_idx]
    top_resid = X[np.arange(len(X)), top_idx]

    scores = pd.DataFrame({
        "anomaly_score": np.round(anomaly_score, 4),
        "iso_flag": anomaly_score > threshold,
        "top_series": top_series,
        "top_resid": np.round(top_resid, 3),
        "is_drop": top_resid < 0,
    }, index=index)
    scores.index.name = "timestamp"

    return {
        "scores": scores,
        "model": model,
        "X": X,
        "cols": cols,
        "index": index,
        "threshold": float(threshold),
    }


def shap_reason_codes(iso_out: dict) -> dict:
    """Compute SHAP values for the Isolation Forest.

    Returns dict:
        shap_values : np.ndarray (n_hours × n_series)
        global_importance : Series — mean |SHAP| per series (ranked)
        cols  : list[str]
        index : DatetimeIndex
    """
    import shap

    model = iso_out["model"]
    X = iso_out["X"]
    cols = iso_out["cols"]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=False)
    sv = np.asarray(sv)

    # SHAP sign convention for IsolationForest: more negative = pushes toward
    # anomalous. Flip so positive = pushes toward anomalous (easier to read).
    contrib = -sv

    global_importance = (
        pd.Series(np.abs(contrib).mean(axis=0), index=cols)
        .sort_values(ascending=False)
    )

    return {
        "shap_values": contrib,
        "global_importance": global_importance,
        "cols": cols,
        "index": iso_out["index"],
    }


def top_reason_for_hour(shap_out: dict, timestamp: pd.Timestamp) -> pd.Series:
    """SHAP contribution per series for one hour, ranked — the reason code."""
    idx = shap_out["index"].get_loc(timestamp)
    return (
        pd.Series(shap_out["shap_values"][idx], index=shap_out["cols"])
        .sort_values(ascending=False)
    )


def consensus_with_zscore(anomalies: pd.DataFrame, iso_scores: pd.DataFrame) -> dict:
    """Overlap of Isolation-Forest drop flags with z-score critical flags."""
    critical = anomalies[
        anomalies["flag_type"].isin(["drop_zscore", "contextual_zero"])
    ].copy()
    critical["timestamp"] = pd.to_datetime(critical["timestamp"])

    iso_hits = iso_scores[iso_scores["iso_flag"] & iso_scores["is_drop"]]
    iso_lookup = set(zip(iso_hits.index, iso_hits["top_series"]))

    n_agree = int(
        critical.apply(lambda r: (r["timestamp"], r["series"]) in iso_lookup, axis=1).sum()
    ) if not critical.empty else 0

    return {
        "n_agree": n_agree,
        "n_critical": len(critical),
        "n_iso_drop_flags": len(iso_hits),
    }


def save_scores(iso_out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iso_out["scores"].to_csv(path)
