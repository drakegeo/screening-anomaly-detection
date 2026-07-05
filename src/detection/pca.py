"""Second detector: PCA reconstruction-error anomaly detection (multivariate).

Idea
----
The primary z-score detector scores each series on its own. A multivariate view
asks a different question: at a given hour, is the *joint pattern across all 15
series* unusual? PCA learns the normal joint structure from the data, projects
each hour onto a few principal components, and reconstructs it. Hours that do not
reconstruct well (high reconstruction error, a.k.a. the SPE / Q-statistic) break
the normal cross-series structure and are flagged.

Honest caveat for this dataset
------------------------------
The 15 series are near-independent (correlation ~0, except ListA ~0.41). PCA
compresses by exploiting correlation, so with independent series it cannot
compress much — reconstruction error tends toward the raw per-series deviation.
We therefore expect PCA to broadly echo the z-score detector rather than add a
truly orthogonal signal. Running it makes that limitation explicit and testable,
which is itself a defensible finding.

Implementation uses numpy SVD only (no sklearn) to keep the dependency stack
minimal.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.load import INACTIVE_SERIES

VAR_TARGET = 0.90       # keep enough components to explain this much variance
SPE_MAD_K = 3.0         # robust flag threshold: median + k * scaled-MAD of SPE


def _standardised_residual_matrix(
    df: pd.DataFrame, series_cols: list[str], baseline: dict[str, pd.DataFrame]
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex]:
    """Build (n_timestamps × n_series) matrix of (obs − hourly_median)/hourly_MAD.

    Removing the hour-of-day baseline means PCA sees *deviations from normal*,
    not the diurnal pattern itself. Missing values map to 0 (= at baseline).
    """
    cols = [c for c in series_cols if c not in INACTIVE_SERIES]
    hours = df["hour_of_day"].values

    mat = np.zeros((len(df), len(cols)))
    for j, col in enumerate(cols):
        med = baseline["median"].loc[hours, col].values
        mad = baseline["mad"].loc[hours, col].values
        resid = (df[col].values - med) / mad
        resid = np.where(np.isfinite(resid), resid, 0.0)  # NaN/inf → 0
        mat[:, j] = resid
    return mat, cols, df.index


def pca_reconstruction_flags(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    var_target: float = VAR_TARGET,
    spe_k: float = SPE_MAD_K,
) -> dict:
    """Fit PCA on standardised residuals, flag hours with high reconstruction error.

    Returns dict:
        flags        : DataFrame [spe, pca_flag, top_series, top_resid] per timestamp
        var_explained: float — variance captured by kept components
        k            : int   — number of components kept
        spe          : Series — reconstruction error per timestamp
        threshold    : float — SPE flag threshold
        components   : np.ndarray — principal axes (k × n_series)
        cols         : list[str]
    """
    X, cols, index = _standardised_residual_matrix(df, series_cols, baseline)

    # Correlation-PCA: centre AND scale each column to unit variance. Without
    # this, sparse series (MAD near the floor) produce huge residuals and a
    # single series dominates every component. Unit-variance columns give each
    # series equal weight, so the component structure reflects genuine shared
    # variation, not a scaling artifact.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xc = (X - mean) / std

    # SVD → principal components
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    var_ratio = var / var.sum()
    cum = np.cumsum(var_ratio)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = max(1, min(k, len(S)))

    # reconstruct with k components
    Vk = Vt[:k]                       # k × n_series
    proj = Xc @ Vk.T                  # n × k  (scores)
    Xhat = proj @ Vk                  # n × n_series (reconstruction, centred)
    resid = Xc - Xhat                 # reconstruction residual

    spe = np.sum(resid ** 2, axis=1)  # squared prediction error per hour

    # robust threshold on SPE
    med = np.median(spe)
    mad = np.median(np.abs(spe - med)) * 1.4826
    threshold = med + spe_k * (mad if mad > 0 else spe.std())

    # which series drove each flagged hour, and its signed residual (drop = negative)
    top_idx = np.argmax(resid ** 2, axis=1)
    top_series = [cols[i] for i in top_idx]
    top_resid = X[np.arange(len(X)), top_idx]  # signed standardised residual

    flags = pd.DataFrame({
        "spe": np.round(spe, 4),
        "pca_flag": spe > threshold,
        "top_series": top_series,
        "top_resid": np.round(top_resid, 3),
    }, index=index)
    flags.index.name = "timestamp"

    return {
        "flags": flags,
        "var_explained": float(cum[k - 1]),
        "k": k,
        "n_series": len(cols),
        "var_ratio": var_ratio,          # scree: per-component variance share
        "cum_var": cum,
        "scores": U[:, :2] * S[:2],      # first 2 PC scores for scatter
        "spe": pd.Series(spe, index=index, name="spe"),
        "threshold": float(threshold),
        "components": Vk,
        "cols": cols,
    }


def consensus_with_zscore(anomalies: pd.DataFrame, pca_flags: pd.DataFrame) -> dict:
    """How many critical z-score flags fall on a PCA-flagged hour for the same series?"""
    critical = anomalies[
        anomalies["flag_type"].isin(["drop_zscore", "contextual_zero"])
    ].copy()
    critical["timestamp"] = pd.to_datetime(critical["timestamp"])

    pf = pca_flags[pca_flags["pca_flag"]]
    pf_lookup = set(zip(pf.index, pf["top_series"]))

    def _agree(row):
        return (row["timestamp"], row["series"]) in pf_lookup

    n_agree = int(critical.apply(_agree, axis=1).sum()) if not critical.empty else 0
    return {"n_agree": n_agree, "n_critical": len(critical)}


def save_pca_scores(pca_out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pca_out["flags"].to_csv(path)
