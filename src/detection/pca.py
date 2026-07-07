"""PCA independence check — justifies the per-series univariate design.

Before choosing a detector we ask: do the 15 series share structure? If they
did, a joint multivariate model would be the right tool. PCA answers directly —
it compresses by exploiting correlation, so the share of variance captured by
the first few components measures how much shared structure exists.

Finding: the series are independent. PC1 explains only ~18% of variance and it
takes 11 of 14 components to reach 90% — variance is spread almost evenly.
Correlated data would concentrate 60–90% in PC1. Independence is therefore a
*proof* that the per-series baseline is the correct architecture (and, later,
exactly what a tree-based detector like Isolation Forest needs — see
isoforest.py).

numpy SVD only (no sklearn) to keep the dependency stack minimal.
"""

import numpy as np
import pandas as pd

from src.processing.load import INACTIVE_SERIES

VAR_TARGET = 0.90       # components needed to explain this much variance


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


def pca_independence_check(
    df: pd.DataFrame,
    series_cols: list[str],
    baseline: dict[str, pd.DataFrame],
    var_target: float = VAR_TARGET,
) -> dict:
    """Fit PCA on standardised residuals; measure how concentrated variance is.

    Concentrated variance (PC1 large) = shared structure = correlated series.
    Evenly-spread variance = independent series = per-series baseline is correct.

    Returns dict:
        var_ratio     : np.ndarray — per-component variance share (scree bars)
        cum_var       : np.ndarray — cumulative variance share
        k             : int   — components needed to reach var_target
        n_series      : int
        var_explained : float — variance captured by the first k components
    """
    X, cols, _index = _standardised_residual_matrix(df, series_cols, baseline)

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
    _U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    var_ratio = var / var.sum()
    cum = np.cumsum(var_ratio)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = max(1, min(k, len(S)))

    return {
        "var_ratio": var_ratio,
        "cum_var": cum,
        "k": k,
        "n_series": len(cols),
        "var_explained": float(cum[k - 1]),
    }
