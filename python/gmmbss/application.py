"""Helpers for fitting the model to real time series."""

import numpy as np

from .gmm import fit_gmm_de
from .kernels import gamma_acf


def interpolate_missing(x):
    """Linear interpolation of NaNs (ends are held at the nearest valid value)."""
    x = np.asarray(x, dtype=float).copy()
    bad = np.isnan(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def stl_remainder(x, period=261):
    """Remainder of a periodic STL decomposition, matching R's
    stl(ts(x, frequency = period), s.window = "periodic")."""
    from statsmodels.tsa.seasonal import STL

    x = interpolate_missing(x)
    n = len(x)
    seasonal = 10 * n + 1  # R's choice for s.window = "periodic"
    trend = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
    trend += trend % 2 == 0
    low_pass = period + 1 if period % 2 == 0 else period + 2  # statsmodels needs low_pass > period
    res = STL(x, period=period, seasonal=seasonal, trend=trend, low_pass=low_pass,
              seasonal_deg=0, trend_deg=1, low_pass_deg=1,
              seasonal_jump=int(np.ceil(seasonal / 10)), trend_jump=int(np.ceil(trend / 10)),
              low_pass_jump=int(np.ceil(low_pass / 10)), robust=False).fit(inner_iter=2, outer_iter=0)
    # R replaces a periodic seasonal component by its cycle-subseries means
    cycle = np.arange(n) % period
    means = np.bincount(cycle, weights=res.seasonal) / np.bincount(cycle)
    seasonal_comp = means[cycle]
    return x - seasonal_comp - res.trend


def fit_series(x, n=1, lags=(*range(1, 11), 15, 20, 25, 30), **kwargs):
    """Fit the gamma-kernel BSS model to an observed series. E[sigma^2] is unknown,
    so the model variance is set to the sample variance."""
    return fit_gmm_de(np.asarray(x, float), n, lags, variance="sample", **kwargs)


def plot_acf_fit(x, nu, lam, n=1, max_lag=50, level=0.95, ax=None):
    """Empirical acf (bars) with the fitted gamma-kernel acf (red line)."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    x = np.asarray(x, float) - np.mean(x)
    N = len(x)
    lags = np.arange(max_lag + 1)
    acov = np.array([x[: N - h] @ x[h:] / N for h in lags])
    acf = acov / acov[0]
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.vlines(lags / n, 0, acf, color="black", lw=1)
    ax.plot(lags / n, gamma_acf(lags / n, nu, lam), color="red", label="fitted acf")
    band = norm.ppf((1 + level) / 2) / np.sqrt(N)
    ax.axhline(0, color="black", lw=0.5)
    for s in (-1, 1):
        ax.axhline(s * band, color="blue", ls="--", lw=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("acf")
    return ax
