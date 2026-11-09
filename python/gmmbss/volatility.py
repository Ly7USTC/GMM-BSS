"""Gamma-OU volatility process."""

import numpy as np
from scipy.signal import lfilter


def gamma_ou(lam, a, b, T, dt, y0=None, method="euler", rng=None):
    """Simulate a Gamma(shape a, rate b)-OU process on [0, T] with step dt.

        dx_t = -lam x_t dt + dz_{lam t},

    where z is compound Poisson with rate a and Exp(rate b) jumps, so the
    stationary marginal is Gamma(a, b) for every lam.

    method="euler": x_i = (1 - lam dt) x_{i-1} + J_i  (Schoutens 2003, as used in
                    the paper; needs lam * dt << 1, and inflates the variance by
                    roughly 1 / (1 - lam dt / 2)).
    method="exact": x_i = exp(-lam dt) x_{i-1} + sum_j exp(-lam (t_i - tau_j)) E_j.

    y0=None starts from a draw of the stationary Gamma(a, b) distribution.
    Returns an array of length round(T / dt) + 1 at times 0, dt, ..., T.
    """
    rng = np.random.default_rng(rng)
    M = int(round(T / dt))
    if y0 is None:
        y0 = rng.gamma(shape=a, scale=1.0 / b)
    n_jumps = rng.poisson(lam * a * T)
    tau = rng.uniform(0, T, n_jumps)
    E = rng.exponential(scale=1.0 / b, size=n_jumps)
    step = np.clip(np.ceil(tau / dt).astype(int), 1, M)  # jump in ((s-1)dt, s dt]
    if method == "exact":
        phi = np.exp(-lam * dt)
        E = E * np.exp(-lam * (step * dt - tau))
    elif method == "euler":
        phi = 1.0 - lam * dt
    else:
        raise ValueError("method must be 'euler' or 'exact'")
    J = np.bincount(step - 1, weights=E, minlength=M)
    x, _ = lfilter([1.0], [1.0, -phi], J, zi=[phi * y0])
    return np.concatenate([[y0], x])
