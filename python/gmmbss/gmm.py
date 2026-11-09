"""GMM estimation of the shifted gamma-kernel BSS model theta = (mu, nu, lambda).

Moment vector (identity-weighted GMM, as in the paper):

    G_bar(theta) = ( mean(X) - mu,
                     gamma_hat(h; mu) - R_{nu,lambda}(h / n),  h in H )

    gamma_hat(h; mu) = (1/N) sum_{t=1}^{N-h} (X_t - mu)(X_{t+h} - mu)

The sums over t do not depend on theta, so they are computed once and each
objective evaluation costs O(|H|) instead of O(N |H|).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .kernels import gamma_acf, gamma_acov

DEFAULT_BOUNDS = ((-10.0, 10.0), (-0.499, 0.499), (0.001, 100.0))


@dataclass
class GMMResult:
    mu: float
    nu: float
    lam: float
    objective: float
    iterations: int = 1

    @property
    def theta(self):
        return np.array([self.mu, self.nu, self.lam])


def _clean_lags(lags):
    lags = np.asarray(lags, dtype=int)
    unique = np.unique(lags)
    if len(unique) < len(lags):
        warnings.warn("Duplicate lags removed (they would be double-weighted).", stacklevel=3)
    if unique.min() < 1:
        raise ValueError("Lags must be positive integers.")
    return unique


class MomentSetup:
    """Precomputed sums for the autocovariance moments of a series Y."""

    def __init__(self, Y, lags):
        Y = np.asarray(Y, dtype=float)
        self.lags = _clean_lags(lags)
        self.N = len(Y)
        if self.lags.max() >= self.N:
            raise ValueError("Largest lag must be smaller than the series length.")
        self.mean = Y.mean()
        self.var = Y.var(ddof=1)
        self.S = np.array([Y[:-h] @ Y[h:] for h in self.lags])
        self.A = np.array([Y[:-h].sum() for h in self.lags])
        self.B = np.array([Y[h:].sum() for h in self.lags])
        self.cnt = self.N - self.lags

    def empirical_acov(self, mu):
        return (self.S - mu * (self.A + self.B) + mu**2 * self.cnt) / self.N


def model_acov(lags, n, nu, lam, variance="model", sigma2=None, sample_var=None):
    """Model autocovariance at lags/n.

    variance="model":  E[sigma^2] = sigma2 is known (simulation studies).
    variance="sample": R = var(X) * acf, for real data where E[sigma^2] is unknown.
    """
    t = np.asarray(lags) / n
    if variance == "model":
        if sigma2 is None:
            raise ValueError("variance='model' needs sigma2 = E[sigma^2].")
        return gamma_acov(t, nu, lam, sigma2)
    if variance == "sample":
        return sample_var * gamma_acf(t, nu, lam)
    raise ValueError("variance must be 'model' or 'sample'")


def moments(theta, setup, n, variance="model", sigma2=None):
    mu, nu, lam = theta
    model = model_acov(setup.lags, n, nu, lam, variance, sigma2, setup.var)
    return np.concatenate([[setup.mean - mu], setup.empirical_acov(mu) - model])


def objective(theta, setup, n, variance="model", sigma2=None, W=None):
    g = moments(theta, setup, n, variance, sigma2)
    return float(g @ g) if W is None else float(g @ W @ g)


def fit_gmm_de(Y, n, lags, variance="model", sigma2=None, bounds=DEFAULT_BOUNDS,
               maxiter=50, popsize=10, polish=False, seed=None, W=None):
    """Identity-weighted GMM minimised by differential evolution.

    The defaults follow R's DEoptim as used in the paper: population 10 x 3,
    50 generations, F = 0.8, CR = 0.5, no local polishing. Set polish=True to
    refine the result with L-BFGS-B.
    """
    setup = Y if isinstance(Y, MomentSetup) else MomentSetup(Y, lags)
    res = differential_evolution(objective, bounds, args=(setup, n, variance, sigma2, W),
                                 maxiter=maxiter, popsize=popsize, mutation=0.8,
                                 recombination=0.5, polish=polish, seed=seed, tol=0)
    mu, nu, lam = res.x
    return GMMResult(mu, nu, lam, float(res.fun))


def hac_matrix(Y, theta, n, lags, variance="model", sigma2=None, m=None, weights="bartlett"):
    """Long-run covariance of G(X_i, theta):

        S_N = Omega_0 + sum_{j=1}^m w_j (Omega_j + Omega_j^T),  default m = floor(N^(1/5)).

    weights="bartlett":  w_j = 1 - j/(m+1) (Newey-West; always positive semi-definite).
    weights="truncated": w_j = 1, the formula written in the paper; it can be
                         indefinite, which makes the weighted objective unbounded below.
    """
    if weights not in ("bartlett", "truncated"):
        raise ValueError("weights must be 'bartlett' or 'truncated'")
    Y = np.asarray(Y, dtype=float)
    lags = _clean_lags(lags)
    N = len(Y)
    m = int(np.floor(N ** 0.2)) if m is None else int(m)
    mu, nu, lam = theta
    c = Y - mu
    L = N - lags.max()
    model = model_acov(lags, n, nu, lam, variance, sigma2, Y.var(ddof=1))
    G = np.column_stack([c[:L]] + [c[:L] * c[h:h + L] - model[j] for j, h in enumerate(lags)])
    S = G.T @ G / N
    for j in range(1, m + 1):
        Om = G[j:].T @ G[:-j] / N
        w = 1 - j / (m + 1) if weights == "bartlett" else 1.0
        S += w * (Om + Om.T)
    return S


def fit_gmm_two_step(Y, n, lags, variance="model", sigma2=None, m=None, weights="bartlett",
                     tol=1e-4, max_iter=10, bounds=DEFAULT_BOUNDS, seed=None, maxiter=50):
    """Iterated GMM with a Newey-West weight matrix (Section 4.3 algorithm).

    1. theta_orig = argmin G_bar' G_bar  (differential evolution)
    2. W = S_N(theta)^{-1}
    3. theta_opti = argmin G_bar' W G_bar  (L-BFGS-B from the previous theta)
    4. repeat 2-3 until max |theta_opti - theta_orig| < tol.
    """
    setup = MomentSetup(Y, lags)
    theta = fit_gmm_de(setup, n, lags, variance, sigma2, bounds, maxiter=maxiter, seed=seed).theta
    for it in range(1, max_iter + 1):
        W = np.linalg.inv(hac_matrix(Y, theta, n, setup.lags, variance, sigma2, m, weights))
        res = minimize(objective, theta, args=(setup, n, variance, sigma2, W),
                       method="L-BFGS-B", bounds=bounds)
        done = np.max(np.abs(res.x - theta)) < tol
        theta = res.x
        if done:
            break
    return GMMResult(*theta, objective=float(res.fun), iterations=it)
