"""Hybrid scheme simulation of gamma-kernel BSS processes (Bennedsen et al., 2017)."""

import numpy as np
from scipy.signal import fftconvolve

from .kernels import gamma_kernel, hybrid_covariance, hybrid_cross_covariance


def _check_lengths(W, sigma, N, nT, kappa):
    if W.shape != (N + nT, kappa + 1):
        raise ValueError(f"W must have shape {(N + nT, kappa + 1)}, got {W.shape}")
    if len(sigma) != N + nT + 1:
        raise ValueError(f"sigma must have length N + n*T + 1 = {N + nT + 1}, got {len(sigma)}")


def hybrid_path_from_W(W, sigma, n, nu, lam, kappa, N, nT, sigma_alignment="corrected"):
    """Build a BSS path on the grid 0, 1/n, ..., T from hybrid-scheme variables.

    W      : (N + nT, kappa + 1) Gaussian variables; column 0 holds the Brownian
             increments, columns 1..kappa the near-singular terms.
    sigma  : volatility at grid points -N/n, ..., T (length N + nT + 1).
    sigma_alignment:
        "corrected"   - the lower sum pairs sigma[j] with W[j], as the upper sum does.
        "bss_package" - reproduces R's BSS::gammaKernelBSS exactly, where the lower
                        sum takes sigma in reverse order (differs only if kappa > 1
                        and sigma is not constant).
    Returns an array of length nT + 1.
    """
    W = np.asarray(W, float)
    sigma = np.asarray(sigma, float)
    _check_lengths(W, sigma, N, nT, kappa)
    k_upper = np.arange(kappa + 1, N + 1, dtype=float)
    b_star = ((k_upper ** (nu + 1) - (k_upper - 1) ** (nu + 1)) / (nu + 1)) ** (1 / nu)
    g_b = np.concatenate([np.zeros(kappa), gamma_kernel(b_star / n, nu, lam)])
    upper = fftconvolve(g_b, sigma[:-1] * W[:, 0])[N - 1:N + nT]

    i = np.arange(nT + 1)
    lower = np.zeros(nT + 1)
    for k in range(1, kappa + 1):
        rows = i + N - k
        if sigma_alignment == "corrected":
            sig = sigma[rows]
        elif sigma_alignment == "bss_package":
            sig = sigma[i + N - kappa + k - 1]
        else:
            raise ValueError("sigma_alignment must be 'corrected' or 'bss_package'")
        lower += np.exp(-lam * k / n) * sig * W[rows, k]
    return lower + upper


def _default_N(n, N):
    return 100 * n if N is None else int(N)


def simulate_bss(n, T, nu, lam, sigma=None, N=None, kappa=3, mu=0.0,
                 sigma_alignment="corrected", rng=None):
    """Simulate X_t = mu + int g(t - s) sigma_s dW_s on 0, 1/n, ..., T.

    N is the truncation of the Riemann sum (in grid steps), default 100 n.
    sigma must have length N + n*T + 1 (defaults to a constant 1).
    """
    rng = np.random.default_rng(rng)
    nT = int(round(n * T))
    N = _default_N(n, N)
    sigma = np.ones(N + nT + 1) if sigma is None else np.asarray(sigma, float)
    S = hybrid_covariance(kappa, n, nu)
    W = rng.multivariate_normal(np.zeros(kappa + 1), S, size=N + nT, method="eigh")
    return mu + hybrid_path_from_W(W, sigma, n, nu, lam, kappa, N, nT, sigma_alignment)


def simulate_bivariate_bss(n, T, nu1, lam1, nu2, lam2, rho, sigma1=None, sigma2=None,
                           N=None, kappa=3, mu=(0.0, 0.0), cross_covariance="paper",
                           sigma_alignment="corrected", rng=None):
    """Simulate a bivariate gamma-kernel BSS process whose driving Brownian
    motions have correlation rho. Returns (Y1, Y2).

    cross_covariance="paper" uses rho * sd1 sd2^T between the hybrid-scheme
    vectors (as in the paper); "exact" uses the exact cross-covariance.
    """
    rng = np.random.default_rng(rng)
    nT = int(round(n * T))
    N = _default_N(n, N)
    sigma1 = np.ones(N + nT + 1) if sigma1 is None else np.asarray(sigma1, float)
    sigma2 = np.ones(N + nT + 1) if sigma2 is None else np.asarray(sigma2, float)
    S1 = hybrid_covariance(kappa, n, nu1)
    S2 = hybrid_covariance(kappa, n, nu2)
    C = hybrid_cross_covariance(kappa, n, nu1, nu2, rho, cross_covariance)
    S = np.block([[S1, C], [C.T, S2]])
    if np.linalg.eigvalsh(S).min() < -1e-12 * np.abs(S).max():
        raise ValueError("Joint hybrid-scheme covariance is not positive semi-definite; "
                         "try cross_covariance='exact'.")
    W = rng.multivariate_normal(np.zeros(2 * (kappa + 1)), S, size=N + nT, method="eigh")
    Y1 = hybrid_path_from_W(W[:, :kappa + 1], sigma1, n, nu1, lam1, kappa, N, nT, sigma_alignment)
    Y2 = hybrid_path_from_W(W[:, kappa + 1:], sigma2, n, nu2, lam2, kappa, N, nT, sigma_alignment)
    return mu[0] + Y1, mu[1] + Y2
