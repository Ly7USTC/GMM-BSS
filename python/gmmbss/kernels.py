"""Gamma kernel, its autocorrelation/autocovariance, and hybrid-scheme covariances."""

import numpy as np
from scipy import integrate
from scipy.special import gamma as gamma_fn
from scipy.special import hyp2f1, kv


def gamma_kernel(x, nu, lam):
    """g(x) = x^nu exp(-lam x) for x > 0."""
    x = np.asarray(x, dtype=float)
    return x**nu * np.exp(-lam * x)


def gamma_acf(t, nu, lam):
    """Autocorrelation of a gamma-kernel BSS process at lag t (in time units).

    acf(t) = 2^(1 - a) / Gamma(a) * (lam t)^a * K_a(lam t),  a = nu + 1/2.
    """
    t, nu, lam = np.broadcast_arrays(np.asarray(t, float), np.asarray(nu, float),
                                     np.asarray(lam, float))
    a = nu + 0.5
    x = lam * np.abs(t)
    out = np.ones(x.shape)
    pos = x > 0
    out[pos] = 2.0 ** (1 - a[pos]) / gamma_fn(a[pos]) * x[pos] ** a[pos] * kv(a[pos], x[pos])
    return out


def kernel_l2(nu, lam):
    """int_0^inf g(x)^2 dx = Gamma(2 nu + 1) / (2 lam)^(2 nu + 1)."""
    return gamma_fn(2 * nu + 1) / (2 * lam) ** (2 * nu + 1)


def kernel_cross(nu1, lam1, nu2, lam2):
    """int_0^inf g1(x) g2(x) dx = Gamma(nu1 + nu2 + 1) / (lam1 + lam2)^(nu1 + nu2 + 1)."""
    return gamma_fn(nu1 + nu2 + 1) / (lam1 + lam2) ** (nu1 + nu2 + 1)


def gamma_acov(t, nu, lam, sigma2):
    """Autocovariance R(t) = E[sigma^2] * int g^2 * acf(t)."""
    return sigma2 * kernel_l2(nu, lam) * gamma_acf(t, nu, lam)


def gamma_ou_second_moment(a, b):
    """E[sigma^2] when sigma is a stationary Gamma(shape a, rate b)-OU process."""
    return a / b**2 + (a / b) ** 2


def hybrid_covariance(kappa, n, alpha):
    """Covariance matrix of the hybrid-scheme Gaussian vector, (kappa+1) x (kappa+1).

    Same as BSS::hybridSchemeCovarianceMatrix in R.
    """
    S = np.zeros((kappa + 1, kappa + 1))
    S[0, 0] = 1.0 / n
    for j in range(2, kappa + 2):  # 1-based column index, as in the R code
        S[0, j - 1] = ((j - 1) ** (alpha + 1) - (j - 2) ** (alpha + 1)) / (alpha + 1) / n ** (alpha + 1)
        S[j - 1, j - 1] = (((j - 1) ** (2 * alpha + 1) - (j - 2) ** (2 * alpha + 1))
                           / (2 * alpha + 1) / n ** (2 * alpha + 1))
        for k in range(j + 1, kappa + 2):
            t1 = (j - 1) ** (alpha + 1) * (k - 1) ** alpha * hyp2f1(-alpha, 1, alpha + 2, (j - 1) / (k - 1))
            t2 = 0.0
            if j > 2:
                t2 = (j - 2) ** (alpha + 1) * (k - 2) ** alpha * hyp2f1(-alpha, 1, alpha + 2, (j - 2) / (k - 2))
            S[j - 1, k - 1] = (t1 - t2) / (alpha + 1) / n ** (2 * alpha + 1)
    return S + np.triu(S, 1).T


def hybrid_cross_covariance(kappa, n, nu1, nu2, rho, method="paper"):
    """Cross-covariance between the hybrid-scheme vectors of two Brownian motions
    with instantaneous correlation rho.

    method="paper": rho * sd1 sd2^T, as in the paper (approximate; the joint
                    matrix is not guaranteed to be positive definite).
    method="exact": rho * int_0^{1/n} (k/n - u)^nu1 (l/n - u)^nu2 du.
    """
    if method == "paper":
        s1 = np.sqrt(np.diag(hybrid_covariance(kappa, n, nu1)))
        s2 = np.sqrt(np.diag(hybrid_covariance(kappa, n, nu2)))
        return rho * np.outer(s1, s2)
    if method != "exact":
        raise ValueError("method must be 'paper' or 'exact'")
    C = np.zeros((kappa + 1, kappa + 1))
    for k in range(kappa + 1):
        for l in range(kappa + 1):
            e1 = nu1 if k > 0 else 0.0
            e2 = nu2 if l > 0 else 0.0
            if k <= 1 and l <= 1:
                # (1 - v)^(e1 + e2) on [0, 1]
                val = 1.0 / (e1 + e2 + 1)
            elif k == 1 or l == 1:
                # singular factor (1 - v)^e handled by the algebraic weight
                e_sing, e_other, c = (e1, e2, l) if k == 1 else (e2, e1, k)
                val = integrate.quad(lambda v: (c - v) ** e_other, 0, 1,
                                     weight="alg", wvar=(0, e_sing), epsabs=1e-13, epsrel=1e-12)[0]
            else:
                val = integrate.quad(lambda v: (k - v) ** e1 * (l - v) ** e2, 0, 1,
                                     epsabs=1e-13, epsrel=1e-12)[0]
            C[k, l] = val / n ** (e1 + e2 + 1)
    return rho * C
