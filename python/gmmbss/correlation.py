"""Correlation estimator for the bivariate gamma-kernel BSS model."""

import numpy as np

from .kernels import kernel_cross, kernel_l2


def rho_hat(Y1, Y2, nu1, lam1, nu2, lam2, e_sigma12=None):
    """rho_hat = cov(Y1, Y2) / ( E[sigma1 sigma2] * int g1 g2 ).

    e_sigma12=None uses the plug-in sqrt(E[sigma1^2] E[sigma2^2]) with
    E[sigma_j^2] = var(Y_j) / int g_j^2. That plug-in equals E[sigma1 sigma2]
    only when the volatilities are constant; pass the true value when known.
    """
    Y1 = np.asarray(Y1, float)
    Y2 = np.asarray(Y2, float)
    if e_sigma12 is None:
        e1 = Y1.var(ddof=1) / kernel_l2(nu1, lam1)
        e2 = Y2.var(ddof=1) / kernel_l2(nu2, lam2)
        e_sigma12 = np.sqrt(e1 * e2)
    cov = np.cov(Y1, Y2, ddof=1)[0, 1]
    return cov / (e_sigma12 * kernel_cross(nu1, lam1, nu2, lam2))
