"""Bivariate gamma-kernel BSS process: simulate, fit each component, estimate rho."""

import matplotlib.pyplot as plt
import numpy as np

import gmmbss as gb

rng = np.random.default_rng(2022)
n, T, N, kappa = 100, 100, 2000, 5
nu1, lam1, nu2, lam2, rho = -0.4, 1.0, -0.1, 1.5, 0.5
a, b, ou_rate = 10, 100, 10
lags = [*range(1, 11), 15, 20, 25, 30]

s1 = gb.gamma_ou(ou_rate, a, b, (N + n * T) / n, 1 / n, method="exact", rng=rng)
s2 = gb.gamma_ou(ou_rate, a, b, (N + n * T) / n, 1 / n, method="exact", rng=rng)
Y1, Y2 = gb.simulate_bivariate_bss(n, T, nu1, lam1, nu2, lam2, rho, s1, s2, N=N, kappa=kappa,
                                   cross_covariance="exact", rng=rng)

sigma2 = gb.gamma_ou_second_moment(a, b)
f1 = gb.fit_gmm_de(Y1, n, lags, sigma2=sigma2, seed=1)
f2 = gb.fit_gmm_de(Y2, n, lags, sigma2=sigma2, seed=2)
e_sigma12 = (a / b) ** 2                     # independent volatilities: E[s1] E[s2]
r = gb.rho_hat(Y1, Y2, f1.nu, f1.lam, f2.nu, f2.lam, e_sigma12=e_sigma12)
print("component 1:", f1.theta.round(3), " component 2:", f2.theta.round(3), " rho_hat:", round(r, 3))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].plot(Y1[:500], color="blue", lw=0.7)
axes[0].plot(Y2[:500], color="red", lw=0.7)
axes[0].set_title("first 500 observations")
gb.plot_acf_fit(Y1, f1.nu, f1.lam, n=n, max_lag=100, ax=axes[1])
gb.plot_acf_fit(Y2, f2.nu, f2.lam, n=n, max_lag=100, ax=axes[2])
fig.tight_layout()
fig.savefig("bivariate_example.png", dpi=130)
