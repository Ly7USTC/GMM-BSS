"""Simulation study: GMM estimates of (mu, nu, lambda) over independent paths.

Each replication draws its own Gamma-OU volatility path, so replications are
independent. Usage:  python examples/simulation_study.py [M]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import gmmbss as gb

M = int(sys.argv[1]) if len(sys.argv) > 1 else 50
n, T, N = 100, 100, 2000                     # 10,001 observations per path
mu, nu = 2.0, -0.2
a, b, ou_rate = 10, 100, 10                  # Gamma(10, 100)-OU volatility
sigma2 = gb.gamma_ou_second_moment(a, b)
lags = [*range(1, 11), 15, 20, 25, 30]
rng = np.random.default_rng(2022)

fig, axes = plt.subplots(3, 3, figsize=(11, 8))
for row, lam in enumerate([1, 10, 100]):
    est = []
    for _ in range(M):
        sigma = gb.gamma_ou(ou_rate, a, b, (N + n * T) / n, 1 / n, method="exact", rng=rng)
        Y = gb.simulate_bss(n, T, nu, lam, sigma=sigma, N=N, mu=mu, rng=rng)
        est.append(gb.fit_gmm_de(Y, n, lags, sigma2=sigma2, seed=rng).theta)
    est = np.array(est)
    for col, (name, true) in enumerate(zip(["mu", "nu", "lambda"], [mu, nu, lam])):
        ax = axes[row, col]
        ax.hist(est[:, col], bins=20, density=True, color="0.7")
        ax.axvline(true, color="red")
        ax.set_title(f"{name}  (true lambda = {lam})", fontsize=9)
    print(f"lambda = {lam:>3}: mean estimates", est.mean(0).round(4), " sd", est.std(0).round(4))
fig.tight_layout()
fig.savefig("simulation_study.png", dpi=130)
