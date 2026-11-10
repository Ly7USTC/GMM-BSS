# GMM estimation of Brownian semistationary processes

Simulation and generalised-method-of-moments (GMM) estimation for gamma-kernel
Brownian semistationary (BSS) processes. Python package with an R reference
implementation; the Python code is cross-checked against R in the test suite.

## Model

$$
X_t = \mu + \int_{-\infty}^{t} g(t-s)\sigma_sdW_s, \qquad g(x) = x^{\nu} e^{-\lambda x},
\quad \nu \in \left(-\tfrac12, \tfrac12\right)\setminus\lbrace 0\rbrace,\ \lambda > 0 .
$$

The volatility $\sigma$ is a stationary process independent of $W$ (a Gamma-OU
process in the simulations). The autocovariance has the closed form

$$
R_{\nu,\lambda}(t) = \mathbb{E}[\sigma_0^2]\frac{\Gamma(2\nu+1)}{(2\lambda)^{2\nu+1}}
\tilde K_{\nu+\frac12}(\lambda t), \qquad
\tilde K_a(x) = \frac{2^{1-a}}{\Gamma(a)}\,x^{a} K_a(x),
$$

where $K_a$ is the modified Bessel function of the second kind.

## Estimator

For observations $X_0, \dots, X_N$ at frequency $n$ and a set of lags $H$, the
moment conditions are

$$
\mathbb{E}[X_i - \mu] = 0, \qquad
\mathbb{E}\big[(X_i-\mu)(X_{i+h}-\mu)\big] - R_{\nu,\lambda}(h/n) = 0, \quad h \in H,
$$

and $\hat\theta = \arg\min_\theta \bar G_N(\theta)^\top W_N \bar G_N(\theta)$ with
$\theta = (\mu, \nu, \lambda)$.

* `fit_gmm_de`: identity weight, minimised by differential evolution.
* `fit_gmm_two_step`: iterated GMM with a Newey–West weight matrix.
* `rho_hat`: correlation of the driving Brownian motions in the bivariate model.
* `fit_series`: real data, with $\mathbb{E}[\sigma^2]$ replaced by the sample variance.

## Repository layout

```
python/gmmbss/       Python package
  volatility.py      Gamma-OU process (Euler or exact)
  kernels.py         acf, autocovariance, hybrid-scheme covariances
  hybrid.py          hybrid scheme, univariate and bivariate
  gmm.py             moments, GMM estimators, HAC weight matrix
  correlation.py     correlation estimator
  application.py     STL remainder, fitting and plotting for real data
python/tests/        pytest suite (cross-checks against R + statistical tests)
python/examples/     simulation study, bivariate example, VIX application
R/gmmbss.R           R reference implementation
tests/r/             R checks against the BSS package; writes the test fixtures
tests/fixtures/      reference values produced by R
```

## Quick start

```bash
cd python
pip install -e ".[test]"
pytest
```

```python
import numpy as np
import gmmbss as gb

rng = np.random.default_rng(1)
n, T, N = 100, 1000, 10000
sigma = gb.gamma_ou(10, 10, 100, (N + n * T) / n, 1 / n, method="exact", rng=rng)
X = gb.simulate_bss(n, T, nu=-0.2, lam=1, sigma=sigma, N=N, mu=2, rng=rng)

fit = gb.fit_gmm_de(X, n, lags=[*range(1, 11), 15, 20, 25, 30],
                    sigma2=gb.gamma_ou_second_moment(10, 100), seed=1)
print(fit.mu, fit.nu, fit.lam)
```

## Implementation notes

* **Fast moments.** The lag sums do not depend on θ, so they are computed once;
  each objective evaluation then costs O(|H|) instead of O(N·|H|).
* **Hybrid scheme.** `sigma_alignment="corrected"` pairs the volatility with the
  matching Brownian term in the near-singular part of the scheme.
  `"bss_package"` reproduces R's `BSS::gammaKernelBSS` exactly, which takes the
  volatility in reverse order there. The two agree when σ is constant or κ = 1.
* **Bivariate cross-covariance.** `cross_covariance="paper"` uses
  ρ·sd₁·sd₂ᵀ between the hybrid-scheme vectors; `"exact"` computes the exact
  cross-covariance, which always gives a valid joint covariance matrix.
* **Weight matrix.** `weights="bartlett"` (default) uses Newey–West weights and is
  always positive semi-definite; `"truncated"` uses equal weights.
* **Gamma-OU.** `method="euler"` needs λΔ ≪ 1 and inflates the variance by about
  1/(1 − λΔ/2); `method="exact"` has the exact Gamma marginal.

## R reference

```bash
Rscript tests/r/make_fixtures.R      # needs BSS, MASS, DEoptim, hypergeo
```

This checks the clean R code against `BSS::gammaKernelBSS` and the original
research scripts, then writes the fixtures used by `python/tests/test_against_r.py`.

## References

* O. E. Barndorff-Nielsen and J. Schmiegel (2009). Brownian semistationary processes and volatility/intermittency.
* M. Bennedsen, A. Lunde and M. S. Pakkanen (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics*, 21(4), 931–965.
* W. K. Newey and K. D. West (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
* W. Schoutens (2003). *Lévy Processes in Finance*. Wiley.

## Author

Yuan Li, Department of Mathematics, Imperial College London
