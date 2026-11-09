# Fits 10 simulated paths with the R estimator; python/tests/compare_estimates.py
# fits the same paths in Python and compares.
suppressPackageStartupMessages({ library(BSS); library(MASS); library(DEoptim) })
source("R/gmmbss.R")
set.seed(21)
lags <- c(1:10, 15, 20, 25, 30)
paths <- sapply(1:10, function(j) {
  v <- gamma_ou(10, 10, 100, 120, 0.01, method = "exact")
  2 + simulate_bss(100, 100, -0.2, 10, sigma = v, N = 2000)
})
est <- t(apply(paths, 2, function(Y) unlist(fit_gmm_de(Y, 100, lags, sigma2 = 0.011, itermax = 300))))
write.csv(paths, "tests/fixtures/compare_paths.csv", row.names = FALSE)
write.csv(est, "tests/fixtures/compare_estimates_r.csv", row.names = FALSE)
