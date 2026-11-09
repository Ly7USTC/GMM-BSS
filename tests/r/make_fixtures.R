# Checks the clean R code against the BSS package and the original draft
# functions, then writes fixtures used by the Python tests.
# Run from the repository root:  Rscript tests/r/make_fixtures.R
suppressPackageStartupMessages({ library(BSS); library(MASS); library(DEoptim) })
source("R/gmmbss.R")
out <- "tests/fixtures"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
wcsv <- function(x, f) write.csv(x, file.path(out, f), row.names = FALSE)
check <- function(label, a, b, tol = 1e-10) {
  err <- max(abs(a - b))
  cat(sprintf("%-62s max abs diff = %.2e  %s\n", label, err, if (err < tol) "OK" else "FAIL"))
  if (err >= tol) quit(status = 1)
}

# ---- 1. hybrid scheme covariance matrices ----------------------------------
cov_cases <- expand.grid(kappa = c(1, 3, 5), n = c(10, 100), alpha = c(-0.4, -0.2, 0.3))
cov_rows <- do.call(rbind, lapply(seq_len(nrow(cov_cases)), function(r) {
  p <- cov_cases[r, ]
  S <- hybridSchemeCovarianceMatrix(p$kappa, p$n, p$alpha)
  data.frame(kappa = p$kappa, n = p$n, alpha = p$alpha,
             i = rep(seq_len(nrow(S)), ncol(S)), j = rep(seq_len(ncol(S)), each = nrow(S)),
             value = as.vector(S))
}))
wcsv(cov_rows, "hybrid_covariance.csv")

# ---- 2. hybrid scheme paths: clean code vs BSS::gammaKernelBSS -------------
N <- 500; n <- 50; T <- 4; kappa <- 3; nu <- -0.3; lam <- 2
set.seed(1)
sig <- gamma_ou(10, 10, 100, (N + n * T) / n, 1 / n)
set.seed(2); pkg <- gammaKernelBSS(N, n, T, kappa, nu, lam, sig)
set.seed(2); W <- mvrnorm(N + n * T, mu = rep(0, kappa + 1),
                          Sigma = hybridSchemeCovarianceMatrix(kappa, n, nu))
check("clean hybrid ('bss_package' alignment) vs BSS::gammaKernelBSS",
      hybrid_path_from_W(W, sig, n, nu, lam, kappa, N, n * T, "bss_package"), pkg$bss)
check("clean hybrid with sigma = 1 vs BSS Gaussian core",
      hybrid_path_from_W(W, rep(1, length(sig)), n, nu, lam, kappa, N, n * T), pkg$core)
wcsv(data.frame(W), "hybrid_W.csv")
wcsv(data.frame(sigma = sig), "hybrid_sigma.csv")
wcsv(data.frame(bss_package = pkg$bss, core = pkg$core,
                corrected = hybrid_path_from_W(W, sig, n, nu, lam, kappa, N, n * T, "corrected")),
     "hybrid_paths.csv")
wcsv(data.frame(N = N, n = n, T = T, kappa = kappa, nu = nu, lambda = lam), "hybrid_params.csv")

# ---- 3. bivariate: clean code vs draft multigammaKernelBSS_mod_2 -----------
source("tests/r/draft_functions.R")
nu2 <- -0.1; lam2 <- 1.5; rho <- 0.5
set.seed(3); sig2 <- gamma_ou(10, 10, 100, (N + n * T) / n, 1 / n)
set.seed(4); draft <- multigammaKernelBSS_mod_2(N, n, T, kappa, nu, lam, nu2, lam2, sig, sig2, rho)
S1 <- hybridSchemeCovarianceMatrix(kappa, n, nu); S2 <- hybridSchemeCovarianceMatrix(kappa, n, nu2)
C <- hybrid_cross_covariance(kappa, n, nu, nu2, rho, "paper")
set.seed(4); Wb <- mvrnorm(N + n * T, mu = rep(0, 2 * (kappa + 1)),
                           Sigma = rbind(cbind(S1, C), cbind(t(C), S2)))
Y1 <- hybrid_path_from_W(Wb[, 1:(kappa + 1)], sig, n, nu, lam, kappa, N, n * T, "bss_package")
Y2 <- hybrid_path_from_W(Wb[, (kappa + 2):(2 * kappa + 2)], sig2, n, nu2, lam2, kappa, N, n * T, "bss_package")
check("clean bivariate component 1 vs draft multigammaKernelBSS_mod_2", Y1, draft$bss_1)
check("clean bivariate component 2 vs draft multigammaKernelBSS_mod_2", Y2, draft$bss_2)
wcsv(data.frame(Wb), "bivariate_W.csv")
wcsv(data.frame(sigma2 = sig2), "bivariate_sigma2.csv")
wcsv(data.frame(Y1 = Y1, Y2 = Y2), "bivariate_paths.csv")

Cex <- hybrid_cross_covariance(kappa, n, nu, nu, rho, "exact")
check("exact cross-covariance with nu1 = nu2 equals rho * Sigma", Cex, rho * S1, 1e-8)
Cex2 <- hybrid_cross_covariance(kappa, n, nu, nu2, rho, "exact")
wcsv(data.frame(Cex2), "cross_covariance_exact.csv")

# ---- 4. acf / autocovariance vs BSS::gammaKernelCorrelation ----------------
h <- c(0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 3)
acf_rows <- do.call(rbind, lapply(c(-0.4, -0.2, 0.2, 0.45), function(a) do.call(rbind,
  lapply(c(0.5, 1, 10), function(l) data.frame(nu = a, lambda = l, t = h,
    acf = gammaKernelCorrelation(a, l, h), acov = gamma_acov(h, a, l, 0.011))))))
check("gamma_acf vs BSS::gammaKernelCorrelation",
      with(acf_rows, gamma_acf(t, nu, lambda)), acf_rows$acf, 1e-12)
wcsv(acf_rows, "acf.csv")

# ---- 5. GMM objective: vectorised vs draft criterion functions -------------
set.seed(5)
Y <- 2 + simulate_bss(100, 20, -0.2, 10, sigma = gamma_ou(10, 10, 100, 120, 0.01), N = 10000)
lags <- c(1:10, 15, 20, 25, 30)
setup <- gmm_moment_setup(Y, lags)
thetas <- rbind(c(2, -0.2, 10), c(1.9, -0.1, 5), c(2.1, 0.3, 30), c(0, -0.45, 0.5))
obj_model <- apply(thetas, 1, gmm_objective, setup = setup, n = 100, variance = "model", sigma2 = 0.011)
obj_sample <- apply(thetas, 1, gmm_objective, setup = setup, n = 100, variance = "sample")
check("objective ('model') vs draft gmm_DEoptim criterion",
      obj_model, apply(thetas, 1, draft_criterion_model, Y = Y, n = 100, select_lags = lags), 1e-12)
check("objective ('sample') vs draft gmm_DEoptim_CO criterion",
      obj_sample, apply(thetas, 1, draft_criterion_sample, Y = Y, n = 100, select_lags = lags), 1e-12)
wcsv(data.frame(Y = Y), "gmm_Y.csv")
wcsv(data.frame(mu = thetas[, 1], nu = thetas[, 2], lambda = thetas[, 3],
                obj_model = obj_model, obj_sample = obj_sample), "gmm_objective.csv")
for (wt in c("bartlett", "truncated")) {
  wcsv(data.frame(hac_matrix(Y, c(2, -0.2, 10), 100, lags, "model", 0.011, weights = wt)),
       paste0("hac_matrix_", wt, ".csv"))
}

# ---- 6. rho estimator ------------------------------------------------------
r_known <- rho_hat(Y1, Y2, nu, lam, nu2, lam2, e_sigma12 = 0.01)
r_plug <- rho_hat(Y1, Y2, nu, lam, nu2, lam2)
check("closed-form int g1 g2 vs numerical integration",
      kernel_cross(nu, lam, nu2, lam2),
      integrate(function(x) x^(nu + nu2) * exp(-(lam + lam2) * x), 0, Inf)$value, 1e-7)
wcsv(data.frame(rho_known = r_known, rho_plugin = r_plug), "rho.csv")

# ---- 7. STL remainder ------------------------------------------------------
set.seed(7)
tt <- 1:(261 * 6)
x <- 20 + 3 * sin(2 * pi * tt / 261) + 0.002 * tt + as.numeric(arima.sim(list(ar = 0.9), length(tt)))
x[c(50, 51, 400)] <- NA
wcsv(data.frame(x = x, remainder = stl_remainder(x, 261, start = 2010)), "stl.csv")

cat("All R reference checks passed; fixtures written to", out, "\n")
