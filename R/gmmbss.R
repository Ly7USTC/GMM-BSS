# =============================================================================
# GMM estimation of gamma-kernel Brownian semistationary (BSS) processes
# Clean R reference implementation.
#
#   X_t = mu + int_{-inf}^t g(t-s) sigma_s dW_s,   g(x) = x^nu exp(-lambda x)
#
# Contents
#   1. Gamma-OU volatility
#   2. Gamma-kernel autocorrelation / autocovariance
#   3. Hybrid scheme (univariate and bivariate)
#   4. GMM estimation (identity weight with DEoptim; two-step HAC weight)
#   5. Correlation estimator for the bivariate model
#   6. Real data helper (STL remainder)
#
# Requires: BSS (hybridSchemeCovarianceMatrix), MASS, DEoptim
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Gamma(a, b)-OU volatility:  dx_t = -lambda x_t dt + dz_{lambda t}
#    z is compound Poisson with rate a and Exp(rate b) jumps, so the stationary
#    marginal is Gamma(shape a, rate b).
#    method = "euler": x_i = (1 - lambda dt) x_{i-1} + J_i  (Schoutens 2003;
#                      needs lambda * dt << 1)
#    method = "exact": x_i = exp(-lambda dt) x_{i-1} + sum_j exp(-lambda (t_i - tau_j)) E_j
#    Returns a vector of length round(T / dt) + 1 at times 0, dt, ..., T.
# -----------------------------------------------------------------------------
gamma_ou <- function(lambda, a, b, T, dt, y0 = NULL, method = c("euler", "exact")) {
  method <- match.arg(method)
  M <- round(T / dt)
  if (is.null(y0)) y0 <- rgamma(1, shape = a, rate = b)   # start in stationarity
  n_jumps <- rpois(1, lambda * a * T)
  tau <- runif(n_jumps, 0, T)
  E <- rexp(n_jumps, rate = b)
  step <- pmax(1L, ceiling(tau / dt))                      # jump falls in ((s-1)dt, s dt]
  J <- numeric(M)
  if (n_jumps > 0) {
    w <- if (method == "exact") exp(-lambda * (step * dt - tau)) else 1
    J <- as.numeric(tapply(c(E * w, numeric(M)), c(step, seq_len(M)), sum))
  }
  phi <- if (method == "exact") exp(-lambda * dt) else 1 - lambda * dt
  x <- as.numeric(stats::filter(J, phi, method = "recursive", init = y0))
  c(y0, x)
}

gamma_ou_second_moment <- function(a, b) a / b^2 + (a / b)^2

# -----------------------------------------------------------------------------
# 2. Gamma-kernel autocorrelation and autocovariance
#    acf(t) = 2^(1 - nu') / Gamma(nu') (lambda t)^nu' K_nu'(lambda t),  nu' = nu + 1/2
#    R(t)   = E[sigma^2] * int_0^inf g^2 * acf(t),  int g^2 = Gamma(2nu+1) / (2 lambda)^(2nu+1)
# -----------------------------------------------------------------------------
gamma_acf <- function(t, nu, lambda) {
  len <- max(length(t), length(nu), length(lambda))
  a <- rep_len(nu + 1/2, len)
  x <- rep_len(lambda, len) * abs(rep_len(t, len))
  out <- rep(1, len)
  for (i in which(x > 0)) out[i] <- 2^(1 - a[i]) / gamma(a[i]) * x[i]^a[i] * besselK(x[i], nu = a[i])
  out
}

kernel_l2 <- function(nu, lambda) gamma(2 * nu + 1) / (2 * lambda)^(2 * nu + 1)

gamma_acov <- function(t, nu, lambda, sigma2) sigma2 * kernel_l2(nu, lambda) * gamma_acf(t, nu, lambda)

# int_0^inf g1 g2 for two gamma kernels (closed form)
kernel_cross <- function(nu1, lambda1, nu2, lambda2) {
  gamma(nu1 + nu2 + 1) / (lambda1 + lambda2)^(nu1 + nu2 + 1)
}

# -----------------------------------------------------------------------------
# 3. Hybrid scheme (Bennedsen, Lunde & Pakkanen 2017)
#    W:     (N + nT) x (kappa + 1) matrix of hybrid-scheme Gaussian variables
#    sigma: volatility at grid points -N/n, ..., T, length N + nT + 1
#    sigma_alignment = "corrected": the lower sum pairs sigma[j] with W[j, ]
#                                   (as the upper sum does)
#                    = "bss_package": reproduces BSS::gammaKernelBSS exactly,
#                                   where the lower sum takes sigma in reverse order
# -----------------------------------------------------------------------------
hybrid_path_from_W <- function(W, sigma, n, nu, lambda, kappa, N, nT,
                               sigma_alignment = c("corrected", "bss_package")) {
  sigma_alignment <- match.arg(sigma_alignment)
  k_upper <- (kappa + 1):N
  b_star <- ((k_upper^(nu + 1) - (k_upper - 1)^(nu + 1)) / (nu + 1))^(1 / nu)
  g_b <- c(rep(0, kappa), (b_star / n)^nu * exp(-lambda * b_star / n))
  L_g <- exp(-lambda * (1:kappa) / n)
  upper <- convolve(g_b, rev(head(sigma, -1) * W[, 1]), type = "open")[N:(N + nT)]
  i <- 1:(nT + 1)
  lower <- numeric(nT + 1)
  for (k in 1:kappa) {
    rows <- i + N - k
    sig <- if (sigma_alignment == "corrected") sigma[rows] else sigma[i + N - kappa + k - 1]
    lower <- lower + L_g[k] * sig * W[cbind(rows, k + 1)]
  }
  lower + upper
}

simulate_bss <- function(n, T, nu, lambda, sigma = NULL, N = NULL, kappa = 3, mu = 0,
                         sigma_alignment = "corrected") {
  nT <- n * T
  if (is.null(N)) N <- 100 * n
  if (is.null(sigma)) sigma <- rep(1, N + nT + 1)
  stopifnot(length(sigma) == N + nT + 1)
  Sigma_W <- BSS::hybridSchemeCovarianceMatrix(kappa, n, nu)
  W <- MASS::mvrnorm(N + nT, mu = rep(0, kappa + 1), Sigma = Sigma_W)
  mu + hybrid_path_from_W(W, sigma, n, nu, lambda, kappa, N, nT, sigma_alignment)
}

# Cross-covariance between the hybrid-scheme vectors of two Brownian motions
# with correlation rho.
#   "paper": rho * sd1 sd2'  (as in the paper and the drafts; approximate)
#   "exact": rho * int_0^{1/n} (k/n - u)^nu1 (l/n - u)^nu2 du
hybrid_cross_covariance <- function(kappa, n, nu1, nu2, rho, method = c("paper", "exact")) {
  method <- match.arg(method)
  S1 <- BSS::hybridSchemeCovarianceMatrix(kappa, n, nu1)
  S2 <- BSS::hybridSchemeCovarianceMatrix(kappa, n, nu2)
  if (method == "paper") return(rho * sqrt(diag(S1)) %*% t(sqrt(diag(S2))))
  C <- matrix(0, kappa + 1, kappa + 1)
  for (k in 0:kappa) for (l in 0:kappa) {
    f <- function(v) (if (k == 0) 1 + 0 * v else (k - v)^nu1) * (if (l == 0) 1 + 0 * v else (l - v)^nu2)
    C[k + 1, l + 1] <- integrate(f, 0, 1, rel.tol = 1e-10)$value /
      n^((k > 0) * nu1 + (l > 0) * nu2 + 1)
  }
  rho * C
}

simulate_bivariate_bss <- function(n, T, nu1, lambda1, nu2, lambda2, rho,
                                   sigma1 = NULL, sigma2 = NULL, N = NULL, kappa = 3,
                                   mu = c(0, 0), cross_covariance = "paper",
                                   sigma_alignment = "corrected") {
  nT <- n * T
  if (is.null(N)) N <- 100 * n
  if (is.null(sigma1)) sigma1 <- rep(1, N + nT + 1)
  if (is.null(sigma2)) sigma2 <- rep(1, N + nT + 1)
  S1 <- BSS::hybridSchemeCovarianceMatrix(kappa, n, nu1)
  S2 <- BSS::hybridSchemeCovarianceMatrix(kappa, n, nu2)
  C <- hybrid_cross_covariance(kappa, n, nu1, nu2, rho, cross_covariance)
  Sigma_W <- rbind(cbind(S1, C), cbind(t(C), S2))
  W <- MASS::mvrnorm(N + nT, mu = rep(0, 2 * (kappa + 1)), Sigma = Sigma_W)
  W1 <- W[, 1:(kappa + 1)]
  W2 <- W[, (kappa + 2):(2 * kappa + 2)]
  list(Y1 = mu[1] + hybrid_path_from_W(W1, sigma1, n, nu1, lambda1, kappa, N, nT, sigma_alignment),
       Y2 = mu[2] + hybrid_path_from_W(W2, sigma2, n, nu2, lambda2, kappa, N, nT, sigma_alignment))
}

# -----------------------------------------------------------------------------
# 4. GMM
#    G_bar(theta) = ( mean(X) - mu,
#                     gamma_hat(h; mu) - R(h/n),  h in H )
#    gamma_hat(h; mu) = (1/N) sum_{t=1}^{N-h} (X_t - mu)(X_{t+h} - mu)
#    The sums over t do not depend on theta, so they are precomputed once.
#    variance = "model":  R uses E[sigma^2] = sigma2 (simulation studies)
#    variance = "sample": R = var(X) * acf      (real data, E[sigma^2] unknown)
# -----------------------------------------------------------------------------
gmm_moment_setup <- function(Y, lags) {
  lags <- sort(unique(lags))
  N <- length(Y)
  list(N = N, lags = lags, mean = mean(Y), var = var(Y),
       S = sapply(lags, function(h) sum(Y[1:(N - h)] * Y[(1 + h):N])),
       A = sapply(lags, function(h) sum(Y[1:(N - h)])),
       B = sapply(lags, function(h) sum(Y[(1 + h):N])),
       cnt = N - lags)
}

gmm_moments <- function(theta, setup, n, variance = "model", sigma2 = NULL) {
  mu <- theta[1]; nu <- theta[2]; lambda <- theta[3]
  emp <- (setup$S - mu * (setup$A + setup$B) + mu^2 * setup$cnt) / setup$N
  model <- if (variance == "model") gamma_acov(setup$lags / n, nu, lambda, sigma2)
           else setup$var * gamma_acf(setup$lags / n, nu, lambda)
  c(setup$mean - mu, emp - model)
}

gmm_objective <- function(theta, setup, n, variance = "model", sigma2 = NULL, W = NULL) {
  g <- gmm_moments(theta, setup, n, variance, sigma2)
  if (is.null(W)) sum(g^2) else as.numeric(t(g) %*% W %*% g)
}

default_bounds <- list(lower = c(-10, -0.499, 0.001), upper = c(10, 0.499, 100))

fit_gmm_de <- function(Y, n, lags, variance = "model", sigma2 = NULL,
                       lower = default_bounds$lower, upper = default_bounds$upper,
                       itermax = 50, W = NULL) {
  setup <- gmm_moment_setup(Y, lags)
  opt <- DEoptim::DEoptim(gmm_objective, lower = lower, upper = upper,
                          control = DEoptim::DEoptim.control(trace = FALSE, itermax = itermax),
                          setup = setup, n = n, variance = variance, sigma2 = sigma2, W = W)
  th <- as.numeric(opt$optim$bestmem)
  list(mu = th[1], nu = th[2], lambda = th[3], value = opt$optim$bestval)
}

# Long-run covariance of G(X_i, theta):
#   S = Omega_0 + sum_{j=1}^m w_j (Omega_j + Omega_j')
#   weights = "bartlett":  w_j = 1 - j/(m+1)  (Newey-West; always positive semi-definite)
#   weights = "truncated": w_j = 1            (formula in the paper; can be indefinite)
hac_matrix <- function(Y, theta, n, lags, variance = "model", sigma2 = NULL, m = NULL,
                       weights = c("bartlett", "truncated")) {
  weights <- match.arg(weights)
  lags <- sort(unique(lags))
  N <- length(Y); hmax <- max(lags)
  if (is.null(m)) m <- floor(N^(1/5))
  c0 <- Y - theta[1]
  idx <- 1:(N - hmax)
  model <- if (variance == "model") gamma_acov(lags / n, theta[2], theta[3], sigma2)
           else var(Y) * gamma_acf(lags / n, theta[2], theta[3])
  G <- cbind(c0[idx], sapply(seq_along(lags), function(j) c0[idx] * c0[idx + lags[j]] - model[j]))
  S <- crossprod(G) / N
  for (j in seq_len(m)) {
    Om <- crossprod(G[(j + 1):nrow(G), , drop = FALSE], G[1:(nrow(G) - j), , drop = FALSE]) / N
    w <- if (weights == "bartlett") 1 - j / (m + 1) else 1
    S <- S + w * (Om + t(Om))
  }
  S
}

fit_gmm_two_step <- function(Y, n, lags, variance = "model", sigma2 = NULL, m = NULL,
                             weights = "bartlett", tol = 1e-4, max_iter = 10, itermax = 50,
                             lower = default_bounds$lower, upper = default_bounds$upper) {
  first <- fit_gmm_de(Y, n, lags, variance, sigma2, lower, upper, itermax)
  theta <- c(first$mu, first$nu, first$lambda)
  setup <- gmm_moment_setup(Y, lags)
  for (it in seq_len(max_iter)) {
    W <- solve(hac_matrix(Y, theta, n, lags, variance, sigma2, m, weights))
    opt <- optim(theta, gmm_objective, method = "L-BFGS-B", lower = lower, upper = upper,
                 setup = setup, n = n, variance = variance, sigma2 = sigma2, W = W)
    done <- max(abs(opt$par - theta)) < tol
    theta <- opt$par
    if (done) break
  }
  list(mu = theta[1], nu = theta[2], lambda = theta[3], iterations = it)
}

# -----------------------------------------------------------------------------
# 5. Correlation estimator (bivariate model)
#    rho_hat = cov(Y1, Y2) / ( E[sigma1 sigma2] * int g1 g2 )
#    e_sigma12 = NULL uses the plug-in sqrt(E[sigma1^2] E[sigma2^2]) with
#    E[sigma_j^2] = var(Y_j) / int g_j^2.
# -----------------------------------------------------------------------------
rho_hat <- function(Y1, Y2, nu1, lambda1, nu2, lambda2, e_sigma12 = NULL) {
  if (is.null(e_sigma12)) {
    e_sigma12 <- sqrt(var(Y1) / kernel_l2(nu1, lambda1) * var(Y2) / kernel_l2(nu2, lambda2))
  }
  cov(Y1, Y2) / (e_sigma12 * kernel_cross(nu1, lambda1, nu2, lambda2))
}

# -----------------------------------------------------------------------------
# 6. Real data: remainder of a periodic STL decomposition
# -----------------------------------------------------------------------------
stl_remainder <- function(x, frequency = 261, start = 1) {
  x <- approx(seq_along(x), x, xout = seq_along(x))$y   # linear interpolation of NAs
  dec <- stl(ts(x, start = start, frequency = frequency), s.window = "periodic")
  as.numeric(dec$time.series[, "remainder"])
}
