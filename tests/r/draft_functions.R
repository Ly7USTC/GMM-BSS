# Functions copied from the original drafts, used only to check that the clean
# code reproduces them. Not part of the package.

multigammaKernelBSS_mod_2 <- function(N, n, T, kappa = 3, alpha_1=-0.2, lambda_1=1, alpha_2=-0.3, lambda_2=1.5, sigma_1 = rep(1, N + n*T + 1), sigma_2 = rep(1, N + n*T + 1),rho=0) {
  Y_lower_1 <- numeric(n*T + 1)
  Y_lower_2 <- numeric(n*T + 1)
  g_1 <- function(x) x^alpha_1 * exp(-lambda_1*x)
  g_2 <- function(x) x^alpha_2 * exp(-lambda_2*x)
  k_lower <- 1:kappa
  k_upper <- (kappa + 1):N
  b_star_1 <- function(k) ((k^(alpha_1 + 1) - (k-1)^(alpha_1 + 1))/(alpha_1 + 1))^(1/alpha_1)
  b_star_2 <- function(k) ((k^(alpha_2 + 1) - (k-1)^(alpha_2 + 1))/(alpha_2 + 1))^(1/alpha_2)
  L_g_1 <- exp(-lambda_1*k_lower/n)
  L_g_2 <- exp(-lambda_2*k_lower/n)
  g_b_star_1 <- c(rep(0, kappa), g_1(b_star_1(k_upper)/n))
  g_b_star_2 <- c(rep(0, kappa), g_2(b_star_2(k_upper)/n))
  Sigma_W_1 <- hybridSchemeCovarianceMatrix(kappa, n, alpha_1)
  Sigma_W_2 <- hybridSchemeCovarianceMatrix(kappa, n, alpha_2)
  sd_W_1 <- sqrt(diag(Sigma_W_1))
  sd_W_2 <- sqrt(diag(Sigma_W_2))
  Cross_cov <- rho*sd_W_1%*%t(sd_W_2)
  Sigma_W <- rbind(cbind(Sigma_W_1,Cross_cov),cbind(t(Cross_cov),Sigma_W_2))
  W <- MASS::mvrnorm(N + n*T, mu = rep(0,2*(kappa + 1)), Sigma = Sigma_W)
  for (i in 1:(n*T + 1)) {
    Y_lower_1[i] <- sum( L_g_1  * sigma_1[(i + N - kappa):(i+N-1)] * diag(W[(i + N - kappa):(i+N-1), 2:(kappa + 1)][kappa:1, 1:kappa]))
    Y_lower_2[i] <- sum( L_g_2  * sigma_2[(i + N - kappa):(i+N-1)] * diag(W[(i + N - kappa):(i+N-1), (kappa+3):(2*kappa + 2)][kappa:1, 1:kappa]))
  }
  Y_1 <- Y_lower_1 + convolve( g_b_star_1, rev(head(sigma_1,-1) * W[,1]), type = 'open')[N:(N+n*T)]
  Y_2 <- Y_lower_2 + convolve( g_b_star_2, rev(head(sigma_2,-1) * W[,kappa+2]), type = 'open')[N:(N+n*T)]
  list(bss_1=Y_1,bss_2=Y_2)
}

.draft_true_acf <- function(lambda, t) {
  tilde_K <- function(nu, x) 2^{1-nu}/gamma(nu)*x^nu*besselK(x, nu)   # Bessel::BesselK in the draft
  tilde_K(lambda[1]+1/2, lambda[2]*t)
}

# criterionfunction from gmm_DEoptim (simulation studies, E[sigma^2] = 0.011)
draft_criterion_model <- function(theta, Y, n, select_lags) {
  var_BSS <- function(lambda) 0.011*gamma(2*lambda[1]+1)/(2*lambda[2])^(2*lambda[1]+1)
  true_acovf <- function(lambda, t) var_BSS(lambda)*.draft_true_acf(lambda, t)
  cen_Y <- ts(Y - theta[1]); g_1 <- mean(Y) - theta[1]
  emp <- sapply(select_lags, function(h) sum(cen_Y*stats::lag(cen_Y, -h))/length(cen_Y))
  g_s <- c(g_1, emp - sapply(select_lags/n, true_acovf, lambda = theta[2:3]))
  sum(g_s^2)
}

# criterionfunction from gmm_DEoptim_CO / gmm_DEoptim_VIX (real data, var(Y))
draft_criterion_sample <- function(theta, Y, n, select_lags) {
  true_acovf <- function(lambda, t) var(Y)*.draft_true_acf(lambda, t)
  cen_Y <- ts(Y - theta[1]); g_1 <- mean(Y) - theta[1]
  emp <- sapply(select_lags, function(h) sum(cen_Y*stats::lag(cen_Y, -h))/length(cen_Y))
  g_s <- c(g_1, emp - sapply(select_lags/n, true_acovf, lambda = theta[2:3]))
  sum(g_s^2)
}
