"""Statistical and property tests that do not depend on R."""

import numpy as np
import pytest
from scipy import integrate

import gmmbss as gb

LAGS = [*range(1, 11), 15, 20, 25, 30]


# ---------------------------------------------------------------- kernels
def test_acf_is_one_at_zero_and_decreasing():
    t = np.linspace(0, 2, 50)
    r = gb.gamma_acf(t, -0.2, 3)
    assert r[0] == 1 and np.all(np.diff(r) < 0) and r[-1] > 0


@pytest.mark.parametrize("nu, lam", [(-0.4, 1), (-0.2, 10), (0.3, 2)])
def test_acf_equals_normalised_kernel_convolution(nu, lam):
    g = lambda x: x**nu * np.exp(-lam * x)
    for t in (0.05, 0.3):
        num = integrate.quad(lambda x: g(x) * g(x + t), 0, np.inf, limit=200)[0]
        assert gb.gamma_acf(t, nu, lam) == pytest.approx(num / gb.kernel_l2(nu, lam), rel=1e-6)


def test_kernel_cross_closed_form():
    num = integrate.quad(lambda x: x**-0.5 * np.exp(-2.5 * x), 0, np.inf)[0]
    assert gb.kernel_cross(-0.4, 1, -0.1, 1.5) == pytest.approx(num, rel=1e-8)


def test_exact_cross_covariance_reduces_to_sigma():
    S = gb.hybrid_covariance(4, 50, -0.3)
    C = gb.hybrid_cross_covariance(4, 50, -0.3, -0.3, 0.7, "exact")
    np.testing.assert_allclose(C, 0.7 * S, rtol=1e-8)


def test_exact_joint_covariance_is_psd():
    S1, S2 = gb.hybrid_covariance(5, 100, -0.4), gb.hybrid_covariance(5, 100, -0.1)
    C = gb.hybrid_cross_covariance(5, 100, -0.4, -0.1, 0.9, "exact")
    assert np.linalg.eigvalsh(np.block([[S1, C], [C.T, S2]])).min() > -1e-14


# ---------------------------------------------------------------- volatility
def test_gamma_ou_exact_has_gamma_marginal():
    x = gb.gamma_ou(10, 10, 100, 2000, 0.01, method="exact", rng=1)
    assert x.mean() == pytest.approx(0.1, rel=0.03)
    assert x.var() == pytest.approx(0.001, rel=0.08)
    assert len(x) == 200001


def test_gamma_ou_euler_variance_inflation():
    x = gb.gamma_ou(10, 10, 100, 2000, 0.01, method="euler", rng=2)
    assert x.mean() == pytest.approx(0.1, rel=0.03)
    assert x.var() == pytest.approx(0.001 / (1 - 0.1 / 2), rel=0.08)


def test_gamma_ou_handles_zero_and_one_jump():
    # the draft crashed when exactly one jump occurred
    for seed in range(30):
        x = gb.gamma_ou(1, 0.05, 1, 10, 0.1, y0=0.0, rng=seed)
        assert np.all(np.isfinite(x)) and len(x) == 101


# ---------------------------------------------------------------- hybrid scheme
def test_simulated_path_has_theoretical_acf():
    n, nu, lam = 100, -0.2, 5
    Y = gb.simulate_bss(n, 1000, nu, lam, N=5000, rng=3)
    Yc = Y - Y.mean()
    emp = np.array([Yc[:-h] @ Yc[h:] / (Yc @ Yc) for h in (1, 5, 20, 50)])
    np.testing.assert_allclose(emp, gb.gamma_acf(np.array([1, 5, 20, 50]) / n, nu, lam), atol=0.03)
    assert Y.var() == pytest.approx(gb.kernel_l2(nu, lam), rel=0.1)


def test_alignment_only_matters_for_varying_sigma():
    rng = np.random.default_rng(4)
    N, nT, k = 300, 200, 3
    W = rng.standard_normal((N + nT, k + 1))
    ones = np.ones(N + nT + 1)
    a = gb.hybrid_path_from_W(W, ones, 50, -0.3, 1, k, N, nT, "corrected")
    b = gb.hybrid_path_from_W(W, ones, 50, -0.3, 1, k, N, nT, "bss_package")
    np.testing.assert_allclose(a, b)
    sig = 1 + rng.random(N + nT + 1)
    assert not np.allclose(gb.hybrid_path_from_W(W, sig, 50, -0.3, 1, k, N, nT, "corrected"),
                           gb.hybrid_path_from_W(W, sig, 50, -0.3, 1, k, N, nT, "bss_package"))


def test_bivariate_correlation():
    Y1, Y2 = gb.simulate_bivariate_bss(100, 500, -0.4, 1, -0.1, 1.5, 0.5, N=3000, kappa=5,
                                       cross_covariance="exact", rng=5)
    r = gb.rho_hat(Y1, Y2, -0.4, 1, -0.1, 1.5, e_sigma12=1.0)
    assert r == pytest.approx(0.5, abs=0.06)


def test_wrong_sigma_length_raises():
    with pytest.raises(ValueError):
        gb.simulate_bss(10, 5, -0.2, 1, sigma=np.ones(10), N=100)


# ---------------------------------------------------------------- GMM
def test_fast_moments_equal_naive_loop():
    rng = np.random.default_rng(6)
    Y = 2 + rng.standard_normal(3000).cumsum() * 0.01
    setup = gb.MomentSetup(Y, LAGS)
    for mu in (1.5, 2.0, 2.3):
        c = Y - mu
        naive = np.array([c[:-h] @ c[h:] / len(Y) for h in LAGS])
        np.testing.assert_allclose(setup.empirical_acov(mu), naive, rtol=1e-11)


def test_duplicate_lags_are_removed_with_warning():
    with pytest.warns(UserWarning):
        s = gb.MomentSetup(np.random.default_rng(0).standard_normal(500), [1, 2, 5, 5, 10])
    assert list(s.lags) == [1, 2, 5, 10]


def test_moments_vanish_at_truth_on_average():
    # population moments: use the theoretical acov as the "data" acov
    setup = gb.MomentSetup(np.zeros(100), LAGS)
    setup.mean, setup.var = 2.0, gb.kernel_l2(-0.2, 10) * 0.011
    setup.empirical_acov = lambda mu: gb.gamma_acov(np.array(LAGS) / 100, -0.2, 10, 0.011)
    np.testing.assert_allclose(gb.moments((2.0, -0.2, 10), setup, 100, "model", 0.011), 0, atol=1e-15)


@pytest.fixture(scope="module")
def long_path():
    rng = np.random.default_rng(7)
    n, T, N = 100, 1000, 10000
    sigma = gb.gamma_ou(10, 10, 100, (N + n * T) / n, 1 / n, method="exact", rng=rng)
    return 2 + gb.simulate_bss(n, T, -0.2, 1, sigma=sigma, N=N, rng=rng)


def test_de_recovers_parameters(long_path):
    sigma2 = gb.gamma_ou_second_moment(10, 100)
    fit = gb.fit_gmm_de(long_path, 100, LAGS, sigma2=sigma2, seed=8, maxiter=200, polish=True)
    assert fit.mu == pytest.approx(2, abs=0.02)
    assert fit.nu == pytest.approx(-0.2, abs=0.03)
    assert fit.lam == pytest.approx(1, rel=0.15)


def test_two_step_recovers_parameters(long_path):
    sigma2 = gb.gamma_ou_second_moment(10, 100)
    fit = gb.fit_gmm_two_step(long_path, 100, LAGS, sigma2=sigma2, seed=9)
    assert fit.mu == pytest.approx(2, abs=0.02)
    assert fit.nu == pytest.approx(-0.2, abs=0.05)
    assert fit.lam == pytest.approx(1, rel=0.25)


def test_bartlett_hac_matrix_is_positive_definite(long_path):
    S = gb.hac_matrix(long_path[:20000], (2, -0.2, 1), 100, LAGS, "model", 0.011)
    np.testing.assert_allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > 0


def test_truncated_hac_matrix_can_be_indefinite(long_path):
    # documents why Bartlett weights are the default
    S = gb.hac_matrix(long_path[:20000], (2, -0.2, 1), 100, LAGS, "model", 0.011, weights="truncated")
    assert np.linalg.eigvalsh(S).min() < 0


def test_fit_series_on_simulated_data(long_path):
    fit = gb.fit_series(long_path, n=100, seed=10, maxiter=200, polish=True)
    assert fit.nu == pytest.approx(-0.2, abs=0.05)
    assert fit.lam == pytest.approx(1, rel=0.25)


# ---------------------------------------------------------------- application
def test_interpolate_missing():
    x = gb.interpolate_missing([1, np.nan, 3, np.nan, np.nan, 6])
    np.testing.assert_allclose(x, [1, 2, 3, 4, 5, 6])


def test_stl_remainder_removes_seasonality():
    t = np.arange(261 * 5)
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(len(t)) * 0.1
    x = 10 + 2 * np.sin(2 * np.pi * t / 261) + 0.001 * t + noise
    r = gb.stl_remainder(x, 261)
    s = np.sin(2 * np.pi * t / 261)
    assert abs(r @ s / (s @ s)) < 0.05 * 2          # < 5% of the seasonal amplitude 2 is left
    assert abs(np.polyfit(t, r, 1)[0]) < 0.05 * 0.001  # < 5% of the trend slope is left
    assert np.corrcoef(r, noise)[0, 1] > 0.8
