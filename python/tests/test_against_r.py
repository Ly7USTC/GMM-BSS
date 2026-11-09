"""Exact cross-checks of the Python code against the R reference implementation.

Fixtures are produced by tests/r/make_fixtures.R, which first checks the clean R
code against the BSS package and the original draft functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gmmbss as gb

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def load(name):
    return pd.read_csv(FIX / name)


@pytest.fixture(scope="module")
def hybrid_case():
    p = load("hybrid_params.csv").iloc[0]
    return dict(N=int(p.N), n=int(p.n), T=int(p["T"]), kappa=int(p.kappa), nu=p.nu, lam=p["lambda"],
                W=load("hybrid_W.csv").to_numpy(), sigma=load("hybrid_sigma.csv")["sigma"].to_numpy(),
                paths=load("hybrid_paths.csv"))


def test_hybrid_covariance_matches_bss_package():
    df = load("hybrid_covariance.csv")
    for (kappa, n, alpha), g in df.groupby(["kappa", "n", "alpha"]):
        S = gb.hybrid_covariance(int(kappa), int(n), alpha)
        np.testing.assert_allclose(S[g.i - 1, g.j - 1], g.value, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("alignment, column", [("bss_package", "bss_package"), ("corrected", "corrected")])
def test_hybrid_path_matches_r(hybrid_case, alignment, column):
    c = hybrid_case
    Y = gb.hybrid_path_from_W(c["W"], c["sigma"], c["n"], c["nu"], c["lam"], c["kappa"],
                              c["N"], c["n"] * c["T"], alignment)
    np.testing.assert_allclose(Y, c["paths"][column], rtol=1e-9, atol=1e-12)


def test_gaussian_core_matches_r(hybrid_case):
    c = hybrid_case
    Y = gb.hybrid_path_from_W(c["W"], np.ones_like(c["sigma"]), c["n"], c["nu"], c["lam"],
                              c["kappa"], c["N"], c["n"] * c["T"])
    np.testing.assert_allclose(Y, c["paths"]["core"], rtol=1e-9, atol=1e-12)


def test_bivariate_matches_draft(hybrid_case):
    c = hybrid_case
    Wb = load("bivariate_W.csv").to_numpy()
    sig2 = load("bivariate_sigma2.csv")["sigma2"].to_numpy()
    ref = load("bivariate_paths.csv")
    k, nT = c["kappa"], c["n"] * c["T"]
    Y1 = gb.hybrid_path_from_W(Wb[:, :k + 1], c["sigma"], c["n"], c["nu"], c["lam"], k, c["N"], nT, "bss_package")
    Y2 = gb.hybrid_path_from_W(Wb[:, k + 1:], sig2, c["n"], -0.1, 1.5, k, c["N"], nT, "bss_package")
    np.testing.assert_allclose(Y1, ref.Y1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(Y2, ref.Y2, rtol=1e-9, atol=1e-12)


def test_exact_cross_covariance_matches_r(hybrid_case):
    C = gb.hybrid_cross_covariance(hybrid_case["kappa"], hybrid_case["n"], hybrid_case["nu"], -0.1, 0.5, "exact")
    np.testing.assert_allclose(C, load("cross_covariance_exact.csv").to_numpy(), rtol=1e-6)


def test_acf_and_acov_match_r():
    df = load("acf.csv")
    np.testing.assert_allclose(gb.gamma_acf(df.t, df.nu, df["lambda"]), df.acf, rtol=1e-12)
    np.testing.assert_allclose(gb.gamma_acov(df.t, df.nu, df["lambda"], 0.011), df.acov, rtol=1e-12)


@pytest.fixture(scope="module")
def gmm_case():
    Y = load("gmm_Y.csv")["Y"].to_numpy()
    return Y, gb.MomentSetup(Y, [*range(1, 11), 15, 20, 25, 30]), load("gmm_objective.csv")


def test_objective_matches_r_and_drafts(gmm_case):
    _, setup, ref = gmm_case
    for row in ref.itertuples():
        theta = (row.mu, row.nu, row._3)
        assert gb.objective(theta, setup, 100, "model", 0.011) == pytest.approx(row.obj_model, rel=1e-10)
        assert gb.objective(theta, setup, 100, "sample") == pytest.approx(row.obj_sample, rel=1e-10)


@pytest.mark.parametrize("weights", ["bartlett", "truncated"])
def test_hac_matrix_matches_r(gmm_case, weights):
    Y, setup, _ = gmm_case
    S = gb.hac_matrix(Y, (2, -0.2, 10), 100, setup.lags, "model", 0.011, weights=weights)
    np.testing.assert_allclose(S, load(f"hac_matrix_{weights}.csv").to_numpy(), rtol=1e-9)


def test_rho_hat_matches_r():
    ref = load("rho.csv").iloc[0]
    Y = load("bivariate_paths.csv")
    assert gb.rho_hat(Y.Y1, Y.Y2, -0.3, 2, -0.1, 1.5, e_sigma12=0.01) == pytest.approx(ref.rho_known, rel=1e-10)
    assert gb.rho_hat(Y.Y1, Y.Y2, -0.3, 2, -0.1, 1.5) == pytest.approx(ref.rho_plugin, rel=1e-10)


def test_stl_remainder_matches_r():
    df = load("stl.csv")
    np.testing.assert_allclose(gb.stl_remainder(df.x.to_numpy(), 261), df.remainder, atol=1e-6)
