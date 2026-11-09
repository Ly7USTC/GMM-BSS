"""Fit the R-simulated paths in Python and compare with the R estimates."""
from pathlib import Path
import numpy as np, pandas as pd
import gmmbss as gb

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
paths = pd.read_csv(FIX / "compare_paths.csv").to_numpy()
r = pd.read_csv(FIX / "compare_estimates_r.csv")
lags = [*range(1, 11), 15, 20, 25, 30]
py = np.array([gb.fit_gmm_de(paths[:, j], 100, lags, sigma2=0.011, maxiter=300, seed=j).theta
               for j in range(paths.shape[1])])
out = pd.DataFrame({"mu_R": r.mu, "mu_py": py[:, 0], "nu_R": r.nu, "nu_py": py[:, 1],
                    "lambda_R": r["lambda"], "lambda_py": py[:, 2]})
print(out.round(4).to_string(index=False))
print("max |diff|  mu %.1e  nu %.1e  lambda %.1e" % tuple(np.abs(py - r[["mu", "nu", "lambda"]].to_numpy()).max(0)))
