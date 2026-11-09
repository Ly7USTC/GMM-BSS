"""Fit the gamma-kernel BSS model to ten years of daily VIX.

Download VIXCLS.csv from FRED (https://fred.stlouisfed.org/series/VIXCLS) and run
    python examples/vix_application.py path/to/VIXCLS.csv
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

import gmmbss as gb

df = pd.read_csv(sys.argv[1], na_values=".")
date_col, value_col = df.columns[0], df.columns[1]
df[date_col] = pd.to_datetime(df[date_col])
vix = df[(df[date_col] >= "2010-01-01") & (df[date_col] <= "2019-12-31")][value_col].to_numpy(float)

remainder = gb.stl_remainder(vix, period=261)         # 261 trading days per year
fit = gb.fit_series(remainder, n=261, seed=1)         # lags measured in years
print(f"mu = {fit.mu:.4f}, nu = {fit.nu:.4f}, lambda = {fit.lam:.4f}")

ax = gb.plot_acf_fit(remainder, fit.nu, fit.lam, n=261, max_lag=50)
ax.set_title("VIX remainder: empirical and fitted acf")
plt.savefig("vix_acf.png", dpi=130, bbox_inches="tight")
