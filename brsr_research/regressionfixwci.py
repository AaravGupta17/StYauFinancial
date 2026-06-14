"""
BRSR DiD regressions — bid-ask spread (Corwin-Schultz)
Improved version:
  - Two-way clustered SEs (firm + date) instead of HC3
  - Model 3 now has firm AND year-month fixed effects
  - Negative/zero CS spreads are floored (not silently dropped) before logging
  - Sample sizes and dropped-row diagnostics are printed
  - Correct degrees of freedom reported for the two-way FE model
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

OUTPUT_FOLDER = r"./output"

print("Loading spreads...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/spreads.parquet")

# ── Trim pre-Oct-2021 rows ────────────────────────────────────────────────────
n_before = len(df)
df = df[df["DATE"] >= "2021-10-01"].copy()
print(f"Dropped {n_before - len(df):,} rows before 2021-10-01. "
      f"Remaining: {len(df):,}")

# ── DiD variables ──────────────────────────────────────────────────────────
df["POST1"] = (df["DATE"] >= "2022-04-01").astype(int)  # BRSR announcement
df["POST2"] = (df["DATE"] >= "2023-04-01").astype(int)  # BRSR enforcement
df["DiD1"] = df["TREATED"] * df["POST1"]  # announcement effect
df["DiD2"] = df["TREATED"] * df["POST2"]  # enforcement effect

# ── Handle non-positive CS spreads BEFORE logging ────────────────────────────
# Corwin-Schultz spreads can come out negative/zero by construction; dropping
# them outright can bias the sample toward more volatile/illiquid firm-days.
n_nonpos = (df["CS_SPREAD"] <= 0).sum()
print(f"CS_SPREAD <= 0: {n_nonpos:,} rows ({n_nonpos / len(df):.2%}) — flooring at 0")
df["CS_SPREAD_ADJ"] = df["CS_SPREAD"].clip(lower=0)
df["LOG_SPREAD"] = np.log1p(df["CS_SPREAD_ADJ"])  # log(1 + x), handles zeros

# (no rows dropped here — log1p is defined at 0)

# ── Time variable for fixed effects ──────────────────────────────────────────
df["YM"] = df["DATE"].dt.to_period("M").astype(str)

# ── Cluster IDs ────────────────────────────────────────────────────────────
df["FIRM_ID"] = df["SYMBOL"].astype("category").cat.codes
df["DATE_ID"] = df["DATE"].astype("category").cat.codes

CLUSTER_KW = lambda d: {"groups": d[["FIRM_ID", "DATE_ID"]]}

# ── Model 1: Simple DiD — announcement ───────────────────────────────────────
print("\nModel 1: Announcement effect (Apr 2022)")
m1 = smf.ols("LOG_SPREAD ~ TREATED + POST1 + DiD1", data=df).fit(
    cov_type="cluster", cov_kwds=CLUSTER_KW(df)
)
print(m1.summary().tables[1])
print(f"N = {int(m1.nobs):,}")

# ── Model 2: Simple DiD — enforcement ────────────────────────────────────────
print("\nModel 2: Enforcement effect (Apr 2023)")
m2 = smf.ols("LOG_SPREAD ~ TREATED + POST2 + DiD2", data=df).fit(
    cov_type="cluster", cov_kwds=CLUSTER_KW(df)
)
print(m2.summary().tables[1])
print(f"N = {int(m2.nobs):,}")

# ── Model 3: Both events + firm + year-month fixed effects ───────────────────
print("\nModel 3: Both events + firm + time fixed effects (main specification)")

vars_to_demean = ["LOG_SPREAD", "DiD1", "DiD2", "POST1", "POST2"]
df_within = df[vars_to_demean + ["SYMBOL", "YM", "FIRM_ID", "DATE_ID"]].copy()

# Two-way demean: firm effects, then time effects (iterative within-transform)
firm_means = df_within.groupby("SYMBOL")[vars_to_demean].transform("mean")
df_within[vars_to_demean] = df_within[vars_to_demean] - firm_means

time_means = df_within.groupby("YM")[vars_to_demean].transform("mean")
df_within[vars_to_demean] = df_within[vars_to_demean] - time_means

# Add back grand mean (keeps intercept interpretable, doesn't change other coefs)
df_within[vars_to_demean] += df[vars_to_demean].mean()

m3 = smf.ols("LOG_SPREAD ~ DiD1 + DiD2 + POST1 + POST2", data=df_within).fit(
    cov_type="cluster", cov_kwds=CLUSTER_KW(df_within)
)

# ── Correct degrees of freedom for two-way FE absorption ─────────────────────
n_firms = df["SYMBOL"].nunique()
n_periods = df["YM"].nunique()
k = len(m3.params)
correct_df_resid = len(df_within) - k - (n_firms - 1) - (n_periods - 1)
print(f"statsmodels df_resid = {m3.df_resid:.0f}  "
      f"(correct df_resid for 2-way FE ~ {correct_df_resid:.0f})")
print("Note: coefficients & clustered SEs from statsmodels are valid; "
      "df_resid above is informational only (affects e.g. F-tests/R2 adjustment).")

params = m3.params[["DiD1", "DiD2", "POST1", "POST2"]]
pvals = m3.pvalues[["DiD1", "DiD2", "POST1", "POST2"]]
conf = m3.conf_int().loc[["DiD1", "DiD2", "POST1", "POST2"]]

summary_tbl = pd.DataFrame({
    "Coef": params.round(4),
    "SE": m3.bse[["DiD1", "DiD2", "POST1", "POST2"]].round(4),
    "P-value": pvals.round(4),
    "CI_low": conf[0].round(4),
    "CI_high": conf[1].round(4),
})
print(summary_tbl)
print(f"\nR-squared (uncorrected): {m3.rsquared:.4f}")
print(f"N: {len(df_within):,}")
print(f"Firms: {n_firms:,} | Months: {n_periods}")

summary_tbl.to_csv(f"{OUTPUT_FOLDER}/regression_results.csv")
print(f"\nSaved regression_results.csv")