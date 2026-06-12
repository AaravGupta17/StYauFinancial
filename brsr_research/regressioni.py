import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

OUTPUT_FOLDER = r"./output"
print("Loading spreads...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/spreads.parquet")
# Trim COVID-affected period (India's second wave distorted microstructure)
df = df[df["DATE"] >= "2021-10-01"].copy()
print(f"After trimming pre-COVID rows: {len(df):,}")
# ── Create DiD variables ──────────────────────────────────────────────────────
# POST1 = after BRSR announcement (Apr 2022)
# POST2 = after BRSR enforcement  (Apr 2023)
df["POST1"] = (df["DATE"] >= "2022-04-01").astype(int)
df["POST2"] = (df["DATE"] >= "2023-04-01").astype(int)

# DiD interaction terms
df["DiD1"] = df["TREATED"] * df["POST1"]  # announcement effect
df["DiD2"] = df["TREATED"] * df["POST2"]  # enforcement effect

# Log spread (standard in microstructure — normalizes skewness)
df["LOG_SPREAD"] = np.log(df["CS_SPREAD"].replace(0, np.nan))
df = df.dropna(subset=["LOG_SPREAD"])

# Time fixed effects (year-month)
df["YM"] = df["DATE"].dt.to_period("M").astype(str)

# ── Model 1: Simple DiD — announcement ───────────────────────────────────────
print("\nModel 1: Announcement effect (Apr 2022)")
m1 = smf.ols(
    "LOG_SPREAD ~ TREATED + POST1 + DiD1",
    data=df
).fit(cov_type="HC3")
print(m1.summary().tables[1])

# ── Model 2: Simple DiD — enforcement ────────────────────────────────────────
print("\nModel 2: Enforcement effect (Apr 2023)")
m2 = smf.ols(
    "LOG_SPREAD ~ TREATED + POST2 + DiD2",
    data=df
).fit(cov_type="HC3")
print(m2.summary().tables[1])

# ── Model 3: Both events + firm fixed effects (memory efficient) ──────────────
print("\nModel 3: Both events + firm fixed effects (main specification)")

# Demean by firm (within-transformation — equivalent to firm FE)
vars_to_demean = ["LOG_SPREAD", "DiD1", "DiD2", "POST1", "POST2"]
df_demean = df[vars_to_demean + ["SYMBOL"]].copy()
firm_means = df_demean.groupby("SYMBOL")[vars_to_demean].transform("mean")
df_within  = df_demean[vars_to_demean] - firm_means

m3 = smf.ols(
    "LOG_SPREAD ~ DiD1 + DiD2 + POST1 + POST2",
    data=df_within
).fit(cov_type="HC3")

params = m3.params[["DiD1", "DiD2", "POST1", "POST2"]]
pvals  = m3.pvalues[["DiD1", "DiD2", "POST1", "POST2"]]
conf   = m3.conf_int().loc[["DiD1", "DiD2", "POST1", "POST2"]]

print(pd.DataFrame({
    "Coef"   : params.round(4),
    "P-value": pvals.round(4),
    "CI_low" : conf[0].round(4),
    "CI_high": conf[1].round(4)
}))
print(f"\nR-squared : {m3.rsquared:.4f}")
print(f"N         : {len(df):,}")

results = pd.DataFrame({
    "Coef": m3.params[["DiD1", "DiD2", "POST1", "POST2"]],
    "SE"  : m3.bse[["DiD1", "DiD2", "POST1", "POST2"]],
    "P"   : m3.pvalues[["DiD1", "DiD2", "POST1", "POST2"]],
})
results.to_csv(f"{OUTPUT_FOLDER}/regression_results.csv")
print(f"\nSaved regression_results.csv")