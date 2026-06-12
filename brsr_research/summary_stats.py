"""
summary_stats.py
Generates all descriptive statistics tables needed for the paper.
  Table 1: Sample composition
  Table 2: Variable means by group and period
  Table 3: Pre vs post comparison with t-tests
  Table 4: Correlation matrix of liquidity measures
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "./output"

print("Loading data...")
liq      = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")
liq      = liq.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")

liq["LOG_CS"]     = np.log(liq["CS_SPREAD"].replace(0, np.nan))
liq["LOG_AMIHUD"] = np.log(liq["AMIHUD"].replace(0, np.nan))
liq["LOG_ROLL"]   = np.log(liq["ROLL_SPREAD"].replace(0, np.nan))
liq["PERIOD"]     = pd.cut(
    liq["DATE"],
    bins=[pd.Timestamp("2021-01-01"),
          pd.Timestamp("2022-03-31"),
          pd.Timestamp("2023-03-31"),
          pd.Timestamp("2025-01-01")],
    labels=["Pre (Oct21-Mar22)","Post1 (Apr22-Mar23)","Post2 (Apr23-Mar24)"]
)

T = liq["TREATED"] == 1
C = liq["TREATED"] == 0


# ── Table 1: Sample composition ───────────────────────────────────────────────
print("\n" + "="*65)
print("TABLE 1: SAMPLE COMPOSITION")
print("="*65)

total_firms    = liq["SYMBOL"].nunique()
treated_firms  = liq[T]["SYMBOL"].nunique()
control_firms  = liq[C]["SYMBOL"].nunique()
total_obs      = len(liq)
date_min       = liq["DATE"].min().date()
date_max       = liq["DATE"].max().date()

print(f"  Total firm-day observations : {total_obs:>10,}")
print(f"  Total unique firms          : {total_firms:>10,}")
print(f"  Treatment firms (rank 1-1000) : {treated_firms:>8,}")
print(f"  Control firms (rank 1001-1500): {control_firms:>8,}")
print(f"  Date range                  : {date_min} to {date_max}")
print(f"  Trading days                : {liq['DATE'].nunique():>10,}")

print(f"\n  By period:")
print(f"  {'Period':<25} {'Obs':>10} {'Treated':>10} {'Control':>10}")
print(f"  {'─'*55}")
for period in ["Pre (Oct21-Mar22)","Post1 (Apr22-Mar23)","Post2 (Apr23-Mar24)"]:
    sub = liq[liq["PERIOD"] == period]
    print(f"  {period:<25} {len(sub):>10,} "
          f"{len(sub[sub['TREATED']==1]):>10,} "
          f"{len(sub[sub['TREATED']==0]):>10,}")


# ── Table 2: Variable means by group and period ───────────────────────────────
print("\n" + "="*65)
print("TABLE 2: DESCRIPTIVE STATISTICS BY GROUP AND PERIOD")
print("="*65)

vars_labels = [
    ("CS_SPREAD",  "CS Spread"),
    ("AMIHUD",     "Amihud ILLIQ"),
    ("ROLL_SPREAD","Roll Spread"),
    ("TURNOVER",   "Turnover"),
    ("VOLATILITY", "Volatility (30d)"),
    ("LOG_VOLUME", "Log Volume"),
]

for var, label in vars_labels:
    if var not in liq.columns:
        continue
    print(f"\n  {label}")
    print(f"  {'Period':<25} {'Treated':>10} {'Control':>10} {'Diff':>10} {'t-stat':>8}")
    print(f"  {'─'*60}")
    for period in ["Pre (Oct21-Mar22)","Post1 (Apr22-Mar23)","Post2 (Apr23-Mar24)"]:
        sub = liq[liq["PERIOD"] == period]
        t_vals = sub[sub["TREATED"]==1][var].dropna()
        c_vals = sub[sub["TREATED"]==0][var].dropna()
        if len(t_vals) < 10 or len(c_vals) < 10:
            continue
        t_mean = t_vals.mean()
        c_mean = c_vals.mean()
        diff   = t_mean - c_mean
        tstat, pval = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {period:<25} {t_mean:>10.4f} {c_mean:>10.4f} "
              f"{diff:>10.4f} {tstat:>8.2f}{sig}")


# ── Table 3: Pre vs post within treated ───────────────────────────────────────
print("\n" + "="*65)
print("TABLE 3: PRE VS POST COMPARISON WITHIN TREATED FIRMS")
print("="*65)
print(f"\n  {'Variable':<20} {'Pre':>8} {'Post1':>8} {'Post2':>8} {'Chg1':>8} {'Chg2':>8}")
print(f"  {'─'*60}")

treated_df = liq[liq["TREATED"] == 1]
for var, label in vars_labels:
    if var not in liq.columns:
        continue
    pre   = treated_df[treated_df["PERIOD"]=="Pre (Oct21-Mar22)"][var].mean()
    post1 = treated_df[treated_df["PERIOD"]=="Post1 (Apr22-Mar23)"][var].mean()
    post2 = treated_df[treated_df["PERIOD"]=="Post2 (Apr23-Mar24)"][var].mean()
    print(f"  {label:<20} {pre:>8.4f} {post1:>8.4f} {post2:>8.4f} "
          f"{post1-pre:>8.4f} {post2-pre:>8.4f}")


# ── Table 4: Correlation matrix ───────────────────────────────────────────────
print("\n" + "="*65)
print("TABLE 4: CORRELATION MATRIX — LIQUIDITY MEASURES")
print("="*65)

corr_vars = ["CS_SPREAD","AMIHUD","ROLL_SPREAD","TURNOVER","VOLATILITY"]
corr_vars = [v for v in corr_vars if v in liq.columns]
corr_df   = liq[corr_vars].dropna()
corr_mat  = corr_df.corr().round(3)

print()
header = f"  {'':>15}" + "".join(f"{v:>13}" for v in corr_vars)
print(header)
print("  " + "─" * (15 + 13*len(corr_vars)))
for v in corr_vars:
    row = f"  {v:<15}" + "".join(f"{corr_mat.loc[v,c]:>13.3f}" for c in corr_vars)
    print(row)


# ── Save all tables to CSV ────────────────────────────────────────────────────
corr_mat.to_csv(f"{OUTPUT_FOLDER}/correlation_matrix.csv")

summary_rows = []
for var, label in vars_labels:
    if var not in liq.columns:
        continue
    summary_rows.append({
        "Variable": label,
        "Mean_All"    : round(liq[var].mean(), 4),
        "Std_All"     : round(liq[var].std(),  4),
        "Mean_Treated": round(liq[T][var].mean(), 4),
        "Mean_Control": round(liq[C][var].mean(), 4),
        "Min"         : round(liq[var].quantile(0.01), 4),
        "P25"         : round(liq[var].quantile(0.25), 4),
        "Median"      : round(liq[var].median(), 4),
        "P75"         : round(liq[var].quantile(0.75), 4),
        "Max"         : round(liq[var].quantile(0.99), 4),
    })

pd.DataFrame(summary_rows).to_csv(f"{OUTPUT_FOLDER}/summary_stats.csv", index=False)
print(f"\nSaved → summary_stats.csv, correlation_matrix.csv")
print("\nDone.")