"""
event_study.py — FIXED
Hyphens in month names (2021-11) break statsmodels formula parser.
Fix: replace hyphens with underscores in all dummy column names.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "./output"

print("Loading data...")
liq = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")
liq = liq.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")

liq["LOG_CS"]     = np.log(liq["CS_SPREAD"].replace(0, np.nan))
liq["LOG_AMIHUD"] = np.log(liq["AMIHUD"].replace(0, np.nan))
liq["YM"]         = liq["DATE"].dt.to_period("M").astype(str)

BASELINE = "2022-03"
months   = sorted(liq["YM"].unique())
months   = [m for m in months if m != BASELINE]

def safe(m): return "m_" + m.replace("-","_")

print(f"Months: {len(months)} | Baseline: {BASELINE}\n")


def run_event_study(df, dep_var):
    df = df.dropna(subset=[dep_var]).copy()

    int_cols = []
    dum_cols = []

    for m in months:
        d_col   = f"D_{safe(m)}"
        int_col = f"I_{safe(m)}"
        df[d_col]   = (df["YM"] == m).astype(int)
        df[int_col] = df["TREATED"] * df[d_col]
        int_cols.append(int_col)
        dum_cols.append(d_col)

    ctrl_cols = [c for c in ["LOG_MKTCAP","VOLATILITY","LOG_VOLUME"]
                 if c in df.columns]
    all_cols  = [dep_var] + int_cols + dum_cols + ctrl_cols

    within = df[all_cols] - df.groupby(df["SYMBOL"])[all_cols].transform("mean")

    formula = (dep_var + " ~ " +
               " + ".join(int_cols) + " + " +
               " + ".join(dum_cols) +
               (" + " + " + ".join(ctrl_cols) if ctrl_cols else ""))

    try:
        fit = smf.ols(formula, data=within).fit(cov_type="HC3")

        results = []
        for m in months:
            var = f"I_{safe(m)}"
            if var in fit.params:
                coef = fit.params[var]
                se   = fit.bse[var]
                results.append({
                    "Month" : m,
                    "Coef"  : round(coef, 4),
                    "SE"    : round(se,   4),
                    "CI_low": round(coef - 1.96*se, 4),
                    "CI_hi" : round(coef + 1.96*se, 4),
                    "P"     : round(fit.pvalues[var], 4),
                })

        results.append({
            "Month":"2022-03","Coef":0,"SE":0,
            "CI_low":0,"CI_hi":0,"P":np.nan
        })
        return sorted(results, key=lambda x: x["Month"])

    except Exception as e:
        print(f"Error: {e}")
        return []


for dep, label in [("LOG_CS","CS Spread"), ("LOG_AMIHUD","Amihud ILLIQ")]:
    print(f"\n{'='*65}")
    print(f"EVENT STUDY — {label}")
    print(f"Baseline = {BASELINE} | Negative = more liquid")
    print(f"{'='*65}")
    print(f"{'Month':<10} {'Coef':>8} {'SE':>8} {'CI_low':>8} {'CI_hi':>8} {'P':>7}")
    print(f"{'─'*58}")

    res = run_event_study(liq, dep)

    for r in res:
        tag = ""
        if r["Month"] == "2022-04": tag = " ← BRSR announced"
        if r["Month"] == "2023-04": tag = " ← BRSR enforced"
        if r["Month"] == BASELINE:  tag = " ← baseline"
        sig = "*" if isinstance(r["P"], float) and r["P"] < 0.05 else ""
        print(f"{r['Month']:<10} {r['Coef']:>8.4f} {r['SE']:>8.4f} "
              f"{r['CI_low']:>8.4f} {r['CI_hi']:>8.4f} "
              f"{str(r['P']):>7}{sig}{tag}")

    pd.DataFrame(res).to_csv(
        f"{OUTPUT_FOLDER}/event_study_{dep.lower()}.csv", index=False
    )
    print(f"\nSaved → {OUTPUT_FOLDER}/event_study_{dep.lower()}.csv")

    pre  = [r["Coef"] for r in res if "2021" in r["Month"] or r["Month"] in ["2022-01","2022-02"]]
    post = [r["Coef"] for r in res if r["Month"] >= "2022-04"]
    print(f"Pre-treatment mean  (should be ~0): {np.mean(pre):.4f}  std={np.std(pre):.4f}")
    print(f"Post-treatment mean (should be <0): {np.mean(post):.4f}  std={np.std(post):.4f}")

print("\nDone.")