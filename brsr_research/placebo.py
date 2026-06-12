"""
placebo.py — FIXED v2
Correct placebo strategy for this dataset:
  - Data starts Oct 2021, treatment Apr 2022
  - Pre-period is too short for early fake dates
  - Anticipation effects contaminate Jan-Feb 2022

Correct approach: use FULL sample, place fake dates AFTER actual
treatment but shift them by ±6 months. If the effect is real and
tied to BRSR specifically, arbitrary post-treatment dates should
not replicate it.

Also run: control-group-only placebo — assign fake treatment to
control firms (rank 1001-1250) vs (rank 1251-1500). Should be zero.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "./output"

print("Loading data...")
liq = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
liq["LOG_CS"]     = np.log(liq["CS_SPREAD"].replace(0, np.nan))
liq["LOG_AMIHUD"] = np.log(liq["AMIHUD"].replace(0, np.nan))
print(f"Rows: {len(liq):,} | Date range: {liq['DATE'].min().date()} to {liq['DATE'].max().date()}\n")


def run_placebo_date(df, fake_date, dep_var, label=None):
    d = df.copy()
    d["POST_FAKE"] = (d["DATE"] >= fake_date).astype(int)
    d["DiD_FAKE"]  = d["TREATED"] * d["POST_FAKE"]

    if d["POST_FAKE"].nunique() < 2 or d["DiD_FAKE"].nunique() < 2:
        return None

    d = d.dropna(subset=[dep_var])
    vars_ = [dep_var, "DiD_FAKE", "POST_FAKE", "LOG_MKTCAP", "VOLATILITY", "LOG_VOLUME"]
    avail = [v for v in vars_ if v in d.columns]
    within = d[avail] - d.groupby(d["SYMBOL"])[avail].transform("mean")

    if within["DiD_FAKE"].std() < 1e-10:
        return None

    controls = " + ".join([v for v in ["LOG_MKTCAP","VOLATILITY","LOG_VOLUME"]
                           if v in within.columns])
    formula = f"{dep_var} ~ DiD_FAKE" + (f" + {controls}" if controls else "")
    try:
        m = smf.ols(formula, data=within).fit(cov_type="HC3")
        return {
            "Label"    : label or str(fake_date)[:10],
            "Dep"      : dep_var,
            "Coef"     : round(m.params["DiD_FAKE"], 4),
            "P"        : round(m.pvalues["DiD_FAKE"], 4),
            "N"        : int(m.nobs),
            "Flag"     : "⚠ SIGNIFICANT" if m.pvalues["DiD_FAKE"] < 0.1 else "✓ insignificant",
        }
    except Exception as e:
        return None


def print_table(results, title):
    print(f"\n{'='*62}")
    print(title)
    print(f"{'─'*62}")
    print(f"  {'Label':<20} {'Coef':>8} {'P':>8} {'N':>8}  Result")
    print(f"  {'─'*54}")
    for r in results:
        if r:
            print(f"  {r['Label']:<20} {r['Coef']:>8.4f} "
                  f"{r['P']:>8.4f} {r['N']:>8,}  {r['Flag']}")


results_all = []

# ── Test 1: Shifted fake dates (between announcement and enforcement) ─────────
# Real DiD1 = Apr 2022, Real DiD2 = Apr 2023
# Fake dates between these two real events — should show nothing extra
print("=" * 62)
print("TEST 1: Shifted fake dates (Jun 2022, Sep 2022, Dec 2022)")
print("These fall AFTER announcement but before enforcement.")
print("If result is robust, these arbitrary dates should be weaker.")
print("=" * 62)

shifted_dates = [
    ("2022-06-01", "Jun 2022 (fake)"),
    ("2022-09-01", "Sep 2022 (fake)"),
    ("2022-12-01", "Dec 2022 (fake)"),
    ("2023-06-01", "Jun 2023 (fake)"),  # after enforcement
]

for dep in ["LOG_CS", "LOG_AMIHUD"]:
    res = []
    for fd, label in shifted_dates:
        r = run_placebo_date(liq, pd.Timestamp(fd), dep, label)
        if r:
            res.append(r)
    print_table(res, f"Shifted Dates — {dep}")
    results_all.extend([r for r in res if r])

# ── Test 2: Control-group-only placebo ───────────────────────────────────────
# Assign fake treatment to rank 1001-1250 vs rank 1251-1500 (both control)
# Neither group got BRSR — so DiD should be zero
print("\n" + "=" * 62)
print("TEST 2: Within-control placebo")
print("Fake treated = rank 1001-1250, Fake control = rank 1251-1500")
print("Neither group got BRSR — effect should be zero")
print("=" * 62)

rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")
liq_ctrl = liq[liq["TREATED"] == 0].copy()
liq_ctrl = liq_ctrl.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")
liq_ctrl["FAKE_TREATED"] = (liq_ctrl["RANK"] <= 1250).astype(int)

for dep in ["LOG_CS", "LOG_AMIHUD"]:
    res = []
    for fd, label in [("2022-04-01","Apr 2022 cutoff"),
                      ("2023-04-01","Apr 2023 cutoff")]:
        d = liq_ctrl.copy()
        d["POST_FAKE"] = (d["DATE"] >= pd.Timestamp(fd)).astype(int)
        d["DiD_FAKE"]  = d["FAKE_TREATED"] * d["POST_FAKE"]
        d = d.dropna(subset=[dep])
        vars_ = [dep, "DiD_FAKE", "POST_FAKE"]
        within = d[vars_] - d.groupby(d["SYMBOL"])[vars_].transform("mean")
        if within["DiD_FAKE"].std() < 1e-10:
            continue
        try:
            m = smf.ols(f"{dep} ~ DiD_FAKE", data=within).fit(cov_type="HC3")
            r = {
                "Label": label,
                "Dep"  : dep,
                "Coef" : round(m.params["DiD_FAKE"], 4),
                "P"    : round(m.pvalues["DiD_FAKE"], 4),
                "N"    : int(m.nobs),
                "Flag" : "⚠ SIGNIFICANT" if m.pvalues["DiD_FAKE"] < 0.1 else "✓ insignificant",
            }
            res.append(r)
            results_all.append(r)
        except Exception as e:
            print(f"  Error: {e}")
    print_table(res, f"Within-Control Placebo — {dep}")

# ── Save ──────────────────────────────────────────────────────────────────────
if results_all:
    pd.DataFrame(results_all).to_csv(f"{OUTPUT_FOLDER}/placebo_results.csv", index=False)
    print(f"\nSaved → {OUTPUT_FOLDER}/placebo_results.csv")

print("\nDone.")