"""
rd.py
Regression Discontinuity Design around the rank-1000 BRSR cutoff.
Firms ranked 990-1010 are nearly identical in size but one group
got BRSR mandated. This is a much cleaner identification than DiD alone.

Strategy:
  - Load rankings from sampli.py output
  - Keep only firms in bandwidth around cutoff (default ±50 ranks)
  - Run RD regression: liquidity ~ f(rank) + treatment_indicator
  - Show results are robust to bandwidth choice (±25, ±50, ±100)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "./output"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
liq      = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")

liq = liq.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")

# Running variable: distance from cutoff (negative = treated, positive = control)
liq["RUNNING"] = liq["RANK"] - 1000   # <0 → treated, >0 → control
liq["ABOVE"]   = (liq["RUNNING"] > 0).astype(int)  # 1 = control (above cutoff)

liq["POST"] = (liq["DATE"] >= "2022-04-01").astype(int)

for col, logcol in [("CS_SPREAD","LOG_CS"), ("AMIHUD","LOG_AMIHUD"),
                    ("ROLL_SPREAD","LOG_ROLL")]:
    liq[logcol] = np.log(liq[col].replace(0, np.nan))

print(f"Rows: {len(liq):,} | Rank range: {liq['RANK'].min()}–{liq['RANK'].max()}\n")


# ── RD estimator ──────────────────────────────────────────────────────────────
def run_rd(df, bandwidth, dep_var, post_only=True):
    sub = df[df["RUNNING"].abs() <= bandwidth].copy()
    if post_only:
        sub = sub[sub["POST"] == 1]
    sub = sub.dropna(subset=[dep_var])
    if len(sub) < 100:
        return None

    formula = f"{dep_var} ~ ABOVE + RUNNING + ABOVE:RUNNING"
    try:
        m = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["SYMBOL"]}
        )
        return {
            "BW"        : bandwidth,
            "Dep"       : dep_var,
            "Coef_ABOVE": round(m.params["ABOVE"], 4),
            "SE"        : round(m.bse["ABOVE"], 4),
            "P"         : round(m.pvalues["ABOVE"], 4),
            "N"         : int(m.nobs),
            "Stars"     : ("***" if m.pvalues["ABOVE"] < 0.01 else
                           "**"  if m.pvalues["ABOVE"] < 0.05 else
                           "*"   if m.pvalues["ABOVE"] < 0.10 else ""),
        }
    except Exception as e:
        return {"BW": bandwidth, "Dep": dep_var, "Error": str(e)}


# ── Run across bandwidths and outcomes ────────────────────────────────────────
outcomes = {
    "LOG_CS"    : "CS Spread (log)",
    "LOG_AMIHUD": "Amihud ILLIQ (log)",
    "LOG_ROLL"  : "Roll Spread (log)",
}

bandwidths = [25, 50, 100, 200]
all_results = []

print("=" * 70)
print("REGRESSION DISCONTINUITY — Local Linear, Firm-Clustered SEs")
print("Positive coef on ABOVE = control firms have HIGHER spreads")
print("(i.e. BRSR treatment REDUCED spreads for treated firms)")
print("=" * 70)

for dep, label in outcomes.items():
    print(f"\n{label}")
    print(f"  {'BW':>6}  {'Coef':>8}  {'SE':>8}  {'P':>7}  {'N':>8}  Stars")
    print(f"  {'─'*55}")
    for bw in bandwidths:
        r = run_rd(liq, bw, dep, post_only=True)
        if r and "Error" not in r:
            print(f"  {r['BW']:>6}  {r['Coef_ABOVE']:>8.4f}  "
                  f"{r['SE']:>8.4f}  {r['P']:>7.4f}  "
                  f"{r['N']:>8,}  {r['Stars']}")
            all_results.append(r)
        else:
            print(f"  {bw:>6}  insufficient data or error")

# ── Pre-period placebo RD ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PLACEBO RD — Pre-period (before Apr 2022), should show NO effect")
print("=" * 70)

liq_pre = liq[liq["DATE"] < "2022-04-01"].copy()

for dep, label in outcomes.items():
    print(f"\n{label}")
    for bw in [50, 100]:
        sub = liq_pre[liq_pre["RUNNING"].abs() <= bw].dropna(subset=[dep])
        if len(sub) < 100:
            continue
        try:
            m = smf.ols(
                f"{dep} ~ ABOVE + RUNNING + ABOVE:RUNNING", data=sub
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["SYMBOL"]})
            sig = "⚠ significant" if m.pvalues["ABOVE"] < 0.1 else "← good, ~0"
            print(f"  BW={bw}: coef={m.params['ABOVE']:.4f}  "
                  f"p={m.pvalues['ABOVE']:.4f}  N={int(m.nobs):,}  {sig}")
        except Exception as e:
            print(f"  BW={bw}: error — {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
if all_results:
    pd.DataFrame(all_results).to_csv(f"{OUTPUT_FOLDER}/rd_results.csv", index=False)
    print(f"\nSaved → {OUTPUT_FOLDER}/rd_results.csv")

print("\nDone.")