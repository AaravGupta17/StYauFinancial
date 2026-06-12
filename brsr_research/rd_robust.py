"""
rd_robust.py
Strengthens the RD result three ways:
  1. Donut RD — exclude firms ranked 995-1005 right at cutoff
     (rules out result driven by 10 firms straddling the threshold)
  2. Bandwidth sensitivity — plot coefficient across every bandwidth
     from 10 to 300 in steps of 10
  3. McCrary (2008) density test — checks firms didn't manipulate
     their ranking to sort just below 1000 (gaming the threshold)
     If density is smooth at 1000, assignment is as-good-as-random
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "./output"

print("Loading data...")
liq      = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")
liq      = liq.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")

liq["LOG_CS"]  = np.log(liq["CS_SPREAD"].replace(0, np.nan))
liq["RUNNING"] = liq["RANK"] - 1000
liq["ABOVE"]   = (liq["RUNNING"] > 0).astype(int)
liq["POST"]    = (liq["DATE"] >= "2022-04-01").astype(int)

liq_post = liq[liq["POST"] == 1].copy()
print(f"Post-treatment rows: {len(liq_post):,}\n")


# ── Helper ────────────────────────────────────────────────────────────────────
def rd_estimate(df, bw, dep="LOG_CS", donut=0):
    sub = df[df["RUNNING"].abs() <= bw].copy()
    if donut > 0:
        sub = sub[sub["RUNNING"].abs() > donut]
    sub = sub.dropna(subset=[dep])
    if len(sub) < 50 or sub["SYMBOL"].nunique() < 10:
        return None
    try:
        m = smf.ols(
            f"{dep} ~ ABOVE + RUNNING + ABOVE:RUNNING", data=sub
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["SYMBOL"]})
        return {
            "coef": m.params["ABOVE"],
            "se"  : m.bse["ABOVE"],
            "p"   : m.pvalues["ABOVE"],
            "n"   : int(m.nobs),
        }
    except:
        return None


# ── 1. Donut RD ───────────────────────────────────────────────────────────────
print("=" * 62)
print("TEST 1: Donut RD (BW=100, excluding firms within donut of cutoff)")
print("Result should remain significant — rules out cutoff manipulation")
print("=" * 62)
print(f"\n{'Donut':>8} {'Coef':>8} {'SE':>8} {'P':>8} {'N':>8}  Stars")
print("─" * 52)

donut_results = []
for donut in [0, 5, 10, 15, 20, 25]:
    r = rd_estimate(liq_post, bw=100, donut=donut)
    if r:
        stars = ("***" if r["p"] < 0.01 else
                 "**"  if r["p"] < 0.05 else
                 "*"   if r["p"] < 0.10 else "")
        label = "baseline" if donut == 0 else f"excl ±{donut}"
        print(f"{label:>8} {r['coef']:>8.4f} {r['se']:>8.4f} "
              f"{r['p']:>8.4f} {r['n']:>8,}  {stars}")
        donut_results.append({"donut": donut, **r})


# ── 2. Bandwidth sensitivity ──────────────────────────────────────────────────
print("\n" + "=" * 62)
print("TEST 2: Bandwidth sensitivity (BW = 10 to 300, step 10)")
print("Look for stable significant region — not just one lucky BW")
print("=" * 62)
print(f"\n{'BW':>6} {'Coef':>8} {'SE':>8} {'P':>8} {'N':>8}  Stars")
print("─" * 52)

bw_results = []
for bw in range(10, 310, 10):
    r = rd_estimate(liq_post, bw=bw)
    if r:
        stars = ("***" if r["p"] < 0.01 else
                 "**"  if r["p"] < 0.05 else
                 "*"   if r["p"] < 0.10 else "")
        print(f"{bw:>6} {r['coef']:>8.4f} {r['se']:>8.4f} "
              f"{r['p']:>8.4f} {r['n']:>8,}  {stars}")
        bw_results.append({"bw": bw, **r})


# ── 3. McCrary density test (manual implementation) ──────────────────────────
print("\n" + "=" * 62)
print("TEST 3: Density test (McCrary 2008)")
print("Tests if firms bunched just below rank 1000 to get BRSR")
print("Insignificant = no manipulation = clean RD")
print("=" * 62)

# Use one observation per firm (ranks are firm-level not daily)
firm_ranks = rankings[["SYMBOL","RANK"]].drop_duplicates()
firm_ranks = firm_ranks[(firm_ranks["RANK"] >= 800) &
                        (firm_ranks["RANK"] <= 1200)].copy()

# Count firms in each bin of width 20 around cutoff
bin_width = 20
firm_ranks["BIN"] = ((firm_ranks["RANK"] - 1000) // bin_width) * bin_width + bin_width//2

bin_counts = firm_ranks.groupby("BIN").size().reset_index(name="COUNT")
bin_counts["ABOVE_CUT"] = (bin_counts["BIN"] > 0).astype(int)
bin_counts["BIN_C"]     = bin_counts["BIN"]  # centered

print(f"\nFirm counts by rank bin (bin width={bin_width}):")
print(f"{'Bin center':>12} {'Count':>8} {'Side':>10}")
print("─" * 35)
for _, row in bin_counts.iterrows():
    side = "Control" if row["ABOVE_CUT"] else "Treated"
    marker = " ← cutoff" if abs(row["BIN"]) <= bin_width else ""
    print(f"{int(row['BIN']):>12} {int(row['COUNT']):>8} {side:>10}{marker}")

# Simple local linear density test: regress count on bin + discontinuity
if len(bin_counts) >= 6:
    try:
        m_density = smf.ols(
            "COUNT ~ BIN_C + ABOVE_CUT + BIN_C:ABOVE_CUT",
            data=bin_counts
        ).fit()
        coef_d = m_density.params["ABOVE_CUT"]
        pval_d = m_density.pvalues["ABOVE_CUT"]
        print(f"\nDensity discontinuity at cutoff:")
        print(f"  Coef = {coef_d:.4f}  P = {pval_d:.4f}")
        if pval_d > 0.10:
            print("  ✓ Insignificant — no evidence of rank manipulation")
        else:
            print("  ⚠ Significant — possible bunching at cutoff, discuss in paper")
    except Exception as e:
        print(f"  Density test error: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(donut_results).to_csv(f"{OUTPUT_FOLDER}/rd_donut.csv", index=False)
pd.DataFrame(bw_results).to_csv(f"{OUTPUT_FOLDER}/rd_bandwidth.csv", index=False)
print(f"\nSaved → rd_donut.csv, rd_bandwidth.csv")
print("\nDone.")