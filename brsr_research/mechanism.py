"""
mechanism.py
WHY did spreads fall for firms near the BRSR cutoff?
Tests three information-based channels:
  1. Trading activity  — did volume increase post-BRSR?
  2. Volatility reduction — did return volatility fall?
  3. Price efficiency — did return autocorrelation fall?
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

liq["POST1"] = (liq["DATE"] >= "2022-04-01").astype(int)
liq["POST2"] = (liq["DATE"] >= "2023-04-01").astype(int)
liq["DiD1"]  = liq["TREATED"] * liq["POST1"]
liq["DiD2"]  = liq["TREATED"] * liq["POST2"]

print(f"Rows: {len(liq):,}\n")


def did_mechanism(df, dep_var, label):
    d = df.dropna(subset=[dep_var]).copy()
    if len(d) < 1000:
        print(f"  {label} — insufficient data")
        return None

    vars_ = [dep_var, "DiD1", "DiD2", "POST1", "POST2"]
    avail = [v for v in vars_ if v in d.columns]
    within = d[avail] - d.groupby(d["SYMBOL"])[avail].transform("mean")

    try:
        m = smf.ols(
            f"{dep_var} ~ DiD1 + DiD2 + POST1 + POST2",
            data=within
        ).fit(cov_type="HC3")

        s1 = "*" if m.pvalues["DiD1"] < 0.05 else ""
        s2 = "*" if m.pvalues["DiD2"] < 0.05 else ""
        print(f"\n  {label}")
        print(f"    DiD1 (announcement): {m.params['DiD1']:>8.4f}  p={m.pvalues['DiD1']:.4f} {s1}")
        print(f"    DiD2 (enforcement) : {m.params['DiD2']:>8.4f}  p={m.pvalues['DiD2']:.4f} {s2}")
        print(f"    N={int(m.nobs):,}")
        return {
            "label"  : label,
            "DiD1"   : round(m.params["DiD1"], 4),
            "DiD1_p" : round(m.pvalues["DiD1"], 4),
            "DiD2"   : round(m.params["DiD2"], 4),
            "DiD2_p" : round(m.pvalues["DiD2"], 4),
            "N"      : int(m.nobs),
        }
    except Exception as e:
        print(f"  {label} — error: {e}")
        return None


# ── Channel 1: Trading Activity ───────────────────────────────────────────────
print("=" * 62)
print("CHANNEL 1: Trading Activity")
print("H: BRSR attracted more traders → volume increased")
print("=" * 62)

liq["LOG_VOLUME"] = np.log(liq["TOTTRDVAL"].replace(0, np.nan))
liq["LOG_TRADES"] = np.log(liq["TOTTRDQTY"].replace(0, np.nan))

results = []
for dep, label in [("LOG_VOLUME","Log Trading Value"),
                   ("LOG_TRADES","Log Trading Quantity")]:
    r = did_mechanism(liq, dep, label)
    if r: results.append(r)


# ── Channel 2: Volatility Reduction ──────────────────────────────────────────
print("\n" + "=" * 62)
print("CHANNEL 2: Volatility Reduction")
print("H: BRSR reduced uncertainty → volatility fell → spreads tightened")
print("=" * 62)

liq["LOG_VOL"] = np.log(liq["VOLATILITY"].replace(0, np.nan))
r = did_mechanism(liq, "LOG_VOL", "Log Return Volatility")
if r: results.append(r)


# ── Channel 3: Price Efficiency (Variance Ratio) ─────────────────────────────
print("\n" + "=" * 62)
print("CHANNEL 3: Price Efficiency")
print("H: BRSR improved info environment → prices more efficient")
print("Variance Ratio = Var(2-day ret) / 2*Var(1-day ret)")
print("VR=1 → efficient | VR>1 → positive autocorrelation (momentum)")
print("VR<1 → negative autocorrelation (mean reversion)")
print("=" * 62)

liq = liq.sort_values(["SYMBOL","DATE"])
liq["RET1"]  = liq.groupby("SYMBOL")["CLOSE"].pct_change(1)
liq["RET2"]  = liq.groupby("SYMBOL")["CLOSE"].pct_change(2)

def variance_ratio(df, window=21):
    var1 = df["RET1"].rolling(window).var()
    var2 = df["RET2"].rolling(window).var()
    vr   = var2 / (2 * var1)
    return vr

liq["VR"] = liq.groupby("SYMBOL", group_keys=False).apply(
    lambda x: pd.Series(variance_ratio(x), index=x.index)
)

# Winsorize VR at 1%/99%
from scipy.stats import mstats
valid = liq["VR"].notna()
liq.loc[valid, "VR"] = mstats.winsorize(
    liq.loc[valid, "VR"].values, limits=[0.01, 0.01]
)

# VR closer to 1 = more efficient
# We use |VR - 1| as inefficiency measure — should FALL post-BRSR
liq["INEFFICIENCY"] = (liq["VR"] - 1).abs()
liq["LOG_INEFF"]    = np.log(liq["INEFFICIENCY"].replace(0, np.nan))

r = did_mechanism(liq, "LOG_INEFF", "Log Price Inefficiency |VR-1|")
if r: results.append(r)

# Also show raw VR means by group and period
print("\n  Variance Ratio summary (VR=1 is efficient):")
for period, mask in [("Pre-Apr22",  liq["DATE"] <  "2022-04-01"),
                     ("Apr22-Mar23",(liq["DATE"] >= "2022-04-01") & (liq["DATE"] < "2023-04-01")),
                     ("Post-Apr23",  liq["DATE"] >= "2023-04-01")]:
    sub = liq[mask]
    t   = sub[sub["TREATED"]==1]["VR"].mean()
    c   = sub[sub["TREATED"]==0]["VR"].mean()
    print(f"  {period:<14}  Treated={t:.4f}  Control={c:.4f}  Diff={t-c:.4f}")


# ── Channel 4: Bid-Ask Spread components ─────────────────────────────────────
print("\n" + "=" * 62)
print("CHANNEL 4: High-Low Range Compression")
print("H: If information asymmetry fell, intraday range narrows")
print("Range = (High-Low)/Close — proxy for intraday uncertainty")
print("=" * 62)

liq["HL_RANGE"]     = (liq["HIGH"] - liq["LOW"]) / liq["CLOSE"]
liq["LOG_HL_RANGE"] = np.log(liq["HL_RANGE"].replace(0, np.nan))

r = did_mechanism(liq, "LOG_HL_RANGE", "Log High-Low Range")
if r: results.append(r)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("MECHANISM SUMMARY")
print("=" * 62)
print(f"{'Channel':<35} {'DiD1':>8} {'p':>7} {'DiD2':>8} {'p':>7}")
print("─" * 62)
for r in results:
    if r:
        s1 = "*" if r["DiD1_p"] < 0.05 else ""
        s2 = "*" if r["DiD2_p"] < 0.05 else ""
        print(f"{r['label']:<35} {r['DiD1']:>8.4f} {r['DiD1_p']:>7.4f}{s1} "
              f"{r['DiD2']:>8.4f} {r['DiD2_p']:>7.4f}{s2}")

pd.DataFrame(results).to_csv(f"{OUTPUT_FOLDER}/mechanism_results.csv", index=False)
print(f"\nSaved → {OUTPUT_FOLDER}/mechanism_results.csv")
print("\nDone.")