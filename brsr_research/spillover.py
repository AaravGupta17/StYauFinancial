"""
spillover.py
Formally tests the spillover hypothesis.
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
liq      = liq.merge(rankings[["SYMBOL","RANK","AVG_TURNOVER"]], on="SYMBOL", how="inner")

liq["LOG_CS"] = np.log(liq["CS_SPREAD"].replace(0, np.nan))
liq["POST1"]  = (liq["DATE"] >= "2022-04-01").astype(int)
liq["POST2"]  = (liq["DATE"] >= "2023-04-01").astype(int)

control = liq[liq["TREATED"] == 0].copy()
print(f"Control firms: {control['SYMBOL'].nunique():,}\n")


def within_ols(df, dep, post_col, label):
    d = df.dropna(subset=[dep]).copy()
    d["POST_VAR"] = d[post_col]
    d["TREND"]    = d["POST_VAR"]
    vars_ = [dep, "POST_VAR"]
    within = d[vars_] - d.groupby(d["SYMBOL"])[vars_].transform("mean")
    if within["POST_VAR"].std() < 1e-10:
        return None
    try:
        m = smf.ols(f"{dep} ~ POST_VAR", data=within).fit(cov_type="HC3")
        return {
            "Group"  : label,
            "POST"   : post_col,
            "Coef"   : round(m.params["POST_VAR"], 4),
            "P"      : round(m.pvalues["POST_VAR"], 4),
            "N_firms": df["SYMBOL"].nunique(),
            "Stars"  : ("***" if m.pvalues["POST_VAR"] < 0.01 else
                        "**"  if m.pvalues["POST_VAR"] < 0.05 else
                        "*"   if m.pvalues["POST_VAR"] < 0.10 else ""),
        }
    except:
        return None


def print_table(results, title):
    print(f"\n{'='*65}")
    print(title)
    print(f"{'─'*65}")
    print(f"{'Group':<30} {'POST':>6} {'Coef':>8} {'P':>7} {'Firms':>7}  Stars")
    print(f"{'─'*65}")
    for r in results:
        if r:
            print(f"{r['Group']:<30} {r['POST']:>6} {r['Coef']:>8.4f} "
                  f"{r['P']:>7.4f} {r['N_firms']:>7,}  {r['Stars']}")


# ── 1. Rank proximity gradient ───────────────────────────────────────────────
bands = [
    ("Rank 1001-1100 (closest)", (1001, 1100)),
    ("Rank 1101-1200",           (1101, 1200)),
    ("Rank 1201-1300",           (1201, 1300)),
    ("Rank 1301-1400",           (1301, 1400)),
    ("Rank 1401-1500 (farthest)",(1401, 1500)),
]

prox_results = []
for label, (lo, hi) in bands:
    sub = control[(control["RANK"] >= lo) & (control["RANK"] <= hi)]
    for post in ["POST1","POST2"]:
        r = within_ols(sub, "LOG_CS", post, label)
        if r:
            prox_results.append(r)

print_table(prox_results, "TEST 1: Rank Proximity Gradient")

print("\n  Monotonicity check (POST1 coefs, closest → farthest):")
post1 = [r["Coef"] for r in prox_results if r["POST"] == "POST1"]
print(f"  {post1}")
diffs = [round(post1[i+1] - post1[i], 4) for i in range(len(post1)-1)]
print(f"  Step changes: {diffs}")
if all(d >= 0 for d in diffs):
    print("  ✓ Monotone — spillover gradient confirmed (closest firms improve most)")
elif all(d <= 0 for d in diffs):
    print("  ✗ Reverse monotone — closer firms improve LESS (no spillover)")
else:
    print("  ~ Non-monotone — mixed evidence")


# ── 2. Sector adjacency ───────────────────────────────────────────────────────
HIGH_DISC_SYMBOLS = {
    "HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
    "BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN",
    "INFY","TCS","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT","COFORGE",
    "HDFCLIFE","SBILIFE","ICICIPRULI",
}

control["HIGH_DISC"] = control["SYMBOL"].isin(HIGH_DISC_SYMBOLS).astype(int)

sec_results = []
for flag, label in [(1,"High-disclosure sector"),(0,"Other sectors")]:
    sub = control[control["HIGH_DISC"] == flag]
    for post in ["POST1","POST2"]:
        r = within_ols(sub, "LOG_CS", post, label)
        if r:
            sec_results.append(r)

print_table(sec_results, "TEST 2: Sector Adjacency")


# ── 3. Size proximity ────────────────────────────────────────────────────────
control["SIZE_DIST"] = (control["RANK"] - 1000).abs()
terciles = control["SIZE_DIST"].quantile([0.33, 0.67])
control["SIZE_TERCILE"] = pd.cut(
    control["SIZE_DIST"],
    bins=[0, terciles[0.33], terciles[0.67], control["SIZE_DIST"].max()+1],
    labels=["Close (T1)","Mid (T2)","Far (T3)"]
)

size_results = []
for t in ["Close (T1)","Mid (T2)","Far (T3)"]:
    sub = control[control["SIZE_TERCILE"] == t]
    for post in ["POST1","POST2"]:
        r = within_ols(sub, "LOG_CS", post, t)
        if r:
            size_results.append(r)

print_table(size_results, "TEST 3: Size Proximity Terciles")


# ── 4. Overall spillover magnitude vs treatment effect ───────────────────────
print(f"\n{'='*65}")
print("TEST 4: Spillover magnitude vs direct treatment effect")
print(f"{'─'*65}")

treated = liq[liq["TREATED"] == 1].copy()
all_ctrl = liq[liq["TREATED"] == 0].copy()

for post, label in [("POST1","Announcement (Apr 2022)"),
                    ("POST2","Enforcement  (Apr 2023)")]:
    t_d  = treated.dropna(subset=["LOG_CS"])
    c_d  = all_ctrl.dropna(subset=["LOG_CS"])
    vars_ = ["LOG_CS", post]

    t_w = t_d[vars_] - t_d.groupby(t_d["SYMBOL"])[vars_].transform("mean")
    c_w = c_d[vars_] - c_d.groupby(c_d["SYMBOL"])[vars_].transform("mean")

    try:
        mt = smf.ols(f"LOG_CS ~ {post}", data=t_w).fit(cov_type="HC3")
        mc = smf.ols(f"LOG_CS ~ {post}", data=c_w).fit(cov_type="HC3")
        ratio = mc.params[post] / mt.params[post] if mt.params[post] != 0 else np.nan
        print(f"\n  {label}")
        print(f"    Treated effect : {mt.params[post]:>8.4f}  p={mt.pvalues[post]:.4f}")
        print(f"    Control effect : {mc.params[post]:>8.4f}  p={mc.pvalues[post]:.4f}")
        print(f"    Spillover ratio: {ratio:>8.2f}x  "
              f"({'control improved as much as treated' if abs(ratio) > 0.8 else 'partial spillover'})")
    except Exception as e:
        print(f"  Error: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
all_results = prox_results + sec_results + size_results
pd.DataFrame(all_results).to_csv(f"{OUTPUT_FOLDER}/spillover_results.csv", index=False)
print(f"\nSaved → {OUTPUT_FOLDER}/spillover_results.csv")
print("\nDone.")