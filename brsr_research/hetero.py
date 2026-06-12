"""
hetero.py
Heterogeneity analysis: for WHOM does BRSR matter most?
  1. By firm size (rank quartiles)
  2. By initial liquidity (high vs low pre-BRSR spread)
  3. By sector (financial vs non-financial)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FOLDER = "./output"

FINANCIAL_SYMBOLS = {
    "HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK","INDUSINDBK",
    "BANDHANBNK","FEDERALBNK","IDFCFIRSTB","BANKBARODA","CANBK","PNB",
    "UNIONBANK","MAHABANK","UCOBANK","CENTRALBK","IOB","BANKINDIA",
    "HDFCLIFE","SBILIFE","ICICIPRULI","LICI","GICRE","NIACL","STARHEALTH",
    "BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","MANAPPURAM",
    "LICHSGFIN","PNBHOUSING","CANFINHOME","AAVAS","HOMEFIRST",
    "HDFCAMC","NIPPONLIFE","ABSLAMC","UTIAMC","360ONE",
    "ICICIGI","SBICARD","CAMS","CDSL","BSE","MCX",
}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
liq      = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")
liq      = liq.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")

liq["POST1"] = (liq["DATE"] >= "2022-04-01").astype(int)
liq["POST2"] = (liq["DATE"] >= "2023-04-01").astype(int)
liq["DiD1"]  = liq["TREATED"] * liq["POST1"]
liq["DiD2"]  = liq["TREATED"] * liq["POST2"]
liq["LOG_CS"]     = np.log(liq["CS_SPREAD"].replace(0, np.nan))
liq["LOG_AMIHUD"] = np.log(liq["AMIHUD"].replace(0, np.nan))
liq["FINANCIAL"]  = liq["SYMBOL"].isin(FINANCIAL_SYMBOLS).astype(int)

# Pre-period spread for initial liquidity split
pre_spread = (liq[liq["DATE"] < "2022-04-01"]
              .groupby("SYMBOL")["CS_SPREAD"].mean()
              .reset_index().rename(columns={"CS_SPREAD":"PRE_SPREAD"}))
liq = liq.merge(pre_spread, on="SYMBOL", how="left")
liq["HIGH_SPREAD"] = (liq["PRE_SPREAD"] > liq["PRE_SPREAD"].median()).astype(int)

print(f"Rows: {len(liq):,} | Stocks: {liq['SYMBOL'].nunique():,}\n")


# ── Helper ────────────────────────────────────────────────────────────────────
def did_within(sub, dep_var, label):
    sub = sub.dropna(subset=[dep_var]).copy()
    if len(sub) < 500 or sub["SYMBOL"].nunique() < 20:
        return None
    vars_ = [dep_var, "DiD1", "DiD2", "POST1", "POST2",
             "LOG_MKTCAP", "VOLATILITY", "LOG_VOLUME"]
    avail = [v for v in vars_ if v in sub.columns]
    within = sub[avail] - sub.groupby(sub["SYMBOL"])[avail].transform("mean")
    controls = " + ".join([v for v in ["LOG_MKTCAP","VOLATILITY","LOG_VOLUME"]
                           if v in within.columns])
    formula = f"{dep_var} ~ DiD1 + DiD2" + (f" + {controls}" if controls else "")
    try:
        m = smf.ols(formula, data=within).fit(cov_type="HC3")
        return {
            "Group"    : label,
            "N_stocks" : sub["SYMBOL"].nunique(),
            "DiD1_coef": round(m.params.get("DiD1", np.nan), 4),
            "DiD1_p"   : round(m.pvalues.get("DiD1", np.nan), 4),
            "DiD2_coef": round(m.params.get("DiD2", np.nan), 4),
            "DiD2_p"   : round(m.pvalues.get("DiD2", np.nan), 4),
        }
    except Exception as e:
        print(f"  Error [{label}]: {e}")
        return None


def print_table(results, title):
    print(f"\n{'='*68}")
    print(title)
    print(f"{'─'*68}")
    print(f"{'Group':<26} {'Stocks':>7} {'DiD1':>8} {'p':>7} {'DiD2':>8} {'p':>7}")
    print(f"{'─'*68}")
    for r in results:
        if r:
            print(f"{r['Group']:<26} {r['N_stocks']:>7,} "
                  f"{r['DiD1_coef']:>8.4f} {r['DiD1_p']:>7.4f} "
                  f"{r['DiD2_coef']:>8.4f} {r['DiD2_p']:>7.4f}")


# ── 1. By size quartile (within treated) ─────────────────────────────────────
treated = liq[liq["TREATED"] == 1].copy()
treated["SIZE_Q"] = pd.qcut(treated["RANK"], q=4,
                             labels=["Q1 Largest","Q2","Q3","Q4 Smallest"])
control = liq[liq["TREATED"] == 0]

for dep in ["LOG_CS", "LOG_AMIHUD"]:
    res = []
    for q in ["Q1 Largest","Q2","Q3","Q4 Smallest"]:
        sub = pd.concat([treated[treated["SIZE_Q"] == q], control])
        res.append(did_within(sub, dep, q))
    print_table(res, f"BY SIZE QUARTILE — {dep}")

# ── 2. By initial liquidity ───────────────────────────────────────────────────
for dep in ["LOG_CS", "LOG_AMIHUD"]:
    res = []
    for flag, label in [(1,"High Initial Spread"),(0,"Low Initial Spread")]:
        sub = liq[liq["HIGH_SPREAD"] == flag]
        res.append(did_within(sub, dep, label))
    print_table(res, f"BY INITIAL LIQUIDITY — {dep}")

# ── 3. By sector ─────────────────────────────────────────────────────────────
for dep in ["LOG_CS", "LOG_AMIHUD"]:
    res = []
    for flag, label in [(1,"Financial"),(0,"Non-Financial")]:
        sub = liq[liq["FINANCIAL"] == flag]
        res.append(did_within(sub, dep, label))
    print_table(res, f"BY SECTOR — {dep}")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\nDone. Results printed above — copy into your paper's heterogeneity table.")