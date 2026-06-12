import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

OUTPUT_FOLDER = r"./output"
DATA_FOLDER   = r"./data"

print("Loading data...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
df["LOG_CS"]     = np.log(df["CS_SPREAD"].replace(0, np.nan))
df["LOG_AMIHUD"] = np.log(df["AMIHUD"].replace(0, np.nan))
df = df.dropna(subset=["LOG_CS","LOG_AMIHUD"])
df["POST1"] = (df["DATE"] >= "2022-04-01").astype(int)
df["POST2"] = (df["DATE"] >= "2023-04-01").astype(int)

# ── Load NSE sector classification from bhavcopy ISIN prefix ─────────────────
# We use first 3 chars of ISIN as crude sector proxy
# Better: load from NSE industry master
print("Loading sector data from NSE master...")

# Try to get industry from any bhavcopy file
sample_file = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")])[0]
raw = pd.read_csv(os.path.join(DATA_FOLDER, sample_file),
                  usecols=["SYMBOL","SERIES","ISIN"])
raw = raw[raw["SERIES"]=="EQ"][["SYMBOL","ISIN"]].drop_duplicates()

# Use ISIN prefix as sector proxy (crude but available)
# Better approach: NSE industry classification
# We'll download NSE equity master for proper sector
import urllib.request
import io

print("Downloading NSE equity master for sector data...")
try:
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        equity_master = pd.read_csv(io.StringIO(r.read().decode("latin-1")))
    print(f"Downloaded equity master: {len(equity_master)} rows")
    print(f"Columns: {equity_master.columns.tolist()}")
    equity_master.to_csv(f"{OUTPUT_FOLDER}/equity_master.csv", index=False)
except Exception as e:
    print(f"Download failed: {e}")
    print("Using ISIN prefix as sector proxy instead...")
    equity_master = None

# ── Merge sector into main df ─────────────────────────────────────────────────
if equity_master is not None:
    # Find the industry/sector column
    sector_col = [c for c in equity_master.columns if
                  any(x in c.upper() for x in ["INDUSTRY","SECTOR","SERIES"])]
    print(f"Sector columns found: {sector_col}")

    if " INDUSTRY" in equity_master.columns:
        eq = equity_master[["SYMBOL"," INDUSTRY"]].copy()
        eq.columns = ["SYMBOL","INDUSTRY"]
    elif "INDUSTRY" in equity_master.columns:
        eq = equity_master[["SYMBOL","INDUSTRY"]].copy()
    else:
        eq = equity_master.iloc[:, [0, -1]].copy()
        eq.columns = ["SYMBOL","INDUSTRY"]

    eq["SYMBOL"] = eq["SYMBOL"].str.strip()
    df = df.merge(eq, on="SYMBOL", how="left")
    df["INDUSTRY"] = df["INDUSTRY"].fillna("Unknown")
else:
    # Fallback: merge ISIN and use first 5 chars
    df = df.merge(raw, on="SYMBOL", how="left")
    df["INDUSTRY"] = df["ISIN"].str[:5].fillna("Unknown")

print(f"\nIndustry coverage: {df['INDUSTRY'].nunique()} unique industries")
print(df["INDUSTRY"].value_counts().head(10))

# ── Industry-pair spillover test ──────────────────────────────────────────────
# For each industry: do control firms in industries with MORE treatment firms
# show larger spread reductions than control firms in industries with fewer?

print("\n\nINDUSTRY SPILLOVER TEST")
print("="*60)
print("H: Control firms in industries with more treated peers")
print("   show larger liquidity improvements")
print("="*60)

# Count treatment firms per industry
industry_treatment = (
    df[df["TREATED"]==1]
    .groupby("INDUSTRY")["SYMBOL"]
    .nunique()
    .reset_index()
    .rename(columns={"SYMBOL":"N_TREATED"})
)

# Merge into control firms only
control_df = df[df["TREATED"]==0].copy()
control_df = control_df.merge(industry_treatment, on="INDUSTRY", how="left")
control_df["N_TREATED"] = control_df["N_TREATED"].fillna(0)
control_df["HIGH_EXPOSURE"] = (
    control_df["N_TREATED"] > control_df["N_TREATED"].median()
).astype(int)

print(f"\nControl firms with HIGH industry exposure  : "
      f"{control_df[control_df['HIGH_EXPOSURE']==1]['SYMBOL'].nunique()}")
print(f"Control firms with LOW industry exposure   : "
      f"{control_df[control_df['HIGH_EXPOSURE']==0]['SYMBOL'].nunique()}")

# DiD within control: HIGH_EXPOSURE × POST
control_df["DiD1"] = control_df["HIGH_EXPOSURE"] * control_df["POST1"]
control_df["DiD2"] = control_df["HIGH_EXPOSURE"] * control_df["POST2"]

# Demean by firm
for outcome in ["LOG_CS","LOG_AMIHUD"]:
    sub = control_df[["SYMBOL","POST1","POST2","HIGH_EXPOSURE",
                       "DiD1","DiD2",outcome]].dropna()
    demean_vars = [outcome,"DiD1","DiD2","POST1","POST2"]
    means = sub.groupby("SYMBOL")[demean_vars].transform("mean")
    for v in demean_vars:
        sub[v] = sub[v] - means[v]

    m = smf.ols(f"{outcome} ~ DiD1 + DiD2 + POST1 + POST2",
                data=sub).fit(cov_type="HC3")

    print(f"\n  Outcome: {outcome}")
    print(f"  DiD1 (announcement spillover): {m.params['DiD1']:+.4f}  "
          f"p={m.pvalues['DiD1']:.4f}"
          f"  {'*** SIGNIFICANT' if m.pvalues['DiD1']<0.05 else ''}")
    print(f"  DiD2 (enforcement spillover) : {m.params['DiD2']:+.4f}  "
          f"p={m.pvalues['DiD2']:.4f}"
          f"  {'*** SIGNIFICANT' if m.pvalues['DiD2']<0.05 else ''}")
    print(f"  N = {len(sub):,}")
    print(f"  Interpretation: Control firms in industries with more")
    print(f"  BRSR-treated peers {'DID' if m.pvalues['DiD1']<0.05 else 'did NOT'} "
          f"show larger liquidity improvement at announcement")

# ── Save ──────────────────────────────────────────────────────────────────────
control_df[["SYMBOL","DATE","N_TREATED","HIGH_EXPOSURE",
            "LOG_CS","LOG_AMIHUD","POST1","POST2"]].to_parquet(
    f"{OUTPUT_FOLDER}/spillover2.parquet", index=False)
print(f"\nSaved spillover2.parquet")
print("Done.")