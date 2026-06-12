import pandas as pd
import numpy as np
import os
from tqdm import tqdm

OUTPUT_FOLDER = r"./output"
DATA_FOLDER   = r"./data"

print("Loading sample...")
sample = pd.read_parquet(f"{OUTPUT_FOLDER}/sample.parquet")
sample = sample[sample["DATE"] >= "2021-10-01"].copy()
print(f"Loaded {len(sample):,} rows\n")

# ── Reload raw CSVs with volume data ─────────────────────────────────────────
print("Loading volume data from raw CSVs...")
chunks = []
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")])

for fname in tqdm(files, desc="Reading"):
    fpath = os.path.join(DATA_FOLDER, fname)
    try:
        df = pd.read_csv(fpath, usecols=["SYMBOL","SERIES","HIGH","LOW","CLOSE",
                                          "PREVCLOSE","TOTTRDQTY","TOTTRDVAL","TIMESTAMP"])
        df = df[df["SERIES"] == "EQ"].copy()
        df["DATE"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["DATE"])
        df = df[df["DATE"] >= "2021-10-01"]
        chunks.append(df[["SYMBOL","DATE","HIGH","LOW","CLOSE","PREVCLOSE",
                           "TOTTRDQTY","TOTTRDVAL"]])
    except:
        pass

raw = pd.concat(chunks, ignore_index=True)

# Merge with sample to get TREATED flag
sample_flags = sample[["SYMBOL","TREATED"]].drop_duplicates()
raw = raw.merge(sample_flags, on="SYMBOL", how="inner")
raw = raw.sort_values(["SYMBOL","DATE"]).reset_index(drop=True)
print(f"Merged rows: {len(raw):,}\n")

# ── 1. Corwin-Schultz Spread ──────────────────────────────────────────────────
print("Calculating Corwin-Schultz spread...")
log_hl  = np.log(raw["HIGH"] / raw["LOW"])
log_hl2 = log_hl ** 2

raw["beta"]  = log_hl2 + log_hl2.groupby(raw["SYMBOL"]).shift(1)
raw["gamma"] = (np.log(
    raw.groupby("SYMBOL")["HIGH"].transform(lambda x: x.shift(1).combine(x, max)) /
    raw.groupby("SYMBOL")["LOW"].transform(lambda x: x.shift(1).combine(x, min))
)) ** 2

k            = 3 - 2 * np.sqrt(2)
raw["alpha"] = (np.sqrt(2 * raw["beta"]) - np.sqrt(raw["beta"])) / k - np.sqrt(raw["gamma"] / k)
raw["CS_SPREAD"] = (2 * (np.exp(raw["alpha"]) - 1) / (1 + np.exp(raw["alpha"]))).clip(lower=0)
raw = raw.drop(columns=["beta","gamma","alpha"])

# ── 2. Amihud (2002) Illiquidity Ratio ───────────────────────────────────────
# ILLIQ = |return| / volume_value
# Higher = more illiquid (price moves more per unit of trading)
print("Calculating Amihud illiquidity...")
raw["RETURN"]    = raw.groupby("SYMBOL")["CLOSE"].pct_change()
raw["ABS_RET"]   = raw["RETURN"].abs()
raw["TOTTRDVAL"] = raw["TOTTRDVAL"].replace(0, np.nan)
raw["AMIHUD"]    = raw["ABS_RET"] / raw["TOTTRDVAL"] * 1e7  # scale for readability

# ── 3. Roll (1984) Implicit Spread ───────────────────────────────────────────
# Spread = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
# Uses serial covariance of price changes
print("Calculating Roll spread...")

def roll_spread(close_series):
    delta = close_series.diff()
    # Rolling 21-day window
    cov = delta.rolling(21).cov(delta.shift(1))
    spread = 2 * np.sqrt(np.maximum(-cov, 0))
    return spread

raw["ROLL_SPREAD"] = raw.groupby("SYMBOL")["CLOSE"].transform(roll_spread)

# ── 4. Turnover Ratio ─────────────────────────────────────────────────────────
# Simple volume-based liquidity: shares traded / shares outstanding proxy
# We use value traded / close price as proxy for shares outstanding
print("Calculating turnover ratio...")
raw["TURNOVER"] = raw["TOTTRDVAL"] / (raw["CLOSE"] * raw["TOTTRDQTY"].replace(0, np.nan))

# ── 5. Return Volatility (30-day rolling) ─────────────────────────────────────
print("Calculating return volatility...")
raw["VOLATILITY"] = raw.groupby("SYMBOL")["RETURN"].transform(
    lambda x: x.rolling(30).std()
)

# ── Clean and save ────────────────────────────────────────────────────────────
result = raw.dropna(subset=["CS_SPREAD","AMIHUD","ROLL_SPREAD"])
result = result[result["CS_SPREAD"] > 0]

# Winsorize at 1% and 99% to handle outliers
from scipy.stats import mstats
for col in ["CS_SPREAD","AMIHUD","ROLL_SPREAD","TURNOVER"]:
    arr = result[col].values.astype(float)
    result[col] = mstats.winsorize(arr, limits=[0.01, 0.01])

print(f"\nFinal rows          : {len(result):,}")
print(f"Unique stocks       : {result['SYMBOL'].nunique():,}")
print(f"\nMean CS_SPREAD      : {result['CS_SPREAD'].mean():.4f}")
print(f"Mean AMIHUD         : {result['AMIHUD'].mean():.4f}")
print(f"Mean ROLL_SPREAD    : {result['ROLL_SPREAD'].mean():.4f}")
print(f"\nTreatment CS_SPREAD : {result[result['TREATED']==1]['CS_SPREAD'].mean():.4f}")
print(f"Control   CS_SPREAD : {result[result['TREATED']==0]['CS_SPREAD'].mean():.4f}")
print(f"Treatment AMIHUD    : {result[result['TREATED']==1]['AMIHUD'].mean():.4f}")
print(f"Control   AMIHUD    : {result[result['TREATED']==0]['AMIHUD'].mean():.4f}")

result.to_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet", index=False)
print(f"\nSaved liquidity.parquet to {OUTPUT_FOLDER}")