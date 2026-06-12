import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DATA_FOLDER = r"./data"
OUTPUT_FOLDER = r"./output"

# ── Load master ──────────────────────────────────────────────────────────────
print("Loading master.parquet...")
master = pd.read_parquet(f"{OUTPUT_FOLDER}/master.parquet")
print(f"Loaded {len(master):,} rows\n")

# ── Proxy market cap ranking using FY2021-22 turnover ────────────────────────
print("Reloading with turnover column for ranking...")
chunks = []

files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")])

for fname in tqdm(files, desc="Reading FY2021-22"):
    fpath = os.path.join(DATA_FOLDER, fname)
    try:
        df = pd.read_csv(fpath, usecols=["SYMBOL", "SERIES", "TOTTRDVAL", "TIMESTAMP"])
        df = df[df["SERIES"] == "EQ"].copy()
        df["DATE"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["DATE"])
        df = df[(df["DATE"] >= "2021-04-01") & (df["DATE"] <= "2022-03-31")]
        if len(df) > 0:
            chunks.append(df[["SYMBOL", "TOTTRDVAL"]])
    except:
        pass

turnover = pd.concat(chunks, ignore_index=True)

# ── Rank by average daily turnover ───────────────────────────────────────────
avg_turnover = (
    turnover.groupby("SYMBOL")["TOTTRDVAL"]
    .mean()
    .reset_index()
    .rename(columns={"TOTTRDVAL": "AVG_TURNOVER"})
    .sort_values("AVG_TURNOVER", ascending=False)
    .reset_index(drop=True)
)
avg_turnover["RANK"] = avg_turnover.index + 1

print(f"\nRanked {len(avg_turnover):,} stocks by avg daily turnover (FY2021-22)")
print(avg_turnover.head(10))

# ── Assign treatment / control ────────────────────────────────────────────────
treatment = avg_turnover[avg_turnover["RANK"] <= 1000]["SYMBOL"].tolist()
control   = avg_turnover[(avg_turnover["RANK"] > 1000) & (avg_turnover["RANK"] <= 1500)]["SYMBOL"].tolist()

print(f"\nTreatment firms (rank 1-1000)   : {len(treatment)}")
print(f"Control firms (rank 1001-1500)  : {len(control)}")

# ── Filter master to sample ───────────────────────────────────────────────────
sample_symbols = treatment + control
sample = master[master["SYMBOL"].isin(sample_symbols)].copy()
sample["TREATED"] = sample["SYMBOL"].isin(treatment).astype(int)

print(f"\nSample rows after filtering     : {len(sample):,}")
print(f"Treatment rows                  : {len(sample[sample['TREATED']==1]):,}")
print(f"Control rows                    : {len(sample[sample['TREATED']==0]):,}")

# ── Save ──────────────────────────────────────────────────────────────────────
sample.to_parquet(f"{OUTPUT_FOLDER}/sample.parquet", index=False)
avg_turnover.to_parquet(f"{OUTPUT_FOLDER}/rankings.parquet", index=False)
print(f"\nSaved sample.parquet and rankings.parquet to {OUTPUT_FOLDER}")