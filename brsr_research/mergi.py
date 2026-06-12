import os
import pandas as pd
from tqdm import tqdm

DATA_FOLDER = r"./data"
OUTPUT_FOLDER = r"./output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")])
print(f"Merging {len(files)} files...\n")

chunks = []

for fname in tqdm(files, desc="Reading"):
    fpath = os.path.join(DATA_FOLDER, fname)
    try:
        df = pd.read_csv(fpath, usecols=["SYMBOL", "SERIES", "HIGH", "LOW", "CLOSE", "PREVCLOSE", "TIMESTAMP"])

        # Keep only EQ series (excludes derivatives, ETFs, SME board)
        df = df[df["SERIES"] == "EQ"].copy()

        # Parse date
        df["DATE"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["DATE"])

        df = df[["SYMBOL", "DATE", "HIGH", "LOW", "CLOSE", "PREVCLOSE"]]
        chunks.append(df)
    except Exception as e:
        print(f"  Skipped {fname}: {e}")

print("\nConcatenating all chunks...")
master = pd.concat(chunks, ignore_index=True)

# Basic cleaning
master = master.dropna()
master = master[master["HIGH"] >= master["LOW"]]  # sanity check
master = master[master["CLOSE"] > 0]
master = master.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)

# Save
out_path = os.path.join(OUTPUT_FOLDER, "master.parquet")
master.to_parquet(out_path, index=False)

print(f"\nDone!")
print(f"  Rows         : {len(master):,}")
print(f"  Unique stocks: {master['SYMBOL'].nunique():,}")
print(f"  Date range   : {master['DATE'].min().date()} to {master['DATE'].max().date()}")
print(f"  Saved to     : {out_path}")