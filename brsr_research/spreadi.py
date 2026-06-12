import pandas as pd
import numpy as np

OUTPUT_FOLDER = r"./output"

print("Loading sample...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/sample.parquet")
print(f"Loaded {len(df):,} rows\n")

print("Calculating Corwin-Schultz spreads...")

df = df.sort_values(["SYMBOL", "DATE"]).copy()

log_hl  = np.log(df["HIGH"] / df["LOW"])
log_hl2 = log_hl ** 2

df["beta"]  = log_hl2 + log_hl2.groupby(df["SYMBOL"]).shift(1)
df["gamma"] = (np.log(df.groupby("SYMBOL")["HIGH"].transform(lambda x: x.shift(1).combine(x, max)) /
                      df.groupby("SYMBOL")["LOW"].transform(lambda x: x.shift(1).combine(x, min)))) ** 2

k = 3 - 2 * np.sqrt(2)
df["alpha"]   = (np.sqrt(2 * df["beta"]) - np.sqrt(df["beta"])) / k - np.sqrt(df["gamma"] / k)
df["CS_SPREAD"] = (2 * (np.exp(df["alpha"]) - 1) / (1 + np.exp(df["alpha"]))).clip(lower=0)

result = df.drop(columns=["beta", "gamma", "alpha"]).dropna(subset=["CS_SPREAD"])

print(f"Rows after spread calculation : {len(result):,}")
print(f"Columns                       : {result.columns.tolist()}")
print(f"Mean spread (all)             : {result['CS_SPREAD'].mean():.4f}")
print(f"Mean spread (treatment)       : {result[result['TREATED']==1]['CS_SPREAD'].mean():.4f}")
print(f"Mean spread (control)         : {result[result['TREATED']==0]['CS_SPREAD'].mean():.4f}")

result.to_parquet(f"{OUTPUT_FOLDER}/spreads.parquet", index=False)
print(f"\nSaved spreads.parquet to {OUTPUT_FOLDER}")