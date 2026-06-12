import pandas as pd
import numpy as np

OUTPUT_FOLDER = "./output"

liq = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
rankings = pd.read_parquet(f"{OUTPUT_FOLDER}/rankings.parquet")
liq = liq.merge(rankings[["SYMBOL","RANK"]], on="SYMBOL", how="inner")
liq["LOG_CS"] = np.log(liq["CS_SPREAD"].replace(0, np.nan))

# Monthly averages by group
liq["YM"] = liq["DATE"].dt.to_period("M")

treated = liq[liq["RANK"] <= 1000].groupby("YM")["LOG_CS"].mean().rename("Treated")
ctrl_1  = liq[(liq["RANK"] > 1000) & (liq["RANK"] <= 1250)].groupby("YM")["LOG_CS"].mean().rename("Control_1001_1250")
ctrl_2  = liq[(liq["RANK"] > 1250) & (liq["RANK"] <= 1500)].groupby("YM")["LOG_CS"].mean().rename("Control_1251_1500")

panel = pd.concat([treated, ctrl_1, ctrl_2], axis=1).reset_index()
panel["YM"] = panel["YM"].astype(str)

# Difference between treated and each control over time
panel["DIFF_1"] = panel["Treated"] - panel["Control_1001_1250"]
panel["DIFF_2"] = panel["Treated"] - panel["Control_1251_1500"]

print("Monthly spread gap: Treated minus Control")
print(f"{'Month':<10} {'vs 1001-1250':>14} {'vs 1251-1500':>14}")
print("─" * 42)
for _, row in panel.iterrows():
    marker = " ← Apr 2022" if row["YM"] == "2022-04" else (
             " ← Apr 2023" if row["YM"] == "2023-04" else "")
    print(f"{row['YM']:<10} {row['DIFF_1']:>14.4f} {row['DIFF_2']:>14.4f}{marker}")

print()
print("Pre-treatment gap stability (should be constant if parallel trends hold):")
pre = panel[panel["YM"] < "2022-04"]
print(f"  Std of DIFF_1 pre-treatment: {pre['DIFF_1'].std():.4f}")
print(f"  Std of DIFF_2 pre-treatment: {pre['DIFF_2'].std():.4f}")
print()
print("Post-treatment gap change (should widen if BRSR has effect):")
pre_mean_1  = pre["DIFF_1"].mean()
pre_mean_2  = pre["DIFF_2"].mean()
post = panel[panel["YM"] >= "2022-04"]
print(f"  Pre-mean DIFF_1: {pre_mean_1:.4f} | Post-mean DIFF_1: {post['DIFF_1'].mean():.4f} | Change: {post['DIFF_1'].mean()-pre_mean_1:.4f}")
print(f"  Pre-mean DIFF_2: {pre_mean_2:.4f} | Post-mean DIFF_2: {post['DIFF_2'].mean():.4f} | Change: {post['DIFF_2'].mean()-pre_mean_2:.4f}")