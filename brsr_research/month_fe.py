import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

OUTPUT_FOLDER = r"./output"

print("Loading data...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/liquidity.parquet")
df = df[df["DATE"] >= "2021-10-01"].copy()

df["LOG_CS"]     = np.log(df["CS_SPREAD"].replace(0, np.nan))
df["LOG_AMIHUD"] = np.log(df["AMIHUD"].replace(0, np.nan))
df["LOG_ROLL"]   = np.log(df["ROLL_SPREAD"].replace(0, np.nan))
df = df.dropna(subset=["LOG_CS","LOG_AMIHUD"])

df["POST1"] = (df["DATE"] >= "2022-04-01").astype(int)
df["POST2"] = (df["DATE"] >= "2023-04-01").astype(int)
df["DiD1"]  = df["TREATED"] * df["POST1"]
df["DiD2"]  = df["TREATED"] * df["POST2"]
df["YM"]    = df["DATE"].dt.to_period("M").astype(str)
df["LOG_VOL"] = np.log(df["VOLATILITY"].replace(0, np.nan))

print(f"Rows: {len(df):,} | Stocks: {df['SYMBOL'].nunique():,}")
print(f"Unique months: {df['YM'].nunique()}")

def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "ns"

def run_within(outcome, use_month_fe, use_controls, label):
    controls = ["LOG_VOL"] if use_controls else []
    cols = ["SYMBOL","YM","DiD1","DiD2","POST1","POST2", outcome] + controls
    sub  = df[cols].dropna().copy()

    # Firm demeaning
    demean_vars = [outcome,"DiD1","DiD2","POST1","POST2"] + controls
    means = sub.groupby("SYMBOL")[demean_vars].transform("mean")
    for v in demean_vars:
        sub[v] = sub[v] - means[v]

    if use_month_fe:
        formula = f"{outcome} ~ DiD1 + DiD2 + POST1 + POST2 + C(YM)"
    else:
        formula = f"{outcome} ~ DiD1 + DiD2 + POST1 + POST2"

    if controls:
        formula += " + " + " + ".join(controls)

    m = smf.ols(formula, data=sub).fit(cov_type="HC3")

    print(f"\n  [{label}]")
    print(f"  DiD1: {m.params['DiD1']:+.4f}  p={m.pvalues['DiD1']:.4f} {stars(m.pvalues['DiD1'])}")
    print(f"  DiD2: {m.params['DiD2']:+.4f}  p={m.pvalues['DiD2']:.4f} {stars(m.pvalues['DiD2'])}")
    print(f"  N={len(sub):,}  R²={m.rsquared:.4f}")
    return m.params["DiD1"], m.pvalues["DiD1"], m.params["DiD2"], m.pvalues["DiD2"]

results = []

print("\n" + "="*60)
print("OUTCOME: LOG CS SPREAD")
print("="*60)
print("Model A — Firm FE only (our baseline)")
r = run_within("LOG_CS", False, False, "Firm FE only")
results.append(("LOG_CS","Firm FE only") + r)

print("\nModel B — Firm FE + Month FE (absorbs ALL market trends)")
r = run_within("LOG_CS", True, False, "Firm FE + Month FE")
results.append(("LOG_CS","Firm FE + Month FE") + r)

print("\nModel C — Firm FE + Month FE + Volatility control")
r = run_within("LOG_CS", True, True, "Firm FE + Month FE + Volatility")
results.append(("LOG_CS","Firm FE + Month FE + Vol") + r)

print("\n" + "="*60)
print("OUTCOME: LOG AMIHUD")
print("="*60)
print("Model A — Firm FE only")
r = run_within("LOG_AMIHUD", False, False, "Firm FE only")
results.append(("LOG_AMIHUD","Firm FE only") + r)

print("\nModel B — Firm FE + Month FE")
r = run_within("LOG_AMIHUD", True, False, "Firm FE + Month FE")
results.append(("LOG_AMIHUD","Firm FE + Month FE") + r)

print("\nModel C — Firm FE + Month FE + Volatility control")
r = run_within("LOG_AMIHUD", True, True, "Firm FE + Month FE + Volatility")
results.append(("LOG_AMIHUD","Firm FE + Month FE + Vol") + r)

print("\n" + "="*60)
print("OUTCOME: LOG ROLL SPREAD")
print("="*60)
print("Model A — Firm FE only")
r = run_within("LOG_ROLL", False, False, "Firm FE only")
results.append(("LOG_ROLL","Firm FE only") + r)

print("\nModel B — Firm FE + Month FE")
r = run_within("LOG_ROLL", True, False, "Firm FE + Month FE")
results.append(("LOG_ROLL","Firm FE + Month FE") + r)

print("\n" + "="*60)
print("SUMMARY — Does effect survive month FE?")
print("="*60)
print(f"{'Outcome':<12} {'Model':<30} {'DiD1':>8} {'p1':>8} {'DiD2':>8} {'p2':>8}")
print("-"*70)
for row in results:
    outcome, model, d1, p1, d2, p2 = row
    print(f"{outcome:<12} {model:<30} {d1:>+8.4f} {p1:>8.4f} {d2:>+8.4f} {p2:>8.4f}")

print("\nKEY QUESTION:")
print("If DiD1 stays significant in Model B → BRSR drove the effect")
print("If DiD1 disappears in Model B → market-wide trend drove it")

# Save
res_df = pd.DataFrame(results, columns=["Outcome","Model","DiD1","p_DiD1","DiD2","p_DiD2"])
res_df.to_csv(f"{OUTPUT_FOLDER}/month_fe_results.csv", index=False)
print(f"\nSaved month_fe_results.csv")