import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

OUTPUT_FOLDER = r"./output"

print("Loading spreads...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/spreads.parquet")

# Trim COVID-affected period (India's second wave distorted microstructure)
df = df[df["DATE"] >= "2021-10-01"].copy()
print(f"After trimming pre-COVID rows: {len(df):,}")

df["LOG_SPREAD"] = np.log(df["CS_SPREAD"].replace(0, np.nan))
df = df.dropna(subset=["LOG_SPREAD"])

# ── Event study: quarters relative to Apr 2022 announcement ──────────────────
df["QUARTER"] = pd.PeriodIndex(df["DATE"], freq="Q")
base = pd.Period("2022Q1", freq="Q")
df["REL_Q"] = (df["QUARTER"] - base).apply(lambda x: x.n)

# Keep -4 to +4 quarters around announcement
df_es = df[(df["REL_Q"] >= -4) & (df["REL_Q"] <= 4)].copy()

# Rename to avoid patsy parsing Q-4 as subtraction
# Q_m4 = Q-4, Q_m3 = Q-3, Q_m2 = Q-2, Q_0 = Q0, etc.
def qname(q):
    if q < 0:
        return f"Q_m{abs(q)}"
    else:
        return f"Q_p{q}"

quarters = [q for q in range(-4, 5) if q != -1]
for q in quarters:
    df_es[qname(q)] = ((df_es["REL_Q"] == q) & (df_es["TREATED"] == 1)).astype(int)

q_vars = [qname(q) for q in quarters]
formula = "LOG_SPREAD ~ " + " + ".join(q_vars) + " + TREATED"

# Demean by firm
demean_vars = ["LOG_SPREAD"] + q_vars
df_es2 = df_es[demean_vars + ["SYMBOL", "TREATED"]].copy()
firm_means = df_es2.groupby("SYMBOL")[demean_vars].transform("mean")
for v in demean_vars:
    df_es2[v] = df_es2[v] - firm_means[v]

es_model = smf.ols(formula, data=df_es2).fit(cov_type="HC3")

# ── Plot event study ──────────────────────────────────────────────────────────
all_quarters = list(range(-4, 5))
all_coefs, all_ci_low, all_ci_high = [], [], []

for q in all_quarters:
    if q == -1:
        all_coefs.append(0)
        all_ci_low.append(0)
        all_ci_high.append(0)
    else:
        name = qname(q)
        all_coefs.append(es_model.params[name])
        all_ci_low.append(es_model.conf_int().loc[name, 0])
        all_ci_high.append(es_model.conf_int().loc[name, 1])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(all_quarters, all_coefs, "o-", color="#1f77b4", linewidth=2, label="DiD estimate")
ax.fill_between(all_quarters, all_ci_low, all_ci_high, alpha=0.2, color="#1f77b4", label="95% CI")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(-0.5, color="red", linewidth=1.5, linestyle=":", label="BRSR Announced (Apr 2022)")
ax.set_xlabel("Quarters Relative to BRSR Announcement", fontsize=12)
ax.set_ylabel("DiD Coefficient (Log Spread)", fontsize=12)
ax.set_title("Event Study: Pre-trends Test\nTreatment vs Control Bid-Ask Spreads", fontsize=13, fontweight="bold")
ax.set_xticks(all_quarters)
ax.set_xticklabels([f"Q{q:+d}" for q in all_quarters])
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/pretrends.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved pretrends.png")

# ── Pre-trend Wald test ───────────────────────────────────────────────────────
wald = es_model.f_test([f"{qname(q)} = 0" for q in range(-4, -1)])
print(f"\nPre-trends Wald test (H0: no pre-trends):")
print(f"  F-stat  : {float(wald.statistic):.4f}")
print(f"  P-value : {float(wald.pvalue):.4f}")
print("\nIf p > 0.05, parallel pre-trends assumption holds.")