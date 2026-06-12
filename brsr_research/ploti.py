import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_FOLDER = r"./output"

print("Loading spreads...")
df = pd.read_parquet(f"{OUTPUT_FOLDER}/spreads.parquet")

# ── Monthly average spreads by group ─────────────────────────────────────────
df["MONTH"] = df["DATE"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df.groupby(["MONTH", "TREATED"])["CS_SPREAD"]
    .mean()
    .reset_index()
)

treatment_m = monthly[monthly["TREATED"] == 1]
control_m   = monthly[monthly["TREATED"] == 0]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(treatment_m["MONTH"], treatment_m["CS_SPREAD"],
        label="Treatment (Top 1000)", color="#1f77b4", linewidth=2)
ax.plot(control_m["MONTH"], control_m["CS_SPREAD"],
        label="Control (Rank 1001-1500)", color="#ff7f0e", linewidth=2, linestyle="--")

# Event lines
ax.axvline(pd.Timestamp("2022-04-01"), color="red",   linestyle=":", linewidth=1.5, label="BRSR Announced (Apr 2022)")
ax.axvline(pd.Timestamp("2023-04-01"), color="green", linestyle=":", linewidth=1.5, label="BRSR Effective (Apr 2023)")

# Shade pre/post periods
ax.axvspan(pd.Timestamp("2021-04-01"), pd.Timestamp("2022-04-01"),
           alpha=0.05, color="gray", label="Pre-announcement")
ax.axvspan(pd.Timestamp("2022-04-01"), pd.Timestamp("2023-04-01"),
           alpha=0.05, color="yellow", label="Announcement-to-enforcement")
ax.axvspan(pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31"),
           alpha=0.05, color="green", label="Post-enforcement")

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Avg Corwin-Schultz Bid-Ask Spread", fontsize=12)
ax.set_title("Bid-Ask Spreads: Treatment vs Control Firms\nAround BRSR Disclosure Mandate",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f"{OUTPUT_FOLDER}/spread_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved spread_plot.png to {OUTPUT_FOLDER}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\nAverage spreads by period:")
df["PERIOD"] = pd.cut(
    df["DATE"],
    bins=[pd.Timestamp("2021-04-01"), pd.Timestamp("2022-04-01"),
          pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")],
    labels=["Pre-announcement", "Announcement-to-enforcement", "Post-enforcement"]
)

summary = (
    df.groupby(["PERIOD", "TREATED"])["CS_SPREAD"]
    .mean()
    .unstack()
    .rename(columns={0: "Control", 1: "Treatment"})
)
summary["Difference"] = summary["Treatment"] - summary["Control"]
print(summary.round(4))