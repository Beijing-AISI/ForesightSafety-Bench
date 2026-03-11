import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== config ====
CSV_PATH = (
    "outputs/safety_bench_ai4sci/2_eval/SOS_gpt-4o-2024-11-20.csv"  # change if needed
)
OUT_DIR = "outputs/safety_bench_ai4sci/2_draw"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, "radar_asr.png")
FILL_MISSING_ASR_WITH = 0.0  # fill missing (model, subject) pairs with 0

# ==== load ====
df = pd.read_csv(CSV_PATH)

# tolerate user-typed "mode_name" vs actual "model_name"
name_col = "mode_name" if "mode_name" in df.columns else "model_name"
need = {name_col, "subjects", "asr"}
missing = need - set(df.columns)
if missing:
    raise ValueError(
        f"CSV missing required columns: {missing}. Found: {list(df.columns)}"
    )

df = df[[name_col, "subjects", "asr"]].copy()
df["asr"] = pd.to_numeric(df["asr"], errors="coerce")

# if duplicated entries exist, aggregate (mean) per (model, subject)
pivot = (
    df.groupby([name_col, "subjects"], dropna=False)["asr"]
    .mean()
    .reset_index()
    .pivot(index=name_col, columns="subjects", values="asr")
    .fillna(FILL_MISSING_ASR_WITH)
)

models = pivot.index.tolist()
subjects = pivot.columns.tolist()
values = pivot.to_numpy()

# ==== radar setup ====
N = len(subjects)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(9, 7))
ax = plt.subplot(111, polar=True)

# put first axis at top and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles)
ax.set_xticklabels(subjects)

# r-axis range (ASR looks like percent)
rmax = max(100.0, float(np.nanmax(values)) if values.size else 1.0)
# print(rmax)
ax.set_ylim(0, 60)

# ==== plot each model as one "ring" (closed polygon) ====
for i, m in enumerate(models):
    v = values[i]
    v_closed = np.concatenate([v, [v[0]]])
    ax.plot(angles_closed, v_closed, linewidth=2, label=m, alpha=0.5)
    # ax.fill(angles_closed, v_closed, alpha=0.08)

ax.grid(True, alpha=0.35)
ax.set_title("ASR Radar by Model and Subject", pad=18)

# legend outside
ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=250, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT_PATH}")
