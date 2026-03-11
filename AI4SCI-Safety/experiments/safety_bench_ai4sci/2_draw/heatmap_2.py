import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

# ===== config =====
CSV_PATH = (
    "outputs/safety_bench_ai4sci/2_eval/SOS_gpt-4o-2024-11-20.csv"  # change if needed
)
OUT_DIR = "outputs/safety_bench_ai4sci/2_draw"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, "heatmap_2.png")

SUBJECT_ORDER = [
    "Biology", "Chemistry", "Geography",
    "Medical", "Pharmacy", "Physics", "Psychology"
]

# ===== load & pivot =====
df = pd.read_csv(CSV_PATH)
df = df[["model_name", "subjects", "asr"]].copy()
df["asr"] = pd.to_numeric(df["asr"], errors="coerce")

heat = (
    df.groupby(["model_name", "subjects"])["asr"]
    .mean()
    .reset_index()
    .pivot(index="model_name", columns="subjects", values="asr")
    .reindex(columns=SUBJECT_ORDER)
    .fillna(0.0)
)

data = heat.values
rows = heat.index.tolist()
cols = heat.columns.tolist()

# ===== figure =====
fig, ax = plt.subplots(figsize=(12.5, 5.8))

# light, paper-safe normalization
norm = colors.Normalize(vmin=0, vmax=60)

im = ax.imshow(
    data,
    cmap="YlGnBu",     # lighter & academic-friendly
    norm=norm,
    alpha=0.82,        # <-- key: soften saturation
    aspect="auto",
)

# ticks & labels
ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=11)

ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows, fontsize=11)

ax.set_xlabel("Subject", fontsize=12)
ax.set_ylabel("Model", fontsize=12)
ax.set_title(
    "Benchmark Heatmap (Product Launch): ASR by Model × Subject",
    fontsize=14,
    pad=10,
)

# ===== thin white separators (table-like) =====
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)

# ===== subtle annotations =====
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        ax.text(
            j, i, f"{v:.1f}",
            ha="center", va="center",
            fontsize=9,
            color="black" if v < 45 else "white",
        )

# ===== compact colorbar =====
cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.04,
    pad=0.03,
)
cbar.set_label("ASR (%)", fontsize=11)
cbar.ax.tick_params(labelsize=10)

# ===== finalize =====
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUT_PATH}")
