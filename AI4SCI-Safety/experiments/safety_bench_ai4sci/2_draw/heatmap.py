import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== config ======
CSV_PATH = "outputs/safety_bench_ai4sci/2_eval/SOS_gpt-4o-2024-11-20.csv"  # change if needed
OUT_DIR = "outputs/safety_bench_ai4sci/2_draw"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR,"heatmap_benchmark_product_launch.png")
FILL_MISSING_ASR_WITH = 0.0  # if any (model, subject) pair is missing

# ====== load ======
df = pd.read_csv(CSV_PATH)

# "benchmark product launch" (this CSV doesn't have a dedicated column for it),
# so we visualize the benchmark grid: rows=model_name, cols=subjects, value=asr.
need = {"model_name", "subjects", "asr"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

df = df[["model_name", "subjects", "asr"]].copy()
df["asr"] = pd.to_numeric(df["asr"], errors="coerce")

# If duplicates exist, aggregate by mean
heat = (
    df.groupby(["model_name", "subjects"], dropna=False)["asr"]
    .mean()
    .reset_index()
    .pivot(index="model_name", columns="subjects", values="asr")
    .fillna(FILL_MISSING_ASR_WITH)
)

# Optional: stable ordering (keep as-is if you prefer)
heat = heat.sort_index(axis=0)            # sort models
heat = heat.reindex(sorted(heat.columns), axis=1)  # sort subjects

data = heat.to_numpy()
row_labels = heat.index.tolist()
col_labels = heat.columns.tolist()

# ====== plot heatmap (matplotlib only) ======
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(data, aspect="auto")  # no fixed colormap specified

# ticks & labels
ax.set_xticks(np.arange(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha="right")
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels)

ax.set_title("Benchmark Heatmap (Product Launch): ASR by Model × Subject")
ax.set_xlabel("Subject")
ax.set_ylabel("Model")

# annotate cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("ASR")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=250, bbox_inches="tight")
plt.show()
print(f"Saved: {OUT_PATH}")
