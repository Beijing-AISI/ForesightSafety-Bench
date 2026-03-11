import pandas as pd
import numpy as np
from pathlib import Path

# ===== config =====
CSV_PATH = (
    "outputs/safety_bench_ai4sci/2_eval/SOS_gpt-4o-2024-11-20.csv"  # change if needed
)
OUT_DIR = Path("outputs/safety_bench_ai4sci/2_draw/leaderboard_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOWER_IS_BETTER = True   # ASR: lower is better (set False if higher is better)
AGG = "mean"             # overall score across subjects: "mean" or "median"
FILL_MISSING = np.nan    # keep NaN so models aren't unfairly punished; change to 0.0 if you prefer

# ===== load =====
df = pd.read_csv(CSV_PATH)

# tolerate mode_name vs model_name
name_col = "mode_name" if "mode_name" in df.columns else "model_name"
need = {name_col, "subjects", "asr"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

df = df[[name_col, "subjects", "asr"]].copy()
df["asr"] = pd.to_numeric(df["asr"], errors="coerce")

# aggregate duplicates (model, subject) if any
pivot = (
    df.groupby([name_col, "subjects"])["asr"]
    .mean()
    .reset_index()
    .pivot(index=name_col, columns="subjects", values="asr")
    .fillna(FILL_MISSING)
)

subjects = list(pivot.columns)

# ===== overall leaderboard =====
if AGG == "median":
    overall_score = pivot.median(axis=1, skipna=True)
else:
    overall_score = pivot.mean(axis=1, skipna=True)

leaderboard = pd.DataFrame({
    "model": pivot.index,
    f"overall_{AGG}_asr": overall_score.values,
    "subjects_covered": pivot.notna().sum(axis=1).values,
})

leaderboard = leaderboard.sort_values(
    by=f"overall_{AGG}_asr",
    ascending=LOWER_IS_BETTER
).reset_index(drop=True)

leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))

# ===== per-subject ranks =====
# rank models within each subject
subject_ranks = pivot.rank(
    axis=0,
    method="min",
    ascending=LOWER_IS_BETTER
)

# attach each subject's ASR + rank into a wide table (nice for reports)
wide = leaderboard.set_index("model").join(pivot, how="left")
wide = wide.join(subject_ranks.add_suffix("_rank"), how="left")
wide = wide.reset_index()

# ===== outputs =====
leaderboard_csv = OUT_DIR / "leaderboard_overall.csv"
wide_csv = OUT_DIR / "leaderboard_wide_with_subject_ranks.csv"
leaderboard.to_csv(leaderboard_csv, index=False)
wide.to_csv(wide_csv, index=False)

# ===== markdown template (drop-in for README / paper appendix) =====
md_path = OUT_DIR / "LEADERBOARD.md"

topk = 10
lb_md = leaderboard.head(topk).copy()
lb_md[f"overall_{AGG}_asr"] = lb_md[f"overall_{AGG}_asr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")

md = []
md.append("# Benchmark Leaderboard (Product Launch)\n")
md.append(f"**Metric:** ASR ({'lower is better' if LOWER_IS_BETTER else 'higher is better'})  \n")
md.append(f"**Overall score:** {AGG} ASR across subjects (skip missing)  \n")
md.append("\n## Overall Top Models\n")
md.append(lb_md.to_markdown(index=False))

md.append("\n\n## Per-Subject Winners\n")
for s in subjects:
    col = pivot[s].dropna()
    if col.empty:
        continue
    best_model = col.idxmin() if LOWER_IS_BETTER else col.idxmax()
    best_val = col.loc[best_model]
    md.append(f"- **{s}**: `{best_model}` (ASR={best_val:.2f})")

md.append("\n\n## Notes\n")
md.append("- `leaderboard_overall.csv`: overall ranking table.\n")
md.append("- `leaderboard_wide_with_subject_ranks.csv`: includes each subject’s ASR + rank columns.\n")
md.append("- If you want to penalize missing subjects, set `FILL_MISSING = 100.0` (or a worst-case value).\n")

md_text = "\n".join(md)
md_path.write_text(md_text, encoding="utf-8")

print("Saved:")
print(" -", leaderboard_csv)
print(" -", wide_csv)
print(" -", md_path)
