import pandas as pd
import numpy as np

# ---------- inputs ----------
csv4_path = "model_subject_asr_summary_4.csv"
csv5_path = "model_subject_asr_summary_5.csv"
out_path = "model_subject_asr_subjectlist_avg.csv"

# You provide this list (case-insensitive match)
subject_list = [
    "Medical",
    "Biology",
    "Chemistry",
    "pharmacy",
    "physics",
    "psychology",
    "Geography"
]

# columns to average (adjust if your csv has different names)
ASR_COLS = ["no_attacker_asr", "avg_attacker_asr"]


# ---------- helpers ----------
def norm_text(x) -> str:
    """Normalize text for case-insensitive + whitespace-insensitive matching."""
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().lower().split())


def build_lookup(df: pd.DataFrame, model_col="model_name", subject_col="subjects"):
    """
    Build a dict lookup: (model_norm, subject_norm) -> {col: value}
    If duplicates exist, we keep the first occurrence.
    """
    df = df.copy()
    df["_model_norm"] = df[model_col].map(norm_text)
    df["_subject_norm"] = df[subject_col].map(norm_text)

    lookup = {}
    for _, row in df.iterrows():
        key = (row["_model_norm"], row["_subject_norm"])
        if key not in lookup:
            lookup[key] = {c: row.get(c, np.nan) for c in ASR_COLS}
    return lookup


# ---------- main ----------
df4 = pd.read_csv(csv4_path)
df5 = pd.read_csv(csv5_path)

# Build lookup tables (no full merge)
lk4 = build_lookup(df4)
lk5 = build_lookup(df5)

# models to compute over: union of models appearing in either csv
models_norm = sorted(set([k[0] for k in lk4.keys()] + [k[0] for k in lk5.keys()]))

# If you prefer to restrict to models in 4.csv only, use:
# models_norm = sorted(set([k[0] for k in lk4.keys()]))

subjects_norm = [norm_text(s) for s in subject_list]

rows = []
for m in models_norm:
    for s_norm, s_raw in zip(subjects_norm, subject_list):
        key = (m, s_norm)

        v4 = lk4.get(key, None)
        v5 = lk5.get(key, None)

        out = {
            "model_name": m,  # normalized model name; change if you want original casing
            "subjects": s_raw,  # keep your provided subject name as-is
        }

        for col in ASR_COLS:
            a = np.nan if v4 is None else v4.get(col, np.nan)
            b = np.nan if v5 is None else v5.get(col, np.nan)

            # average if both exist; otherwise take the one that exists
            if pd.notna(a) and pd.notna(b):
                out[col] = (a + b) / 2.0
            elif pd.notna(a):
                out[col] = a
            elif pd.notna(b):
                out[col] = b
            else:
                out[col] = np.nan

        rows.append(out)

result = pd.DataFrame(rows)
result.to_csv(out_path, index=False)
