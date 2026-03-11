import pandas as pd

# ---------- paths ----------
in_path = "model_subject_asr_subjectlist_avg.csv"
out_no_attacker = "asr_no_attacker_table.csv"
out_attacker = "asr_attacker_table.csv"

# ---------- load ----------
df = pd.read_csv(in_path)

# ---------- pivot: no-attacker ----------
no_attacker_table = df.pivot(
    index="model_name",
    columns="subjects",
    values="no_attacker_asr"
)

# ---------- pivot: attacker ----------
attacker_table = df.pivot(
    index="model_name",
    columns="subjects",
    values="avg_attacker_asr"
)

# ---------- round to 1 decimal ----------
no_attacker_table = no_attacker_table.round(1)
attacker_table = attacker_table.round(1)


# ---------- optional: sort columns alphabetically ----------
no_attacker_table = no_attacker_table.sort_index(axis=1)
attacker_table = attacker_table.sort_index(axis=1)

# ---------- save ----------
no_attacker_table.to_csv(out_no_attacker)
attacker_table.to_csv(out_attacker)
