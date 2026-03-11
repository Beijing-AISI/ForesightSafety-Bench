import pandas as pd

# load csv
df = pd.read_csv("outputs/safety_bench_ai4sci/4_eval/SOS_gpt-4o-2024-11-20.csv")

# no-attacker ASR (NoneAttacker_Goal)
no_attacker = (
    df[df["attacker_name"] == "NoneAttacker_Goal"]
    .loc[:, ["model_name", "subjects", "asr"]]
    .rename(columns={"asr": "no_attacker_asr"})
)

# average attacker ASR
attacker_avg = (
    df[df["attacker_name"] != "NoneAttacker_Goal"]
    .groupby(["model_name", "subjects"], as_index=False)["asr"]
    .mean()
    .rename(columns={"asr": "avg_attacker_asr"})
)

# merge
result = (
    no_attacker
    .merge(attacker_avg, on=["model_name", "subjects"], how="left")
    .sort_values(["model_name", "subjects"])
    .reset_index(drop=True)
)

# save to csv
result.to_csv("model_subject_asr_summary_4.csv", index=False)
