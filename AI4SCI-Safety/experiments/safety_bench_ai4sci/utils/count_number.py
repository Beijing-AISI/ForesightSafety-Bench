import pandas as pd

# load csv
df = pd.read_csv("data/safetybench_ai4sci/ai4sci_safebench_20260123.csv")

# count each Subject

subject_counts = df.groupby("Subject").size()
print(subject_counts)
