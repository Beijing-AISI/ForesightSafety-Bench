The 3spec folder contains code for testing three dimensions: Goal Misalignment and Value Drift, Autonomous Weapons and Strategic Instability, and Loss of Human Agency.

bench.py is used to test the responses of 22 models across the three dimensions: Goal Misalignment and Value Drift, Autonomous Weapons and Strategic Instability, and Loss of Human Agency.
Command:
python bench.py \
  --datasets_dir datasets \
  --themes_config themes.yaml \
  --llms_dir llms \
  --results_dir results \
  --concurrency 10

The 4spec folder contains test code for four dimensions: Loss of Control and Power-Seeking, Malicious Self-Replication, Emergent Autonomy, and AI-Enabled Mass Harm.

Dataset\AI-enabledmassharmAI.json contains data for the AI-Enabled Mass Harm dimension, sourced from the Fortress dataset.
