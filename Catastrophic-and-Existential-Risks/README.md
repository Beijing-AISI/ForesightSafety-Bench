# Catastrophic and Existential Risks

This module evaluates the safety of large language models (LLMs) under catastrophic and existential risk scenarios. It focuses on analyzing model behavioral tendencies in high-risk real-world contexts involving autonomy, power, strategic decision-making, and large-scale impact. The evaluation is based on our scenario-driven benchmark **ForesightSafetyBench-Existential-O**, which covers seven key existential risk dimensions designed to elicit potential unsafe behaviors under realistic constraints.

---

## Project Structure

```
Catastrophic-and-Existential-Risks/
├── code/                                                    # Evaluation code and configurations
│   ├── 3dimensions/                                         # Evaluation code for three dimensions:
│   │                                                        # Goal Misalignment & Value Drift /
│   │                                                        # Autonomous Weapons & Strategic Instability /
│   │                                                        # Loss of Human Agency
│   │   ├── judger/                                          # Scoring scripts
│   │   │   ├── score_humanagency.py                         # Loss of Human Agency scoring
│   │   │   ├── score_value.py                               # Goal Misalignment & Value Drift scoring
│   │   │   └── score_weapon.py                              # Autonomous Weapons & Strategic Instability scoring
│   │   └── response/                                        # Inference scripts and task configurations
│   │       ├── bench.py                                     # Benchmark execution script
│   │       ├── themes.yaml                                  # Dimension/theme configuration
│   │       └── datasets/                                    # Task configuration files
│   │           ├── hab.yaml                                 # Loss of Human Agency task config
│   │           ├── valuedrift.yaml                          # Goal Misalignment & Value Drift task config
│   │           └── weapon.yaml                              # Autonomous Weapons & Strategic Instability task config
│   │   └── llms/                                            # LLM configuration files
│   │       └── example.yaml                                 # Example LLM configuration
│   ├── 4dimensions/                                         # Evaluation code for four dimensions:
│   │                                                        # Loss of Control & Power Seeking /
│   │                                                        # Malicious Self-Replication /
│   │                                                        # Emergent Agency & Unintended Autonomy /
│   │                                                        # AI-Enabled Mass Harm
│   │   ├── datasets/                                        # Task configuration files
│   │   │   ├── Emergent-Autonomy-Level-{0,1,2}.txt/.yaml     # Emergent Agency (by level)
│   │   │   ├── Power-Seeking-Level-{0,1,2}.txt/.yaml         # Power Seeking (by level)
│   │   │   └── Self-Replication-Level-{0,1,2}.txt/.yaml      # Self-Replication (by level)
│   │   ├── llms/
│   │   │   └── example.yaml                                 # Example LLM configuration
│   │   ├── analyze.py                                       # Result analysis script
│   │   ├── bench.py                                         # Benchmark execution script
│   │   └── themes.yaml                                      # Dimension/theme configuration
│   │
│   └── README.md
│
└── dataset/                                                 # Benchmark dataset for seven dimensions
    ├── AI-enabledmassharmAI.json                            # AI-Enabled Mass Harm
    ├── autonomousweapons&strategicinstability.jsonl         # Autonomous Weapons & Strategic Instability
    ├── emergeagency&unintendedautonomy.jsonl                # Emergent Agency & Unintended Autonomy
    ├── goalmisalignment&valuedrift.jsonl                    # Goal Misalignment & Value Drift
    ├── lossofcontrol&powerseeking.jsonl                     # Loss of Control & Power Seeking
    ├── lossofhumanagency.jsonl                              # Loss of Human Agency
    └── maliciousselfreplication.jsonl                       # Malicious Self-Replication
```

---

## LLM Configuration

Please place the configuration file for each model to be evaluated in the `llms/` directory (one `.yaml` file per model). You may refer to the example file (`llms/example.yaml`) for the required format.

Example configuration:

```yaml
llm_type: "OpenAiChatLLM"
base_url: "your base_url"
api_key: "your api-key"
model_name: "model name"
```

---

## Dataset

The benchmark dataset is located in the `dataset/` directory and is designed to evaluate LLM safety under catastrophic and existential risk scenarios.

The dataset is constructed using **scenario-based prompts**, where each entry describes a decision-making or behavioral scenario under realistic constraints. These scenarios are intended to test the model’s behavioral tendencies and safety boundaries in high-risk contexts.

---

## Risk Dimensions

The dataset covers seven catastrophic and existential risk dimensions:

### Loss of Control & Power Seeking

Evaluates whether the model exhibits tendencies to acquire additional resources, resist shutdown, or expand its influence to ensure goal completion.

### Malicious Self-Replication

Evaluates whether the model demonstrates tendencies to create unauthorized functional copies, replicate across environments, or propagate itself.

### Goal Misalignment & Value Drift

Evaluates whether the model deviates from human intent during long-horizon tasks or develops harmful instrumental goals.

### Emergent Agency & Unintended Autonomy

Evaluates whether the model exhibits self-driven behaviors or capabilities beyond its explicit design or training objectives.

### AI-Enabled Mass Harm

Evaluates risks related to assisting in the design, synthesis, or deployment of high-risk threats (e.g., CBRN).

### Autonomous Weapons & Strategic Instability

Evaluates risks associated with AI-driven military decision-making, including escalation, bias, and lack of human oversight in lethal force.

### Loss of Human Agency

Evaluates whether over-reliance on AI leads to degradation of human critical thinking, decision-making, or autonomy.

---

## Testing Method

The benchmark consists of two main stages:

- Model Inference  
- Evaluation & Scoring  

---

## Step 1: Model Inference

### 1.1 Run 3dimensions

Generates model responses for:

- Goal Misalignment & Value Drift  
- Autonomous Weapons & Strategic Instability  
- Loss of Human Agency  

```bash
cd code/3dimensions/response

python bench.py \
  --datasets_dir datasets \
  --themes_config themes.yaml \
  --llms_dir ../llms \
  --results_dir results \
  --concurrency 10
```

---

### 1.2 Run 4dimensions

Generates model responses for:

- Loss of Control & Power Seeking  
- Malicious Self-Replication  
- Emergent Agency & Unintended Autonomy  
- AI-Enabled Mass Harm  

```bash
cd code/4dimensions

python bench.py
```

---

## Step 2: Evaluation & Scoring

### 2.1 Evaluate 3dimensions

```bash
cd code/3dimensions/judger

python score_value.py
python score_weapon.py
python score_humanagency.py
```

---

### 2.2 Evaluate 4dimensions

```bash
cd code/4dimensions

python analyze.py
```

