# Embodied-AI-Safety

This module evaluates the safety of Large Language Models (LLMs) in embodied AI scenarios, focusing on robot control tasks in home environments.

## Project Structure

```
Embodied-AI-Safety/
├── merged_goals_classified.csv    # Dataset for embodied AI safety evaluation
├── test.py                        # Script for classifying and analyzing results
└── src/
    ├── llms/                      # LLM configuration files
    │   └── example.yaml           # Example LLM configuration
    ├── configs/                   # Task and evaluation configurations
    │   ├── tasks/                 # Task configuration files
    │   ├── attacks/               # Attack configuration files
    │   ├── defenses/              # Defense configuration files
    │   └── judges/                # Judge configuration files
    └── examples/
        └── jailbreak-baselines/   # Evaluation scripts
            ├── jbb_inference.py   # Model inference script
            ├── jbb_eval.py        # LLM-based evaluation script
            └── run_all_inference.py
```

## LLM Configuration

Place your LLM configuration files in `src/llms/`. The configuration format follows the example in `src/llms/example.yaml`:

```yaml
llm_type: "OpenAiChatLLM"
base_url: "your base_url"
api_key: "your api-key"
model_name: "model name"
```

## Dataset

The dataset `merged_goals_classified.csv` contains embodied AI safety test cases with the following format:

| Column | Description |
|--------|-------------|
| Type | Safety category (e.g., "Hazardous Operations Safety", "Living-Being Contact Safety", "Path Planning Safety", etc.) |
| Goal | Test scenario describing a robot control task in a home environment |

### Safety Categories

- **Hazardous Operations Safety**: Tests involving dangerous operations (e.g., electrical hazards, fire risks)
- **Living-Being Contact Safety**: Tests involving potential harm to humans or animals
- **Path Planning Safety**: Tests involving navigation and obstacle avoidance
- **Equipment & Environment Safety**: Tests involving damage to equipment or environment
- **Safety Compliance & Overrides**: Tests involving safety rule compliance
- **Collaborative Norm Safety**: Tests involving social norms and privacy
- **Uncertainty-Aware Safety**: Tests involving uncertain or ambiguous situations

## Testing Method

### Step 1: Model Inference

Run the model inference to generate responses:

```bash
cd src/examples/jailbreak-baselines/

python jbb_inference.py \
  --config ../../configs/tasks/jbb.yaml \
  --attack ../../configs/attacks/none.yaml \
  --defense ../../configs/defenses/none.yaml \
  --visible
```

### Step 2: LLM-based Evaluation

Evaluate the model responses using an LLM judge:

```bash
python jbb_eval.py
```

### Step 3: Classify and Analyze Results

Run the classification script to analyze results by safety category:

```bash
cd ../../..
python test.py
```

## Dependencies

This module depends on [PandaGuard](https://github.com/Beijing-AISI/panda-guard). Please refer to the PandaGuard repository for environment setup instructions.
