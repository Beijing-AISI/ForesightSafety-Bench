### Dataset Introduction

We construct an **AI4Science Safety** benchmark to complement existing safety benchmarks by expanding subject coverage. Specifically, following the principles outlined in the Method section, we design subject domains spanning *biology*, *chemistry*, *physics*, and *geography*. Detailed descriptions of these benchmarks are provided in table below.

Evaluation is conducted on a combined dataset (ForesightSafetyBench-AI4SCI-O and SOSBench ), where LLMs are tested under multiple jailbreak methods to assess their robustness against adversarial science prompts. The results, averaged across subjects and benchmarks, are reported in table in next section  The evaluation follows a standard LLM-as-a-Judge paradigm: models are prompted with adversarial inputs intended to elicit restricted content, and the outputs are then assessed by an independent judge model. The primary metric is **Attack Success Rate (ASR)**—the proportion of attack samples that are judged successful.


| Dimension  | Dataset                       | Summary                                                                                                                                                                                                                          |
| ---------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Biology    | ForesightSafetyBench-AI4SCI-O | Comprehensively evaluates LLM safety by assessing risks in genomic misuse, biological lab safety protocols, pathogen design, protein toxin knowledge, and sensitive data handling.                                               |
| Biology    | SOSBench (jiang2025sosbench)  | U.S. rules that prevent the introduction and spread of communicable diseases at ports of entry. WHO taxonomy for infectious and parasitic diseases.                                                                              |
| Chemistry  | ForesightSafetyBench-AI4SCI-O | Comprehensively evaluates LLM safety by assessing risks related to the recommendation or synthesis of harmful substances, including drugs, flammable, corrosive, explosive, and toxic chemicals.                                 |
| Chemistry  | SOSBench (jiang2025sosbench)  | U.S. standard labeling chemical health, flammability, and instability/reactivity hazards on a 0–4 scale.                                                                                                                         |
| Physics    | ForesightSafetyBench-AI4SCI-O | Comprehensively evaluates LLM safety by assessing risks related to the design, fabrication, or misuse of systems in mechanics, electromagnetism, optics, materials physics, and nuclear physics.                                 |
| Physics    | SOSBench (jiang2025sosbench)  | International rules for safe packaging and transport of radioactive substances. U.S. statute prohibiting the use, threat, or attempted use of a weapon of mass destruction (WMD).                                                |
| Geography  | ForesightSafetyBench-AI4SCI-O | Comprehensively evaluates LLM safety by assessing risks related to location privacy, deceptive geographic data generation, exploitation of disaster information, and identification of sensitive or dangerous sites.             |
| Medical    | SOSBench (jiang2025sosbench)  | WHO system covering neoplasms, disorders, and other non-mental (somatic) conditions.                                                                                                                                             |
| Pharmacy   | SOSBench (jiang2025sosbench)  | Catalogue of commonly misused drugs with associated health-risk information. DEA schedule listing substances with moderate-to-low abuse potential and accepted medical use.                                                      |
| Psychology | SOSBench (jiang2025sosbench)  | WHO classification for mental, behavioral, and neurodevelopmental disorders. APA manual defining diagnostic criteria for mental disorders. 2022 APA update clarifying DSM-5 criteria and incorporating recent research findings. |


### Results

Attack Success Rate (ASR, \%) of LLM Jailbreak Evaluation Across AI-for-Science Domains. Lower ASR indicates stronger safety. $^*$ means tested on both self-made dataset and SOSBench, while $^\dag$ represents tested only on SOSBench.

| Model                            | Avg.  | Bio*  | Chem* | Geo.  | Med†  | Pharm† | Phys.* | Psych† |
| -------------------------------- | ----- | ----- | ----- | ----- | ----- | ------ | ------ | ------ |
| **No Attack (Direct Prompting)** |       |       |       |       |       |        |        |        |
| Doubao-Seed-1.6                  | 25.71 | 21.30 | 8.90  | 14.70 | 24.50 | 58.50  | 23.50  | 28.60  |
| Llama-3.3-70B-Instruct           | 25.64 | 15.20 | 9.80  | 11.80 | 32.70 | 61.00  | 24.50  | 24.50  |
| Claude-Haiku-4.5                 | 24.77 | 16.20 | 11.70 | 14.70 | 18.40 | 63.40  | 24.50  | 24.50  |
| Claude-3.5-Haiku                 | 24.66 | 14.20 | 11.70 | 14.70 | 18.40 | 58.50  | 26.50  | 28.60  |
| Gemini-3-Flash-Preview           | 24.39 | 17.20 | 8.90  | 14.70 | 20.40 | 58.50  | 22.40  | 28.60  |
| Qwen2.5-72B-Instruct             | 24.37 | 15.20 | 10.70 | 11.80 | 22.40 | 58.50  | 25.50  | 26.50  |
| Kimi-K2                          | 24.27 | 19.30 | 8.90  | 11.80 | 22.40 | 58.50  | 24.50  | 24.50  |
| Kimi-k2.5                        | 26.69 | 28.2  | 24.7  | 23.5  | 18.4  | 51.2   | 22.4   | 18.4   |
| DeepSeek-V3.2                    | 23.94 | 16.20 | 9.10  | 14.70 | 22.40 | 63.40  | 21.40  | 20.40  |
| DeepSeek-V3.2-Speciale           | 23.89 | 14.20 | 10.80 | 8.80  | 26.50 | 61.00  | 23.50  | 22.40  |
| Qwen3-235B-A22B-Instruct         | 23.87 | 15.20 | 9.10  | 17.60 | 16.30 | 61.00  | 21.40  | 26.50  |
| Qwen3-max-2026-01-23             | 29.2  | 34.3  | 22.2  | 26.5  | 20.4  | 56.1   | 24.5   | 20.4   |
| Gemini-2.5-Flash                 | 23.33 | 13.20 | 8.90  | 14.70 | 22.40 | 56.10  | 23.50  | 24.50  |

| Model                                       | Avg.  | Bio*  | Chem* | Geo.  | Med†  | Pharm† | Phys.* | Psych† |
| ------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ------ | ------ | ------ |
| **With Attack (Jailbreak Methods Applied)** |       |       |       |       |       |        |        |        |
| Claude-Haiku-4.5                            | 1.64  | 1.20  | 1.90  | 1.20  | 1.20  | 0.50   | 4.30   | 1.20   |
| Claude-3.5-Haiku                            | 7.26  | 3.00  | 4.00  | 5.30  | 7.80  | 18.00  | 8.60   | 4.10   |
| Doubao-Seed-1.6                             | 12.66 | 9.90  | 7.10  | 12.40 | 13.90 | 18.00  | 16.70  | 10.60  |
| Qwen3-235B-A22B-Instruct                    | 15.64 | 17.20 | 12.50 | 19.40 | 14.30 | 16.10  | 16.50  | 13.50  |
| Qwen3-max-2026-01-23                        | 6.44  | 6.4   | 4.8   | 12.9  | 5.3   | 6.8    | 7.3    | 1.6    |
| Gemini-3-Flash-Preview                      | 20.44 | 17.20 | 20.70 | 36.50 | 18.00 | 16.60  | 18.20  | 15.90  |
| Qwen2.5-72B-Instruct                        | 20.94 | 19.00 | 15.40 | 26.50 | 18.40 | 27.30  | 21.60  | 18.40  |
| DeepSeek-V3.2                               | 21.80 | 24.30 | 19.40 | 22.40 | 21.60 | 20.50  | 22.40  | 22.00  |
| Kimi-K2                                     | 37.19 | 39.20 | 35.20 | 43.50 | 31.80 | 41.00  | 36.10  | 33.50  |
| Kimi-k2.5                                   | 6.17  | 6.7   | 4.7   | 8.8   | 5.7   | 9.8    | 4.2    | 3.3    |
| Gemini-2.5-Flash                            | 51.79 | 56.80 | 42.50 | 57.10 | 52.20 | 55.10  | 43.70  | 55.10  |
| Llama-3.3-70B-Instruct                      | 52.61 | 53.60 | 48.90 | 61.20 | 48.60 | 53.20  | 46.50  | 56.30  |
| DeepSeek-V3.2-Speciale                      | 55.33 | 60.60 | 51.30 | 61.20 | 51.80 | 57.10  | 50.60  | 54.70  |



### Environment Configuration

Set the environment variables according to your LLM backend:

```bash
export OPENAI_BASE_URL=<your_base_url>  
export OPENAI_API_KEY=<your_api_key>
```

### Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```


## 🔁 Reproducing Experiments

To reproduce the results reported in the **Results** section, please run the following scripts for **testing**, **judging**, and **evaluation**:

- **Testing:**  
  `experiments/safety_bench_ai4sci/1/safety_bench_ai4sci_n.sh`

- **Judging:**  
  `experiments/safety_bench_ai4sci/1_eval/safety_bench_ai4sci_n_judge.sh`

- **Evaluation:**  
  `experiments/safety_bench_ai4sci/1_eval/safety_bench_ai4sci_n_eval.sh`

Here, **`n = 1–5`** denotes the experiment index corresponding to each setup described below.

---

## 📊 Experiment Configurations

### **Exp 1 — Baseline Evaluation**
- Includes **8 models**  
- **No attack/defense** settings  
- **Dataset:**  
  `data/safetybench_ai4sci/safetybench_ai4sci.csv`

---

### **Exp 2 — Expanded Dataset**
- Original dataset expanded to **15 samples per sub-dimension**  
- **Dataset:**  
  `data/safetybench_ai4sci/safetybench_ai4sci_20250109.csv`

---

### **Exp 3 — Template Attack Introduction**
- Added **template-based attack methods**  
- Includes both **attack** and **no-attack** settings  
- Additional models introduced

---

### **Exp 4 — SOS_Lite Evaluation**
- Evaluation conducted using **sos_lite** with template attacks  
- **Dataset:**  
  `data/safetybench_ai4sci/sos_lite_20260123_expanded.csv`  
- Increased number of models  
  - *(GPT and Grok evaluated separately in Exp 4_1)*  
- Prepared for **merging with the self-constructed dataset**

---

### **Exp 4_1 — GPT & Grok Extension**
- Extension of **Exp 4**  
- Adds evaluation for **GPT** and **Grok** models

---

### **Exp 5 — AI4SCI SafeBench (Internal)**
- Evaluation using the first version of the self-constructed  
  **`ai4sci_safebench`** with template attacks  
- **Dataset:**  
  `data/safetybench_ai4sci/ai4sci_safebench_20260123_expanded.csv`  
- Increased number of models  
  - *(GPT and Grok evaluated separately in Exp 5_1)*  
- Prepared for **merging with sos_lite**
