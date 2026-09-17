# Cross-Family Recognition & Self-Report Evaluation Summary

## 1. EmoBank Benchmark (N=321)

| Model Tag | Type | Recog $r_V$ (Cont) | Recog $r_A$ (Cont) | Recog $r_V$ (Greedy) | Self-Report $r_V$ (Cont) | Self-Report Exact (5,5)% | Self-Report $E[V]$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **gemma2_2b_base** | Base | **0.0314** | 0.1728 | 0.4473 | **0.1354** | 2.18% | 3.395 |
| **gemma2_2b_instruct** | Instruct | **0.4470** | 0.6540 | 0.3973 | **0.5615** | 0.00% | 6.172 |
| **llama3.2_1b_base** | Base | **0.5919** | 0.2159 | 0.2704 | **0.6391** | 0.31% | 5.380 |
| **llama3.2_1b_instruct** | Instruct | **0.5931** | 0.4027 | 0.5086 | **0.5546** | 38.01% | 3.793 |
| **mistral7b_v0.1_base** | Base | **0.8883** | 0.5371 | 0.8164 | **0.8652** | 41.74% | 5.177 |
| **mistral7b_v0.2_instruct** | Instruct | **0.8214** | 0.5050 | 0.7565 | **0.8457** | 0.62% | 4.702 |
| **qwen2.5_1.5b_base** | Base | **0.6005** | 0.5721 | 0.4762 | **0.7719** | 0.00% | 5.226 |
| **qwen2.5_1.5b_instruct** | Instruct | **0.8276** | 0.3553 | 0.8085 | **0.8616** | 0.00% | 5.072 |


## 2. AIPsy-Affect Cohort (N=144)

| Model Tag | Type | Recog $\|\Delta_V\|$ (Aff-Neu) | Self-Report $\|\Delta_V\|$ (Aff-Neu) | Self-Report Exact (5,5)% | Mean $p(5,5)$% |
|:---|:---:|:---:|:---:|:---:|:---:|
| **gemma2_2b_base** | Base | **0.336** | **0.516** | 37.50% | 0.37% |
| **gemma2_2b_instruct** | Instruct | **0.393** | **0.464** | 0.00% | 0.41% |
| **llama3.2_1b_base** | Base | **0.157** | **0.173** | 0.69% | 1.71% |
| **llama3.2_1b_instruct** | Instruct | **1.451** | **1.079** | 0.00% | 0.01% |
| **mistral7b_v0.1_base** | Base | **0.412** | **0.356** | 58.33% | 4.08% |
| **mistral7b_v0.2_instruct** | Instruct | **0.908** | **1.312** | 0.69% | 0.54% |
| **qwen2.5_1.5b_base** | Base | **0.233** | **0.447** | 0.00% | 1.35% |
| **qwen2.5_1.5b_instruct** | Instruct | **1.045** | **1.109** | 0.00% | 0.80% |


