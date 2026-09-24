# V1 Stage: Internal Representation and Causal Sharing Complete Summary Report

## 1. Linear Probe Decodability and Peak Localization (E1)

| Family | Variant | Task | Valence Layer (d*) | Valence R2 (r) | Arousal Layer (d*) | Arousal R2 (r) | Reader-Self Delta d* |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | Reader | L22 (0.81) | 0.378 (0.617) | L24 (0.89) | 0.130 (0.369) | V:0.00, A:0.07 |
| **Qwen 2.5 1.5B** | Base | Self   | L22 (0.81) | 0.391 (0.626) | L26 (0.96) | 0.123 (0.356) | |
| **Qwen 2.5 1.5B** | Instruct | Reader | L15 (0.56) | 0.215 (0.467) | L17 (0.63) | 0.073 (0.301) | V:0.07, A:0.11 |
| **Qwen 2.5 1.5B** | Instruct | Self   | L13 (0.48) | 0.285 (0.536) | L20 (0.74) | 0.075 (0.300) | |
| **Llama 3.2 1B** | Base | Reader | L14 (0.93) | 0.239 (0.490) | L4 (0.27) | 0.068 (0.288) | V:0.20, A:0.00 |
| **Llama 3.2 1B** | Base | Self   | L11 (0.73) | 0.243 (0.494) | L4 (0.27) | 0.065 (0.282) | |
| **Llama 3.2 1B** | Instruct | Reader | L6 (0.40) | 0.317 (0.564) | L8 (0.53) | 0.113 (0.352) | V:0.13, A:0.00 |
| **Llama 3.2 1B** | Instruct | Self   | L8 (0.53) | 0.355 (0.597) | L8 (0.53) | 0.106 (0.346) | |
| **Gemma 3 1B** | Base | Reader | L16 (0.64) | 0.331 (0.576) | L15 (0.60) | 0.043 (0.251) | V:0.00, A:0.04 |
| **Gemma 3 1B** | Base | Self   | L16 (0.64) | 0.330 (0.576) | L16 (0.64) | 0.044 (0.254) | |
| **Gemma 3 1B** | Instruct | Reader | L12 (0.48) | 0.301 (0.552) | L12 (0.48) | 0.094 (0.323) | V:0.08, A:0.36 |
| **Gemma 3 1B** | Instruct | Self   | L14 (0.56) | 0.375 (0.613) | L21 (0.84) | 0.088 (0.321) | |
| **OLMo 2 1B** | Base | Reader | L14 (0.93) | 0.418 (0.647) | L7 (0.47) | 0.089 (0.314) | V:0.00, A:0.13 |
| **OLMo 2 1B** | Base | Self   | L14 (0.93) | 0.458 (0.677) | L9 (0.60) | 0.106 (0.338) | |
| **OLMo 2 1B** | Instruct | Reader | L7 (0.47) | 0.327 (0.573) | L11 (0.73) | 0.085 (0.307) | V:0.00, A:0.27 |
| **OLMo 2 1B** | Instruct | Self   | L7 (0.47) | 0.352 (0.594) | L7 (0.47) | 0.086 (0.310) | |

## 2. Shared Causal Map and Circuit Sharing (E3)

| Family | Variant | Reader Peak (d*) | Self Peak (d*) | Reader Mag (x10^-3) | Self Mag (x10^-3) | Spearman rho | Mean Cosine |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | L16 (0.59) | L16 (0.59) | 2.56 | 2.46 | 0.805 | 0.390 |
| **Qwen 2.5 1.5B** | Instruct | L1 (0.04) | L8 (0.30) | 0.98 | 0.86 | 0.799 | 0.000 |
| **Llama 3.2 1B** | Base | L4 (0.27) | L6 (0.40) | 0.75 | 1.19 | 0.582 | 0.000 |
| **Llama 3.2 1B** | Instruct | L6 (0.40) | L6 (0.40) | 2.19 | 1.92 | 0.882 | 0.414 |
| **Gemma 3 1B** | Base | L2 (0.08) | L2 (0.08) | 2.87 | 3.70 | 0.802 | 0.712 |
| **Gemma 3 1B** | Instruct | L11 (0.44) | L11 (0.44) | 4.89 | 8.21 | 0.646 | 0.266 |
| **OLMo 2 1B** | Base | L4 (0.27) | L4 (0.27) | 6.14 | 1.66 | 0.715 | 0.210 |
| **OLMo 2 1B** | Instruct | L10 (0.67) | L10 (0.67) | 0.82 | 1.17 | 0.771 | 0.000 |
