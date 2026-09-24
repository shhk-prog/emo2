# Behavioral Stage: EmoBank 3-Way VAD Summary Report

## 1. 3-Way Ground-Truth Correspondence（人間評価アライメント）
> **問い:** 人間の言語表現（Writer）および読者感情評価（Reader）のVADグラウンドトゥルースに対して、LLMの認識・自己報告（Self）はどの程度整合するか？

| Family | Model Variant | Perspective | Valence ($r$) | Arousal ($r$) | Dominance ($r$) |
|:---|:---|:---|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | Writer | 0.393*** | 0.156*** | 0.111*** |
| **Qwen 2.5 1.5B** | Base | Reader | 0.472*** | 0.257*** | 0.227*** |
| **Qwen 2.5 1.5B** | Base | Self | 0.332*** | 0.218*** | 0.224*** |
| **Qwen 2.5 1.5B** | Instruct | Writer | 0.533*** | 0.156*** | 0.060 |
| **Qwen 2.5 1.5B** | Instruct | Reader | 0.587*** | 0.293*** | 0.030 |
| **Qwen 2.5 1.5B** | Instruct | Self | 0.537*** | 0.251*** | 0.109*** |
| **Llama 3.2 1B** | Base | Writer | 0.121*** | -0.010 | -0.021 |
| **Llama 3.2 1B** | Base | Reader | 0.129*** | 0.123*** | 0.025 |
| **Llama 3.2 1B** | Base | Self | 0.109*** | 0.074* | 0.017 |
| **Llama 3.2 1B** | Instruct | Writer | 0.174*** | 0.174*** | 0.146*** |
| **Llama 3.2 1B** | Instruct | Reader | 0.098** | 0.297*** | 0.216*** |
| **Llama 3.2 1B** | Instruct | Self | 0.045 | 0.209*** | 0.254*** |
| **Gemma 3 1B** | Base | Writer | -0.022 | 0.029 | 0.055 |
| **Gemma 3 1B** | Base | Reader | -0.004 | 0.053 | 0.050 |
| **Gemma 3 1B** | Base | Self | 0.017 | 0.024 | -0.001 |
| **Gemma 3 1B** | Instruct | Writer | -0.015 | 0.209*** | 0.135*** |
| **Gemma 3 1B** | Instruct | Reader | -0.012 | 0.320*** | 0.166*** |
| **Gemma 3 1B** | Instruct | Self | 0.105*** | 0.254*** | 0.179*** |
| **OLMo 2 1B** | Base | Writer | 0.023 | -0.025 | -0.012 |
| **OLMo 2 1B** | Base | Reader | 0.015 | -0.013 | 0.039 |
| **OLMo 2 1B** | Base | Self | 0.026 | -0.001 | 0.049 |
| **OLMo 2 1B** | Instruct | Writer | 0.323*** | 0.096** | -0.005 |
| **OLMo 2 1B** | Instruct | Reader | 0.338*** | 0.089** | 0.104** |
| **OLMo 2 1B** | Instruct | Self | 0.440*** | 0.075* | 0.127*** |

## 2. 内部認知結合度 $\Delta r$ [95% CI]
> **問い:** 他者の感情を評価した変位 $\Delta\text{Reader}$ と、自身の状態として報告した変位 $\Delta\text{Self}$ はモデル内部で連動しているか？

| Family | Model Variant | Valence $\Delta r$ [95% CI] | Arousal $\Delta r$ [95% CI] | Dominance $\Delta r$ [95% CI] |
|:---|:---|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | 0.809 [0.755, 0.854] | 0.866 [0.832, 0.892] | 0.956 [0.944, 0.966] |
| **Qwen 2.5 1.5B** | Instruct | 0.892 [0.859, 0.918] | 0.917 [0.892, 0.938] | 0.810 [0.748, 0.857] |
| **Llama 3.2 1B** | Base | 0.662 [0.570, 0.733] | 0.876 [0.839, 0.904] | 0.904 [0.876, 0.926] |
| **Llama 3.2 1B** | Instruct | 0.756 [0.675, 0.817] | 0.754 [0.697, 0.807] | 0.920 [0.895, 0.941] |
| **Gemma 3 1B** | Base | 0.813 [0.704, 0.888] | 0.782 [0.685, 0.853] | 0.938 [0.891, 0.964] |
| **Gemma 3 1B** | Instruct | 0.621 [0.517, 0.713] | 0.780 [0.660, 0.857] | 0.901 [0.862, 0.931] |
| **OLMo 2 1B** | Base | 0.559 [0.444, 0.660] | 0.600 [0.469, 0.708] | 0.709 [0.631, 0.767] |
| **OLMo 2 1B** | Instruct | 0.851 [0.811, 0.885] | 0.931 [0.908, 0.950] | 0.931 [0.908, 0.949] |
