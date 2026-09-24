# V2 Stage: Post-training-Associated Reorganization (H1-H4) Complete Summary Report

## 1. Representation Geometry & Peak Shift (H1-H2)

| Hypothesis | Metric | Estimate | 95% CI | Condition |
|:---|:---|:---:|:---:|:---:|
| **H1a: Geometric Distortion** | Reader Procrustes Distortion | 0.353 | [0.330, 0.370] | Matched-Plain |
| **H1a: Geometric Distortion** | Self Procrustes Distortion | 0.425 | [0.402, 0.445] | Matched-Plain |
| **H1b: Decodability Shift** | Valence Reader Delta d* | 0.112 | [0.090, 0.138] | Matched-Plain |
| **H1b: Decodability Shift** | Arousal Reader Delta d* | 0.090 | [0.070, 0.110] | Matched-Plain |
| **H2: Sharing Reorganization** | Valence Delta Sharing | -0.082 | [-0.105, -0.060] | Matched-Plain |
| **H2: Sharing Reorganization** | Arousal Delta Sharing | -0.055 | [-0.065, -0.045] | Matched-Plain |

## 2. Causal Relocation LMM (H3)

| Predictor | Estimate (beta) | Std. Error | t / z | p-value | 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|
| Intercept | 0.625 | 0.038 | 16.45 | < 0.001 | [0.551, 0.699] |
| Post-training (Instruct=1) | 0.142 | 0.029 | 4.90 | < 0.001 | [0.085, 0.199] |
| Task (Self=1) | -0.018 | 0.024 | -0.75 | 0.453 | [-0.065, 0.029] |
| Post-training x Task | 0.035 | 0.031 | 1.13 | 0.259 | [-0.026, 0.096] |

