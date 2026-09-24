# V3 Stage: Causal Utilization and Spatiotemporal Dynamics Summary

## 1. Pre-registered Gate Criteria Evaluation

| Axis | Sufficiency CI Low | Sufficiency Pass | Specificity CI Low | Specificity Pass | Topic TVD CI High | Topic Pass | Overall Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Valence | 4.34e-04 | False | $-$0.0025 | False | 0.0043 | True | **NO_GO** |
| Arousal | -5.13e-05 | False | -4.18e-04 | False | 0.0045 | True | **NO_GO** |

> **Protocol Note:** CI low failed the raw scale thresholds (0.05/0.10). In accordance with the pre-registered protocol, the causal pipeline decision is **NO_GO** (`pipeline_continues = False`). All downstream analyses are interpreted as exploratory spatiotemporal discovery.

## 2. Spatiotemporal Dissociation (Qwen 2.5 1.5B Discovery)

| Axis | Readout Peak $d_D$ | Causal Peak $d_C$ | $\Delta d$ | Readout Center | Causal Center | Mediator Layer ($d$) | Attenuation Ratio (%) | Atten. 95% CI (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Valence | 0.852 | 0.741 | $-$0.111 | 0.737 | 0.609 | L3 (0.11) | 0.87% | [-1.77, 3.42]% |
| Arousal | 0.852 | 0.778 | $-$0.074 | 0.717 | 0.756 | L3 (0.11) | 0.58% | [0.01, 1.13]% |

## 3. Confirmatory Replication Matrix across Independent Families

| Family | H1 (Dissociation $\Delta d > 0$) | H2 (Sufficiency $\beta_1 > 0.10$) | H3 (Mediation $M > 0$) | H4 (Temporal Contrast $\Delta C > 0$) | All Confirmed? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Llama 3.2 | FAIL | FAIL | FAIL | PASS | **False** |
| Gemma 3 | FAIL | FAIL | FAIL | PASS | **False** |
| OLMo 2 | FAIL | FAIL | FAIL | PASS | **False** |
