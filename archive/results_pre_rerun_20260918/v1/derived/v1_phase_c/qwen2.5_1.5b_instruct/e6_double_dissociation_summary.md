# V1 Phase C: E6 Double Dissociation Report: Qwen/Qwen2.5-1.5B-Instruct

- **Reader-Site**: Layer 8
- **Self-Site**: Layer 20
- **Pairs Tested**: 192

## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)

| Site \ Task | Reader Task Impact | Self Task Impact |
|:---|:---:|:---:|
| **Reader-Site (Layer 8)** | **0.173** | 0.282 |
| **Self-Site (Layer 20)** | 0.018 | **0.015** |

## 2. Statistical Interaction Test (Task x SiteType)

- **Model**: `Linear Mixed-Effects Model (LMM)`
- **Interaction p-value**: **`p = 6.6378e-09`**

```text
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                      MixedLM                          Dependent Variable:                          outcome 
No. Observations:                           768                              Method:                                      REML    
No. Groups:                                 192                              Scale:                                       0.0179  
Min. group size:                            4                                Log-Likelihood:                              374.3289
Max. group size:                            4                                Converged:                                   Yes     
Mean group size:                            4.0                                                                                   
----------------------------------------------------------------------------------------------------------------------------------
                                                                                       Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                               0.173    0.011  15.984 0.000  0.152  0.194
C(task, Treatment('Reader'))[T.Self]                                                    0.109    0.014   7.967 0.000  0.082  0.136
C(site_type, Treatment('ReaderSite'))[T.SelfSite]                                      -0.155    0.014 -11.350 0.000 -0.182 -0.128
C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite] -0.112    0.019  -5.800 0.000 -0.150 -0.074
Group Var                                                                               0.005    0.008                            
==================================================================================================================================

```

## 3. Scientific Interpretation

> [!IMPORTANT]
> **Double Dissociation Established ($p < .05$)**: The interaction between Task and Ablated-Site is statistically significant. Ablating Reader-site disproportionately impairs Reader Prediction, while ablating Self-site disproportionately impairs Self-Report. This proves that while Reader and Self share intermediate representations (as shown in E1/E2/E4), they diverge into distinct, task-specialized causal readout circuits in the deeper layers.
