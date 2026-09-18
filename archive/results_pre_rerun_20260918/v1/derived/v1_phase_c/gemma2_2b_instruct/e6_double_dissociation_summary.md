# V1 Phase C: E6 Double Dissociation Report: google/gemma-2-2b-it

- **Reader-Site**: Layer 6
- **Self-Site**: Layer 18
- **Pairs Tested**: 192

## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)

| Site \ Task | Reader Task Impact | Self Task Impact |
|:---|:---:|:---:|
| **Reader-Site (Layer 6)** | **0.189** | 0.210 |
| **Self-Site (Layer 18)** | 0.025 | **0.037** |

## 2. Statistical Interaction Test (Task x SiteType)

- **Model**: `Linear Mixed-Effects Model (LMM)`
- **Interaction p-value**: **`p = 5.4163e-01`**

```text
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                      MixedLM                          Dependent Variable:                          outcome 
No. Observations:                           768                              Method:                                      REML    
No. Groups:                                 192                              Scale:                                       0.0099  
Min. group size:                            4                                Log-Likelihood:                              599.2789
Max. group size:                            4                                Converged:                                   Yes     
Mean group size:                            4.0                                                                                   
----------------------------------------------------------------------------------------------------------------------------------
                                                                                       Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                               0.189    0.008  23.428 0.000  0.173  0.205
C(task, Treatment('Reader'))[T.Self]                                                    0.021    0.010   2.057 0.040  0.001  0.041
C(site_type, Treatment('ReaderSite'))[T.SelfSite]                                      -0.165    0.010 -16.217 0.000 -0.185 -0.145
C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite] -0.009    0.014  -0.610 0.542 -0.037  0.019
Group Var                                                                               0.003    0.006                            
==================================================================================================================================

```

## 3. Scientific Interpretation

> [!NOTE]
> **No Strong Dissociation**: Interaction did not reach significance at threshold, suggesting shared causal utilization across sites.
