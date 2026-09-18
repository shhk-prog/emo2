# V1 Phase C: E6 Double Dissociation Report: Qwen/Qwen2.5-1.5B

- **Reader-Site**: Layer 8
- **Self-Site**: Layer 20
- **Pairs Tested**: 192

## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)

| Site \ Task | Reader Task Impact | Self Task Impact |
|:---|:---:|:---:|
| **Reader-Site (Layer 8)** | **0.437** | 0.474 |
| **Self-Site (Layer 20)** | 0.022 | **0.023** |

## 2. Statistical Interaction Test (Task x SiteType)

- **Model**: `Linear Mixed-Effects Model (LMM)`
- **Interaction p-value**: **`p = 1.8976e-01`**

```text
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                       MixedLM                          Dependent Variable:                          outcome
No. Observations:                            768                              Method:                                      REML   
No. Groups:                                  192                              Scale:                                       0.0353 
Min. group size:                             4                                Log-Likelihood:                              92.0211
Max. group size:                             4                                Converged:                                   Yes    
Mean group size:                             4.0                                                                                  
----------------------------------------------------------------------------------------------------------------------------------
                                                                                       Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                               0.437    0.016  27.276 0.000  0.406  0.468
C(task, Treatment('Reader'))[T.Self]                                                    0.037    0.019   1.926 0.054 -0.001  0.075
C(site_type, Treatment('ReaderSite'))[T.SelfSite]                                      -0.415    0.019 -21.643 0.000 -0.453 -0.378
C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite] -0.036    0.027  -1.311 0.190 -0.089  0.018
Group Var                                                                               0.014    0.014                            
==================================================================================================================================

```

## 3. Scientific Interpretation

> [!NOTE]
> **No Strong Dissociation**: Interaction did not reach significance at threshold, suggesting shared causal utilization across sites.
