# V1 Phase C: E6 Double Dissociation Report: meta-llama/Llama-3.2-1B

- **Reader-Site**: Layer 4
- **Self-Site**: Layer 12
- **Pairs Tested**: 192

## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)

| Site \ Task | Reader Task Impact | Self Task Impact |
|:---|:---:|:---:|
| **Reader-Site (Layer 4)** | **0.107** | 0.094 |
| **Self-Site (Layer 12)** | 0.024 | **0.021** |

## 2. Statistical Interaction Test (Task x SiteType)

- **Model**: `Linear Mixed-Effects Model (LMM)`
- **Interaction p-value**: **`p = 1.1232e-01`**

```text
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                     MixedLM                          Dependent Variable:                          outcome  
No. Observations:                          768                              Method:                                      REML     
No. Groups:                                192                              Scale:                                       0.0023   
Min. group size:                           4                                Log-Likelihood:                              1170.2791
Max. group size:                           4                                Converged:                                   Yes      
Mean group size:                           4.0                                                                                    
----------------------------------------------------------------------------------------------------------------------------------
                                                                                       Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                               0.107    0.004  28.247 0.000  0.100  0.115
C(task, Treatment('Reader'))[T.Self]                                                   -0.013    0.005  -2.683 0.007 -0.023 -0.004
C(site_type, Treatment('ReaderSite'))[T.SelfSite]                                      -0.084    0.005 -17.278 0.000 -0.093 -0.074
C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite]  0.011    0.007   1.588 0.112 -0.003  0.024
Group Var                                                                               0.001    0.003                            
==================================================================================================================================

```

## 3. Scientific Interpretation

> [!NOTE]
> **No Strong Dissociation**: Interaction did not reach significance at threshold, suggesting shared causal utilization across sites.
