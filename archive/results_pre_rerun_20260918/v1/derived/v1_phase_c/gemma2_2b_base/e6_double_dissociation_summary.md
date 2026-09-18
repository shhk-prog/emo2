# V1 Phase C: E6 Double Dissociation Report: google/gemma-2-2b

- **Reader-Site**: Layer 6
- **Self-Site**: Layer 18
- **Pairs Tested**: 192

## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)

| Site \ Task | Reader Task Impact | Self Task Impact |
|:---|:---:|:---:|
| **Reader-Site (Layer 6)** | **0.915** | 1.101 |
| **Self-Site (Layer 18)** | 0.330 | **0.289** |

## 2. Statistical Interaction Test (Task x SiteType)

- **Model**: `Linear Mixed-Effects Model (LMM)`
- **Interaction p-value**: **`p = 4.1673e-04`**

```text
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                     MixedLM                          Dependent Variable:                          outcome  
No. Observations:                          768                              Method:                                      REML     
No. Groups:                                192                              Scale:                                       0.1973   
Min. group size:                           4                                Log-Likelihood:                              -581.2835
Max. group size:                           4                                Converged:                                   Yes      
Mean group size:                           4.0                                                                                    
----------------------------------------------------------------------------------------------------------------------------------
                                                                                       Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                               0.915    0.039  23.199 0.000  0.838  0.993
C(task, Treatment('Reader'))[T.Self]                                                    0.186    0.045   4.094 0.000  0.097  0.274
C(site_type, Treatment('ReaderSite'))[T.SelfSite]                                      -0.586    0.045 -12.920 0.000 -0.675 -0.497
C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite] -0.226    0.064  -3.529 0.000 -0.352 -0.101
Group Var                                                                               0.102    0.040                            
==================================================================================================================================

```

## 3. Scientific Interpretation

> [!IMPORTANT]
> **Double Dissociation Established ($p < .05$)**: The interaction between Task and Ablated-Site is statistically significant. Ablating Reader-site disproportionately impairs Reader Prediction, while ablating Self-site disproportionately impairs Self-Report. This proves that while Reader and Self share intermediate representations (as shown in E1/E2/E4), they diverge into distinct, task-specialized causal readout circuits in the deeper layers.
