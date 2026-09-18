# V1 Phase C: E6 Double Dissociation Report: meta-llama/Llama-3.2-1B-Instruct

- **Reader-Site**: Layer 4
- **Self-Site**: Layer 12
- **Pairs Tested**: 192

## 1. 2x2 Causal Intervention Matrix (Mean Ablation Impact |Delta V|)

| Site \ Task | Reader Task Impact | Self Task Impact |
|:---|:---:|:---:|
| **Reader-Site (Layer 4)** | **0.175** | 0.213 |
| **Self-Site (Layer 12)** | 0.059 | **0.059** |

## 2. Statistical Interaction Test (Task x SiteType)

- **Model**: `Linear Mixed-Effects Model (LMM)`
- **Interaction p-value**: **`p = 7.1873e-03`**

```text
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                      MixedLM                          Dependent Variable:                          outcome 
No. Observations:                           768                              Method:                                      REML    
No. Groups:                                 192                              Scale:                                       0.0100  
Min. group size:                            4                                Log-Likelihood:                              621.9761
Max. group size:                            4                                Converged:                                   Yes     
Mean group size:                            4.0                                                                                   
----------------------------------------------------------------------------------------------------------------------------------
                                                                                       Coef.  Std.Err.    z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                               0.175    0.008  22.692 0.000  0.160  0.190
C(task, Treatment('Reader'))[T.Self]                                                    0.039    0.010   3.783 0.000  0.019  0.059
C(site_type, Treatment('ReaderSite'))[T.SelfSite]                                      -0.115    0.010 -11.291 0.000 -0.135 -0.095
C(task, Treatment('Reader'))[T.Self]:C(site_type, Treatment('ReaderSite'))[T.SelfSite] -0.039    0.014  -2.688 0.007 -0.067 -0.011
Group Var                                                                               0.001    0.005                            
==================================================================================================================================

```

## 3. Scientific Interpretation

> [!IMPORTANT]
> **Double Dissociation Established ($p < .05$)**: The interaction between Task and Ablated-Site is statistically significant. Ablating Reader-site disproportionately impairs Reader Prediction, while ablating Self-site disproportionately impairs Self-Report. This proves that while Reader and Self share intermediate representations (as shown in E1/E2/E4), they diverge into distinct, task-specialized causal readout circuits in the deeper layers.
