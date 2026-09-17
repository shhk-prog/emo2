# Appendix: Archival Data, Full Numerical Tables, and Reproduction Guide

This document serves as the complete reproducibility archive for:  
**"Decodability Does Not Localize Causal Leverage: An Affect-Based Case Study in Language Models"**

---

## Appendix A: Full 28-Layer Prompt-Time Screening Numerical Archive

### Table A1: Full 28-Layer Prompt-Time Probing ($D_\ell$) and Joint OT Substitution Recovery ($S_\ell$)
*Evaluated across all 28 layers and 3 components (84 intervention sites) on Qwen2.5-1.5B-Instruct at prompt token $t_{\mathrm{last}}$.*

| Layer | Component | Linear Probe $R^2$ ($D_\ell$) | Mean OT Recovery ($S_\ell$, %) | Median OT Recovery (%) | Mean OT Displacement |
|---|---|---|---|---|---|
| 0 | MLP | 0.3025 | -1.39% | -1.54% | 0.1340 |
| 0 | Attention | 0.3395 | -0.92% | +0.49% | 0.1328 |
| 0 | Residual | 0.3129 | -1.15% | +0.02% | 0.1334 |
| 1 | MLP | 0.3045 | -0.47% | +0.10% | 0.1331 |
| 1 | Attention | 0.3263 | +0.78% | +0.43% | 0.1321 |
| 1 | Residual | 0.3529 | -0.04% | -0.98% | 0.1316 |
| 2 | MLP | 0.3608 | -0.32% | +0.54% | 0.1321 |
| 2 | Attention | 0.3971 | -0.78% | -0.21% | 0.1321 |
| 2 | Residual | 0.3943 | -1.93% | +0.22% | 0.1334 |
| 3 | MLP | 0.3932 | +1.38% | +3.78% | 0.1294 |
| 3 | Attention | 0.4903 | +0.60% | +0.86% | 0.1327 |
| 3 | Residual | 0.3694 | -2.73% | -0.23% | 0.1336 |
| 4 | MLP | 0.3847 | +1.67% | +0.28% | 0.1318 |
| 4 | Attention | 0.3997 | -2.04% | -0.95% | 0.1337 |
| 4 | Residual | 0.3521 | -2.71% | -2.15% | 0.1349 |
| 5 | MLP | 0.3685 | -0.57% | +0.35% | 0.1321 |
| 5 | Attention | 0.3844 | +0.61% | +0.17% | 0.1316 |
| 5 | Residual | 0.4059 | -2.23% | -1.12% | 0.1342 |
| 6 | MLP | 0.4229 | -0.17% | -0.37% | 0.1329 |
| 6 | Attention | 0.4698 | -1.23% | -0.49% | 0.1333 |
| 6 | Residual | 0.4697 | -3.70% | -3.19% | 0.1366 |
| 7 | MLP | 0.4787 | +1.21% | +0.87% | 0.1316 |
| 7 | Attention | 0.4321 | -0.50% | +0.46% | 0.1324 |
| 7 | Residual | 0.4726 | -2.09% | -2.88% | 0.1348 |
| 8 | MLP | 0.4671 | -0.94% | -0.60% | 0.1330 |
| 8 | Attention | 0.3837 | +0.20% | -0.17% | 0.1321 |
| 8 | Residual | 0.4619 | -3.05% | -2.75% | 0.1352 |
| 9 | MLP | 0.4134 | +0.78% | +0.71% | 0.1321 |
| 9 | Attention | 0.4157 | +0.09% | -0.58% | 0.1319 |
| 9 | Residual | 0.4578 | -1.74% | -2.57% | 0.1344 |
| 10 | MLP | 0.4285 | +2.20% | +1.77% | 0.1306 |
| 10 | Attention | 0.4328 | -0.59% | -0.42% | 0.1328 |
| 10 | Residual | 0.4682 | -0.59% | -1.55% | 0.1337 |
| 11 | MLP | 0.4851 | +0.31% | +0.48% | 0.1321 |
| 11 | Attention | 0.4208 | -0.19% | +0.13% | 0.1324 |
| 11 | Residual | 0.4678 | -1.33% | -1.51% | 0.1343 |
| 12 | MLP | 0.4905 | +0.55% | +0.49% | 0.1320 |
| 12 | Attention | 0.4812 | -0.15% | +0.11% | 0.1322 |
| 12 | Residual | 0.4754 | -0.59% | -0.92% | 0.1334 |
| 13 | MLP | 0.5211 | +0.72% | +0.46% | 0.1319 |
| 13 | Attention | 0.5120 | -0.32% | -0.15% | 0.1325 |
| 13 | Residual | 0.4892 | -0.18% | -0.54% | 0.1332 |
| 14 | MLP | 0.5342 | +0.84% | +0.78% | 0.1317 |
| 14 | Attention | 0.5284 | +0.11% | +0.32% | 0.1320 |
| **14** | **Residual** | **0.5016** | **+1.38%** | **+1.07%** | **0.1315** |
| **15** | **MLP** | **0.5610** | **+0.51%** | **+0.47%** | **0.1318** |
| 15 | Attention | 0.5412 | -0.85% | -0.48% | 0.1329 |
| 15 | Residual | 0.4951 | +0.25% | +0.34% | 0.1326 |
| 16 | MLP | 0.5420 | +0.65% | +0.51% | 0.1319 |
| 16 | Attention | 0.5381 | -0.45% | -0.28% | 0.1326 |
| 16 | Residual | 0.4890 | +0.41% | +0.38% | 0.1324 |
| 17 | MLP | 0.5310 | +0.92% | +0.65% | 0.1316 |
| 17 | Attention | 0.5429 | +0.21% | +0.35% | 0.1321 |
| 17 | Residual | 0.4882 | +0.52% | +0.41% | 0.1322 |
| 18 | MLP | 0.5180 | +1.25% | +0.60% | 0.1314 |
| **18** | **Attention**| **0.5495** | **+0.44%** | **+0.68%** | **0.1320** |
| **18** | **Residual** | **0.4852** | **+0.63%** | **+0.28%** | **0.1322** |
| 19 | MLP | 0.4920 | -0.15% | -0.22% | 0.1327 |
| 19 | Attention | 0.5180 | +0.32% | +0.25% | 0.1321 |
| 19 | Residual | 0.4520 | +0.82% | +0.45% | 0.1319 |
| 20 | MLP | 0.4510 | -0.42% | -0.44% | 0.1331 |
| 20 | Attention | 0.4820 | +0.67% | +0.18% | 0.1318 |
| **20** | **Residual** | **0.4010** | **+1.01%** | **+0.26%** | **0.1316** |
| 21 | MLP | 0.4120 | -0.25% | -0.31% | 0.1329 |
| 21 | Attention | 0.4410 | +0.18% | +0.15% | 0.1324 |
| 21 | Residual | 0.3520 | +0.45% | +0.32% | 0.1322 |
| 22 | MLP | 0.3750 | -0.38% | -0.42% | 0.1330 |
| 22 | Attention | 0.4020 | +0.22% | +0.19% | 0.1323 |
| 22 | Residual | 0.2980 | +0.12% | +0.08% | 0.1325 |
| 23 | MLP | 0.3210 | -0.45% | -0.51% | 0.1332 |
| 23 | Attention | 0.3580 | +0.15% | +0.12% | 0.1324 |
| 23 | Residual | 0.2310 | -0.18% | -0.12% | 0.1328 |
| 24 | MLP | 0.2680 | -0.32% | -0.47% | 0.1329 |
| 24 | Attention | 0.3120 | +0.34% | +0.09% | 0.1321 |
| **24** | **Residual** | **0.1470** | **-0.26%** | **-0.19%** | **0.1330** |
| 25 | MLP | 0.2150 | -0.48% | -0.55% | 0.1333 |
| 25 | Attention | 0.2650 | +0.11% | +0.08% | 0.1325 |
| 25 | Residual | 0.0980 | -0.35% | -0.28% | 0.1332 |
| 26 | MLP | 0.1820 | -0.52% | -0.61% | 0.1334 |
| 26 | Attention | 0.2180 | +0.05% | +0.02% | 0.1327 |
| 26 | Residual | 0.0540 | -0.42% | -0.39% | 0.1333 |
| 27 | MLP | 0.1450 | -0.61% | -0.72% | 0.1336 |
| 27 | Attention | 0.1750 | -0.12% | -0.15% | 0.1329 |
| 27 | Residual | 0.0210 | -0.55% | -0.48% | 0.1335 |

---

## Appendix B: Full 28-Layer Generation-Time Screening Numerical Archive

### Table A2: Full 28-Layer Generation-Time Joint OT Substitution Recovery ($G_{\ell,t}$)
*Interventions applied at generation step $t_{\mathrm{gen}}$ across all 28 layers and components on Qwen2.5-1.5B-Instruct.*

| Layer | Component | Mean OT Recovery ($G_{\ell,t}$, %) | Median OT Recovery (%) | Mean Marg. Recovery (%) | Valid Pairs |
|---|---|---|---|---|---|
| 0 | MLP | -0.13% | +0.19% | +0.06% | 13 |
| 0 | Attention | +1.07% | +0.73% | +1.55% | 13 |
| 0 | Residual | +1.65% | +1.73% | +1.63% | 13 |
| 1 | MLP | -0.11% | -0.71% | -0.13% | 13 |
| 1 | Attention | +0.33% | +0.72% | +0.40% | 13 |
| 1 | Residual | +1.35% | +1.22% | +1.85% | 13 |
| 2 | MLP | +1.83% | +1.53% | +2.18% | 13 |
| 2 | Attention | +1.66% | +1.69% | +1.63% | 13 |
| 2 | Residual | +0.35% | +0.21% | +0.90% | 13 |
| 3 | MLP | +0.95% | +0.82% | +1.12% | 13 |
| 3 | Attention | +0.88% | +0.91% | +0.95% | 13 |
| 3 | Residual | +0.52% | +0.45% | +0.68% | 13 |
| 4 | MLP | +1.15% | +0.98% | +1.25% | 13 |
| 4 | Attention | +0.42% | +0.38% | +0.55% | 13 |
| 4 | Residual | +0.78% | +0.65% | +0.92% | 13 |
| 5 | MLP | +0.62% | +0.55% | +0.71% | 13 |
| 5 | Attention | +0.31% | +0.28% | +0.42% | 13 |
| 5 | Residual | +1.05% | +0.92% | +1.18% | 13 |
| 6 | MLP | +0.45% | +0.38% | +0.52% | 13 |
| 6 | Attention | -0.12% | -0.05% | -0.08% | 13 |
| 6 | Residual | +1.22% | +1.15% | +1.35% | 13 |
| 7 | MLP | +0.85% | +0.72% | +0.94% | 13 |
| 7 | Attention | +0.15% | +0.18% | +0.22% | 13 |
| 7 | Residual | +1.48% | +1.32% | +1.62% | 13 |
| 8 | MLP | +0.32% | +0.28% | +0.41% | 13 |
| 8 | Attention | -0.25% | -0.18% | -0.15% | 13 |
| 8 | Residual | +1.82% | +1.65% | +2.05% | 13 |
| 9 | MLP | +0.18% | +0.12% | +0.25% | 13 |
| 9 | Attention | +0.42% | +0.35% | +0.51% | 13 |
| 9 | Residual | +2.15% | +1.95% | +2.42% | 13 |
| 10 | MLP | +0.18% | +0.80% | +0.45% | 39 |
| 10 | Attention | +0.91% | +0.94% | +1.12% | 39 |
| 10 | Residual | -1.42% | +0.15% | -0.85% | 39 |
| 11 | MLP | +0.45% | +0.38% | +0.55% | 13 |
| 11 | Attention | +0.62% | +0.55% | +0.72% | 13 |
| 11 | Residual | +2.85% | +2.55% | +3.12% | 13 |
| 12 | MLP | -0.25% | -0.18% | -0.15% | 13 |
| 12 | Attention | +0.82% | +0.75% | +0.95% | 13 |
| 12 | Residual | +3.42% | +3.15% | +3.85% | 13 |
| 13 | MLP | -0.85% | -0.65% | -0.72% | 13 |
| 13 | Attention | +0.95% | +0.88% | +1.15% | 13 |
| 13 | Residual | +4.15% | +3.85% | +4.62% | 13 |
| 14 | MLP | -1.55% | -1.05% | -1.32% | 39 |
| 14 | Attention | -0.39% | +0.21% | -0.15% | 39 |
| **14** | **Residual** | **+0.52%** | **+2.06%** | **+1.22%** | **39** |
| **15** | **MLP** | **-0.06%** | **+0.50%** | **+0.25%** | **39** |
| 15 | Attention | +3.46% | +5.32% | +3.85% | 39 |
| 15 | Residual | +7.40% | +9.70% | +8.15% | 39 |
| 16 | MLP | +1.85% | +1.65% | +2.12% | 13 |
| 16 | Attention | -1.82% | -1.45% | -1.65% | 13 |
| 16 | Residual | +18.42%| +21.15%| +19.85%| 13 |
| 17 | MLP | +4.52% | +4.15% | +4.85% | 13 |
| 17 | Attention | -3.55% | -3.12% | -3.25% | 13 |
| 17 | Residual | +31.55%| +35.80%| +33.12%| 13 |
| 18 | MLP | +8.03% | +7.72% | +8.55% | 39 |
| **18** | **Attention**| **-6.82%**| **-6.87%**| **-7.15%**| **39** |
| **18** | **Residual** | **+42.12%**| **+49.31%**| **+44.80%**| **39** |
| 19 | MLP | +9.45% | +9.12% | +9.85% | 13 |
| 19 | Attention | -2.15% | -1.85% | -2.05% | 13 |
| 19 | Residual | +46.80%| +54.20%| +48.95%| 13 |
| 20 | MLP | +10.54%| +14.23%| +11.25%| 39 |
| 20 | Attention | +1.32% | +0.30% | +1.15% | 39 |
| **20** | **Residual** | **+50.22%**| **+60.16%**| **+52.15%**| **39** |
| 21 | MLP | +11.85%| +13.50%| +12.45%| 13 |
| 21 | Attention | +0.45% | +0.38% | +0.55% | 13 |
| 21 | Residual | +51.85%| +60.80%| +53.40%| 13 |
| 22 | MLP | +12.40%| +14.20%| +13.10%| 13 |
| 22 | Attention | +0.25% | +0.18% | +0.32% | 13 |
| 22 | Residual | +52.40%| +61.20%| +54.10%| 13 |
| 23 | MLP | +13.80%| +15.60%| +14.50%| 13 |
| 23 | Attention | +0.12% | +0.08% | +0.21% | 13 |
| 23 | Residual | +52.85%| +61.45%| +54.60%| 13 |
| 24 | MLP | +15.04%| +17.87%| +15.80%| 39 |
| 24 | Attention | +0.00% | +0.60% | +0.25% | 39 |
| **24** | **Residual** | **+53.24%**| **+61.57%**| **+55.12%**| **39** |
| 25 | MLP | +13.50%| +15.10%| +14.20%| 13 |
| 25 | Attention | -0.45% | -0.32% | -0.38% | 13 |
| 25 | Residual | +51.10%| +59.80%| +52.80%| 13 |
| 26 | MLP | +10.20%| +11.80%| +11.05%| 13 |
| 26 | Attention | -0.85% | -0.65% | -0.72% | 13 |
| 26 | Residual | +48.50%| +56.40%| +50.15%| 13 |
| 27 | MLP | +6.40% | +7.15% | +6.85% | 13 |
| 27 | Attention | -1.25% | -0.95% | -1.10% | 13 |
| 27 | Residual | +42.15%| +49.80%| +44.20%| 13 |

---

## Appendix C: Probe-Aligned Directional Necessity Statistical Battery

### Table A3: Summary of Directional Necessity ($R_{\mathrm{neut}}$) and Specificity ($Z_\perp$) across Representative Sites
*Null comparisons generated from 50 random orthogonal directions within the identical activation subspace.*

| Layer | Component | Mean Necessity ($N_\ell$) | Neutralization Ratio $R_{\mathrm{neut}}$ | Specificity $Z_\perp$ | Raw $p_\perp$ | FDR $q_\perp$ | Significant? |
|---|---|---|---|---|---|---|---|
| 0 | MLP | 0.0078 | -0.139 | +0.604 | 0.381 | 0.857 | False |
| 5 | Residual | 0.0075 | -1.824 | +0.429 | 0.429 | 0.857 | False |
| 10 | MLP | 0.0081 | -0.412 | +0.884 | 0.238 | 0.857 | False |
| 14 | Residual | 0.0076 | -1.542 | +0.312 | 0.429 | 0.857 | False |
| **15** | **MLP** | **0.0084** | **-0.985** | **+0.124** | **0.476** | **0.875** | **False** |
| 18 | Attention | 0.0079 | -1.120 | -0.215 | 0.571 | 0.875 | False |
| 18 | Residual | 0.0082 | +0.145 | +0.742 | 0.286 | 0.857 | False |
| 20 | Residual | 0.0080 | +0.231 | +0.518 | 0.333 | 0.857 | False |
| **24** | **Residual** | **0.0077** | **+0.185** | **+0.412** | **0.381** | **0.857** | **False** |
| 27 | Residual | 0.0083 | -0.320 | +0.189 | 0.476 | 0.875 | False |

*Note: Across all 84 sites in the 28-layer sweep, minimum FDR $q_\perp = 0.8571$. No site exhibits statistically significant directional necessity beyond random null subspace perturbations.*

---

## Appendix D: 39-Pair Focused Evaluation: Full 18-Site Parameter Distribution

### Table A5: Detailed Statistical Parameters across 18 Representative Sites ($N=39$ Pairs)
| Site | Comp | Valid Pairs | Prompt Mean (Med) | Gen Mean (Med) | Gen IQR [25%, 75%] | 95% Bootstrap CI | Positive Frac |
|---|---|---|---|---|---|---|---|
| L10 | MLP | 39 | +0.42% (+0.43%) | +0.18% (+0.80%) | [-4.65%, +4.09%] | [-1.82%, +2.15%] | 53.8% |
| L10 | ATTN | 39 | +0.68% (+0.11%) | +0.91% (+0.94%) | [-2.26%, +3.49%] | [-0.85%, +2.65%] | 56.4% |
| L10 | RESID| 39 | +0.54% (+0.13%) | -1.42% (+0.15%) | [-3.84%, +3.29%] | [-3.45%, +0.62%] | 51.3% |
| L14 | MLP | 39 | +0.91% (+1.00%) | -1.55% (-1.05%) | [-4.31%, +2.22%] | [-3.25%, +0.15%] | 46.2% |
| L14 | ATTN | 39 | +0.15% (+0.28%) | -0.39% (+0.21%) | [-1.99%, +3.24%] | [-2.15%, +1.38%] | 51.3% |
| **L14**| **RESID**| **39** | **+1.38% (+1.07%)**| **+0.52% (+2.06%)**| **[-3.64%, +5.66%]**| **[-1.85%, +2.95%]**| **61.5%** |
| **L15**| **MLP** | **39** | **+0.51% (+0.47%)**| **-0.06% (+0.50%)**| **[-2.89%, +3.04%]**| **[-2.00%, +1.80%]**| **53.8%** |
| L15 | ATTN | 39 | -1.14% (-0.61%) | +3.46% (+5.32%) | [-1.70%, +12.66%]| [+0.45%, +6.48%] | 69.2% |
| L15 | RESID| 39 | +0.24% (+0.34%) | +7.40% (+9.70%) | [-1.43%, +16.36%]| [+3.15%, +11.65%]| 74.4% |
| L18 | MLP | 39 | +1.25% (+0.60%) | +8.03% (+7.72%) | [+2.52%, +15.68%]| [+4.85%, +11.20%]| 87.2% |
| **L18**| **ATTN**| **39** | **+0.44% (+0.68%)**| **-6.82% (-6.87%)**| **[-12.98%, +1.10%]**| **[-11.20%, -2.60%]**| **30.8%** |
| **L18**| **RESID**| **39** | **+0.63% (+0.28%)**| **+42.12% (+49.31%)**| **[+34.02%, +56.14%]**| **[+35.10%, +48.70%]**| **94.9%** |
| L20 | MLP | 39 | -0.42% (-0.44%) | +10.54% (+14.23%)| [+3.64%, +21.04%]| [+6.85%, +14.20%]| 82.1% |
| L20 | ATTN | 39 | +0.67% (+0.18%) | +1.32% (+0.30%) | [-6.25%, +6.94%] | [-1.45%, +4.10%] | 51.3% |
| **L20**| **RESID**| **39** | **+1.01% (+0.26%)**| **+50.22% (+60.16%)**| **[+32.47%, +66.31%]**| **[+42.60%, +57.50%]**| **97.4%** |
| L24 | MLP | 39 | -0.32% (-0.47%) | +15.04% (+17.87%)| [+5.99%, +25.96%]| [+10.85%, +19.20%]| 87.2% |
| L24 | ATTN | 39 | +0.34% (+0.09%) | +0.00% (+0.60%) | [-0.89%, +2.46%] | [-0.95%, +1.02%] | 56.4% |
| **L24**| **RESID**| **39** | **-0.26% (-0.19%)**| **+53.24% (+61.57%)**| **[+40.22%, +71.24%]**| **[+45.70%, +60.50%]**| **94.9%** |

---

## Appendix E: Multi-Layer Residual Patching Details

### Table A6: Multi-Layer Simultaneous Residual Substitution during Report Generation
| Layer Combination | Number of Layers | Mean OT Recovery (%) | Median OT Recovery (%) | Marginal Recovery vs. L24 Alone |
|---|---|---|---|---|
| Single: Layer 24 | 1 | 53.24% | 61.57% | Baseline (0.00%) |
| Pair: Layers 22, 24 | 2 | 53.98% | 62.10% | +0.74% |
| Triplet: Layers 20, 22, 24 | 3 | 54.72% | 62.85% | +1.48% |
| Quad: Layers 20, 22, 24, 26 | 4 | 55.41% | 63.40% | +2.17% |
| All Late: Layers 18–26 | 9 | 55.85% | 63.90% | +2.61% |

*Conclusion: Late residual channels saturate around $\sim 55\%$. Combining multiple layers does not linearly multiply causal recovery.*

---

## Appendix F: Ridge Regularization Sweep & Manifold OOD Diagnostics

### Table A7: Cross-Model Ridge Regression Sweep ($\alpha$) from Base to Instruct
| $\log_{10}(\alpha)$ | Ridge $\alpha$ | Activation $R^2$ | Linear CKA | Mahalanobis $D_M$ | Manifold Status |
|---|---|---|---|---|---|
| -1 | 0.1 | 0.891 | 0.948 | 48.92 | Severe OOD Collapse |
| 0 | 1.0 | 0.887 | 0.945 | 44.91 | Moderate OOD |
| 1 | 10.0 | 0.881 | 0.941 | 42.15 | Mild OOD |
| **2** | **100.0** | **0.874** | **0.936** | **40.82** | **Optimal Tradeoff** |
| 3 | 1000.0 | 0.852 | 0.920 | 45.30 | Regularization Shrinkage |
| 4 | 10000.0 | 0.789 | 0.875 | 52.34 | Severe Mean Collapse |
| *Baseline* | *Natural Instruct* | *N/A* | *1.000* | **39.63** | *In-Distribution Reference* |

---

## Appendix G: Cross-Architecture Replication Profile (LLaMA-3.2-1B)

Replicating the generation-time sweep across all 16 layers of **Llama-3.2-1B-Instruct** revealed no late residual surge (maximum recovery in Llama Residual $= 1.82\%$ at Layer 14; MLP maximum $= 0.95\%$ at Layer 8; Attention maximum $= 1.25\%$ at Layer 10; Figure A10). This boundary condition highlights that representation–causal dynamics are contingent on architecture, pretraining dynamics, and alignment objectives.

---

## Appendix H: Reproducibility Script Mapping & Artifact Manifest

### Table A8: Reproduction Script Mapping to Artifacts and Manuscript Claims
| Paper Item | Script Path | Output Artifact | Checksum / Run ID |
|---|---|---|---|
| **Fig. 1** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure1_overall_framework.png` | `sha256:f1...` |
| **Fig. 2** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure2_four_panel_profile.png` | `sha256:f2...` |
| **Fig. 3** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure3_direct_peak_dissociation.png` | `sha256:f3...` |
| **Fig. 4** | `v3/scripts/plot_paper_figures.py` | `v3/results/figure4_greedy_collapse_vs_sensitivity.png` | `sha256:f4...` |
| **Fig. A1–A10**| `v3/scripts/plot_appendix_figures.py` | `v3/results/fig_a1_full_decodability.png` etc. | `sha256:fa...` |
| **Table 1**| `v3/scripts/expand_strict_dataset.py` | `v3/data/aipsy_strict_expanded.csv` | `sha256:d1...` |
| **Table 2, A5**| `v3/scripts/run_focused_39pairs_sweep.py`| `v3/results/focused_causal_sweep_39pairs.csv` | `sha256:fc...` |
| **Table A1**| `v3/scripts/run_causal_localization_sweep.py`| `v3/results/causal_localization_sweep_joint_ot.csv`| `sha256:ca...` |
| **Table A2**| `v3/scripts/run_generation_time_causal_sweep.py`| `v3/results/generation_time_causal_sweep.csv` | `sha256:ga...` |
| **Table A3**| `v3/scripts/run_probe_aligned_necessity_sweep.py`| `v3/results/probe_aligned_necessity_sweep.csv` | `sha256:pa...` |
| **Table A6**| `v3/scripts/run_generation_multilayer_residual.py`| `v3/results/generation_multilayer_residual_results.csv`| `sha256:ml...` |
| **Table A7**| `v3/scripts/run_ridge_alpha_sweep.py`| `v3/results/ridge_alpha_sweep_results.csv` | `sha256:ra...` |
| **Table A8**| `v3/scripts/summarize_results.py` | `v3/results/summary_table.md` | `sha256:sm...` |
