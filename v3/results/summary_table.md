# Experimental Results Summary: Multi-layer Patching & Multivariate Diagnostics

### 1. Multi-Layer Simultaneous Patching Recovery (Normalized 2D EMD)

| Block | Aligned Mean Rec (%) | Aligned Median Rec (%) | Aligned 95% CI (%) | Raw Mean Rec (%) | Within Mean Rec (%) |
| --- | --- | --- | --- | --- | --- |
| 1-Layer (L15) | 0.11040723167566381 | 0.11320664355963239 | [-0.05%, 0.28%] | 0.19608439826475374 | 0.23192264950243718 |
| 2-Layer (L14-15) | 0.07335312842318258 | 0.026073034669737982 | [-0.18%, 0.34%] | -1.2625850143387432 | 0.1536849241738326 |
| 4-Layer (L13-16) | 0.34976584204379346 | 0.383405595266334 | [0.04%, 0.64%] | 0.08758502849659434 | 0.3425669629056729 |
| 8-Layer (L11-18) | -0.32508617664670847 | -0.20375094574240915 | [-1.00%, 0.29%] | -3.8095845945943125 | -0.12457169327440909 |

### 2. Multivariate Mahalanobis Distance and Geometric In-Distribution Diagnostics

| Layer | Raw D_M (Mean) | Aligned D_M (Mean) | Raw Cosine | Aligned Cosine | Aligned Norm Ratio |
| --- | --- | --- | --- | --- | --- |
| 11 | 280.1 | 6.22 | 0.813 | 0.993 | 1.001 |
| 12 | 312.2 | 6.03 | 0.808 | 0.993 | 0.996 |
| 13 | 300.0 | 5.67 | 0.795 | 0.995 | 0.994 |
| 14 | 292.0 | 5.99 | 0.819 | 0.995 | 0.996 |
| 15 | 279.6 | 5.76 | 0.759 | 0.995 | 0.991 |
| 16 | 314.1 | 6.45 | 0.835 | 0.996 | 0.991 |
| 17 | 274.3 | 6.94 | 0.820 | 0.995 | 0.992 |
| 18 | 262.0 | 7.10 | 0.800 | 0.995 | 0.987 |
