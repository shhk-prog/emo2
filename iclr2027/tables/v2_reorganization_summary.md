# V2 Stage: Post-training-Associated Reorganization (H1-H4) Complete Summary Report

## 1. Confirmatory Hypotheses Testing Summary

| Hypothesis | Key Pre-registered Metric | Estimate | 95% CI | p / FDR q | Supported? |
|:---|:---|:---:|:---:|:---:|:---:|
| H1a: Geometry Reorganization | Reader Procrustes Distortion | 1.4553 | [0.6387, 2.8126] | --- | Supported |
| H1a: Geometry Reorganization | Self Procrustes Distortion | 2.8716 | [0.6603, 6.9954] | --- | Supported |
| H1b: Decodability Peak Shift | Valence Reader Peak Shift Delta d* | $-$0.1185 | [$-$0.2185, $-$0.0333] | --- | Not Supported |
| H1b: Decodability Peak Shift | Valence Self Peak Shift Delta d* | 0.0067 | [$-$0.0533, 0.1000] | --- | Not Supported |
| H1b: Decodability Peak Shift | Arousal Reader Peak Shift Delta d* | 0.1996 | [$-$0.0267, 0.4259] | --- | Not Supported |
| H1b: Decodability Peak Shift | Arousal Self Peak Shift Delta d* | 0.0167 | [$-$0.2000, 0.2500] | --- | Not Supported |
| H2: Sharing Reorganization | Valence Delta Sharing | $-$0.5529 | [$-$1.3236, 0.0760] | --- | Not Supported |
| H2: Sharing Reorganization | Arousal Delta Sharing | $-$0.2784 | [$-$0.6151, 0.0489] | --- | Not Supported |
| H3: Causal Reorganization | Valence Alignment x Depth (Primary) | 0.0001 | [$-$0.0000, 0.0003] | q = 0.287 | Not Supported |
| H3: Causal Reorganization | Valence Alignment x Task (Primary) | 0.0002 | [0.0000, 0.0004] | q = 0.122 | Not Supported |
| H3: Causal Reorganization | Valence Alignment x Task x Depth (Primary) | $-$0.0002 | [$-$0.0005, 0.0001] | q = 0.287 | Not Supported |
| H3: Alignment Main Effect | Valence Alignment main effect (Secondary) | $-$0.0001 | [$-$0.0002, 0.0000] | p = 0.051 | Not Supported |
| H3: Causal Reorganization | Arousal Alignment x Depth (Primary) | $-$0.0000 | [$-$0.0001, 0.0000] | q = 0.796 | Not Supported |
| H3: Causal Reorganization | Arousal Alignment x Task (Primary) | 0.0000 | [$-$0.0000, 0.0000] | q = 0.796 | Not Supported |
| H3: Causal Reorganization | Arousal Alignment x Task x Depth (Primary) | 0.0000 | [$-$0.0001, 0.0001] | q = 0.796 | Not Supported |
| H3: Alignment Main Effect | Arousal Alignment main effect (Secondary) | $-$0.0000 | [$-$0.0000, 0.0000] | p = 0.740 | Not Supported |
| H4: Recovery Asymmetry | Self--Reader AUC diff (Primary) | 0.0319 | --- | --- | Incomplete / Not evaluated |
| H4: Recovery Asymmetry | Matched-Plain AUC (Descriptive) | $-$0.0953 | --- | --- | --- |

## 2. Causal Relocation LMM (H3)

| Axis | Term | Estimate (beta) | 95% CI | p-value | FDR q |
|:---|:---|:---:|:---:|:---:|:---:|
| Valence | Intercept | 0.000012 | --- | --- | --- |
| Valence | C(family)[T.olmo] | $-$0.000008 | [$-$0.000051, 0.000035] | 0.717 | --- |
| Valence | C(alignment)[T.inst] | $-$0.000114 | [$-$0.000229, 0.000000] | 0.051 | --- |
| Valence | C(task)[T.self] | $-$0.000132 | [$-$0.000247, $-$0.000017] | 0.024 | --- |
| Valence | C(alignment)[T.inst]:C(task)[T.self] | 0.000192 | [0.000030, 0.000354] | 0.020 | 0.122 |
| Valence | relative_depth | $-$0.000067 | [$-$0.000206, 0.000072] | 0.345 | --- |
| Valence | C(alignment)[T.inst]:relative_depth | 0.000147 | [$-$0.000050, 0.000343] | 0.144 | 0.287 |
| Valence | C(task)[T.self]:relative_depth | 0.000148 | [$-$0.000048, 0.000345] | 0.138 | --- |
| Valence | C(alignment)[T.inst]:C(task)[T.self]:relative_depth | $-$0.000213 | [$-$0.000491, 0.000065] | 0.133 | 0.287 |
| Valence | Group Var | 0.000000 | --- | --- | --- |
| Arousal | Intercept | 0.000006 | --- | --- | --- |
| Arousal | C(family)[T.olmo] | 0.000006 | [$-$0.000006, 0.000017] | 0.349 | --- |
| Arousal | C(alignment)[T.inst] | $-$0.000005 | [$-$0.000036, 0.000025] | 0.740 | --- |
| Arousal | C(task)[T.self] | 0.000007 | [$-$0.000024, 0.000037] | 0.669 | --- |
| Arousal | C(alignment)[T.inst]:C(task)[T.self] | 0.000006 | [$-$0.000038, 0.000049] | 0.796 | 0.796 |
| Arousal | relative_depth | 0.000010 | [$-$0.000027, 0.000047] | 0.591 | --- |
| Arousal | C(alignment)[T.inst]:relative_depth | $-$0.000008 | [$-$0.000061, 0.000044] | 0.760 | 0.796 |
| Arousal | C(task)[T.self]:relative_depth | $-$0.000030 | [$-$0.000082, 0.000023] | 0.266 | --- |
| Arousal | C(alignment)[T.inst]:C(task)[T.self]:relative_depth | 0.000016 | [$-$0.000058, 0.000090] | 0.676 | 0.796 |
| Arousal | Group Var | 0.000000 | --- | --- | --- |
