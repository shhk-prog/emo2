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
| H3: Causal Reorganization | Valence Alignment x Depth (Primary) | 0.0001 | [$-$0.0001, 0.0002] | q = 0.304 | Not Supported |
| H3: Causal Reorganization | Valence Alignment x Task (Primary) | 0.0002 | [0.0000, 0.0003] | q = 0.044 | Supported |
| H3: Causal Reorganization | Valence Alignment x Task x Depth (Primary) | $-$0.0002 | [$-$0.0004, 0.0000] | q = 0.196 | Not Supported |
| H3: Alignment Main Effect | Valence Alignment main effect (Secondary) | $-$0.0001 | [$-$0.0002, $-$0.0000] | p = 0.027 | Supported |
| H3: Causal Reorganization | Arousal Alignment x Depth (Primary) | $-$0.0000 | [$-$0.0001, 0.0000] | q = 0.304 | Not Supported |
| H3: Causal Reorganization | Arousal Alignment x Task (Primary) | $-$0.0000 | [$-$0.0000, 0.0000] | q = 0.304 | Not Supported |
| H3: Causal Reorganization | Arousal Alignment x Task x Depth (Primary) | 0.0000 | [$-$0.0000, 0.0001] | q = 0.304 | Not Supported |
| H3: Alignment Main Effect | Arousal Alignment main effect (Secondary) | 0.0000 | [$-$0.0000, 0.0000] | p = 0.549 | Not Supported |
| H4: Recovery Asymmetry | Self--Reader AUC diff (Primary) | 0.0111 | [$-$0.0104, 0.0380] | --- | Not Supported |
| H4: Recovery Asymmetry | Matched-Plain AUC (Descriptive) | $-$0.0015 | --- | --- | --- |

## 2. Causal Relocation LMM (H3)

| Axis | Term | Estimate (beta) | 95% CI | p-value | FDR q |
|:---|:---|:---:|:---:|:---:|:---:|
| Valence | Intercept | 0.000058 | --- | --- | --- |
| Valence | C(family)[T.llama] | $-$0.000001 | [$-$0.000047, 0.000044] | 0.961 | --- |
| Valence | C(family)[T.olmo] | $-$0.000008 | [$-$0.000053, 0.000037] | 0.730 | --- |
| Valence | C(family)[T.qwen] | 0.000014 | [$-$0.000025, 0.000053] | 0.494 | --- |
| Valence | C(alignment)[T.inst] | $-$0.000095 | [$-$0.000179, $-$0.000011] | 0.027 | --- |
| Valence | C(task)[T.self] | $-$0.000098 | [$-$0.000182, $-$0.000014] | 0.023 | --- |
| Valence | C(alignment)[T.inst]:C(task)[T.self] | 0.000163 | [0.000044, 0.000282] | 0.007 | 0.044 |
| Valence | relative_depth | $-$0.000032 | [$-$0.000134, 0.000070] | 0.538 | --- |
| Valence | C(alignment)[T.inst]:relative_depth | 0.000090 | [$-$0.000054, 0.000234] | 0.223 | 0.304 |
| Valence | C(task)[T.self]:relative_depth | 0.000108 | [$-$0.000037, 0.000252] | 0.144 | --- |
| Valence | C(alignment)[T.inst]:C(task)[T.self]:relative_depth | $-$0.000192 | [$-$0.000396, 0.000012] | 0.065 | 0.196 |
| Valence | Group Var | 0.000000 | --- | --- | --- |
| Arousal | Intercept | $-$0.000006 | --- | --- | --- |
| Arousal | C(family)[T.llama] | $-$0.000004 | [$-$0.000016, 0.000007] | 0.471 | --- |
| Arousal | C(family)[T.olmo] | 0.000006 | [$-$0.000006, 0.000017] | 0.343 | --- |
| Arousal | C(family)[T.qwen] | $-$0.000002 | [$-$0.000012, 0.000007] | 0.621 | --- |
| Arousal | C(alignment)[T.inst] | 0.000006 | [$-$0.000015, 0.000028] | 0.549 | --- |
| Arousal | C(task)[T.self] | 0.000001 | [$-$0.000020, 0.000023] | 0.902 | --- |
| Arousal | C(alignment)[T.inst]:C(task)[T.self] | $-$0.000017 | [$-$0.000047, 0.000013] | 0.277 | 0.304 |
| Arousal | relative_depth | 0.000002 | [$-$0.000023, 0.000028] | 0.849 | --- |
| Arousal | C(alignment)[T.inst]:relative_depth | $-$0.000019 | [$-$0.000055, 0.000017] | 0.304 | 0.304 |
| Arousal | C(task)[T.self]:relative_depth | $-$0.000012 | [$-$0.000049, 0.000024] | 0.500 | --- |
| Arousal | C(alignment)[T.inst]:C(task)[T.self]:relative_depth | 0.000034 | [$-$0.000017, 0.000085] | 0.196 | 0.304 |
| Arousal | Group Var | 0.000000 | --- | --- | --- |
