# V2 Stage: Post-training-Associated Reorganization (H1-H4) Complete Summary Report

## 1. Representation Geometry and Sharing Reorganization (H1--H2)

| Hypothesis | Metric | Estimate | 95% CI | Condition |
|:---|:---|:---:|:---:|:---:|
| H1a: Geometric Distortion | Reader Procrustes Distortion | 1.4553 | [0.6387, 2.8126] | Matched-Plain |
| H1a: Geometric Distortion | Self Procrustes Distortion | 2.8716 | [0.6603, 6.9954] | Matched-Plain |
| H1b: Decodability Peak Shift | Valence Reader $\Delta d^*$ | $-$0.1185 | [$-$0.2185, $-$0.0333] | Matched-Plain |
| H1b: Decodability Peak Shift | Valence Self $\Delta d^*$ | 0.0067 | [$-$0.0533, 0.1000] | Matched-Plain |
| H1b: Decodability Peak Shift | Arousal Reader $\Delta d^*$ | 0.1996 | [$-$0.0267, 0.4259] | Matched-Plain |
| H1b: Decodability Peak Shift | Arousal Self $\Delta d^*$ | 0.0167 | [$-$0.2000, 0.2500] | Matched-Plain |
| H2: Sharing Reorganization | $\Delta\text{Sharing}$ (Valence) | $-$0.5529 | [$-$1.3236, 0.0760] | Matched-Plain |
| H2: Sharing Reorganization | $\Delta\text{Sharing}$ (Arousal) | $-$0.2784 | [$-$0.6151, 0.0489] | Matched-Plain |

> **Note:** $\Delta d^* = d^*_{\text{instruct}} - d^*_{\text{base}} > 0$ は、事後学習によって表現デコードピークが後段側（deeper側）へシフトしたことを示す。また、$\Delta\text{Sharing} < 0$ はReaderとSelfの表現共有度合いが事後学習によってタスク分離方向に再編されたことを示す。

## 2. Causal Peak Relocation (H3a)

| Family | Task | Axis | Base Peak ($d^*$) | Instruct Peak ($d^*$) | $\Delta d_C^*$ | $\Delta d_{\text{center}}$ |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| GEMMA | Reader | Arousal | 0.760 | 0.560 | $-$0.200 | $-$0.106 |
| GEMMA | Reader | Valence | 0.520 | 0.080 | $-$0.440 | 0.098 |
| GEMMA | Self | Arousal | 0.560 | 0.240 | $-$0.320 | 0.010 |
| GEMMA | Self | Valence | 0.000 | 0.520 | 0.520 | 0.044 |
| LLAMA | Reader | Arousal | 0.467 | 0.467 | 0.000 | 0.013 |
| LLAMA | Reader | Valence | 0.533 | 0.000 | $-$0.533 | $-$0.089 |
| LLAMA | Self | Arousal | 0.067 | 0.800 | 0.733 | 0.254 |
| LLAMA | Self | Valence | 0.467 | 0.933 | 0.467 | 0.341 |
| OLMO | Reader | Arousal | 0.200 | 0.600 | 0.400 | 0.043 |
| OLMO | Reader | Valence | 0.600 | 0.267 | $-$0.333 | $-$0.170 |
| OLMO | Self | Arousal | 0.067 | 0.333 | 0.267 | $-$0.087 |
| OLMO | Self | Valence | 0.600 | 0.133 | $-$0.467 | $-$0.296 |
| QWEN | Reader | Arousal | 0.037 | 0.000 | $-$0.037 | $-$0.144 |
| QWEN | Reader | Valence | 0.778 | 0.111 | $-$0.667 | $-$0.042 |
| QWEN | Self | Arousal | 0.296 | 0.852 | 0.556 | 0.027 |
| QWEN | Self | Valence | 0.593 | 0.259 | $-$0.333 | $-$0.178 |

> **Note:** 因果介入におけるピークおよび重心深度変位 $\Delta d_C^* = d_{C,\text{Instruct}}^* - d_{C,\text{Base}}^*$ は、事後学習に伴う因果部位の後段移行量を示す。なお本集約表は記述的集約値であり、ピーク再配置等の仮説検証の根拠とはせず、H3の統計的推論はsample-level LMMに基づく。またBase/Instruct差はrandomized interventionではないため、post-trainingのcausal effectではなくpost-training-associated reorganizationとして扱う。

## 3. Causal Specificity Controls (H3b)

| Family | Task | Condition | Axis | $C_{\text{raw}}$ | $C_{\text{rand}}$ | $C_{\text{perp}}$ | $C_{\text{net,rand}}$ (Primary) | Zero-Ablation |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| OLMO | Reader | Base-Plain | Valence | 0.002 | 0.002 | 0.002 | $-$0.000 | 0.016 |
| OLMO | Reader | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.019 |
| OLMO | Self | Base-Plain | Valence | 0.002 | 0.002 | 0.002 | $-$0.000 | 0.016 |
| OLMO | Self | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.020 |
| OLMO | Reader | Matched-Plain | Valence | 0.004 | 0.004 | 0.004 | $-$0.000 | 0.053 |
| OLMO | Reader | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.016 |
| OLMO | Self | Matched-Plain | Valence | 0.004 | 0.004 | 0.004 | $-$0.000 | 0.075 |
| OLMO | Self | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.037 |
| LLAMA | Reader | Base-Plain | Valence | 0.003 | 0.003 | 0.003 | 0.000 | 0.033 |
| LLAMA | Reader | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.015 |
| LLAMA | Self | Base-Plain | Valence | 0.003 | 0.003 | 0.003 | 0.000 | 0.032 |
| LLAMA | Self | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.014 |
| LLAMA | Reader | Matched-Plain | Valence | 0.003 | 0.003 | 0.003 | $-$0.000 | 0.020 |
| LLAMA | Reader | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.011 |
| LLAMA | Self | Matched-Plain | Valence | 0.003 | 0.003 | 0.003 | $-$0.000 | 0.020 |
| LLAMA | Self | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.012 |
| GEMMA | Reader | Base-Plain | Valence | 0.005 | 0.005 | 0.005 | 0.000 | 0.031 |
| GEMMA | Reader | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.051 |
| GEMMA | Self | Base-Plain | Valence | 0.005 | 0.005 | 0.005 | $-$0.000 | 0.036 |
| GEMMA | Self | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.054 |
| GEMMA | Reader | Matched-Plain | Valence | 0.006 | 0.006 | 0.006 | $-$0.000 | 0.054 |
| GEMMA | Reader | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.036 |
| GEMMA | Self | Matched-Plain | Valence | 0.006 | 0.006 | 0.006 | 0.000 | 0.059 |
| GEMMA | Self | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.030 |
| QWEN | Reader | Base-Plain | Valence | 0.004 | 0.004 | 0.004 | 0.000 | 0.029 |
| QWEN | Reader | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.008 |
| QWEN | Self | Base-Plain | Valence | 0.004 | 0.004 | 0.004 | $-$0.000 | 0.022 |
| QWEN | Self | Base-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.007 |
| QWEN | Reader | Matched-Plain | Valence | 0.007 | 0.007 | 0.007 | $-$0.000 | 0.122 |
| QWEN | Reader | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | 0.000 | 0.028 |
| QWEN | Self | Matched-Plain | Valence | 0.007 | 0.007 | 0.007 | 0.000 | 0.121 |
| QWEN | Self | Matched-Plain | Arousal | 0.001 | 0.001 | 0.001 | $-$0.000 | 0.027 |

> **Note:** $C_{\mathrm{net,rand}}>0$ は、affect-related directionの平均介入効果がrandom-direction controlより大きい方向にあることを示す。なお本集約表は記述的集約値であり、H3の統計的結論はsample-level LMMを根拠とする。またBase/Instruct差はrandomized interventionではないため、post-trainingのcausal effectではなくpost-training-associated reorganizationとして解釈する。

## 4. Causal Relocation Mixed-Effects Model (H3c LMM)

### 4.1 Primary Interventions and Post-training Effects

| Axis | Predictor / Parameter | Estimate ($\beta$) | 95% CI | $p$-value | FDR $q$ |
|:---|:---|:---:|:---:|:---:|:---:|
| Valence | Post-training $\times$ Depth (Primary) | 0.000090 | [$-$0.000054, 0.000234] | 0.223 | 0.304 |
| Valence | Post-training $\times$ Task (Primary) | 0.000163 | [0.000044, 0.000282] | 0.007 | 0.044 |
| Valence | Post-training $\times$ Task $\times$ Depth (Primary) | $-$0.000192 | [$-$0.000396, 0.000012] | 0.065 | 0.196 |
| Valence | Post-training (Instruct = 1) (Secondary) | $-$0.000095 | [$-$0.000179, $-$0.000011] | 0.027 | --- |
| Arousal | Post-training $\times$ Depth (Primary) | $-$0.000019 | [$-$0.000055, 0.000017] | 0.304 | 0.304 |
| Arousal | Post-training $\times$ Task (Primary) | $-$0.000017 | [$-$0.000047, 0.000013] | 0.277 | 0.304 |
| Arousal | Post-training $\times$ Task $\times$ Depth (Primary) | 0.000034 | [$-$0.000017, 0.000085] | 0.196 | 0.304 |
| Arousal | Post-training (Instruct = 1) (Secondary) | 0.000006 | [$-$0.000015, 0.000028] | 0.549 | --- |

> **Note:** Primary confirmatory inferenceは Alignment $\times$ Depth、Alignment $\times$ Task、Alignment $\times$ Task $\times$ Depth のprespecified interaction termsに基づく。Valenceにおいて $\text{Post-training} \times \text{Task}$ がFDR補正後も有意（$q = 0.044$）となり部分的に支持されたが、層深度の再配置（$\text{Post-training} \times \text{Depth}$等）およびArousalの全interactionはFDR補正後の基準を満たさなかった。なおBase/Instruct差はrandomized interventionではないため、post-trainingのcausal effectではなくpost-training-associated reorganizationとして解釈する。

### 4.2 Full LMM Parameter Estimates

| Axis | Term | Estimate ($\beta$) | 95% CI | $p$-value | FDR $q$ |
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

## 5. Output Distribution Recovery (H4)

| Family | Task | Matched AUC | $\Delta\text{EMD AUC}$ | Max Recovery | Best Depth ($d^*$) | Native AUC | Aligned AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| QWEN | Reader | 0.189 | 0.056 | 0.439 | 0.370 | 0.007 | $-$0.001 |
| QWEN | Self | 0.181 | 0.062 | 0.417 | 0.407 | 0.052 | $-$0.000 |
| LLAMA | Reader | 0.005 | 0.001 | 0.113 | 0.400 | 0.158 | 0.001 |
| LLAMA | Self | $-$0.006 | $-$0.000 | 0.078 | 0.400 | 0.125 | $-$0.000 |
| GEMMA | Reader | $-$0.052 | $-$0.012 | 0.142 | 0.040 | $-$0.072 | $-$0.011 |
| GEMMA | Self | 0.001 | 0.005 | 0.326 | 0.000 | $-$0.025 | $-$0.009 |
| OLMO | Reader | $-$0.170 | $-$0.037 | 0.006 | 0.800 | 0.044 | 0.001 |
| OLMO | Self | $-$0.160 | $-$0.045 | 0.000 | 1.000 | 0.070 | $-$0.001 |

> **Note:** Instructモデルの内部表現を同一familyのBaseモデル由来表現で置換またはalignmentした場合に、InstructのVA output distributionがBase distributionへどの程度接近するかを評価した（Matched AUC, $\Delta\text{EMD AUC}$, Max Recovery）。事前定義された4ファミリー設計（Qwen, Llama, Gemma, OLMo）に基づき、全モデルでMatched-Plain recoveryの実測値が得られた。なおBase/Instruct差はrandomized interventionではないため、post-trainingのcausal effectではなくpost-training-associated reorganizationとして解釈する。

## 6. Pre-registered Hypotheses Testing Summary (H1--H4)

| Hypothesis | Key Pre-registered Metric | Estimate | 95% CI | $p$ / FDR $q$ | Supported? |
|:---|:---|:---:|:---:|:---:|:---:|
| H1a: Geometry Reorganization | Reader Procrustes Distortion | 1.4553 | [0.6387, 2.8126] | --- | Descriptive |
| H1a: Geometry Reorganization | Self Procrustes Distortion | 2.8716 | [0.6603, 6.9954] | --- | Descriptive |
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

> **Note:** H1aはcross-family descriptive summaryとして扱う。Base--Instruct間にはReader / Self双方でrepresentation-geometric disparityが観測されたが、Procrustes distortionのCIが0を除外すること自体をnull-hypothesis testとは解釈しない。一方、H1bのprespecified positive peak shiftおよびH2のReader--Self sharing reorganizationは、4-family bootstrap CIに基づく事前定義criterionを満たさなかった。H3（Causal Reorganization）は全面的な棄却ではなく、sample-level LMMにおいてValenceのtask-dependentな変化（Alignment $\times$ Task, FDR $q = 0.044$）のみ部分的に支持されたが、層深度の再配置（depth relocation; Alignment $\times$ Depth等）は支持されなかった（なおBase/Instruct差はrandomized interventionではないため、post-trainingのcausal effectではなくpost-training-associated reorganizationとして解釈する）。H4のRecovery Asymmetry（Self--Reader AUC差）も95% CIがゼロを跨ぎ支持されなかった。

