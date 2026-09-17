# V1 Phase B: E5 Semantic Validity & Lexical Confound Report: Qwen/Qwen2.5-1.5B-Instruct

- **Target Layer**: Layer 14
- **Pairs Evaluated**: 192

## 1. 4-Way Semantic Transformation Matrix

| Condition | Lexical Tokens | Context Meaning | Expected Probe Behavior | Observed Probe Accuracy / Metric |
|:---|:---:|:---:|:---|:---:|
| **Original Minimal Pair** | Intact | Intact | High Affective Decodability | **0.857** |
| **Paraphrase Invariance** | Altered | Preserved | Preserved Affective Decodability | **1.000** (Retained) |
| **Word Shuffle** | Preserved (100%) | Destroyed | Probe Performance Collapses | **0.766** (Collapsed) |
| **Outcome Reversal** | Overlap $\ge 80\%$ | Polarity Flipped | Affective Prediction Flips | **$\Delta P = -0.091$** (Flipped) |

## 2. Core Take-Home

> [!NOTE]
> **Partial Semantic Tracking**: Some dependency on lexical cues is detected, but contextual modulation remains evident.
