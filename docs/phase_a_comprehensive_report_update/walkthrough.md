# 修正内容の確認 (Walkthrough): Phase A 総合レポート E1-A / E1-B 表の拡充

## 実施概要
ユーザーからの要請に基づき、以下を実施しました。

1. **E1-A (EmoBank) および E1-B (AIPsy-Affect) の実行状況確認**
   - 全8モデル（Qwen 2.5 1.5B, Llama 3.2 1B, Gemma 2 2B, Mistral 7B の各 Base / Instruct）の派生データディレクトリ（`v1/results/derived/v1_phase_a/<model_prefix>/`）を確認。
   - `e1_emobank_decodability.csv`、`e1_aipsy_decodability.csv`、`e2_emobank_geometry.csv`、`e2_aipsy_geometry.csv`、および各要約ファイル `phase_a_summary.md` がすべて完全に存在し、**E1-A および E1-B の両実験が全モデル全層で正常に完走・保存されていること**を確認しました。

2. **総合レポートの修正**
   - 対象ファイル: [`v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md)
   - Part 1 において、従来の表1を **表 1-A: E1-A (EmoBank)** に変更し、新たに **表 1-B: E1-B (AIPsy-Affect)** を追加しました。

## 追加・更新されたテーブル内容

### 表 1-A: E1-A (Decodability) ピーク層と復元精度（EmoBank: 人間VAD評定 $R^2$）
| モデルファミリー | アライメント | 総層数 | Reader ピーク層 ($l^*_R$) | Reader $R^2$ | Self ピーク層 ($l^*_S$) | Self $R^2$ | ピーク層間乖離 $|l^*_R - l^*_S|$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | 28層 | Layer 25 | **0.371** | Layer 23 | **0.375** | **2 層** |
| **Qwen 2.5 1.5B** | Instruct | 28層 | Layer 14 | **0.234** | Layer 14 | **0.288** | **0 層** |
| **Llama 3.2 1B** | Base | 16層 | Layer 14 | **0.327** | Layer 13 | **0.275** | **1 層** |
| **Llama 3.2 1B** | Instruct | 16層 | Layer 10 | **0.326** | Layer 10 | **0.390** | **0 層** |
| **Gemma 2 2B** | Base | 26層 | Layer 26 | **-0.006** | Layer 26 | **0.007** | **0 層** |
| **Gemma 2 2B** | Instruct | 26層 | Layer 26 | **-0.032** | Layer 26 | **-0.027** | **0 層** |
| **Mistral 7B** | Base | 32層 | Layer 32 | **0.172** | Layer 32 | **0.197** | **0 層** |
| **Mistral 7B** | Instruct | 32層 | Layer 21 | **0.487** | Layer 18 | **0.480** | **3 層** |

### 表 1-B: E1-B (Decodability) ピーク層と識別精度（AIPsy-Affect: 臨床・感情刺激分類 ROC-AUC）
| モデルファミリー | アライメント | 総層数 | Reader ピーク層 ($l^*_R$) | Reader AUC | Self ピーク層 ($l^*_S$) | Self AUC | ピーク層間乖離 $|l^*_R - l^*_S|$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen 2.5 1.5B** | Base | 28層 | Layer 17 | **0.977** | Layer 17 | **0.975** | **0 層** |
| **Qwen 2.5 1.5B** | Instruct | 28層 | Layer 18 | **0.964** | Layer 14 | **0.963** | **4 層** |
| **Llama 3.2 1B** | Base | 16層 | Layer 10 | **0.966** | Layer 10 | **0.967** | **0 層** |
| **Llama 3.2 1B** | Instruct | 16層 | Layer 10 | **0.982** | Layer 9 | **0.979** | **1 層** |
| **Gemma 2 2B** | Base | 26層 | Layer 14 | **0.863** | Layer 14 | **0.875** | **0 層** |
| **Gemma 2 2B** | Instruct | 26層 | Layer 14 | **0.835** | Layer 14 | **0.832** | **0 層** |
| **Mistral 7B** | Base | 32層 | Layer 11 | **0.977** | Layer 15 | **0.980** | **4 層** |
| **Mistral 7B** | Instruct | 32層 | Layer 16 | **0.991** | Layer 16 | **0.992** | **0 層** |

## 検証結果
- [`phase_a_comprehensive_report.md`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md) の Part 1 にて表 1-A、表 1-B、表 2 が連続して正しくレンダリングされていることを確認しました。
