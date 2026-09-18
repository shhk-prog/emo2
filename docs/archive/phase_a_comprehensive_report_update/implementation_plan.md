# 実装計画 (Implementation Plan): Phase A 総合レポート E1-A / E1-B 表の拡充

## 目的
`/mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md` において、Part 1 に掲載されている表が現在「EmoBank（Valence）」のみ（表1）となっているため、
1. E1-A (EmoBank)
2. E1-B (AIPsy-Affect)
の2つのデータセットごとのピーク層・性能対比表をそれぞれ明示的に分離・追加する。

## 現状の確認（実行状況）
- `v1/results/derived/v1_phase_a/<model_prefix>/` の全8モデルディレクトリにおいて：
  - `e1_emobank_decodability.csv`
  - `e1_aipsy_decodability.csv`
  - `e2_emobank_geometry.csv`
  - `e2_aipsy_geometry.csv`
  がすべて存在しており、**E1-A (EmoBank)・E1-B (AIPsy-Affect) ともに全8モデルで完全に実行完了**していることを確認。
- 各モデルの `phase_a_summary.md` にも、EmoBank（Valence $R^2$）および AIPsy-Affect（Condition 分類 ROC-AUC）のピーク層および指標が記録されている。

## 変更内容
`v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md` の Part 1 において、
- 従来の「表 1: E1 (Decodability) ピーク層と復元精度（EmoBank Valence）」を
  **「表 1-A: E1-A (Decodability) ピーク層と復元精度（EmoBank: 人間VAD評定 $R^2$）」** に改称。
- 新たに
  **「表 1-B: E1-B (Decodability) ピーク層と識別精度（AIPsy-Affect: 臨床・感情刺激分類 ROC-AUC）」** を追加。
- 従来の表2（E2 幾何構造パターンと転移性能）はそのまま維持。

### 表 1-B のデータ内容
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

## 検証
- `v1/results/derived/v1_phase_a/phase_a_comprehensive_report.md` の記述・Markdownレンダリングの整合性を確認。
