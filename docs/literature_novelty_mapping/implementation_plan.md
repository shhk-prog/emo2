# Implementation Plan: 既存研究と本研究の差異・新規性のマッピング計画

## 1. 整理の軸（4つの次元）
1. **対象（What）**: 認識（他者） vs 自己報告（自己）
2. **測定手法（How）**: 単一トークン・Greedy生成 vs 全候補列尤度分布（Sequence Likelihood Protocol）
3. **因果性と表現の関係（Mechanism）**: プロービング（相関） vs ステアリング（強制介入） vs アラインメント・パッチング（結合因果性）
4. **モデル比較（Post-training）**: 単一モデルの観察 vs BaseとInstructの対照による結合変化の局所化

## 2. 構成案
- **「すでにあるところ」の総括**:
  - 表現の存在（プロービング成功）
  - 感情認識タスクの高性能
  - Activation Steering による生成テキストの操作
  - Instructモデルの自己報告における中立化現象の観察
- **「まだないところ（本研究の新規性）」の総括**:
  - 「表現が存在すること（Decodability）」と「自己報告への結合（Causal substitutability）」の乖離の証明
  - 事後学習による中立化の4大競合仮説（消去・幾何変換・一様抑制・分散的結合変化）の識別
  - Base-to-Instruct Aligned Cross-Model Patching による回路代替性の検証
  - Greedy崩壊の背後に潜む尤度空間での情動感度の実証
  - 最終出力層（lm_head / RMSNorm）単独原因説の否定
