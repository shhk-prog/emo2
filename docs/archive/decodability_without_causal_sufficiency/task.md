# タスクリスト: 論文の中心命題転換と実測データの再解釈

- [x] 1. 中心命題の転換とタイトル変更 <!-- id: 0 -->
  - 旧: 「Post-Training Decouples...」「Why Predictively Aligned Representations Fail...」
  - 新: `Decodability Without Causal Sufficiency: A Case Study of Affect-Relevant Representations Across Base and Post-Trained Language Models`
  - 主張の核: 「Linear decodability does not imply local causal sufficiency」
- [x] 2. Within-model実験の位置づけ修正 <!-- id: 1 -->
  - 「Positive control（正の対照）」という呼称を排除し、「Within-model causal sufficiency test」「Within-model substitution control」へ改称
  - 局所活性化（L15 MLP / 残差ストリーム最終トークン）が同一モデル内ですら下流報告分布を動かせない事実（0.00%）を、プロービングの限界を示す決定的証拠として整理
- [x] 3. 多様体診断（Mahalanobis $D_M$ / Cosine / AUC）の解釈の反転・適正化 <!-- id: 2 -->
  - $D_M \approx 4.98$ は「ID適合」ではなく「高次元ガウス薄殻定理（Annulus theorem）から外れた、平均 $\mu$ への異常な中心凝縮（Center Collapse / Over-regularization）」であることを明記
  - Two-sample AUC = 0.6386 は「ほぼ識別不能」ではなく「統計的に非典型的（atypical）」として正確に記述
  - Cosineの異常な類似性（0.9950）も共有平均成分の過剰復元として考察
- [x] 4. All-tokenパッチングのアーティファクト処理 <!-- id: 3 -->
  - +1088% のEV Shiftは分母不安定性によるアーティファクトであることを注記し、主結果から除外
- [x] 5. `v3/docs/paper.md` の全面更新 <!-- id: 4 -->
- [x] 6. ドキュメント保存（`task.md`, `implementation_plan.md`, `walkthrough.md`） <!-- id: 5 -->
