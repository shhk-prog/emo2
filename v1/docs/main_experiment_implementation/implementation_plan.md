# [実装計画(改訂版)] LLM情動反応性評価・本実験（Main Experiment）コードの実装

ユーザーからのフィードバックに基づき、論文の主張（「機能的内部表現」の検証）と実装の整合性を大幅に高めるための修正・厳格化計画です。

## 概要と目的の修正

本実験では「機能的内部表現」の検証に特化し、抽出テンソルの巨大化を防ぐため **段階的検証 (Phase B1 ~ B4)** を導入します。また、抽出位置の厳格な固定、`stimulus_id` 単位のデータ分割 (dev/test分離)、因果介入 (Patching/Ablation/Steering) のクリーンな設計を徹底します。

---

## 修正されるモジュール構成 (Proposed Changes)

以下の構造に再編・拡張します。

### `src/affective_empathy_eval/`
- **`metrics.py`** [MODIFY]
  - 主指標: $\Delta V, \Delta A$
  - 副指標: $R$, Anchor Direction Alignment (ADA)
  - 補助/探索指標: 事後距離 $D$, Stimulus-to-response gain
  - 品質指標: 非反応率 $P(R=0)$, 有効率
  - *修正: Over-empathy Index は削除し、ADA 等の命名を固定。*
- **`manifests.py`** [NEW]
  - 保存する抽出テンソル (float16/bfloat16) に対して、モデルrevision、層番号、トークン対象位置、条件、stimulus_id などを一元管理・追跡するクラス群。
- **`splits.py`** [NEW]
  - `stimulus_id` で grouped split を行い、同一刺激が train/dev/test にまたがらない分割を行うロジック。
  - 層選択(dev)と最終介入検証(test)の分離。
- **`extraction.py`** [MODIFY]
  - 対象位置の明確化: `stimulus_last_token`, `prompt_last_token`, `stimulus_mean_pool`, `first_generated_token_input`
  - トークン offset の管理とテンソルの float16 保存・manifest 連携。
- **`probing.py`** [MODIFY]
  - 評価指標の拡充 ($R^2$, MAE, RMSE, Pearson, Spearman)。
  - PCA/正規化の train fold 内 fit。
  - 4種類の RSA 計算 (対 human_VA, 対 reported_post, 対 delta, 対 expressed_response) を分離。
- **`intervention.py`** [MODIFY]
  - **Activation patching**: Clean, Corrupted, Patched の 3 run 必須化および `Recovery` スコアの計算。近接刺激対 (Patch pair) の設計導入。
  - **Ablation**: Neutral replacement を主分析とする設計。
  - **Steering**: 開発データ (high/low) の平均差分で方向推定し、hold-out test で複数強度 $\alpha$ の注入と単調性評価。
- **`evaluation.py`** [NEW]
  - 介入効果を「直接的評価（次トークン logit, VA出力確率）」と「生成的評価（free response）」の二系統で評価するロジック。
- **`controls.py`** [MODIFY]
  - Lexical, Contextual, Neutral, Label-shuffled (分析段階でのシード置換) への定義の厳格化。

### `scripts/`
- **`run_main_experiment.py`** [MODIFY]
  - 段階的検証 (Phase B1 向けに hidden state のみを抽出)。
- **`run_probing.py`** [MODIFY]
  - `splits.py` を用いた dev セット上の実行。
- **`run_causal_intervention.py`** [MODIFY]
  - `evaluation.py` を用いた直接 logit 評価。
- **`validate_main_run.py`** [NEW]
  - Activation manifest とログの完全性・整合性検査。
- **`summarize_main_run.py`** [NEW]
  - 層別結果・介入結果の最終的な統合と出力。

---

## 検証計画 (Verification Plan) の補強

### 1. 拡充される Unit tests (`tests/`)
- $R=0$ やゼロベクトル時などの ADA・Gain 例外処理。
- 刺激トークン mask と `stimulus_last_token` 位置抽出の正確性。
- `splits.py` による同一 `stimulus_id` の情報漏洩 (leakage) 防止確認。
- Patching 時の指定層・位置以外の不変性チェック。
- Shuffle 統制のシード再現性・置換不変性。

### 2. 実機スモークテスト (Tiny Model Integration Test)
- `Dry-run` でのランダム数値テストに加え、CPU で動く極小サイズの Causal LM (例: `HuggingFaceM4/tiny-random-LlamaForCausalLM` など) を用い、実際に hook 注入、forward pass, patching, ablation, steering が正しくエラーなく動作するかを通しで結合テストします。

---

## 実装の優先順位

今回の承認後、安全かつ確実に進めるため以下の順序で実装・修正を行います。
1. `metrics.py`, `splits.py`, `manifests.py`, および `controls.py` の統制・ラベル処理ロジックの修正/新規作成
2. `extraction.py` の抽出位置厳密化 (Phase B1: hidden state のみ)
3. `probing.py` と RSA の修正・`splits.py` への適合
4. `intervention.py` における Patching (Clean/Corrupted) と Ablation の厳密化
5. `evaluation.py` の直接 logit 評価追加
6. 実機スモークテストと全体のテスト通過確認
