# Behavioral / V1 / V2 / V3 科学的・実験的リファインメント実装計画

本計画は、論文の中心RQ「LLMの情動関連内部表現は自己報告とどのように結びついており、その関係はタスク、post-training、計算過程を通じてどのように変化するのか」を支持する単一の論理チェーン（Covariation → Representation / Causality → Post-training Reorganization → Causal Utilization）を厳密かつ堅牢にするための、全ステージ（Behavioral, V1, V2, V3）の実装・統計・因果介入オペレータの抜本的修正を定義します。

---

## 1. ユーザー確認事項 (User Review Required)

- **V3 介入オペレータの数式的統一**:
  - 全V3実験（RQ1, RQ2, Confirmatory, Controls）で、従来の全活性化置換（`register_patch_hook`）を完全廃止し、以下の加算介入（additive injection）に統一します：
    $$h' = h + \alpha \cdot \sigma_h \cdot \hat{d}$$
    ここで $\hat{d}$ は単位ノルム方向、$\sigma_h$ は対象層・位置での活性化スケール、$\alpha$ は無次元介入強度です。
  - `mode="inject"` を論文の Primary に統一し、`mode="replace"` は Supplementary control のみに限定します。
- **Behavioral カップリング指標の刷新**:
  - 従来の raw 相関（$\text{corr}(reader_{raw}, self_{raw})$）を Primary から外し、matched pair の刺激変化に対する変位カップリング $\text{corr}(\Delta_{reader}, \Delta_{self})$ （$\Delta = \text{affective} - \text{neutral}$）を Primary metric とします。
- **データリークの完全排除**:
  - V3 Confirmatory において、同一サンプルを用いた direction 推定と intervention 評価を禁止し、pair_id 単位の cross-fitting / holdout に移行します。
  - V1 Phase B において、同一 pair が train/test をまたぐことを禁止する assert を追加します。

---

## 2. 変更提案 (Proposed Changes)

### A. V3 Direction Intervention オペレータの統一 (`src/affective_empathy_eval`, `v3/`)

#### [MODIFY] [hooks.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/hooks.py)
- `register_direction_intervention_hook` および `apply_direction_intervention` 共通APIを拡充。
- 数式 $h' = h + \alpha \cdot \sigma_h \cdot \hat{d}$ を厳密に実装。`direction` は常に unit-normalize され、scale は $\alpha \cdot \sigma_h$ 側でスケーリング。
- `mode="inject"` (デフォルト) と `mode="replace"` (Supplementary) を明示制御。

#### [MODIFY] [likelihood.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)
- `compute_sequence_likelihoods_for_candidates` の `generation_patch` パラメータを改修。
- `patch_tensor` の無差別置換だけでなく、`mode="inject"`, `direction`, `alpha`, `hidden_std` を受け取り、加算注入を実行可能にする。

#### [MODIFY] [run_rq1_state_induction.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq1_state_induction.py)
- `register_patch_hook` で全置換していた箇所（Line 392, 412, 425等）を `register_direction_intervention_hook(mode="inject")` による加算介入へ変更。
- affective direction, random direction, orthogonal direction のすべてで同一の介入オペレータ・スケール定義を使用。

#### [MODIFY] [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- `generation_patch` に `mode="inject"` を設定し、`likelihood.py` と連携して加算注入を実施。

#### [MODIFY] [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
- `hidden_std` が抜けていた注入ロジックを $\alpha \cdot \sigma_h \cdot \hat{d}$ に統一。
- **Data Reuse 防止**: 全体で fit した probe で全サンプルに介入する構造を廃止し、pair_id 単位の cross-fitting または holdout 分割を導入。test fold のみで介入効果を集計。

---

### B. V3 Intervention Sanity Tests (`tests/`)

#### [NEW] [test_v3_interventions_sanity.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_v3_interventions_sanity.py)
1. $\alpha = 0$ で baseline 出力と一致
2. inject で $h' - h = \alpha \cdot \sigma_h \cdot \hat{d}$ と完全一致
3. replace で activation が指定ベクトルへ置換
4. direction norm を変えても unit normalize 後の介入量が変わらない
5. random direction, orthogonal direction, affective direction で scale 定義が同一
6. Confirmatory と Discovery で同一の $\alpha$ が同一の activation norm 変化を生む

---

### C. Behavioral 解析・指標の修正 (`behavioral/`)

#### [MODIFY] [summarize_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
- **RQ4 Coupling**: Primary metric を $\text{corr}(\Delta_{reader}, \Delta_{self})$ に刷新。
  - $\Delta_{reader} = reader_{affective} - reader_{neutral}$
  - $\Delta_{self} = self_{affective} - self_{neutral}$
  - Valence と Arousal を個別に算出。
  - pair_id 単位のリサンプリングによる 95% Bootstrap CI を計算。
  - 出力カラム: `model, alignment, dimension, n_pairs, correlation_type, r, ci_low, ci_high, p_value`
  - raw 相関は secondary / supplementary として別出力または別行に保存。
- **RQ2 Dose-Response**: 単純縦積み独立観測を Primary から外し、triplet 単位の repeated-measures（within-triplet linear slope または ordered contrast）を Primary に変更。
  - 独立単位: `triplet_id`。
  - bootstrap も `triplet_id` 単位で実施。
  - BH-FDR の補正 family をコード・コメントで明確化。
- **RQ1 Sensitivity**: paired mean difference, Cohen's $d_z$ について 95% Bootstrap CI を追加。

---

### D. Layer Indexing の全面統一 (`src/affective_empathy_eval`, `v1/`, `v2/`, `v3/`)

#### [NEW/MODIFY] [adapters.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/models/adapters.py) / [geometry.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/geometry.py)
- 共通関数 `get_block_hidden_state(hidden_states, layer_idx)` を提供:
  - `hidden_states[0]` は embedding output。
  - Transformer block $l$ ($0 \le l < L$) の出力は `hidden_states[l + 1]`。
- relative depth を一律 `layer_idx / (num_hidden_layers - 1)` と定義。
- V1 Phase A/B/C, V2, V3 の hidden_states 取得箇所を共通関数に置換。

---

### E. V1 Phase B の Generalization 再設計 (`v1/primary/run_phase_b.py`)

#### [MODIFY] [run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
- "held-out semantic generalization" の表現を "semantic transformation sensitivity" に全面改訂。
- 追加実験として **Pair-Aware Evaluation** を実装:
  - train: training pair_ids の original stimuli
  - test: held-out pair_ids の paraphrase / reversal
  - 同一 pair 由来刺激が train/test をまたがないよう `assert len(set(train_pairs).intersection(set(test_pairs))) == 0` を導入。

---

### F. V2 Aligned Activation Patch Control (`v2/primary/run_rq4_recovery_patching.py`)

#### [MODIFY] [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)
- 以下の2条件を明示的に比較可能な制御実験として追加:
  - Condition A: Direct Base -> Instruct activation patch
  - Condition B: Aligned Base -> Instruct activation patch (train 分割のみから学習した Orthogonal Procrustes 変換 $R$ を適用)
- 評価用サンプルを alignment fitting に使用しない厳格な holdout を保証。

---

### G. V2 Bootstrap Config Bug 修正 (`v2/primary/`)

#### [MODIFY] [run_rq1_rq2_cross_decoding.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py) & [run_rq4_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)
- `v2_config.get("statistics", {}).get("n_boot", 1000)` を `v2_config.get("bootstrap", {}).get("n_boot", 1000)` に修正。安全なフォールバックヘルパー関数を導入。

---

### H. Result Cache の安全化 & 出力ディレクトリ統一 (`manifests.py`, 各Stage)

#### [MODIFY] [manifests.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py)
- `RunManifest` に `intervention_version`, `dataset_hash`, `prompt_version`, `candidate_space` を追加。
- キャッシュ再利用時に現在の実行パラメータと manifest の整合性を検証する `verify_run_cache()` を提供。
- V3 介入バージョンを `"v3_additive_injection_v2"` とし、旧 replacement キャッシュの誤読を完全防止。

#### [MODIFY] [run.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py) & 各Stage Runner
- Behavioral の出力先を `results/raw/behavioral/` および `results/derived/behavioral/` に統一。
- `python -m affective_empathy_eval.run --stage behavioral` で raw 実行から summary / derived CSV 生成まで自動実行。

---

### I. 依存関係の再現性 & 総合ユニットテスト

- `uv.lock` の生成。
- 新規テストスイート `tests/test_v3_interventions_sanity.py`, `tests/test_behavioral_coupling.py`, `tests/test_layer_indexing_and_cache.py` を追加。

---

## 3. 検証計画 (Verification Plan)

### 自動テスト
- `pytest tests/test_v3_interventions_sanity.py -q`
- `pytest tests/test_behavioral_coupling.py -q`
- `pytest tests/test_layer_indexing_and_cache.py -q`
- `pytest tests/ -k "not test_all_production_entrypoints_execute" -q`
- `ruff check .`

### Dry-run 検証
- `python -m affective_empathy_eval.run --stage behavioral --model-set primary_small --dry-run`
- `python -m affective_empathy_eval.run --stage v1 --model-set primary_small --dry-run`
- `python -m affective_empathy_eval.run --stage v2 --model-set primary_small --dry-run`
- `python -m affective_empathy_eval.run --stage v3 --model-set primary_small --dry-run`
