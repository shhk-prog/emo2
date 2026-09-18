# 科学的リファインメント実装検証レポート (Walkthrough)

## 概要
本作業では、機械学習研究リポジトリ (`Behavioral / v1 / v2 / v3`) において、論文の科学的主張チェーン（`Covariation → Representation / Causality → Post-training Reorganization → Causal Utilization`）を厳密に支えるため、測定妥当性、因果介入オペレータ、統計モデル、データリーク防止、およびレイヤーインデックス定義の全面修正を実施しました。

---

## 実施した変更内容

### 1. V3 因果介入オペレータの加算注入統一 (`h' = h + α * σ_h * d_hat`)
- **修正ファイル**:
  - `src/affective_empathy_eval/models/hooks.py`: `apply_direction_intervention`, `register_direction_intervention_hook` を実装。単位方向ベクトル $\hat{d}$ と活性化スケール $\alpha \cdot \sigma_h$ の乗算分離を保証。
  - `src/affective_empathy_eval/likelihood.py`: `generation_patch` に `mode` ("inject" / "replace") を導入し加算注入をサポート。
  - `v3/primary/run_rq1_state_induction.py`: 全置換フックを廃止し、Dose-response, Specificity (d_rand, d_perp), Topic control すべてで加算注入に統一。
  - `v3/primary/run_rq2_spatiotemporal_maps.py`: `generation_patch` を加算注入に統一。
  - `v3/primary/run_confirmatory_replication.py`: Discovery と同一の加算注入オペレータ・同一スケール定義に統一。

### 2. V3 介入サニティテストの作成と検証
- **ファイル**: `tests/test_v3_interventions_sanity.py`
  1. $\alpha = 0$ で baseline 出力と一致
  2. inject で $h' - h = \alpha \cdot \sigma_h \cdot \hat{d}$ と完全一致
  3. replace で指定ベクトルへ置換
  4. direction norm を変えても unit normalize 後の介入量は不変
  5. random / orthogonal / affective direction で scale 定義が同一
  6. Confirmatory と Discovery で同一の $\alpha$ が同一の activation norm change を生む

### 3. Behavioral カップリング・Dose-Response・Bootstrap CI の刷新
- **修正ファイル**:
  - `src/affective_empathy_eval/statistics.py`: `compute_correlation_bootstrap_ci` を追加。
  - `behavioral/analysis/summarize_behavioral_aipsy.py`:
    - **RQ4 Coupling**: Primary metric を matched pair の刺激変化に対する変位カップリング $\text{corr}(\Delta_{reader}, \Delta_{self})$（$\Delta = \text{clinical} - \text{neutral}$）へ刷新。刺激間の生の静的相関は secondary/reference として分離。pair_id 単位の 95% Bootstrap CI を計算。
    - **RQ2 Dose-Response**: 単純縦積み集計を Primary から外し、同一 triplet/pair 内の反復測定構造を保持するトリプレット内線形スロープ $b_i = (E[Y_i^{\text{clinical}}] - E[Y_i^{\text{neutral}}]) / 2.0$、対応のある 1 標本 $t$ 検定 ($H_0: \bar{b} = 0$)、triplet 単位の 95% Bootstrap CI を Primary 化。
    - **RQ1 Sensitivity**: paired mean difference および Cohen's $d_z$ の 95% Bootstrap CI を出力。
    - **BH-FDR**: Family 単位（Sensitivity, Dose-Response, Coupling）の多重比較補正を明記。
- **検証テスト**: `tests/test_behavioral_coupling.py` (2 passed)

### 4. レイヤーインデックス定義の共通化
- **修正ファイル**:
  - `src/affective_empathy_eval/geometry.py`: `get_block_hidden_state(hidden_states, layer_idx)` を実装。$0 \le l < L$ に対し Transformer ブロック出力である `hidden_states[l + 1]` を厳密に取得。relative depth $l / (L - 1)$ の定義を統一。
  - `v1/primary/run_phase_a.py`, `v1/primary/run_phase_b.py`: レイヤー取得を共通関数へ移行。

### 5. V1 Phase B Generalization 設計刷新とリーク防止
- **修正ファイル**:
  - `v1/primary/run_phase_b.py`: 呼称を "semantic transformation sensitivity" に変更。Pair-Aware evaluation（train: training pair_ids, test: held-out pair_ids）を実装し、同一 pair 由来データのリーク防止アサーション（`assert len(train_pairs.intersection(test_pairs)) == 0`）を導入。

### 6. V2 Aligned Activation Patch 統制実験 & Config 参照バグ修正
- **修正ファイル**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py`: `v2_config.get("bootstrap", {}).get("n_boot")` 参照バグを修正。
  - `v2/primary/run_rq4_recovery_patching.py`: `n_boot` 参照バグを修正。直接パッチ（Condition A）に加え、train split のみから直交 Procrustes 行列 $R_l$ を学習して適用する Aligned Procrustes パッチ（Condition B）を比較制御実験として実装。

### 7. V3 Confirmatory の Data Reuse 完全排除
- **修正ファイル**:
  - `v3/primary/run_confirmatory_replication.py`: 全データ fit probe を廃止し、`pair_id` 単位の GroupKFold による cross-fitting を導入。train fold で direction と std を推定し、独立な test fold のみで介入効果を集約。

### 8. 結果キャッシュの安全化
- **修正ファイル**:
  - `src/affective_empathy_eval/manifests.py`: `RunManifest` に `intervention_version` (`v3_additive_injection_v2`), `config_hash`, `dataset_hash`, `candidate_space` を追加。旧 replacement キャッシュとの混在を防止する検証関数 `is_manifest_matching` を実装。

### 9. Behavioral Unified Runner & 再現性固定
- **修正ファイル**:
  - `src/affective_empathy_eval/run.py`: `run_behavioral` 内で raw 評価後に自動要約 (`summarize_behavioral_emobank.py`, `summarize_behavioral_aipsy.py`) を実行する一気通貫パイプラインを実装。
  - `uv.lock` を生成・配置。

### 10. ドキュメントおよび論文セクション名の整合
- **修正ファイル**:
  - `README.md`, `behavioral/README.md`, `v1/primary/README.md`, `v1/README.md`, `v2/primary/README.md`, `v2/README.md`, `v3/primary/README.md`, `v3/README.md`
  - 論文 Section 名を確定：
    - **§3 Behavioral Characterization** (Alignment and Coupling between Reader and Self Perspectives)
    - **§4 Shared Representation and Causal Overlap in Base Models**
    - **§5 Post-training-Associated Reorganization of Affect-Relevant Computations**
    - **§6 From Decodability to Causal Leverage: Sufficiency, Specificity, and Spatiotemporal Dynamics**

### 11. 論文主張の査読耐性向上・表現精緻化
- **Core Thesis**: 「単なる出力上の模倣」を削除し、`LLM self-reports are systematically related to affect-relevant internal representations, but this relationship is partial and task-dependent. Across Base–Instruct pairs, the representation–report relationship exhibits post-training-associated reorganization, and decodable affect-relevant information is not uniformly causally relevant: measurable causal leverage over self-report is concentrated at particular layers and stages of computation.` へ刷新。
- **5種類の過大主張表現の適正化**:
  1. `shared mechanism` $\to$ `partially overlapping causally relevant representations and intervention-sensitive sites`
  2. `post-training causes` $\to$ `post-training-associated reorganization`（モデルサイズ差を抑えた複数ファミリーのBase–Instruct対による再現性検証）
  3. `acquire causal leverage` $\to$ `where affect-relevant information exerts measurable causal leverage over self-report`
  4. `utilization` と `causal leverage` の分離（Direction injection: sufficiency/leverage, Subspace removal: necessity/endogenous relevance）
  5. `only at particular stages` $\to$ `concentrated at particular layers and generation stages`（only を削除）
- **Behavioral covariation**: `Reader and Self covary in their responses to controlled affective changes`
- **仮説図**: 内部表現から各タスク計算への分岐として記述し、共有度をV1で検証する論理へ修正
- **729 VAD / 81 VA**: 各Stage内のcontrastとrelative patternを主たる推論対象とし、Supplementary感度分析を明記

---

## テスト実行結果

仮想環境 (`.venv/bin/python`) において、CPU テストスイートを実行し、既存機能のリグレッションゼロと新規機能の正常動作を確認しました。

```bash
.venv/bin/python -m pytest tests/ -k "not test_integration and not test_gpu" -v
```

**結果**: **75 passed in 86.29s**

主な検証対象テスト:
- `tests/test_v3_interventions_sanity.py`: 6 passed (全介入サニティ項目)
- `tests/test_behavioral_coupling.py`: 2 passed (変位カップリング・トリプレット内 dose-response)
- `tests/test_refinement_suite.py`: 6 passed (インデックスマッピング、候補空間サイズ 81/729、pair_id リーク防止アサート、キャッシュ不一致検出、config 伝播)
- 他既存ユニットテスト 61 件すべて passed
