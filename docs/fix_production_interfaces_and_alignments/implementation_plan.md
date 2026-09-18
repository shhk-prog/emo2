# 実装計画: 本番実行インターフェース不整合・位置交絡の解消とプロトコル統一

## 概要
前回のレビューで判明した、dry-run や単体テストをすり抜けていた本番実行（real run）時の致命的なバグ、インターフェース不一致、位置交絡（DとCのtoken position不一致）、未初期化変数、および集計不足を根本的に修正し、新鮮な本番再実行（fresh rerun）を確実に成功させる基盤を確立します。

---

## 提案する変更内容

### 1. Behavioral Stage の修復

#### [MODIFY] [`behavioral/primary/run_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
- `parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")` を追加。
- `device_map="auto"` のハードコードを見直し、指定された `args.device`（例: `cuda:0`）にモデルを確実にロードするように修正。

#### [MODIFY] [`behavioral/primary/run_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
- `device_map="auto" if args.device == "cuda" else None` により `cuda:0` 指定時に CPU ロードされてしまう不具合を解消。指定された GPU デバイスへ確実に配置。

#### [MODIFY] [`behavioral/analysis/summarize_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
- 単なる `sort_values("id")` スライス依存を廃止し、`pair_id`（または `base_id`）に基く厳密なペアマッチングを実装。
- **Sensitivity**: Clinical vs. Neutral ペア差分（Cohen's $d_z$、対応のある $t$ 検定、FDR補正）。
- **Dose-Response**: Neutral $\rightarrow$ Moderate $\rightarrow$ Clinical の単調推移検定（Jonckheere-Terpstra / Spearman 順位相関）。
- **Specificity**: Clinical vs. Complex Neutral 対照（統制文との差分検定）。
- **Reader-Self Coupling**: 3次元（V, A, D）それぞれにおける相関・Bootstrap 95% 信頼区間。

---

### 2. V1 Stage の補完・整合

#### [MODIFY] [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- **Arousal Geometry の追加**: Valence だけでなく Arousal (`geom_a`: `target="Arousal_human"`) に対しても cross-decoding, RSA, Procrustes アライメントを評価・記録。
- **AIPsy 分類プローブ**: `args.dataset in ["aipsy", "both"]` の際、AIPsy 4-Split の感情カテゴリ分類プローブ（Logistic Regression / Ridge 分類、Balanced Accuracy、F1、ROC-AUC）を実行。
- **Secondary Model-Output Target**: モデル自身の自己報告行動値 $E[V], E[A]$ に対するプローブ評価をオプションまたは同時記録可能に拡張。

#### [MODIFY] [`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
- `extract_single_layer_hidden_states()` の全呼び出しにおいて、`task_type=args.task_type`（または Reader/Self）および `is_instruct=args.is_instruct` を明示的に引数として渡すよう修正。

#### [MODIFY] [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py) & [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `run_v1()` において、`run_phase_c.py` に `--all-layers` を渡せるようオプションおよび既定動作を整備し、全層 causal map が取得されるようにする。

---

### 3. V2 Stage の位置交絡解消とバグ修正

#### [MODIFY] [`v2/primary/run_rq1_rq2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py)
- 活性化抽出の意味的アンカーを `stimulus_end` から `prompt_end`（または統一アンカー）に変更し、RQ3（Causal Map）の介入位置と一致させ、D と C の層解離比較におけるトークン位置交絡を完全に排除。

#### [MODIFY] [`v2/primary/run_rq4_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq4_recovery_patching.py)
- non-dry-run 実行時にクラッシュを引き起こす未初期化変数 `layer_mean_ratios_plain = []` をループ前に初期化。
- matched_plain の回復率（Recovery Ratio）計算において、分母を native chat の初期 EMD ではなく、matched_plain 自身の初期 EMD（`sample_initial_emds_plain`）から正しく計算するよう修正。

#### [MOVE] [`v2/primary/run_confirmatory_analysis.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_confirmatory_analysis.py) $\rightarrow$ `v2/legacy/`
- 旧パス依存・非推奨スクリプトを `v2/legacy/` へ移動・退避。

---

### 4. V3 Stage のインターフェース不一致・致命的バグの解消

#### [MODIFY] [`src/affective_empathy_eval/interventions.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/interventions.py)
- `extract_conditional_directions`: `method="ridge"`, `alpha=1.0` 等のキーワード引数を許容し、Ridge または最小二乗回帰で解けるよう拡張。
- `compute_orthonormal_subspace`: `(d_v, d_a)` の 2 引数および `([d_v, d_a])`（リスト渡し）の双方を受け取れるよう引数シグネチャを柔軟化。

#### [MODIFY] [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
- `from affective_empathy_eval.models.adapters import get_model_adapter` のインポートを追加。
- 未初期化変数 `all_H = {}` を層ループ前に初期化。
- `ActivationHookManager` の呼び出しを、正しいシグネチャを持つ `register_direction_injection_hook` および `register_subspace_removal_hook` へ修正。

#### [MODIFY] [`scripts/run_production_v3.sh`](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_v3.sh)
- `DEVICE="${1:-cuda:0}"; shift || true; EXTRA_ARGS=("$@")` とし、`--force-after-no-go` や `--dry-run` などの追加フラグが確実に Python CLI へ渡るように修正。

#### [MODIFY] [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- 因果介入サンプル選択を、先頭スライス固定（`range(n)`）から、シード固定（`seed=42`）のランダムサンプリング（`rng.choice(N, size=n_causal_intervene, replace=False)`）に変更。

---

### 5. 論文ドキュメント・Methods プロトコル表の整備

#### [MODIFY] [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md)
- Methods セクションに共通 protocol table を追加（Candidate space, Prompt 形式, Token 位置, Relative depth, Split 単位, Primary metric, 統計検定）。
- 主張のトーンを適切に調整（コード修正と再実行完了まで V2/V3 の因果利用の断定を弱め、「自己報告は内部表現と部分的・タスク依存に結びつく」とする）。

---

## 検証計画

### 自動テスト
1. **pytest スイート**: `.venv/bin/pytest -q` を実行し、既存の 61 テストが全て PASS することを確認。
2. **修正スクリプトの dry-run テスト**:
   - Behavioral: `python -m affective_empathy_eval.run --stage behavioral --dry-run`
   - V1: `python -m affective_empathy_eval.run --stage v1 --dry-run --max-samples 2`
   - V2: `python -m affective_empathy_eval.run --stage v2 --dry-run --max-samples 2`
   - V3: `python -m affective_empathy_eval.run --stage v3 --dry-run --max-samples 2 --force-after-no-go`
3. **Bash ランナー引数渡しテスト**:
   - `bash scripts/run_production_v3.sh cpu --dry-run --force-after-no-go` が正しく引数を受け取って完了することを確認。
