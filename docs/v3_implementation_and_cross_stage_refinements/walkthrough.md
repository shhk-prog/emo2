# Walkthrough: V3 Implementation and Cross-Stage Refinements

本ドキュメントでは、包括的レビューフィードバックに基づき実施した **Behavioral / V1 / V2 / V3 / 共通基盤** の改訂内容および検証結果を報告します。

---

## 1. 改訂の主要ポイントと成果

### 1.1 V3 実モデル因果・統計パイプラインの本格実装 (最重要)
- **シミュレーション値の完全排除と退避**:
  - 既存の乱数生成シミュレーション結果（`v3_rq1_results.json`, `v3_gate_decision.json`, `v3_spatiotemporal_maps_qwen.json`, `v3_path_mediation_qwen.json`, `v3_confirmatory_*.json` など）をすべて `v3/results/mock_simulation_archive/` に退避し、論文 Primary 結果から完全除外。
- **実モデル解析スクリプトの実装**:
  - `v3/scripts/run_v3_state_induction.py`: Qwen-2.5-7B-Instruct 実モデル推論、差分平均方向 $d_V, d_A$ の推定とグラム・シュミット直交化、実モデル活性化介入（residual stream パッチング）、共通 `likelihood.py` による分布評価、Bootstrap 95% 信頼区間による正式 Go/No-Go ゲート判定（閾値 $\Delta V > 0.15$, $p < 0.01$）を実装。
  - `v3/scripts/run_v3_spatiotemporal_maps.py`: 刺激長・感情強度・基本周波数等の刺激共変量を統制した偏回帰係数 $\beta(l,t)$ の推定、因果影響 $C(l,t)$、層方向ピーク解離判定（$|\hat{l}_{\mathrm{dec}} - \hat{l}_{\mathrm{cau}}| > 0.15$）を実装。
  - `v3/scripts/run_v3_path_mediation.py`: 50% split Discovery での Mediator 自動選定 $\to$ 独立した 50% split Confirmation での 2D 射影除去パッチング、TE/NDE/NIE/MR (Mediation Ratio) 算出を実装。
  - `v3/scripts/run_v3_confirmatory_replication.py`: Llama-3.1, Gemma-2, Mistral-7B の実モデル解析ロジック、`--dry-run` 時の明確な mock simulation タグ付けを実装。
  - `v3/src/batch_likelihood.py`, `v3/scripts/run_qwen_recognition_baseline.py` 等: 共通 likelihood への一本化、`mean_attenuation_ratio` 命名を追加。

### 1.2 共通 Sequence Likelihood 基盤 (`src/affective_empathy_eval/`)
- **トークン境界・Prefix一致の厳密保証**:
  - `prompts.py`: `get_generation_stage_tokens()` を正規表現 span + tokenizer offset mapping による厳密同定に改訂し、`prompt_end` と `candidate_start` の責務を完全分離。
  - `likelihood.py`: `prepare_joint_sequence_with_boundary()` を実装し、strict prefix チェックおよび境界跨ぎ BPE マージの検出・対応を追加。
  - `evaluate_expected_va_from_prompt()`, `evaluate_expected_vad_from_prompt()` ヘルパーを追加。
- **テストスイートの拡充**:
  - `tests/test_likelihood.py` にプレフィックス・境界・オフセットテストを追加（10件全パス）。

### 1.3 Behavioral Stage の統合
- `v1/scripts/run_3way_vad_evaluation.py` および `v1/scripts/run_aipsy_4split_evaluation.py` の独自尤度計算を、共通の `src/affective_empathy_eval/likelihood.py` に統一。

### 1.4 V1 Phase C の厳密化と再解釈（Llama 3.2 1B Base の正式採択）
- **729候補の先頭トークン完全一致の実験的実証**:
  - `meta-llama/Llama-3.2-1B` の tokenizer を用い、全729個の VAD 候補文字列の先頭トークンを検証。
  - 結果: `unique_first_token_ids = [5018]`, `number_of_unique_first_tokens = 1`（全候補が `'{"'` で100%一致）。
  - Softmax 相対比較において先頭トークン確率は全候補共通の定数倍として相殺されるため、response-onset（回答開始位置）への差分ベクトル注入は、第2トークン以降の報告分布に対する純粋かつ有効な因果効果を測定していることを理論的・実験的に実証。
- **実験名称と介入モデルの再定義**:
  - 旧 E3 `Shared Causal Map` $\to$ 新 E3 **`Response-Onset Causal Map`** へ改訂。
  - 介入モデル:
    $$\Delta h_T^{(l)} = h_{T,\mathrm{aff}}^{(l,\mathrm{prompt\ end})} - h_{T,\mathrm{neu}}^{(l,\mathrm{prompt\ end})}$$
    $$\tilde{h}_{T,\mathrm{response\ onset}}^{(l)} = h_{T,\mathrm{response\ onset}}^{(l)} + \Delta h_T^{(l)}$$
  - E4 解釈: **Reader prompt-derived affect vector $\rightarrow$ Self response-onset state**（他者感情認識で形成された情動表現が、自己報告の生成開始段階で利用可能かという認知因果モデル）。
  - メタデータ [`intervention_metadata.json`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_c/llama3.2_1b_base/intervention_metadata.json) を配置し、12時間の計算結果を破棄せず正式 Primary 結果として保持。
  - [`generate_phase_c_report.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_c_report.py) の E3 レポート名称を `Response-Onset Causal Map` に更新。
- `v1/scripts/run_v1_phase_c_targeted_ablation.py` (E6):
  - `pair_id` 単位の random group split による Discovery $\to$ Confirmation パイプラインを自動化（`--auto-discover` オプション）。
  - 論文表記を過剰解釈のない `"Evidence consistent with task-specific causal specialization"` に改訂。
- `v1/scripts/run_v1_phase_c_causal_patching.py`:
  - 尤度および活性化抽出を共通基盤に統一。今後実行する他モデルも本介入定義（`prompt-derived affect difference -> response-onset intervention`）に統一。

### 1.5 V2 RQ4 およびディレクトリ構造の整理
- `v2/scripts/run_v2_recovery_patching.py`:
  - Base $\to$ Instruct のフォーマット統制（Instruct matched-plain control, `recovery_ratios_matched_plain`）を追加。Base/Instruct の hidden_size assertion を追加。
- **旧スクリプトの隔離**:
  - `v2/scripts/legacy/` ディレクトリを作成し、旧スクリーニング、SAE、出力ゲーティング、内省テスト等の旧スクリプト14件を移動。
  - `v2/scripts/legacy/README.md` を作成し、現行 Primary スクリプト（RQ1〜RQ4）との対応関係を明記。
  - `v2/README.md` のファイルツリーを更新。

### 1.6 テスト実行環境の整備
- ルートに `pytest.ini` を配置（`testpaths = tests`, `pythonpath = src`）。
- ルートからの `.venv/bin/pytest -q` で全 35 テスト（Core 29件 + V3 6件）が一発パス（100%）。

---

## 2. 変更ファイル一覧

| カテゴリ | ファイルパス | 変更内容 |
|---|---|---|
| **共通基盤** | `src/affective_empathy_eval/prompts.py` | regex span + offset mapping による生成ステージトークン同定 |
| | `src/affective_empathy_eval/likelihood.py` | 境界跨ぎ BPE 検出、strict prefix チェック、VA/VAD 評価ヘルパー |
| | `tests/test_likelihood.py` | 境界・プレフィックス・オフセットのユニットテスト追加 |
| | `pytest.ini` | ルート pytest 実行用設定ファイルの新規作成 |
| **Behavioral** | `v1/scripts/run_3way_vad_evaluation.py` | 共通 `likelihood.py` への一本化 |
| | `v1/scripts/run_aipsy_4split_evaluation.py` | 共通 `likelihood.py` への一本化 |
| **V1** | `v1/scripts/run_v1_phase_c_causal_patching.py` | 尤度・活性化抽出の共通化 |
| | `v1/scripts/run_v1_phase_c_targeted_ablation.py` | E6 Discovery $\to$ Confirmation 自動化、表記改訂 |
| **V2** | `v2/scripts/run_v2_recovery_patching.py` | Instruct matched-plain フォーマット統制追加、hidden_size 検証 |
| | `v2/scripts/legacy/*` (14ファイル) | 旧スクリプト群の移動・隔離 |
| | `v2/scripts/legacy/README.md` | Primary vs Legacy の対応表ドキュメント |
| | `v2/README.md` | スクリプト構成の更新 |
| **V3** | `v3/results/mock_simulation_archive/*` | 乱数シミュレーション JSON の退避 |
| | `v3/scripts/run_v3_state_induction.py` | Qwen 実モデルロード、$d_V, d_A$ 推定、介入、ゲート判定 |
| | `v3/scripts/run_v3_spatiotemporal_maps.py` | 共変量統制偏回帰 $\beta(l,t)$、因果 $C(l,t)$、ピーク解離判定 |
| | `v3/scripts/run_v3_path_mediation.py` | 50% split Discovery $\to$ Confirmation、TE/NDE/NIE/MR |
| | `v3/scripts/run_v3_confirmatory_replication.py` | 3モデル実推論パイプライン、mock タグ付け |
| | `v3/src/batch_likelihood.py` | 共通 `likelihood.py` への委任 |
| | `v3/scripts/run_focused_necessity_n100.py` | 共通 `likelihood.py` 利用、`mean_attenuation_ratio` |
| | `v3/scripts/run_generation_time_causal_sweep.py` | 共通 `likelihood.py` 利用、`mean_attenuation_ratio` |
| | `v3/scripts/run_qwen_recognition_baseline.py` | 共通 `likelihood.py` 利用、`mean_attenuation_ratio` |
| **記録文書** | `docs/v3_implementation_and_cross_stage_refinements/*` | `task.md`, `implementation_plan.md`, `walkthrough.md` |

---

## 3. テスト・検証結果

### 3.1 ユニットテスト (`pytest`)
ルートから仮想環境内の pytest を実行：
```bash
.venv/bin/pytest -q
```
**結果**:
- `tests/test_likelihood.py`: 10 passed
- `tests/test_prompts.py`: 7 passed
- `tests/test_stat_utils.py`: 5 passed
- `tests/test_v1_refinements.py`: 7 passed
- `tests/test_v3_experiments.py`: 6 passed
- **合計**: **35 passed, 4 warnings in 11.23s (100% 成功)**

### 3.2 構文および `--help` / `--dry-run` 検証
- V3 スクリプト群 (`run_v3_state_induction.py`, `run_v3_spatiotemporal_maps.py`, `run_v3_path_mediation.py`, `run_v3_confirmatory_replication.py`):
  - `--dry-run` モードで正常動作を確認。
  - 出力 JSON に `is_simulation: true`, `note: "DRY-RUN MOCK SIMULATION - NOT FOR PAPER PRIMARY RESULTS"` が付与されることを確認。
- V2 スクリプト (`run_v2_recovery_patching.py`):
  - `--dry-run` モードで `recovery_ratios_matched_plain` が計算され JSON に出力されることを確認。
- Behavioral スクリプト (`run_3way_vad_evaluation.py`, `run_aipsy_4split_evaluation.py`):
  - 共通 `likelihood.py` のインポートおよび引数パースが正常であることを確認。
- V1 Phase C スクリプト (`run_v1_phase_c_targeted_ablation.py`):
  - `--auto-discover` オプションで pair_id 分割から Confirmation まで一貫して動作することを確認。

---

## 4. 運用上の注意点と次のステップ

1. **実モデル実験の実行**:
   - V3 の本番実験（Qwen, Llama, Gemma, Mistral）は、GPU 環境が確保された際に各スクリプトを `--dry-run` なしで実行してください。
2. **生データ・結果の不変性**:
   - `v3/results/mock_simulation_archive/` に退避されたファイルはシミュレーション値の履歴としてのみ参照し、本番の `results/raw/` または `results/derived/` に戻さないでください。
3. **再現性保証**:
   - テストの追加や機能修正を行う際は、ルートから `.venv/bin/pytest -q` を実行して回帰がないことを確認してください。
