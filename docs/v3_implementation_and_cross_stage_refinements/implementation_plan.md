# 実装計画: V3実モデル解析の実装および全ステージ共通基盤の統合改訂

## 概要
最新版コードベースにおいて、Behavioral / V1 / V2 は大幅に改善されたものの、**V3（新Primaryスクリプト群）に実モデル解析を行わずシミュレーション値を生成する重大な未実装**が残っています。また、各ステージ間で Sequence Likelihood のトークン境界処理（`full_ids[prompt_len:]`）の不整合や境界跨ぎトークンの問題、V1 E6の二重解離パイプラインの自動化不備、V2 RQ4のフォーマット効果の統制不足、およびリポジトリ全体の pytest 収集エラーが存在します。

本計画では、以下の優先度に基づきコードベース全体を包括的かつ厳密に修正します。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> **1. V3 のシミュレーション結果 JSON の除外と退避**
> 現在 `v3/results/raw/` および `v3/results/derived/` に保存されている `v3_rq1_results.json`, `v3_gate_decision.json`, `v3_spatiotemporal_maps_qwen.json`, `v3_path_mediation_qwen.json`, `v3_confirmatory_*.json` はすべて乱数シミュレーションで生成されたものです。これらは論文結果として使用できないため、`v3/results/mock_simulation_archive/` に退避・隔離し、論文 Primary 結果から完全に除外します。
>
> **2. GPU 利用と実モデル実行の運用方針**
> プロジェクト規約（AGENTS.md 1.7「GPUを勝手に用いない」）に従い、スクリプトは `--dry-run` オプション付きで高速に動作確認可能な設計を維持しつつ、通常実行時には本物のモデル重み（Qwen等）をロードして実際の活性化抽出・介入・尤度計算を行う実装へ修正します。本計画承認後の検証では、テストおよび `--dry-run` を中心に自動検証し、大規模GPU実行は別途指示に基づいて行います。

---

## 変更内容 (Proposed Changes)

### 1. リポジトリ全体のテスト環境整備 (Pytest Configuration)

#### [NEW] [pytest.ini](file:///mnt/nas/home/hiromi/src/emo/pytest.ini)
- `pythonpath = src .` を設定し、`PYTHONPATH` を手動指定せずとも全テストから `affective_empathy_eval` および `v3` をインポート可能にする。
- `testpaths = tests v3/tests` を指定。
- `norecursedirs = scratch v2/scripts .venv` を指定し、未整理のスクリプトや scratch 内ファイルがテストとして誤収集されるのを防止。

---

### 2. 共通 Sequence Likelihood と Token 境界保証 (Core Library)

#### [MODIFY] [likelihood.py](file:///mnt/nas/home/hiromi/src/emo/src/affective_empathy_eval/likelihood.py)
- **Token 境界保証の導入**:
  - プロンプト末尾と JSON 候補の間に tokenization 上安定した delimiter（または canonical prefix）を事前固定。
  - `tokens(prompt_with_delim)` が `tokens(prompt_with_delim + candidate)` の厳密な prefix であることを実行時にアサートし、境界跨ぎトークンの脱落や位置ずれを数学的・情報理論的に排除。
  - `cand_start = len(prompt_tokens)` による厳密なスライスと教師強制尤度計算。
  - 81 通り（VA）および 729 通り（VAD）のバッチ計算に対応。

#### [MODIFY] [prompts.py](file:///mnt/nas/home/hiromi/src/emo/src/affective_empathy_eval/prompts.py)
- `get_generation_stage_tokens()` の厳密化:
  - 単純な数字探索（`any(char.isdigit())`）を廃止。
  - JSON 文字列に対する正規表現マッチ（`"valence":\s*(\d+)`, `"arousal":\s*(\d+)`）による character span を取得し、tokenizer の `offset_mapping` を用いて V/A 数値および直前区切りトークンの正確なトークン位置を特定。
- `find_semantic_anchors()` における命名の完全分離:
  - プロンプト側の終端は `prompt_end`
  - 生成側の先頭は `candidate_start`
  - 後方互換性エイリアスを整理し、2つの異なる概念が同一キーで混同されるリスクを解消。

#### [MODIFY] [test_likelihood.py](file:///mnt/nas/home/hiromi/src/emo/tests/test_likelihood.py)
- 厳密 prefix 保証テスト、境界跨ぎトークン処理テスト、delimiter 安定性テストを追加。

---

### 3. V3 実モデル解析の実装とシミュレーション値の排除 (V3 Stage - 最優先)

#### [MODIFY] [run_v3_state_induction.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_state_induction.py)
- `--dry-run` 時のみ `simulate_mock_intervention_responses` を呼び出すように完全分離。
- 実モデル実行パイプラインの本格実装:
  - Qwen-2.5-1.5B (Instruct) をロード。
  - Train split の活性化から外部ラベル（`reader_V`, `reader_A`）に対する重回帰により情動方向 $d_V, d_A$ を推定。
  - Held-out test split に対し、中間層への活性化介入（$\alpha$-sweep: $[-1.0, 1.0]$）、Centered projection removal、直交・ランダムコントロール、および Topic control を実行。
  - 共通 `compute_sequence_likelihoods_for_candidates` により Self / Reader / Topic の実対数尤度と期待値を測定。
  - サンプル単位・ペア単位の変位から Bootstrap 95% 信頼区間を算出し、信頼区間下限に基づく正式な Go/No-Go ゲート判定（`evaluate_go_no_go_gate`）を実施。

#### [MODIFY] [run_v3_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_spatiotemporal_maps.py)
- 実モデルから 4-Map（$D, \beta, \gamma, C$）を算出するパイプラインを実装。
- **$\beta(l,t)$ の定義修正**: 単なる「予測重みノルム $||\beta_V||$」ではなく、「刺激共変量を統制した internal score $\to$ report の偏回帰係数（partial regression coefficient controlling for stimulus covariates）」として実装。
- 意味的アンカー（`prompt_end`, `candidate_start`, `pre_V`, `V_value`, `pre_A`, `A_value`）× 28層のグリッド計算。

#### [MODIFY] [run_v3_path_mediation.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_path_mediation.py)
- Discovery (50%) における Mediator 層自動選定（刺激提示時ピーク $l_{\text{stim}}^*$ と生成時因果ピーク $l_{\text{med}}^*$ の特定）。
- Confirmation (50%) において固定した Mediator 層を 2D 部分空間除去（$P_A = Q Q^\top$）で遮断する実 Activation / Path Patching パイプラインを実装。
- Total Effect (TE), Natural Direct Effect (NDE), Natural Indirect Effect (NIE), Mediation Ratio を Bootstrap CI とともに算出。

#### [MODIFY] [run_v3_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_confirmatory_replication.py)
- Llama-3.2, Gemma-2, Mistral の実モデルロード・検証ロジックを実装。
- 乱数シミュレーションの自動出力（"CONFIRMED: All 4 major hypotheses replicated..."）を停止し、実測定に基づいた結果記録へ改訂。

#### [MODIFY] V3 旧実験群の共通尤度統一・命名整理
- [v3/src/batch_likelihood.py](file:///mnt/nas/home/hiromi/src/emo/v3/src/batch_likelihood.py), [run_generation_time_causal_sweep.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_generation_time_causal_sweep.py), [run_focused_necessity_n100.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_focused_necessity_n100.py) 等の独自尤度計算を共通関数へ移行。
- 旧 `neutralization_ratio` 命名を `projection_removal_effect` / `attenuation_ratio` へ整理。

---

### 4. Behavioral & V1 Phase C の共通化とパイプライン堅牢化 (Behavioral & V1 Stage)

#### [MODIFY] [run_3way_vad_evaluation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_3way_vad_evaluation.py) & [run_aipsy_4split_evaluation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_aipsy_4split_evaluation.py)
- 独自の `evaluate_expected_vad` / `evaluate_expected_va` を廃止し、共通 `src/affective_empathy_eval/likelihood.py` へ一本化。
- `exact_neutral_argmax_pct` を Primary endpoint から除外し、descriptive 指標として位置付け。
- 論文ストーリーと整合させるため、`collapse` / `neutralization` の残存表記を整理。

#### [MODIFY] [run_v1_phase_c_causal_patching.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_causal_patching.py)
- E3 / E4 の Sequence Likelihood を共通 `likelihood.py` に統一。
- 活性化抽出側・尤度計算側の両方で `encode_prompt_canonical()` を一貫して使用（special tokens のズレを完全防止）。

#### [MODIFY] [run_v1_phase_c_targeted_ablation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py)
- E6 の Discovery $\to$ Confirmation パイプラインを完全自動化:
  - データを seed 固定の `pair_id` 単位 random group split で 50/50 に分割。
  - Discovery split で各タスクの最適介入層（Reader-site / Self-site）を自動同定。
  - その同定層を固定し、Confirmation split のみを用いて 2×2 因果マトリクスおよび LMM 交互作用検定を実行。
- 表記を「Double Dissociation Established」から「Evidence consistent with task-specific causal specialization」へ改名。
- 本番解析向けに `--limit 20` に依存せず全ペアを評価可能な構成を徹底。

---

### 5. V2 RQ4 コントロール追加および Legacy スクリプト分離 (V2 Stage)

#### [MODIFY] [run_v2_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_recovery_patching.py)
- **Prompt-format control の追加**:
  - Base (plain) $\to$ Instruct (chat) に加え、Base (plain) $\to$ Instruct (matched-plain) の回復率測定を追加。
  - Post-training による表現変位と chat template によるフォーマット効果を厳密に分離。
- Base と Instruct の hidden state 次元一致のアサーション（`assert adapter_base.hidden_size == adapter_inst.hidden_size`）を追加。
- 4 family サマリを一般化推論ではなく descriptive summary として位置づけ、family 別結果を Primary とする。

#### [NEW] `v2/scripts/legacy/` ディレクトリの設置
- [run_introspective_accessibility_test.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_introspective_accessibility_test.py) などの旧スクリプト群を `legacy/` に移動・隔離し、新 V2 Primary パイプラインとの混同を防止。

---

## 検証計画 (Verification Plan)

### 自動テスト (Automated Tests)
- `pytest -q`:
  - ルートディレクトリから引数なしで一発実行し、全テスト（Core ライブラリ、トークン境界テスト、V1 拡張テスト、V3 OT テストなど）がパスすることを確認。
  - コマンド: `.venv/bin/pytest -q`

### スクリプト動作検証 (Dry-run / Smoke Tests)
- **V3 実モデルスクリプトのドライラン動作確認**:
  - `python v3/scripts/run_v3_state_induction.py --dry-run`
  - `python v3/scripts/run_v3_spatiotemporal_maps.py --dry-run`
  - `python v3/scripts/run_v3_path_mediation.py --dry-run`
  - `python v3/scripts/run_v3_confirmatory_replication.py --dry-run`
- **V1 Phase C / E6 の動作確認**:
  - `python v1/scripts/run_v1_phase_c_targeted_ablation.py --help`
- **V2 RQ4 の動作確認**:
  - `python v2/scripts/run_v2_recovery_patching.py --dry-run`
