# タスクリスト: 一本化論文の厳密な測定不一致解消と実行停止バグ修正

- [x] 1. 実装計画の策定と承認 <!-- id: 1 -->
  - [x] 該当箇所（Behavioral, V1, V2, V3, README, outline）のコードベース詳細調査 <!-- id: 1.1 -->
  - [x] `task.md` および `implementation_plan.md` の作成 <!-- id: 1.2 -->
  - [x] ユーザー承認（Proceed）の受領 <!-- id: 1.3 -->

- [x] 2. 【P0】実モデル実行停止バグ（3系統）の即時修正 <!-- id: 2 -->
  - [x] `behavioral/primary/run_behavioral_emobank.py` に `from typing import Optional` を追加 <!-- id: 2.1 -->
  - [x] `behavioral/primary/run_behavioral_aipsy.py` に `from typing import Optional` を追加 <!-- id: 2.2 -->
  - [x] `v1/primary/run_phase_a.py`, `v1/primary/run_phase_c.py`, `v1/primary/phase_c/run_e6_specialization.py`, `v1/primary/run_phase_b.py` の `args.device == "cuda"` を `str(device).startswith("cuda")` 系に統一 <!-- id: 2.3 -->
  - [x] `v3/primary/run_confirmatory_replication.py` の `d_profile_a = []` 未定義バグを修正 <!-- id: 2.4 -->

- [x] 3. 【P1】Behavioral & V1 測定条件・候補の統一 <!-- id: 3 -->
  - [x] `behavioral/primary/` の独自 VAD 候補文字列生成を廃止し、`src/affective_empathy_eval/likelihood.py::build_vad_candidates()`（compact JSON, `separators=(',', ':')`）に一本化 <!-- id: 3.1 -->
  - [x] `behavioral/analysis/summarize_behavioral_aipsy.py` の neutral 不在時 `else 5.0` fallback を廃止（NA/エラー化） <!-- id: 3.2 -->
  - [x] V1 Phase A/B/C/E6 の tokenization 条件を `add_special_tokens=False` に完全統一 <!-- id: 3.3 -->
  - [x] 同一プロンプトに対して `Phase A prompt_end == Phase B prompt_end == Phase C prompt_end` を検証する単体テストを追加 <!-- id: 3.4 -->
  - [x] `v1/primary/run_phase_a.py` の probe において group 不足時の通常 KFold/StratifiedKFold fallback を禁止（`n_splits = min(requested_cv, n_unique_groups)`、2未満は NA） <!-- id: 3.5 -->

- [x] 4. 【P1】V3 Confirmatory 方向推定プロトコルを Discovery RQ2 と完全統一 <!-- id: 4 -->
  - [x] `v3/primary/run_confirmatory_replication.py` の H4 (Temporal Emergence) において、prompt-end 固定方向の transport を廃止し、Discovery RQ2 と同様に各 layer × generation stage の表現から direction を学習 ($d_{l,t}$) して介入する局所プロトコルへ統一 <!-- id: 4.1 -->
  - [x] V3 RQ2 の D-map (V=5,A=5 prefix) と C-map (全81候補) の trajectory 差異について、`pre_V` などの candidate-independent stage を Primary とし、それ以降を canonical neutral teacher-forced trajectory として明記・整理 <!-- id: 4.2 -->

- [x] 5. 【P1】モデル管理・キャッシュ・設定の整理 <!-- id: 5 -->
  - [x] `v1/configs/eval_cohort.yaml` を `v1/configs/legacy/` へ移動し、`configs/models.yaml` を唯一の正本（single source of truth）として確定 <!-- id: 5.1 -->
  - [x] `v2/primary/run_rq1_rq2_cross_decoding.py` の cache manifest 検証を強化（dataset hash, config hash, code hash, prompt version, dtype, seed, max_samples 照合） <!-- id: 5.2 -->

- [x] 6. 【P1】論文構成・図・表記・主張の統一 <!-- id: 6 -->
  - [x] `docs/v3_prerun_five_fixes/paper_outline.md` の「Shared representation を前提にした図」を、刺激から分岐する並列計算から共有度を検証する論理図に修正 <!-- id: 6.1 -->
  - [x] 論文 Section 番号を統一（§3 Unified Framework, §4 Behavioral, §5 V1 Representation Sharing, §6 V2 Reorganization, §7 V3 Causal Utilization, §8 Discussion） <!-- id: 6.2 -->
  - [x] V1 section 名から `in Base Models` を削除 <!-- id: 6.3 -->
  - [x] 中心主張文を指示された文言に統一 <!-- id: 6.4 -->
  - [x] 729 VAD と 81 VA の sensitivity analysis の補足および直接比較不可の明記 <!-- id: 6.5 -->
  - [x] `README.md` を上記構成・主張に同期 <!-- id: 6.6 -->

- [/] 7. 検証とテスト <!-- id: 7 -->
  - [/] 全変更対象スクリプトの `py_compile` <!-- id: 7.1 -->
  - [ ] 全本番エントリーポイントの `--help` / mock smoke test <!-- id: 7.2 -->
  - [ ] pytest テストスイートの実行 <!-- id: 7.3 -->
  - [ ] `docs/unified_paper_rigorous_alignment/walkthrough.md` の作成 <!-- id: 7.4 -->
