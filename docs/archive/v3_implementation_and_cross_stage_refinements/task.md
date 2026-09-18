# Task: V3 Implementation and Cross-Stage Refinements

## 1. V3 統計・因果・パイプラインの本格実装 (最高優先度)
- [x] 1.1 乱数シミュレーション結果ファイルの完全除外・退避 (`v3/results/mock_simulation_archive/` に退避)
- [x] 1.2 `run_v3_state_induction.py`: 実モデル（Qwen-2.5-7B-Instruct）での $d_V, d_A$ 推定、介入、尤度評価、Bootstrap CI によるゲート判定
- [x] 1.3 `run_v3_spatiotemporal_maps.py`: 刺激共変量を統制した偏回帰係数 $\beta(l,t)$ の推定、因果影響 $C(l,t)$、ピーク解離判定の実装
- [x] 1.4 `run_v3_path_mediation.py`: 50% split による Discovery での Mediator 選定 $\to$ 50% split Confirmation での 2D 射影除去パッチング、TE/NDE/NIE/MR 計算
- [x] 1.5 `run_v3_confirmatory_replication.py`: Llama-3.1, Gemma-2, Mistral-7B の実モデル解析ロジック、`--dry-run` 時の明確な mock simulation タグ付け
- [x] 1.6 `v3/src/batch_likelihood.py` および旧 V3 スクリプトの共通尤度関数 (`src/affective_empathy_eval/likelihood.py`) への一本化、`mean_attenuation_ratio` 命名対応

## 2. Behavioral / V1 / V2 / 共通基盤の修正
- [x] 2.1 共通 Sequence Likelihood (`src/affective_empathy_eval/likelihood.py`): BPE境界（strict prefix、boundary detection、offset mapping）の厳密保証
- [x] 2.2 Behavioral スクリプト (`run_3way_vad_evaluation.py`, `run_aipsy_4split_evaluation.py`): 独自尤度を共通 `likelihood.py` に統一
- [x] 2.3 V1 Phase C (`run_v1_phase_c_targeted_ablation.py`): E6 を pair_id 単位の Discovery $\to$ Confirmation 自動化パイプラインに改訂、"Evidence consistent with task-specific causal specialization" 表記へ更新
- [x] 2.4 V2 RQ4 (`run_v2_recovery_patching.py`): Base $\to$ Instruct のフォーマット統制（Instruct matched-plain control）を追加
- [x] 2.5 V2 旧スクリプトの隔離: `v2/scripts/legacy/` ディレクトリを作成し、旧探索・スクリーニングスクリプトを移動、README を配置
- [x] 2.6 ルートでの `pytest.ini` 整備と全テスト実行（Core 29件 + V3 6件 = 35件パス）

## 3. ドキュメント・再現性の整理
- [x] 3.1 `implementation_plan.md` の作成・承認
- [x] 3.2 `walkthrough.md` の作成
- [x] 3.3 `docs/v3_implementation_and_cross_stage_refinements/` 配下への記録整理
