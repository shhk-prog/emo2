# タスクリスト: 最終論文用コード監査40項目の包括的修正

## 1. P1: 論文の結果・推論に影響する重要項目の修正 (Item 1 - 15)
- [x] 1.1 **【Item 1】V1 E4 Confirmatory selection leakage 修正**: `run_phase_c.py` で `discovery_mag_reader` から candidate layer を選択、fallback 廃止、E3 なしは production error <!-- id: p1_item1 -->
- [x] 1.2 **【Item 2】V1 Phase A 測定不能の NaN 保持**: `run_phase_a.py` の `evaluate_regression_probe` で 0 置換を撤廃し、`NaN`, `status="failed"`, `failure_reason`, `n_valid_folds` を返却 <!-- id: p1_item2 -->
- [x] 1.3 **【Item 3】V1 Intensity 解析の未知ラベル検査**: `run_phase_a.py` で `.fillna(0.0)` を撤廃し、未知ラベル検出時に `ValueError` を送出 <!-- id: p1_item3 -->
- [x] 1.4 **【Item 4】V1 Phase A/B 非有限 hidden state の例外化**: `run_phase_a.py`, `run_phase_b.py` で `np.nan_to_num`/`clip` による 0 置換を撤廃し、非有限値検出時は `FloatingPointError`、`max_abs` を記録 <!-- id: p1_item4 -->
- [x] 1.5 **【Item 5】V1 Phase B Primary N 修正**: `run_phase_b.py` で held-out test の valid 数（`test_para_valid_mask`, `test_rev_valid_mask`）を記録、train/test/nonfallback 総数も保存 <!-- id: p1_item5 -->
- [x] 1.6 **【Item 6】Phase B `validated` 表現の削除**: `n_validated_*` を `n_nonfallback_*` へ完全改名 <!-- id: p1_item6 -->
- [x] 1.7 **【Item 7】Phase B セマンティックコントロールの品質監査追加**: `transformation_method`, `quality_status` の保存 <!-- id: p1_item7 -->
- [x] 1.8 **【Item 8】V2 RQ3 seed の config 連動化**: `run_rq3_causal_map.py` 内の `seed=42` 固定を config 由来へ変更 <!-- id: p1_item8 -->
- [x] 1.9 **【Item 9】V2 RQ4 ΔEMD のサンプル単位計算**: `run_rq4_recovery_patching.py` で `initial_emd_mean * layer_mean_ratio` を廃止し、$\Delta EMD_i = EMD_i^{init} - EMD_i^{patch}$ の平均として算出 <!-- id: p1_item9 -->
- [x] 1.10 **【Item 10】V2 RQ4 Primary/Secondary 出力分離**: JSON を `primary_matched_plain`（AUC）と `secondary_peak_localization`（best_layer/max_recovery）に分離 <!-- id: p1_item10 -->
- [x] 1.11 **【Item 11】V3 RQ2 causal subset seed の config 連動化**: `run_rq2_spatiotemporal_maps.py` の `seed=42` を引数 seed 由来へ変更 <!-- id: p1_item11 -->
- [x] 1.12 **【Item 12】V3 RQ3 subset seed の config 連動化**: `run_rq3_path_mediation.py` の `seed=42` を引数 seed 由来へ変更 <!-- id: p1_item12 -->
- [x] 1.13 **【Item 13】V3 RQ3 group 不足時の例外化**: `run_rq3_path_mediation.py` で `n_splits < 2` の場合に $R^2=0$ ではなく `ValueError` 送出（production） <!-- id: p1_item13 -->
- [x] 1.14 **【Item 14】V3 Confirmatory cache key の V/A 分離 & frozen_sites_hash 追加**: `run_confirmatory_replication.py` の manifest_config に `temporal_relative_depth_v/a`, `temporal_stage_v/a`, `frozen_sites_hash` を設定、単一値 fallback 削除 <!-- id: p1_item14 -->
- [x] 1.15 **【Item 15】V3 Confirmatory コメント・説明文の改訂**: Discovery 推定 site の empirical freeze に記述を統一 <!-- id: p1_item15 -->

## 2. P2: 最終論文の再現性向上のための修正 (Item 16 - 40)
- [x] 2.1 **【Item 16 & 37】Behavioral キャッシュ検証 & チェックポイント分離**: `results/checkpoints/` へ分離、manifest 照合厳格化、`checkpoint_used` 保存 <!-- id: p2_item16_37 -->
- [x] 2.2 **【Item 17 & 18】V1 Phase A/B/C キャッシュ判定 & 内容 Hash 必須化**: `is_manifest_matching()` に config/dataset/prompt/model_rev/git を伝播、実ファイル SHA256 を保証 <!-- id: p2_item17_18 -->
- [x] 2.3 **【Item 19, 20, 21】HF モデル・Tokenizer revision 固定**: `configs/models.yaml` にコミット SHA 固定、AutoConfig/AutoTokenizer/AutoModel に revision 伝播、manifest に個別保存・照合 <!-- id: p2_item19_21 -->
- [x] 2.4 **【Item 22, 23, 24】Prompt 共通化・SHA-256 Hash・ChatTemplate 判定**: `src/affective_empathy_eval/prompts.py` へ VAD/VA 生成を集約、組み込み `hash()` 廃止し SHA-256 化、template_mode の manifest 記録 <!-- id: p2_item22_24 -->
- [x] 2.5 **【Item 25 & 26】run_id 保存構造 & 衝突防止**: `generate_run_id` にマイクロ秒+UUID8桁を追加、`results/raw/<run_id>/` 保存対応および `latest.json` 生成 <!-- id: p2_item25_26 -->
- [x] 2.6 **【Item 27】Archive tarball 名のタイムスタンプ化**: `archive_and_clean_results.py` で固有 tarball 名を生成し、既存アーカイブ削除を禁止 <!-- id: p2_item27 -->
- [x] 2.7 **【Item 28】`run_production_reruns.sh` の Confirmatory 重複呼び出し解消**: loop 外で 1 回のみ実行 <!-- id: p2_item28 -->
- [x] 2.8 **【Item 29】V1 Phase A config 読み込み対応**: `--config` 引数追加、`configs/v1_experiments.yaml` の seed/cv_folds/alpha を probe へ反映 <!-- id: p2_item29 -->
- [x] 2.9 **【Item 30】V1 E3/E4 Discovery 失敗時の fallback 禁止**: production では `RuntimeError` を送出 <!-- id: p2_item30 -->
- [x] 2.10 **【Item 31】Candidate-space sensitivity の実モデル matched Δ 実装**: AIPsy matched pair 上で $\Delta V_{729}$ vs $\Delta V_{81}$, $\Delta A_{729}$ vs $\Delta A_{81}$ の相関・方向一致度・MAE を算出、モデルレジストリ連携 <!-- id: p2_item31 -->
- [x] 2.11 **【Item 32】Phase A/B candidate_space manifest 名整理**: `measurement_space="prompt_end_hidden_state"`, `candidate_space="N/A"` に変更 <!-- id: p2_item32 -->
- [x] 2.12 **【Item 33】Stage 間 dtype 設定 & manifest 記録**: `configs/models.yaml` に `inference_dtype: bfloat16`、manifest に `actual_dtype` 保存 <!-- id: p2_item33 -->
- [x] 2.13 **【Item 34】Scale 定義の統一**: Primary を 1–9 raw scale に統一、文書・コメントを整合 <!-- id: p2_item34 -->
- [x] 2.14 **【Item 35】V2 RQ1/RQ2 分割の非空アサーション**: `len(train_df) == 0 or len(test_df) == 0` で `ValueError` 送出 <!-- id: p2_item35 -->
- [x] 2.15 **【Item 36】Phase A 分類での単一クラス fold 処理**: `np.unique(y_train).size < 2` を検出し適切にスキップ／エラーハンドリング <!-- id: p2_item36 -->
- [x] 2.16 **【Item 38】V3 unmatched pair の除外記録**: `exclusions_v3.csv` への保存および N 値整合 <!-- id: p2_item38 -->
- [x] 2.17 **【Item 39 & 40】V3 RQ1 depth fallback 撤廃 & frozen 互換キー完全削除**: `resolved_relative_depth` 必須化、旧 `temporal_relative_depth` 削除 <!-- id: p2_item39_40 -->

## 3. テストと検証
- [x] 3.1 `compileall` 検証 (`behavioral v1 v2 v3 src scripts`) <!-- id: test_compile -->
- [x] 3.2 単体・統合テスト検証 (`pytest tests/`) <!-- id: test_pytest -->
- [x] 3.3 Stage all dry-run スモークテスト (`affective_empathy_eval.run --dry-run`) <!-- id: test_dryrun -->
- [x] 3.4 `walkthrough.md` の作成と完了報告 <!-- id: test_walkthrough -->
