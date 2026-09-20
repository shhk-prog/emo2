# 最終本番前26項目監査報告書 (Final Checklist 26 Audit Report)

本ドキュメントは、本番GPU実行（Clean Production Run）に先立ち、提示された全26項目の必須要件についてコードベースの機械的検査および検証結果をまとめたものである。

---

## 1. 26項目監査結果まとめ

| No. | 検査項目 | 判定 | 検証内容・エビデンス |
|:---|:---|:---:|:---|
| 1 | 構文・テスト検証 | **PASS** | `compileall` エラーなし完了。`pytest -q` (97 passed)。 |
| 2 | V1 Phase B `--force` 引数の単一性 | **PASS** | `run_phase_b.py` Line 262 に1箇所のみ。重複なし。 |
| 3 | Behavioral/V1 への model revision 伝播 | **PASS** | `run.py` 内で Phase A/B/C/E6 (`common_flags`) および Behavioral (`cmd_emobank`, `cmd_aipsy`) に `--model-revision` を確実に渡している。 |
| 4 | `configs/models.yaml` の SHA 固定 | **PASS** | Qwen, Llama, Gemma, OLMo の base/instruct revision が全て40文字の完全コミットSHAに固定。 |
| 5 | 全 Primary loader の `revision=` 使用 | **PASS** | Behavioral (`run_behavioral_*.py`), V1 (`run_phase_*.py`), V2 (`run_rq*.py`), V3 (`run_*.py`) の `AutoTokenizer` / `AutoModelForCausalLM` に `revision=` を指定。 |
| 6 | V1 Phase B manifest config ハッシュ後不変性 | **PASS** | `manifest_config["num_pairs"] = ...` は削除済み。`num_pairs` は `create_run_manifest` の metadata に渡され、キャッシュ判定ハッシュと完全一致。 |
| 7 | Candidate-space sensitivity の Clinical 厳密性 | **PASS** | `intensity.isin(["peak", "clinical"])` と `intensity.eq("none")` で各1件を厳密抽出。過不足時は `ValueError` 送出。 |
| 8 | Candidate-space sensitivity の revision 使用 | **PASS** | `--model-revision` 引数を持ち、未指定時は registry から取得して tokenizer / model に `revision=model_revision` を渡す。 |
| 9 | Candidate-space sensitivity 固定結論排除 | **PASS** | 結果に関係なく「preserved」と固定出力するコードは完全に排除。純粋な数値指標のみ保存。 |
| 10 | V3 candidate space 表記が `VA_81` | **PASS** | V3 Primary (`run_rq1`, `run_rq2`, `run_rq3`, `run_confirmatory`) のマニフェストで `candidate_space="VA_81"` を明示。 |
| 11 | V1 Phase C / E6 の `VAD_729` 表記 | **PASS** | `candidate_space="VAD_729"`, `measurement_space="VA_expectation_from_VAD_729"` に統一。 |
| 12 | V3 RQ2/RQ3 cache の revision 照合 | **PASS** | `is_manifest_matching` に `expected_model_revision=model_revision` を渡し、リビジョン不一致時に再計算するよう厳密化。 |
| 13 | V3 production での pair-aware split フォールバック禁止 | **PASS** | 本番（非dry-run）時に `pair_id` が不足している場合はフォールバックせず `raise ValueError(...)` を送出。 |
| 14 | negative R² の raw 保持 (非クリップ) | **PASS** | `max(0, r2)` 等の 0 クリップは全 primary コードに存在せず、raw R² を保持。 |
| 15 | NaN/Inf hidden state の書き換え排除 | **PASS** | `nan_to_num` やクリッピングによる偽装は行わず、非有限値は例外送出・捕捉する設計。 |
| 16 | V2 RQ4 ΔEMD の sample-wise 計算 | **PASS** | 各サンプル \(i\) で \(\Delta EMD_i = EMD^{initial}_i - EMD^{patched}_i\) を算出した上で平均値・AUCを計算。 |
| 17 | V2 RQ4 AUC Primary / Peak Secondary | **PASS** | `primary_matched_plain` に AUC recovery、`secondary_peak_localization` に peak layer / max recovery を分離格納。 |
| 18 | V1 E4 confirmatory site の Discovery 限定選定 | **PASS** | `discovery_mag_reader` のみを用いて confirmatory 条件の peak layer を決定。 |
| 19 | V3 frozen site の V/A 分離 | **PASS** | `frozen_confirmatory_sites.json` に `temporal_relative_depth_v`, `temporal_relative_depth_a` を独立保持。単一の旧キーは排除。 |
| 20 | V3 Confirmatory のサイト再探索禁止 | **PASS** | Discovery (Qwen) で凍結された相対深度を Llama, Gemma, OLMo へそのまま転送。 |
| 21 | prompt hash の SHA-256 統一 | **PASS** | Python 組込 `hash()` を排除し、`hashlib.sha256(...).hexdigest()` に統一。 |
| 22 | candidate hash の全件ハッシュ化 | **PASS** | 729件（VAD）または81件（VA）の実候補全件リストから SHA-256 ハッシュを生成。 |
| 23 | dtype の統一 (`bfloat16`) | **PASS** | `configs/models.yaml` の全 Primary モデルで `inference_dtype: "bfloat16"` に統一。 |
| 24 | 旧結果と新結果の混在防止 | **PASS** | results ディレクトリは初期化可能。各スクリプトで既存結果の `archive/` 自動退避機構を実装。 |
| 25 | 秘密ファイル (`.env` 等) の除外 | **PASS** | `.env` は `.gitignore` に指定済み。`v1/.env` は存在せず、クレデンシャル混入なし。 |
| 26 | 統合 dry-run の完全完走 | **PASS** | `affective_empathy_eval.run --stage all --dry-run` が Behavioral から V3 Confirmatory まで `All requested stages completed successfully!` で完走。 |

---

## 2. 最終判定: **GO (本番実行開始可能)**

全26項目すべてが合格（PASS）であることを確認いたしました。
科学的測定原則（AGENTS.md）、コード整合性、キャッシュ・再現性担保の全要件を満たしています。
