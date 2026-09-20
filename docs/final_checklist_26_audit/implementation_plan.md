# 最終本番前26項目監査計画 (Final Checklist 26 Audit Plan)

本計画は、ユーザーから提示された本番実行前の26項目チェックリストを網羅的・機械的かつ厳密に検証し、本番GPU実行（Clean Production Run）の安全性を最終確認するためのものである。

## 監査項目と検証方法

1. **構文・テスト検証**:
   - `python -m compileall behavioral v1 v2 v3 src scripts`
   - 指定7テストおよび全テスト（`pytest -q`）の実行
2. **引数定義検証**:
   - `v1/primary/run_phase_b.py` の `--force` 重複の有無
3. **リビジョン伝播検証**:
   - `src/affective_empathy_eval/run.py` 内で Behavioral, Phase A/B/C/E6 に `--model-revision` を渡しているか
4. **モデル設定ファイル検証**:
   - `configs/models.yaml` の commit SHA の確認（`main` ではなくフルSHA）
5. **Loader リビジョン指定検証**:
   - `from_pretrained` を持つ全スクリプトで `revision=` が渡されているか
6. **Phase B マニフェストハッシュ後変更検証**:
   - `manifest_config` のハッシュ計算後にキーが追加されていないか
7. **Clinical 刺激抽出厳密性**:
   - `scripts/run_candidate_space_sensitivity.py` で `isin(["peak", "clinical"])` かつ neutral 1件・clinical 1件のチェックが行われているか
8. **Candidate-space sensitivity リビジョン指定**:
   - `AutoTokenizer` / `AutoModelForCausalLM` に `revision=model_revision` が指定されているか
9. **Candidate-space sensitivity 固定結論排除**:
   - 固定テキスト（`preserved`, `consistent` 等）の有無の確認
10. **V3 Candidate Space 表記**:
    - `VA_81` に統一されているか（`VAD_729` の混入がないか）
11. **V1 Phase C / E6 Candidate Space 表記**:
    - `candidate_space = VAD_729`, `measurement_space = VA_expectation_from_VAD_729` になっているか
12. **V3 RQ2 / RQ3 キャッシュリビジョン照合**:
    - `is_manifest_matching` に `expected_model_revision`, `expected_tokenizer_revision` が渡されているか
13. **V3 Production Fallback 禁止**:
    - Production 時に pair-aware split から KFold/index split へのフォールバックが `raise ValueError` になっているか
14. **Negative R² の Raw 保持**:
    - `max(0, r2)` 等でクリップされていないか
15. **Hidden State の NaN/Inf 処理**:
    - `nan_to_num` やクリッピングで誤魔化さず、有限値チェックが行われているか
16. **V2 RQ4 ΔEMD の Sample-wise 計算**:
    - 各サンプルで差分をとってから平均しているか
17. **V2 RQ4 AUC Primary / Peak Secondary**:
    - AUC recovery が Primary、peak layer が Secondary となっているか
18. **V1 E4 Confirmatory Site Selection**:
    - Discovery 側の指標のみでサイト選定されているか
19. **V3 Frozen Site の V/A 分離**:
    - `frozen_confirmatory_sites.json` が V/A 別キーを持っているか
20. **V3 Confirmatory サイト再探索禁止**:
    - Discovery サイトの相対深度をそのまま転送しているか
21. **Prompt Hash の SHA-256 使用**:
    - Python の組み込み `hash()` が使われていないか
22. **Candidate Hash の全件生成**:
    - 729 / 81 件の全候補からハッシュを計算しているか
23. **Dtype の統一**:
    - registry 設定（`bfloat16`）に統一されているか
24. **結果ディレクトリの分離・アーカイブ方針**:
    - `run_id` またはアーカイブ体制の確認
25. **シークレット管理**:
    - `.env` や秘密鍵が Git 管理外になっているか
26. **統合 Dry-run の完全完走**:
    - `affective_empathy_eval.run` の完走確認

## 検証結果の記録と報告
- 各項目の検証コマンドと結果を `docs/final_checklist_26_audit/walkthrough.md` に記録し、総合判定（GO / NO-GO）を下す。
