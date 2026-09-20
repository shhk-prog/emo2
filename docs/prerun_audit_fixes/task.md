# タスクリスト: 本番実行前監査指摘事項の完全修正 (Pre-run Audit Fixes)

- [ ] 0. ドキュメント整備 (`docs/prerun_audit_fixes/`) <!-- id: 0 -->
- [ ] 1. 【P0】V1 Phase B の `--force` 二重定義の削除と entrypoint テスト追加 <!-- id: 1 -->
- [ ] 2. 【P0】Behavioral / V1 への HF revision SHA 伝播とスクリプト側での安全弁 <!-- id: 2 -->
- [ ] 3. 【P1】Behavioral EmoBank の dtype を `bfloat16` に統一 <!-- id: 3 -->
- [ ] 4. 【P1】V1/V2/V3 のキャッシュ判定における `is_manifest_matching` 厳密照合の実装 <!-- id: 4 -->
- [ ] 5. 【P1】`create_run_manifest` への `dataset_hash` 引数追加と V1 Phase A 複合ハッシュ対応 <!-- id: 5 -->
- [ ] 6. 【P1】V2 manifest への Base/Instruct revision 保存とキャッシュハッシュ統合 <!-- id: 6 -->
- [ ] 7. 【P1】V3 manifest の candidate_space / measurement_space 表記を `VA_81` へ修正 <!-- id: 7 -->
- [ ] 8. 【P1】V1 Phase C / E6 / Behavioral の measurement_space 表記適正化 <!-- id: 8 -->
- [ ] 9. 【P1】AIPsy / EmoBank の candidate hash を 729 候補全体に修正 <!-- id: 9 -->
- [ ] 10. 【P1】`is_manifest_matching()` の `expected_intervention_version` デフォルトを `None` に変更 <!-- id: 10 -->
- [ ] 11. 【P1】V1 Phase B dry-run スキーマの `n_validated_*` 削除・統一 <!-- id: 11 -->
- [ ] 12. 【P1】`configs/v3_experiments.yaml` の旧 `temporal_relative_depth` 削除・V/A 分離統一 <!-- id: 12 -->
- [ ] 13. 【P1】V2/V3 の dtype を registry (`fam_cfg.inference_dtype`) 参照に統一 <!-- id: 13 -->
- [ ] 14. 【P2】`src/affective_empathy_eval/extraction.py` の mock extractor を決定論的 sha256 シード化 <!-- id: 14 -->
- [ ] 15. 【P2】AIPsy の stimuli path 存在確認と明示的 FileNotFoundError 送出 <!-- id: 15 -->
- [ ] 16. 【P2】`v1/.env` の削除と漏洩防止確認 <!-- id: 16 -->
- [ ] 17. 全体検証: `compileall`、`pytest`、`--stage all --family qwen --dry-run` の完走確認 <!-- id: 17 -->
