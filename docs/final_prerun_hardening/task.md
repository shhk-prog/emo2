# タスクリスト: 最終本番前ハーデニング (Final Pre-run Hardening)

- [x] 0. ドキュメント整備 (`docs/final_prerun_hardening/`) <!-- id: 0 -->
- [x] 1. V1 Phase B: manifest hash 保存直前変更の削除 (`num_pairs` を metadata に移動) とコメント内の `validated` 表現修正 <!-- id: 1 -->
- [x] 2. Candidate-space sensitivity: `peak`/`clinical` の厳密選択への修正 <!-- id: 2 -->
- [x] 3. Candidate-space sensitivity: HF revision 固定の適用 (CLI引数 & registry 参照) <!-- id: 3 -->
- [x] 4. Candidate-space sensitivity: 事前結論（`conclusion` 固定文）の削除と純粋な数値保存化 <!-- id: 4 -->
- [x] 5. V3 RQ2/RQ3: cache 照合への `model_revision` 明示的追加 <!-- id: 5 -->
- [x] 6. V3 Confirmatory & RQ1: 本番実行時の KFold fallback / index split 禁止 (dry-run のみ許容) <!-- id: 6 -->
- [x] 7. Unified Runner: Behavioral dry-run で子スクリプト（`run_behavioral_emobank.py`, `run_behavioral_aipsy.py`）を `--dry-run` で実起動するよう修正 <!-- id: 7 -->
- [x] 8. V2 RQ1/RQ2 manifest: `candidate_space="N/A"`, `measurement_space="prompt_end_hidden_state"` の明示 <!-- id: 8 -->
- [x] 9. `v1/.env` ファイルの無害化確認と完全消去手順整備 <!-- id: 9 -->
- [x] 10. 全体検証: `compileall`, `pytest -q`, 全ステージ dry-run (`--stage all --family qwen --dry-run --force`) のパス <!-- id: 10 -->
