# タスク: behavioral 実行時の CUDA OOM エラー解消

## 状況
- `bash scripts/run_production_behavioral.sh cuda:0` 実行時、EmoBank 3-Way VAD 評価の 79/1000 サンプル目で `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.51 GiB` が発生し停止した。

## 目標
1. `src/affective_empathy_eval/likelihood.py` の `compute_sequence_likelihoods_for_candidates` におけるメモリ消費を抜本的に削減（系列全体への巨大 `log_softmax` 計算を廃止し、候補トークン部分のみをスライスして計算）。
2. `behavioral/primary/run_behavioral_emobank.py`, `behavioral/primary/run_behavioral_aipsy.py` のデフォルト `batch_size` を 243 から安全な 81 に調整し、`src/affective_empathy_eval/run.py` からも `--batch-size` を指定可能にする。
3. `scripts/run_production_behavioral.sh` に `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定し、PyTorch のキャッシュ断片化による OOM を防止する。
4. 単体テストおよび小規模 dry-run / 動作確認を行い、正常に動作することを確認する。

## タスクリスト
- [x] 計画策定とタスクファイルの作成 (`docs/fix_behavioral_oom/`) <!-- id: 0 -->
- [x] `src/affective_empathy_eval/likelihood.py` のスライス計算による省メモリ化実装 <!-- id: 1 -->
- [x] `run_behavioral_emobank.py`, `run_behavioral_aipsy.py`, `run.py` の batch_size 設定調整 <!-- id: 2 -->
- [x] `scripts/run_production_behavioral.sh` の環境変数設定 <!-- id: 3 -->
- [x] テストおよび検証 <!-- id: 4 -->
- [x] 実験結果と振り返りドキュメントの作成 <!-- id: 5 -->
