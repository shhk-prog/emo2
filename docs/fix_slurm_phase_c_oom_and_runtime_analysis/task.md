# タスク管理：Slurm Phase C エラー修正および実行時間差異の分析

## 目的
1. `/mnt/nas/home/hiromi/src/emo/v1/results/logs/slurm` で発生しているエラー（CUDA Out of Memory, モジュール不足等）の根本原因を特定し修正する。
2. 実行に非常に時間がかかるジョブ（約22時間）とそうではないジョブ（約20〜35分）の構造的・計算量的な差異を解明し報告する。

## タスク一覧
- [x] ログ解析と原因特定
  - [x] Slurmログ（`phase_c_108271_*.out`, `*.err`）の精査
  - [x] 実行スクリプト（`run_v1_phase_c_causal_patching.py`, `run_v1_phase_c_targeted_ablation.py`, `run_all_phase_c.sh`, `slurm_run_phase_c.sbatch`）の調査
  - [x] 実行時間差異の理論的計算とログ数値の突き合わせ
  - [x] CUDA OOMの発生メカニズムの特定
- [x] ドキュメント作成
  - [x] `task.md` 作成
  - [x] `implementation_plan.md` 作成
  - [x] `walkthrough.md` 作成
- [x] 実装・修正
  - [x] `run_v1_phase_c_causal_patching.py` のメモリ管理・OOM対策・抽出バッチ化
  - [x] `run_v1_phase_c_targeted_ablation.py` のメモリ管理・OOM対策
  - [x] `run_all_phase_c.sh` への `--sub-batch-size` パラメータ透過伝播
  - [x] `slurm_run_phase_c.sbatch` への環境変数設定（`PYTORCH_CUDA_ALLOC_CONF` 等）
  - [x] 仮想環境における `statsmodels` 依存関係の確認・動作確認
- [x] 検証と完了報告
  - [x] 単体・ドライラン検証
  - [x] `walkthrough.md` の作成
  - [x] ユーザーへの詳細解説の提供
- [x] 完了済みジョブの自動スキップ機能追加
  - [x] `run_all_phase_c.sh` に完了判定（成果物ファイルの存在確認）と `--force` オプションの追加
  - [x] `run_v1_phase_c_causal_patching.py` に `--force` オプションと成果物スキップ判定の追加
  - [x] `run_v1_phase_c_targeted_ablation.py` に `--force` オプションと成果物スキップ判定の追加
  - [x] スキップ動作の dry-run / 実機検証
  - [x] `walkthrough.md` の更新
