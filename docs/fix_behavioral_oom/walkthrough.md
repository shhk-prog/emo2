# Behavioral 評価における CUDA OOM エラー解消 修正確認 (Walkthrough)

## 概要
`bash scripts/run_production_behavioral.sh cuda:0` 実行時に発生した CUDA OOM エラー（Qwen2.5-1.5B による EmoBank 3-Way 評価 79/1000 サンプル目、`torch.OutOfMemoryError: Tried to allocate 16.51 GiB`）を根本的に解消しました。

---

## 主な変更内容

### 1. 尤度計算におけるメモリ消費の抜本的最適化
- **対象ファイル**: [`src/affective_empathy_eval/likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)
- **変更前**:
  - `outputs.logits[:, :-1, :].float()` に対して `F.log_softmax` を全系列長（~150 トークン）にわたって計算。
  - 語彙サイズ 152,064 × 系列長 150 × バッチサイズ 243 の場合、単一の float32 テンソルだけで約 22 GB、合計で 50 GB 以上のメモリを消費。
- **変更後**:
  - 各バッチで候補文字列が存在する最小ステップ `min_step = max(0, min(b_start_idxs) - 1)` を求め、必要な候補トークン部分のみをスライス (`logits[:, min_step:-1, :]`) して `float()` キャストおよび `F.log_softmax` を計算。
  - プロンプト部分（140 トークン以上）の無駄なテンソル確保を排除し、中間テンソルサイズを 1/30 〜 1/50（数百 MB 以下）に削減。
  - バッチ毎に `del outputs, logits, sliced_logits, log_probs_slice` を明示して即時破棄。

### 2. バッチサイズおよびメモリ断片化対策の適正化
- **対象ファイル**:
  - [`behavioral/primary/run_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
  - [`behavioral/primary/run_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
  - [`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
  - [`scripts/run_production_behavioral.sh`](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_behavioral.sh)
- **変更点**:
  - デフォルトの `batch_size` を 243 から安全な 81（V3 評価と同様のサイズ）に変更。
  - `run.py` の CLI オプションに `--batch-size` を追加し、実行スクリプト側から指定・伝播できるように整備。
  - サンプル評価ループ内（10サンプルごと）に `torch.cuda.empty_cache()` を追加。
  - `run_production_behavioral.sh` に `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定し、PyTorch キャッシュの断片化による OOM を防止。

---

## 検証結果

### 単体テスト
[`tests/test_likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_likelihood.py) に、全系列 log_softmax 計算とスライス最適化版の計算結果が完全一致するかを検証する回帰テスト `test_compute_sequence_likelihoods_sliced_equivalence` を追加し、全 13 テストが成功することを確認しました。

```bash
.venv/bin/pytest -v tests/test_likelihood.py
============================== 13 passed in 7.48s ==============================
```

- 数値的等価性: スライス計算による対数尤度および確率分布は、従来の全系列計算と完全一致（`atol=1e-5`）。
- 境界条件: パディングや候補長の差異があっても安全にインデックス参照されることを確認。
