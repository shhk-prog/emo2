# Behavioral 評価実行時の CUDA OOM エラー解消 実装計画

## 概要
`bash scripts/run_production_behavioral.sh cuda:0` 実行時、Qwen2.5-1.5B を用いた EmoBank 3-Way 評価の 79/1000 サンプル目で `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.51 GiB` が発生しました。
本計画では、メモリ消費の主因である尤度計算関数内の巨大テンソル生成を根本的に最適化し、安全なバッチサイズとアロケータ設定を導入することで、OOM を防止します。

## 原因分析
1. **語彙サイズと系列長全体の log_softmax 計算による巨大メモリ消費**:
   - `src/affective_empathy_eval/likelihood.py` の `compute_sequence_likelihoods_for_candidates`:
     ```python
     logits = outputs.logits  # (batch_size, max_seq_len, vocab_size)
     log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
     ```
   - Qwen2.5 の語彙数 `vocab_size` は 152,064。
   - `batch_size = 243`、長文の `max_seq_len = 150` の場合、`(243, 149, 152064)` の float32 テンソル 1 つだけで約 22 GB を消費します。
   - 実際には候補文字列（たかだか 2〜5 トークン）の直前ステップの対数確率のみが必要であるにもかかわらず、プロンプト全域（140 トークン以上）の logits を float にキャストして log_softmax を計算しているため、数十 GB の無駄なメモリ確保が発生しています。
2. **デフォルトのバッチサイズ `batch_size=243` が過大**:
   - 1.5B モデルといえど、243 系列を同時に forward すると logits テンソル (float16) だけで 11 GB を超えます。
3. **PyTorch キャッシュの断片化**:
   - 巨大アロケーションと解放の反復により CUDA メモリが断片化し、空き容量があっても連続 16.5 GiB の確保に失敗しています。

---

## 修正方針

### 1. `src/affective_empathy_eval/likelihood.py` の最適化
- 各バッチで評価に必要な最小ステップ `min_step = max(0, min(b_start_idxs) - 1)` を求め、`logits` のうち必要な末尾トークン部分 `logits[:, min_step:-1, :]` のみをスライスして抽出します。
- これにより、系列長 150 ステップのうち 145 ステップ以上の無駄なテンソル確保を排除し、`log_softmax` 計算に必要なテンソルサイズを 1/30 〜 1/50（数百 MB 以下）に削減します。
- バッチ処理ループ毎に `del logits, outputs` を明示的に行い、メモリ解放を確実化します。

### 2. デフォルト `batch_size` の適正化
- `behavioral/primary/run_behavioral_emobank.py` および `behavioral/primary/run_behavioral_aipsy.py` のデフォルト `batch_size` を 243 から 81（V3 評価と同様の値）に設定します。
- `src/affective_empathy_eval/run.py` の `run_behavioral` に `--batch-size` オプションを追加し、CLI 引数経由で指定可能にします。

### 3. 断片化防止設定の追加
- `scripts/run_production_behavioral.sh` で `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定し、PyTorch アロケータの断片化を防止します。

---

## 変更対象ファイル

### [MODIFY] [likelihood.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)
- `compute_sequence_likelihoods_for_candidates` 内で、`b_start_idxs` から `min_step` を算出し、必要なステップ範囲のみをスライスして `F.log_softmax` を計算するように変更。
- スライス基準の相対インデックスで `cand_tokens` の対数確率を取得。

### [MODIFY] [run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
- `--batch-size` のデフォルト値を 81 に変更。
- 定期的な `torch.cuda.empty_cache()` の挿入。

### [MODIFY] [run_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
- `--batch-size` のデフォルト値を 81 に変更。
- 定期的な `torch.cuda.empty_cache()` の挿入。

### [MODIFY] [run.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `--batch-size` 引数をパーサーに追加し、`run_behavioral` で各子スクリプトに渡すよう変更（指定がない場合はデフォルト 81）。

### [MODIFY] [run_production_behavioral.sh](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_production_behavioral.sh)
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を追加。

---

## 検証計画

### 1. 単体テスト
- `pytest tests/test_likelihood.py` （または尤度計算に関連する既存テスト）を実行し、スライス最適化後も対数尤度および確率分布の値が従来と完全に一致することを確認。

### 2. 小規模動作検証
- Qwen2.5-1.5B に対して `--limit 5` や `--limit 100` で `run_behavioral_emobank.py` を実行し、OOM が発生した 79 サンプル目を余裕を持って突破できることを確認。
