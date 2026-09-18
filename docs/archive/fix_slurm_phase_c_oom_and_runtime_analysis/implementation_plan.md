# Slurm Phase C ログエラー修正および実行時間差異の分析・対策計画

## 概要
`/mnt/nas/home/hiromi/src/emo/v1/results/logs/slurm` の実行ログ（Job Array 108271 等）を詳細に解析した結果、以下の2点が明確になりました：
1. **実行時間の極端な差異の要因**:
   - **数十分で完了するジョブ（偶数番: 2, 4, 6...）**: E6 Targeted Ablation（ターゲット層切除）。事前同定された**2層のみ**を評価するため、推論回数が合計1,152回と少なく約20〜35分で終了します。
   - **非常に時間がかかるジョブ（奇数番: 1, 3, 5...）**: E3/E4 Causal Patching（因果パッチング）。`--all-layers` が指定されているため**全層（17〜33層）**を評価し、各層ごとにE3・E4（5つのalpha × 2条件）を192ペアに適用するため、推論回数が合計66,816回（1層あたり2,304回）に達し、1層約46分・全体で約22時間かかります。
2. **発生しているエラーの要因**:
   - **CUDA Out of Memory (OOM)**: `phase_c_108271_3.out` (Qwen2.5-1.5B-Instruct Patching) にて、729候補の尤度計算におけるサブバッチサイズ（`sub_batch_size = 243`）が過大であること、および `extract_hidden_states` で192件の刺激プロンプトを一括フォワードして全層の活性値を一度に確保したことによるメモリ枯渇・断片化。
   - **ライブラリ不在**: `statsmodels` が仮想環境に入っていないため、LMM（線形混合効果モデル）が実行できず fallback（ANOVA / paired t-test）にフォールバックしている。

---

## 実行時間の差異の詳細分析

| 項目 | E6 Targeted Ablation (短時間: ~20-35分) | E3/E4 Causal Patching (長時間: ~22時間) |
| :--- | :--- | :--- |
| **対象スクリプト** | `run_v1_phase_c_targeted_ablation.py` (Job 2, 4, 6, 8, ...) | `run_v1_phase_c_causal_patching.py` (Job 1, 3, 5, 7, ...) |
| **評価対象レイヤー** | **2層のみ**（`reader_layer`, `self_layer`） | **全層（17〜33層）**（`--all-layers` 指定時） |
| **1層あたりの処理内容** | 2サイト × 2タスク = 4条件 × 192ペア | E3 (384回) + E4 (5アルファ × 2条件 × 192ペア = 1,920回) = **2,304回** |
| **合計評価回数** | ベースライン (384) + 切除 (768) = **1,152回** | ベースライン (768) + 29層 × 2,304回 = **67,584回** |
| **729候補 forward 回数** | 1,152 × 3バッチ = **3,456回** | 67,584 × 3バッチ = **202,752回** |
| **計算負荷比** | **1 (基準)** | **約 58.7 倍** |
| **実績所要時間** | Qwen 1.5B: 約35分 / Llama 1B: 約20分 | Qwen 1.5B: **約22.2時間**（46分/層） / Llama 1B: **約13.6時間**（48分/層） |

> [!NOTE]
> 偶数ジョブ（Ablation）は「二重解離の検証」に絞った2層の局所実験であるのに対し、奇数ジョブ（Patching）は「全層にわたる因果ダイナミクスと転移性マップの網羅的探索」であるため、純粋に計算ステップ数が約60倍存在します。

---

## エラーの根本原因と修正計画

### 1. CUDA Out of Memory (OOM) 対策
#### 原因
- `sub_batch_size: int = 243`: Qwen2.5 のボキャブラリ数は 151,936、Llama 3.2 は 128,256。243件 × 系列長 × 語彙サイズの logits テンソルだけで十数GBのGPUメモリを瞬間確保しようとする。
- `extract_hidden_states`: 全192ペアのプロンプトを一括バッチ（batch_size=192）で `output_hidden_states=True` を付けてフォワードパスしているため、全層の活性値が巨大テンソルとして一気にメモリ展開され断片化を招く。
- キャッシュ解放の欠如: ループ内で `torch.cuda.empty_cache()` が呼ばれておらず、断片化が蓄積。

#### 修正方針
1. **`sub_batch_size` の安全化**:
   - デフォルトを `243` から `81` に引き下げ（メモリ使用量を1/3に削減）。
   - コマンドライン引数 `--sub-batch-size` を追加し、モデル規模やGPUに応じて調整可能にする。
2. **`extract_hidden_states` のミニバッチ化**:
   - `extract_hidden_states` で `batch_size=16` 程度に分割してエンコード・推論し、CPUへ逐次退避。
3. **明示的キャッシュクリア**:
   - 各レイヤーのイテレーション終了時、およびバッチ評価ごとに `torch.cuda.empty_cache()` を実行。
4. **メモリ断片化防止環境変数の導入**:
   - `slurm_run_phase_c.sbatch` に `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を指定。

### 2. `statsmodels` 不在エラーの対策
- 仮想環境 `.venv` に `statsmodels` をインストールし、E6 Double Dissociation における本番の線形混合効果モデル（LMM: Fixed Effects + Random Intercepts by Pair）解析が正常動作するようにする。

---

## 変更対象ファイル

### [MODIFY] [run_v1_phase_c_causal_patching.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_causal_patching.py)
- `evaluate_expected_va_batch` の `sub_batch_size` を 81 に変更、`--sub-batch-size` 引数を追加。
- `extract_hidden_states` をミニバッチ（`batch_size=16`）処理に変更。
- レイヤーループ内に `torch.cuda.empty_cache()` を追加。

### [MODIFY] [run_v1_phase_c_targeted_ablation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py)
- `evaluate_expected_va` の `sub_batch_size` を 81 に変更、`--sub-batch-size` 引数を追加。
- `torch.cuda.empty_cache()` を追加。

### [MODIFY] [slurm_run_phase_c.sbatch](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/slurm_run_phase_c.sbatch)
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を追加。

---

## 検証手順
1. **構文チェック・型確認**:
   - Pythonスクリプトの構文チェック (`python -m py_compile ...`)
2. **省メモリ化の動作確認**:
   - `--limit 2` によるドライラン / 少量実行で OOM なく正常終了することを確認。
3. **Slurmジョブ復旧手順の案内**:
   - 失敗した Job 3 (`qwen2.5_1.5b_instruct_patching`) の個別再投入コマンドの提示。
