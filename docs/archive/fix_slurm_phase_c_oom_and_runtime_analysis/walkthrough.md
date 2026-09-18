# Slurm Phase C ログエラー修正・実行時間差異分析・成果物自動スキップ機能追加報告（Walkthrough）

## 1. 作業の目的と達成状況
本作業では、`/mnt/nas/home/hiromi/src/emo/v1/results/logs/slurm` におけるジョブ実行ログ（Job Array 108271 等）を精査し、以下の項目を完了しました：
1. **ジョブ実行時間の極端な差異（約22時間 vs 約20〜35分）の構造的・計算量的なメカニズム解明**
2. **CUDA Out of Memory (OOM) エラーの根本修正（バッチサイズ最適化・ミニバッチ活性値抽出・キャッシュ解放・環境変数設定）**
3. **実行済み・成功済みジョブの自動スキップ機能の配備（二重ガード＆ `--force` オプション対応）**

---

## 2. 実行時間に大きな差が生じる理由

Slurm Job Array（1〜16）は、各モデルに対して以下の2つの異なるタスクを交互に実行しています。

### 計算量・推論回数の構造的比較

| 項目 | E6 Targeted Ablation (短時間: 約20〜35分) | E3/E4 Causal Patching (長時間: 約22時間) |
| :--- | :--- | :--- |
| **対象ジョブ** | **偶数番（Job 2, 4, 6, 8, ...）** | **奇数番（Job 1, 3, 5, 7, ...）** |
| **実行スクリプト** | [`run_v1_phase_c_targeted_ablation.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py) | [`run_v1_phase_c_causal_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_causal_patching.py) |
| **対象レイヤー** | **2層のみ**（事前同定された `reader_layer` と `self_layer`） | **全層（17〜33層）**（`--all-layers` オプション指定時） |
| **1層あたりの処理内容** | 2サイト × 2タスク = 4条件 × 192ペア | E3（384回）＋ E4（5アルファ × 2条件 × 192ペア）＝ **2,304回** |
| **全層合計推論回数** | ベースライン(384) ＋ 切除(768) = **1,152回** | ベースライン(768) ＋ 29層 × 2,304回 = **67,584回** |
| **729候補 forward 回数** | 1,152 × 3バッチ = **3,456回** | 67,584 × 3バッチ = **202,752回** |
| **計算負荷比** | **1（基準）** | **約 58.7 倍** |
| **実測所要時間** | Qwen 1.5B: **約35分**（完了）<br>Llama 1B: **約20分**（完了） | Qwen 1.5B: **約22.2時間**（実測: 46分/層）<br>Llama 1B: **約13.6時間**（実測: 48分/層） |

- **結論**:
  - **Ablation（短時間）**: 事前同定された「Reader局在層」と「Self局在層」の2層のみをゼロ置換・復元し、相互の二重解離（Double Dissociation）を検定する局所実験であるため、非常に軽量です。
  - **Patching（長時間）**: モデルの全層（Qwen 1.5Bは29層、Mistral 7Bは33層）において、感情差分ベクトルを注入した際の因果変位量およびAlphaスイープ（[-1.0, 0.0, 0.5, 1.0, 1.5]）を探索するため、**約60倍のステップ数**が存在します。

---

## 3. エラー修正と最適化の内容

### 3.1 CUDA Out of Memory (OOM) エラーの解消
- **発生箇所**: `phase_c_108271_3.out` (Qwen2.5-1.5B-Instruct Patching)
- **修正内容**:
  1. **サブバッチサイズの圧縮**: 729候補の尤度計算におけるサブバッチサイズを `243` から安全な **`81`** に引き下げ（メモリ使用量を1/3に削減）。
  2. **ミニバッチ活性値抽出**: `extract_hidden_states` で全192ペアのプロンプトを一括フォワードするのをやめ、**`batch_size=16` のミニバッチ処理** でCPUへ逐次退避。
  3. **キャッシュ解放**: 各レイヤーループ終了時に `torch.cuda.empty_cache()` を実行。
  4. **断片化防止**: [`slurm_run_phase_c.sbatch`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/slurm_run_phase_c.sbatch) に **`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** を配備。

### 3.2 完了済みジョブの成果物自動スキップ機能
- **シェルレベルの自動スキップ**:
  [`run_all_phase_c.sh`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_all_phase_c.sh) の `run_job_by_index` にて、該当モデル・タスクの成果物（`e4_interchangeability_results.csv` または `e6_double_dissociation_summary.md`）が既に存在する場合、重いPythonプロセスの起動やGPUモデルロードを行わずに即座に `[SKIP]` して return 0 します。
- **Pythonスクリプトレベルの二重ガード**:
  [`run_v1_phase_c_causal_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_causal_patching.py) および [`run_v1_phase_c_targeted_ablation.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py) でも成果物の存在を確認し、モデル読み込み前に即座にスキップします。
- **強制再実行オプション**:
  既存の成果物を破棄して再実行したい場合は、`--force` または `-f` を付与することで強制再計算が可能です。

---

## 4. 動作検証結果

1. **完了済みジョブのスキップ検証**:
   - `bash v1/scripts/run_all_phase_c.sh --job 2` を実行：
     ```text
     [SKIP] Job 2: qwen2.5_1.5b_base_ablation is already completed.
     Found output: v1/results/derived/v1_phase_c/qwen2.5_1.5b_base/e6_double_dissociation_summary.md
     Skipping execution (use --force to re-run).
     ```
     0.1秒で即座にスキップ完了。
2. **未完了ジョブの正常検知**:
   - `bash v1/scripts/run_all_phase_c.sh --job 3 --all-layers --dry-run` を実行：
     成果物が存在しない Job 3 はスキップされず、`sub-batch-size 81` を付与して正常に実行対象として検知。
3. **`--force` フラグの動作検証**:
   - `bash v1/scripts/run_all_phase_c.sh --job 2 --force --dry-run` を実行：
     既存成果物を無視して実行対象となり、Pythonスクリプトにも `--force` が正しく伝播。

---

## 5. 今後のジョブ投入方法

### 全ジョブの一括再投入（完了済みは自動で瞬時にスキップ）
```bash
sbatch v1/scripts/slurm_run_phase_c.sbatch --all-layers
```
- Job 2, Job 4, Job 6 など完了済みのものは数秒で `[SKIP]` 終了します。
- 未完了のジョブや、先ほどOOMで停止した Job 3 のみが自動的に安全なバッチサイズ（`81`）で実行されます。

### 特定ジョブ（例: Job 3）のみを実行したい場合
```bash
sbatch --array=3 v1/scripts/slurm_run_phase_c.sbatch --all-layers
```

### 完了済みであっても強制的に再計算させたい場合
```bash
sbatch --array=1-16 v1/scripts/slurm_run_phase_c.sbatch --all-layers --force
```

---

## 6. v2・v3 パイプライン（Step 1 〜 Step 7）の実行コマンド

### 6.1 実行前準備（必須環境変数）
ルートディレクトリ配下の共通ライブラリ `affective_empathy_eval` をロードするため、`PYTHONPATH` を設定します。
```bash
cd /mnt/nas/home/hiromi/src/emo
source .venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
```

### 6.2 単独実行コマンド（Step別）

| Step | 解析フェーズ・スクリプト | GPU実実行コマンド | 軽量動作確認（Dry-run）コマンド |
| :--- | :--- | :--- | :--- |
| **Step 1** | **共通基本機能テスト** | `pytest tests/` | `pytest tests/` |
| **Step 2** | **V2-RQ1 & RQ2: 表現幾何・交差デコーディング**<br>([`run_v2_2x2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_2x2_cross_decoding.py)) | `python v2/scripts/run_v2_2x2_cross_decoding.py --device cuda` | `python v2/scripts/run_v2_2x2_cross_decoding.py --dry-run --device cpu` |
| **Step 3** | **V2-RQ3: 因果マップ＆ピーク解離解析**<br>([`run_v2_2x2_causal_map.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_2x2_causal_map.py)) | `python v2/scripts/run_v2_2x2_causal_map.py --device cuda` | `python v2/scripts/run_v2_2x2_causal_map.py --dry-run --device cpu` |
| **Step 4** | **V2-RQ4: 分布回復パッチング (EMD_VA)**<br>([`run_v2_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_recovery_patching.py)) | `python v2/scripts/run_v2_recovery_patching.py --device cuda` | `python v2/scripts/run_v2_recovery_patching.py --dry-run --device cpu` |
| **Step 5** | **V3-RQ1: 内的状態誘導 & Go/No-Go Gate**<br>([`run_v3_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_state_induction.py)) | `python v3/scripts/run_v3_state_induction.py --device cuda` | `python v3/scripts/run_v3_state_induction.py --dry-run --device cpu` |
| **Step 6** | **V3-RQ2 & RQ3: 時空間4マップ & 経路媒介分析 (Qwen)**<br>([`run_v3_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_spatiotemporal_maps.py) & [`run_v3_path_mediation.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_path_mediation.py)) | `python v3/scripts/run_v3_spatiotemporal_maps.py --device cuda && python v3/scripts/run_v3_path_mediation.py --device cuda` | `python v3/scripts/run_v3_spatiotemporal_maps.py --dry-run --device cpu && python v3/scripts/run_v3_path_mediation.py --dry-run --device cpu` |
| **Step 7** | **V3: 確証的追試分析 (Llama, Gemma, Mistral)**<br>([`run_v3_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_confirmatory_replication.py)) | `python v3/scripts/run_v3_confirmatory_replication.py --device cuda` | `python v3/scripts/run_v3_confirmatory_replication.py --dry-run --device cpu` |

### 6.3 一括実行コマンド（Step 1 〜 Step 7 を一気に実行）

[`v3/scripts/run_v2_v3_full_pipeline.sh`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v2_v3_full_pipeline.sh) を用いて一括実行が可能です：

```bash
# GPU で Step 1 〜 Step 7 を一括実行
bash v3/scripts/run_v2_v3_full_pipeline.sh --device cuda

# 全ステップのモック一括検証（Dry-run）
bash v3/scripts/run_v2_v3_full_pipeline.sh --dry-run

# 特定ステップのみの実行（例: Step 3）
bash v3/scripts/run_v2_v3_full_pipeline.sh --step 3 --device cuda

# 途中からの再開（例: Step 4 以降）
bash v3/scripts/run_v2_v3_full_pipeline.sh --from-step 4 --device cuda
```

