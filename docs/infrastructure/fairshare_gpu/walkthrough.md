# FairShare GPUタイプ別最適化 ガイドライン策定完了報告

## 概要
判明した各GPUタイプ（a6000, h100, a6000-ada, h200, all）のFairShare計算式に基づき、総合的なコスト最適化ガイドラインおよびシミュレーションを作成・整理しました。

---

## 主な成果とガイドラインのポイント

### 1. GPUタイプごとのコスト特性一覧
- **a6000 (係数15)**: 最安コスト。7B〜13Bモデルの標準推論・評価ジョブに最適。
- **h100 / a6000-ada (係数30)**: a6000の1.94倍コスト。FP8等の高速化技法が有効な場合や大バッチ処理に最適。
- **h200 (係数40)**: メモリ141GB。70Bモデルを単一GPUで動かす際に最適（H100 2枚指定するより**コスト33%削減**）。
- **all (係数45)**: パーティション未指定のペナルティ。**指定必須**。

### 2. コスト削減のためのアクションルール
1. ジョブ投入時に必ず `--partition` または GPUタイプを明示する（`all` 回避）。
2. CPUコア数は `1`〜`2` コアに厳しく制限する。
3. 1枚のGPUメモリに載る場合は「単一GPU + 最大バッチサイズ」を適用する。
4. 70Bクラスの大型モデルは H100 2枚ではなく H200 1枚を優先選択する。

---

## 作成・保存されたファイル
- [`task.md`](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/task.md)
- [`implementation_plan.md`](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/implementation_plan.md)
- [`walkthrough.md`](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/walkthrough.md)
- [`fairshare_guidelines.md`](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/fairshare_guidelines.md)
