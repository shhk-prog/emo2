# FairShare GPUタイプ別最適化計画 (Implementation Plan)

判明したGPUごとのFairShare計算式に基づき、コスト（FairShareポイント消費）を最小化しつつ処理スループットを最大化するための最適なGPU選定・設定ガイドラインを構築・整理します。

## 背景と判明した計算式

FairShareのポイント消費式:
`Cost = (GPU係数 × GPU枚数 + CPUコア数) × 利用時間`

| GPUタイプ | GPU係数 ($W_{GPU}$) | CPU=1, GPU=1時の1秒あたりコスト | コスト比率 (vs a6000) | メモリ容量 / 特徴 |
| :--- | :---: | :---: | :---: | :--- |
| **a6000** | **15** | 16 pt/s | **1.00x** | 48GB (GDDR6). コスト最安 |
| **h100** | **30** | 31 pt/s | **1.94x** | 80GB (HBM3). 高スループット/FP8対応 |
| **a6000-ada** | **30** | 31 pt/s | **1.94x** | 48GB (GDDR6 Ada). 高速描画・推論 |
| **h200** | **40** | 41 pt/s | **2.56x** | 141GB (HBM3e). 超大容量・高帯域 |
| **all** | **45** | 46 pt/s | **2.88x** | パーティション未指定時の最高ペナルティ |

---

## 主要な分析と最適化ルール

### 1. ノード/パーティション指定の徹底 (最優先ルール)
> [!IMPORTANT]
> `--partition` や GPUタイプを明示せず `all` (ペナルティ係数 45) で実行すると、a6000 (15) の**約2.88倍**、H100 (30) の**1.48倍**のコストが消費されます。
> ジョブ投入時は必ず目的のGPUタイプを明示的に指定してください。

### 2. CPUコア数の厳格な削減 (特に a6000 / H100 で重要)
- a6000 では、CPUコア数を 1 → 16 に増やすと、コスト係数が 16 → 31 (+93.7%) に跳ね上がります（**CPUコア追加だけでH100相当のコストになる**）。
- **対策**: GPU推論メインのタスクでは、CPUコア数は原則 **1〜2コア**（必要最小限）に設定する。

### 3. モデルサイズ・要求メモリに応じたGPU選定戦略
- **7B〜13Bクラス（メモリ48GB以内で十分なモデル）**:
  - **第1選択: a6000**
    - バッチサイズを大きく取れれば、H100の半額近いコストで処理完了可能。
    - ただし、処理時間がH100の約1.94倍以上遅くなる超高負荷タスクの場合はH100を検討。
  - **第2選択: h100 / a6000-ada**
    - a6000でOOMになる大バッチ処理や、FP8化で速度が2倍以上向上する場合。
- **70Bクラスや長文コンテキスト（メモリ > 80GB 要請）**:
  - **第1選択: h200 1枚** (係数 40 + 1 = 41)
    - H100 2枚 (TP=2: 30×2 + 1 = 61) で載せるよりも、**H200 1枚で載せる方がコストが約33%安く**なり、かつノード間/GPU間通信オーバーヘッドもない。
  - **第2選択: h100 2枚以上**
    - H200が空いていない場合や、さらなる並列計算が必要な場合。

### 4. GPU枚数 (Multi-GPU / Tensor Parallelism) の要否判断
- GPUを2枚に増やすと、GPU時間コスト係数はほぼ2倍になります（例: H100 1枚=31, 2枚=61）。
- 処理時間が半分（50%以下）にならない限り、マルチGPU化は総コストを増加させます。
- **原則**: 1枚のGPUメモリに収まる場合は「**1GPU × 最大バッチサイズ**」が最安。

---

## 提案する変更作業 (Proposed Changes)

#### [NEW] [fairshare_guidelines.md](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/fairshare_guidelines.md)
GPUタイプ別計算式を踏まえた詳細なガイドラインと、GPU選定フローチャート、推奨ジョブパラメータ設定例をまとめたドキュメントを新設。

#### [NEW] [task.md](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/task.md)
本タスクの作業履歴およびタスク一覧。

#### [NEW] [implementation_plan.md](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/implementation_plan.md)
本計画書。

#### [NEW] [walkthrough.md](file:///mnt/nas/home/hiromi/src/sst_v2/docs/fairshare_gpu_type_optimization/walkthrough.md)
作業完了時のまとめと確認レポート。
