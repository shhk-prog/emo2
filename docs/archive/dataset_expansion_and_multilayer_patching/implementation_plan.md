# 厳密最小対データセット拡張・多層同時パッチング・Mahalanobis OOD診断の実装計画

本計画は、ICLR本会議での採択可能性を決定的な水準に引き上げるため、以下の3大実証施策を実装・実行する方針をまとめたものである：
1. **Strict Minimal-Pair Dataset の 50〜100 pair への拡張**
2. **Multi-layer Simultaneous Patching による分散性仮説（N3）の直接検証**
3. **共分散に基づく Mahalanobis 距離を用いた多変量 OOD 診断の実装**

---

## 1. User Review Required (ユーザー確認・事前相談事項)

> [!IMPORTANT]
> **GPU利用に関する事前相談（リポジトリ運用ルール 1.7 遵守）**:  
> 本実験では、`Qwen/Qwen2.5-1.5B` および `Qwen/Qwen2.5-1.5B-Instruct` の活性化抽出・Ridgeアライメント・パッチング下での81候補列尤度計算を行います。
> - **使用リソース**: GPU 1基（CUDA:0等）、VRAM約 8〜10GB程度
> - **推定所要時間**: 拡張データセット（Held-out Test 15〜30ペア × 4層条件）で約15〜30分程度
> - **確認事項**: 実装完了後、ローカルGPU環境（CUDA）を用いた推論・実験実行を行ってよいかご確認をお願いいたします。

---

## 2. 実装詳細とコンポーネント設計

### 2.1 Strict Minimal-Pair Dataset の拡張 ($n=10 \rightarrow n=50\sim 100$)

- **入力データプール**:
  - `v1/data/processed/aipsy/train.csv`, `dev.csv`, `test.csv` (1,283行超のプール)
  - `v2/data/processed/aipsy_annotated/train_strict.csv`, `test_strict.csv`
- **厳密抽出アルゴリズム (`v3/scripts/expand_strict_dataset.py`)**:
  - 各 `pair_id` について、`condition == "affective" & intensity == "peak"`、`condition == "affective" & intensity == "moderate"`、および対応する `condition == "neutral"` が完全に1対1対1で揃っている厳密トリプレットのみを抽出。
  - 文長（word_count）の差が極端でないこと、明示的な感情語（happy, sad等）による表面交絡が統制されたシナリオを選定。
  - 目標規模: **50〜80ペア（150〜240サンプル）**。
- **データ分割**:
  - `pair_id` をキーとする `GroupShuffleSplit` により、完全非重複の厳密3分割（Train 40%, Dev 30%, Test 30%）を生成し、`v3/data/aipsy_strict_expanded.csv` に保存。

### 2.2 共分散に基づく Mahalanobis 距離を用いた多変量 OOD 診断

従来の単変量診断（L2ノルム比、コサイン類似度）は多変量空間における相関崩壊（共分散構造からの乖離）を検知できないという理論的批判に対応する。

- **数理モデル (`v3/src/diagnostics.py`)**:
  ターゲット層におけるInstructモデルの自然な隠れ状態分布 $\mathcal{D}_{\mathrm{target}}$ の平均 $\boldsymbol{\mu}$ および共分散行列 $\boldsymbol{\Sigma}$ を推定する。高次元（$d=1536$）かつサンプル数有限に対応するため、**Ledoit-Wolf 正則化共分散推定**（Shrinkage）を採用：
  
  $$\boldsymbol{\Sigma}_{\mathrm{reg}} = (1 - \lambda)\boldsymbol{\Sigma}_{\mathrm{sample}} + \lambda \frac{\mathrm{Tr}(\boldsymbol{\Sigma}_{\mathrm{sample}})}{d}\mathbf{I}$$
  
  パッチ活性化ベクトル $\mathbf{h}$ に対する Mahalanobis 距離：
  
  $$D_M(\mathbf{h}) = \sqrt{(\mathbf{h} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}_{\mathrm{reg}}^{-1} (\mathbf{h} - \boldsymbol{\mu})}$$

- **診断指標**:
  1. In-distribution（自然なInstruct活性化）の $D_M$ 分布（50パーセンタイル, 95パーセンタイル）
  2. Raw Base活性化の $D_M$
  3. Ridge-aligned Base活性化の $D_M$  
  これにより、アライメントによって多変量統計多様体への適合度がどの程度回復しているかを定量化。

### 2.3 Multi-layer Simultaneous Patching スクリプトの実装

N3（分散的変化仮説）を「単一ボトルネックの否定」から「多層介入による相加的・分散的回復の実証」へと引き上げる。

- **スクリプト (`v3/scripts/run_multilayer_aligned_patching.py`)**:
  - **介入ブロック条件**:
    1. **1-Layer (Baseline)**: Layer 15 のみ
    2. **2-Layer**: Layers [14, 15]
    3. **4-Layer**: Layers [13, 14, 15, 16]
    4. **8-Layer**: Layers [11, 12, 13, 14, 15, 16, 17, 18]
  - **アライメント学習**:
    - 各対象層 $\ell$ に対し、Alignment-dev分割を用いて独立に Ridge 写像 $\mathbf{W}_\ell$ を学習。
  - **同時フック注入**:
    - 指定された層集合の全モジュールに対し、同一順伝播パス内で一括してアライメント済み活性化を置換。
  - **評価エンドポイント**:
    - 各ブロックサイズにおける **Normalized 2D EMD Recovery** および期待 Valence $E[V]$ の回復率推移を測定。
    - 層数の増加に伴い回復率が単調増加するか、それとも依然として0%近傍にとどまるかを同定。

---

## 3. 実装・ファイル変更一覧

### [NEW] [`v3/scripts/expand_strict_dataset.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/expand_strict_dataset.py)
データプールから厳密トリプレットを網羅抽出し、統制・フィルタリングして `v3/data/aipsy_strict_expanded.csv` を生成するスクリプト。

### [NEW] [`v3/src/diagnostics.py`](file:///mnt/nas/home/hiromi/src/emo/v3/src/diagnostics.py)
Ledoit-Wolf正則化共分散逆行列と Mahalanobis 距離 $D_M$、主成分射影距離を計算する診断モジュール。

### [NEW] [`v3/scripts/run_multilayer_aligned_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_multilayer_aligned_patching.py)
多層同時パッチング、多変量OOD診断、および2D EMD回復率測定を一括実行する統合スクリプト。

---

## 4. 検証計画

1. **データセット完全性検証**:
   - `v3/data/aipsy_strict_expanded.csv` 内の全ペアについて、peak/moderate/neutralが揃っていること、および Train/Dev/Test 間で同一 `pair_id` が一切重複（リーク）していないことをスクリプトで検証。
2. **多変量OOD診断の数値検証**:
   - 人工的なIn-distributionサンプル vs Out-of-distributionサンプルにおいて、Mahalanobis距離が正しく分離することを確認。
3. **多層パッチングのDry-run**:
   - 1ペアを用いた小規模テストを実行し、フックの多重登録・解除、対数尤度計算が正常に動作することを確認。
