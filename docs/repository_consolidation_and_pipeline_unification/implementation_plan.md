# リポジトリ正本一本化・Primary Pipeline 確立・研究ストーリー統一 実装計画

本計画は、ユーザーからの包括的な指示に基づき、「実験の追加」ではなく**「コードと研究仕様の正本一本化、Primary/Exploratory/Legacy の物理的分離、成果物管理の整理、再現性の担保」**を完遂するための総合設計である。

---

## ユーザー確認が必要な事項 (User Review Required)

> [!IMPORTANT]
> **1. パッケージ正本を root `src/affective_empathy_eval` の1箇所に集約**
> - `v1/src/affective_empathy_eval/` 内のモジュール（`data.py`, `evaluation.py`, `extraction.py`, `intervention.py`, `manifests.py`, `metrics.py`, `probing.py`, `schemas.py`, `splits.py`, `controls.py`）をすべて root `src/` へ統合します。
> - `v1/src/` は削除し、インポート競合を根絶します。
>
> **2. Behavioral ステージの完全独立 (`behavioral/`)**
> - 論文構成（Behavioral → V1 → V2 → V3）に合わせ、`v1` 配下に同居していた EmoBank (3-way) および AIPsy (4-split) の行動評価を `behavioral/` に独立させます。
>
> **3. スクリプトの3階層分離 (`primary/`, `exploratory/`, `legacy/`)**
> - 各ステージ（V1, V2, V3）において、論文の主要結果を生成する Primary スクリプトのみを `primary/` に配置し、それ以外の探索的実験や旧スクリプトは `exploratory/` および `legacy/` へ隔離します。
>
> **4. 相対深度 (Relative Depth) の全ステージ統一**
> - 全ステージで以下に統一します：
>   $$d = \frac{l}{L - 1} \quad (l \in \{0, \dots, L-1\})$$
>   （※ これまで Phase C で使われていた $(l + 1) / L$ や V2/V3 の表記揺れを完全解消）
>
> **5. キャッシュへの実験条件ハッシュ導入**
> - `model_id`, `git_commit`, `prompt_hash`, `dataset_hash`, `n_pairs` を含むハッシュ検証を導入し、条件不一致時の誤再利用を完全防止します。
>
> **6. `docs/` アーカイブ整理と `.gitignore` の厳格化**
> - 過去のタスク・進捗記録（200ファイル以上）を `docs/archive/` に退避し、`.DS_Store`, `._*`, 巨大な生データ/チェックポイントを Git 管理から除外します。

---

## 変更内容 (Proposed Changes)

### Phase 1: 共通パッケージの Root 一本化

#### [MOVE & MERGE] `v1/src/affective_empathy_eval/` $\to$ `src/affective_empathy_eval/`
- 以下のファイルを root `src/affective_empathy_eval/` へ移行：
  - `data.py`: データセット読み込み
  - `evaluation.py`: 評価プロトコル
  - `extraction.py`: 活性抽出
  - `intervention.py`: `PyTorchActivationPatcher`（`interventions.py` と両立させエイリアス提供）
  - `manifests.py`: 実験マニフェスト記録
  - `metrics.py`: 指標算出
  - `probing.py`: 線形プロービング
  - `schemas.py`: 出力スキーマ
  - `splits.py`: データ分割
  - `controls.py`: 統制条件
- `v1/src/` を削除。

#### [MERGE] `v2/src` & `v3/src` $\to$ `src/affective_empathy_eval/`
- `v3/src/ot_utils.py` $\to$ [`src/affective_empathy_eval/optimal_transport.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/optimal_transport.py)
- `v3/src/diagnostics.py` $\to$ [`src/affective_empathy_eval/diagnostics.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/diagnostics.py)
- `v2/src/likelihood.py` および `v3/src/batch_likelihood.py` の独自実装を廃止し、root の正準 [`src/affective_empathy_eval/likelihood.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py) に統一。

---

### Phase 2: Behavioral ステージの独立

#### [NEW] `behavioral/`
- ディレクトリ構成：
  ```text
  behavioral/
  ├── primary/
  │   ├── run_behavioral_emobank.py
  │   ├── run_behavioral_aipsy.py
  │   ├── summarize_behavioral_emobank.py
  │   └── summarize_behavioral_aipsy.py
  ├── data/
  └── README.md
  ```
- 既存の `v1/scripts/run_3way_vad_evaluation.py` および `run_aipsy_4split_evaluation.py` を移行。

---

### Phase 3: V1, V2, V3 の Primary / Exploratory / Legacy 分離

#### V1 (`v1/`)
- `v1/primary/`:
  - `run_phase_a.py`: E1 デコード能 & E2 幾何アライメント
  - `run_phase_b.py`: E5 意味的交絡監査
  - `phase_c/`:
    - `prepare_cache.py`: baseline, hidden states キャッシュ生成
    - `run_e3_causal_map.py`: 全層因果マップ探索
    - `select_e4_sites.py`: Discovery split ピークからの候補層選定
    - `run_e4_interchangeability.py`: Confirmation split 上での matched vs random 評価
    - `run_e6_specialization.py`: Targeted ablation & LMM 交互作用
    - `summarize_phase_c.py`: Phase C レポート作成
- `v1/exploratory/`: スケーリング、抑制実験など
- `v1/legacy/`: 旧スクリプト

#### V2 (`v2/`)
- `v2/primary/`:
  - `run_rq1_rq2_cross_decoding.py`: 幾何構造 & タスク間共有
  - `run_rq3_causal_map.py`: 因果マップ再編成
  - `run_rq4_recovery_patching.py`: 回復パッチング
  - `run_confirmatory_analysis.py`: 追試統計解析
- `v2/exploratory/`: steering, component patching, temperature scaling
- `v2/legacy/`: 過去スクリプト
- README の更新: 重複セクション削除、Primary RQ (RQ1〜RQ4) へ集約。

#### V3 (`v3/`)
- `v3/primary/`:
  - `run_rq1_state_induction.py`: State $\to$ Report 因果依存
  - `run_rq2_spatiotemporal_maps.py`: 空間・時間局在化
  - `run_rq3_path_mediation.py`: 経路媒介分析
  - `run_confirmatory_replication.py`: 他モデル追試
- `v3/exploratory/`: mood congruency, multilayer OT, aligned patching
- `v3/legacy/`: 旧スクリプト
- `v3/docs/legacy/`: 旧論文ドラフト（greedy collapse / neutralization 主張のもの）を完全隔離。

---

### Phase 4: 相対深度・キャッシュハッシュ・メタデータの統一

1. **相対深度の統一**:
   - すべてのスクリプトで `relative_depth = l / (num_layers - 1)` に統一。
2. **キャッシュハッシュの導入**:
   - `hashlib.sha256` により `(model_id, prompt_template, dataset_hash, git_commit)` のハッシュを生成し、キャッシュディレクトリまたはメタデータで検証。
3. **実験マニフェストの保存**:
   - 実行時に `experiment_manifest.json` を全 Stage で共通保存。

---

### Phase 5: 環境設定・成果物管理・クリーンアップ

1. **成果物ディレクトリの3階層統一 (`results/`)**:
   - `raw/`: 巨大応答（Git 管理外）
   - `derived/`: 中間集計（Git 管理外）
   - `release/`: 論文の図表生成用サマリー（Git 管理対象）
2. **不要ファイルの完全除去 & `.gitignore` の厳格化**:
   - `.DS_Store`, `._*` を全削除。
   - `.gitignore` に一時ファイル、大容量バイナリ、キャッシュを追加。
3. **`pyproject.toml` の正本化**:
   - 依存関係および pytest 設定を `pyproject.toml` に一本化し、`pytest.ini` との重複を解消。
4. **`docs/` の整理**:
   - 200 以上の過去実装計画・walkthrough を `docs/archive/` に移動。

---

## 検証計画 (Verification Plan)

### 自動テスト
- `pytest tests/unit/ -v` (共通モジュールの単体テスト)
- `pytest tests/integration/ -v` (モデル・フック・アンカー連携テスト)
- `pytest tests/test_phase_c_tokenization_and_anchors.py -v` (Phase C 整合性テスト)

### 再現性・クリーンアップ確認
- `git status` で `.DS_Store`, `._*` が完全に除外されていることの確認
- `python -c "import affective_empathy_eval; print(affective_empathy_eval.__file__)"` で root `src/` のみがロードされることの確認
- 各 Primary スクリプトの `--help` または dry-run でのインポート検証
