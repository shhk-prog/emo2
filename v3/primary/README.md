# V3 Primary Pipeline (時空間ダイナミクスと因果媒介回路)

本ディレクトリ (`v3/primary/`) は、LLMにおける情動状態の誘発（State Induction）から生成時トークン予測に至る**時空間ダイナミクス（Spatiotemporal Dynamics）**と**因果媒介メカニズム（Path Mediation Mechanism）**を解明するための正式な Primary 実験スイートです。

---

## 1. 論文の主筋：3大リサーチクエスチョン (RQ1〜RQ3)

1. **V3-RQ1: State Induction & Subspace Geometry (情動状態の持続と部分空間幾何)**
   - 刺激提示によって形成された情動表現は、後続プロンプト（指示文・フォーマット指定）の伝播中も持続するか？
   - 状態ベクトルはタスク中立な情動部分空間（Affective Subspace）に直交射影可能か？
2. **V3-RQ2: Spatiotemporal 4-Maps & Peak Dissociation (4-Map 時空間マッピングとピーク解離)**
   - 表現のデコーダビリティ（$D_V, D_A$）と介入因果力（$C_V, C_A$）は、層（Layer）× 意味論的トークンステージ（Semantic Stage: `stimulus_end`, `task_desc`, `format_inst`, `pre_json`, `pre_val`）の 2D 格子上でどのように変移するか？
   - 刺激提示時は中間層でデコードピーク、生成時は後期層プレトークン位置で因果ピークという「時空間ピーク解離」を同定。
3. **V3-RQ3: Causal Mediation & Circuit Mechanism (因果媒介と回路メカニズム)**
   - 刺激受容層（Stimulus Encoding Layer）から出力決定層（Output Generation Layer）への因果的情報伝播において、中間層の情動状態は真の媒介変数（Mediator）として機能しているか？
   - 中心化直交射影除去（Centered Projection Removal: $h - Q Q^\top (h - \mu_{\text{neu}})$）による自然度を保った媒介効果の遮断。

---

## 2. Confirmatory Replication (厳密な事前登録プロトコル追試)

探索的発見（Discovery）を別モデルファミリーおよび独立分割データセット（Confirmation）上で厳密に追試・反証するためのスクリプト：
- **`run_confirmatory_replication.py`**:
  - 5-fold Cross-Validation による held-out $R^2$ 評価
  - 実測ベースラインとの比較
  - 統計的二重解離（Dissociation Metric）の検定

---

## 3. スクリプト構成

```text
v3/primary/
├── README.md                          # 本ドキュメント
├── run_rq1_state_induction.py         # V3-RQ1: 状態誘発と部分空間幾何
├── run_rq2_spatiotemporal_maps.py     # V3-RQ2: 時空間4-Map構築とピーク解離
├── run_rq3_path_mediation.py          # V3-RQ3: 因果媒介解析
└── run_confirmatory_replication.py    # Confirmatory 追試・反証実験
```

---

## 4. 実行方法

### 4.1 RQ1: 状態誘発
```bash
python v3/primary/run_rq1_state_induction.py \
    --config v3/configs/v3_experiments.yaml \
    --models-config v3/configs/models.yaml \
    --device cuda
```

### 4.2 RQ2: 時空間4-Mapマッピング
```bash
python v3/primary/run_rq2_spatiotemporal_maps.py \
    --config v3/configs/v3_experiments.yaml \
    --models-config v3/configs/models.yaml \
    --device cuda
```

### 4.3 RQ3: 因果媒介解析
```bash
python v3/primary/run_rq3_path_mediation.py \
    --config v3/configs/v3_experiments.yaml \
    --models-config v3/configs/models.yaml \
    --device cuda
```

### 4.4 Confirmatory 追試・反証実験
```bash
python v3/primary/run_confirmatory_replication.py \
    --config v3/configs/v3_experiments.yaml \
    --models-config v3/configs/models.yaml \
    --device cuda
```

---

## 5. 評価指標と標準化

- **相対計算深度**:
  - 全モデル共通で $d = \frac{l}{L - 1}$（0-based）を使用。
- **2D Joint Optimal Transport**:
  - `affective_empathy_eval.optimal_transport` による厳密な Wasserstein $W_1$ 距離。
- **多変量 OOD 診断**:
  - Ledoit-Wolf 共分散推定量によるマハラノビス距離 $D_M$、PCA 再構成誤差。
