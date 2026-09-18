# LLM情動反応性評価実験 v3 (Causal Mechanisms: State-to-Report & Mood Congruency)

本リポジトリ (`v3`) は、v1（行動連動および表現存在の解明）および v2（事後学習による表現幾何・因果回路の変容）の知見を受け、**「モデル内部で形成された情動状態は、どのような時空間経路を経て自己報告として出力されるのか？」**、そして **「内部情動状態の操作は、他者認識や認知的判断を因果的にバイアスするのか（気分一致効果）？」** という高度な因果的メカニズムを解明するための第3フェーズ実験環境です。

単一トークンのプロービングにとどまらず、生成プロセス全体（autoregressive generation）における意味的アンカーごとの介入、全層コンポーネント（MLP, Attention, Residual）網羅的因果局在化スイープ、最適輸送（Optimal Transport）多様体整列パッチング、および気分一致因果実験（Mood Congruency Experiment）を実装しています。

---

## 1. 核心的リサーチクエスチョン (Core Research Questions)

v3では、設定ファイル `configs/v3_experiments.yaml` および ICLR 投稿論文の構成に基づき、以下の主要課題を検証します：

```text
【V3-RQ1: 内部情動状態の誘導と Go/No-Go ゲート判定 (State Induction & Gate)】
  問い: 外部刺激ラベルから抽出された感情方向ベクトル (d_V, d_A) の注入によって、
        自己報告を因果的かつ選択的に誘導できるか？
  判定基準: 5大厳格基準 (Sufficiency, Necessity, Specificity, Dose-response, Selectivity)

【V3-RQ2 & RQ3: 時空間全探索とパス仲介分析 (Spatiotemporal 4-Map & Path Mediation)】
  問い: 情動情報は、推論プロンプト時および生成プロセスのどの層・どの意味段階を経て出力へ伝播するか？
        中間層の特定コンポーネントが、最終出力への直接経路・間接経路として機能しているか？
  手法: 28層 × 6意味段階の時空間パッチング（探索的 Discovery マッピング: 全層・全位置探索後、主要候補ウィンドウを Confirmation 拡大サンプルで再検証）, Discovery / Confirmation 分割パス仲介分析

【発展課題 1: 気分一致因果実験 (Mood Congruency Causal Experiment)】
  問い: 内部感情状態（Induced Mood）のステアリングは、感情認識（他者予測）を因果的に歪めるか？
  手法: 感情状態注入下の Reader 推論評価, 気分一致バイアス係数・相関分析

【発展課題 2: デコード可能性と因果的影響の解離 (Causal Localization Sweep)】
  問い: 線形プローブで感情情報が最も強く読み出せる層（argmax D_l）は、
        出力を最も強く動かす因果的層（argmax C_l）と一致するのか？
  仮説: Decodability != Causal Influence (相関 rho ≈ 0 による完全な解離の立証)

【発展課題 3: 最適輸送による多層多様体整列パッチング (Multilayer Aligned Patching)】
  問い: モデル間で異なる座標系を持つ中間表現を、多様体整列（Procrustes / OT）によって移植し、
        多層同時に因果的介入を行うことは可能か？
```

---

## 2. ディレクトリ構成とモジュール設計

v3は、共通基盤ライブラリ (`src/affective_empathy_eval`) と連携しつつ、高速バッチ処理と最適輸送距離計算に最適化された独自のモジュール群と解析スクリプト群で構成されています。

```text
v3/
├── README.md                          # 本ドキュメント
├── src/                               # v3 専用コアモジュール群
│   ├── batch_likelihood.py            # 高速バッチ化 81候補対数尤度計算 (厳密な数値一致を保証)
│   ├── ot_utils.py                    # 2次元最適輸送 (2D Optimal Transport / EMD), 周辺WD, 回復率計算
│   ├── model_utils.py                 # レイヤー・コンポーネント (MLP/Attn/Resid) 抽出, パッチフック管理
│   └── diagnostics.py                 # 介入安定性・確率質量・境界条件の診断関数
│
├── scripts/
│   ├── # --- V3 中核パイプライン (run_v2_v3_full_pipeline.sh 対応) ---
│   ├── run_v3_state_induction.py      # V3-RQ1: 内部情動状態誘導と Go/No-Go ゲート判定
│   ├── run_v3_spatiotemporal_maps.py  # V3-RQ2: 時空間 4-Map (全層 × 6生成ステージ)
│   ├── run_v3_path_mediation.py       # V3-RQ3: パス仲介分析 (Discovery/Confirmation)
│   ├── run_v3_confirmatory_replication.py # 他3モデル (Llama, Gemma, OLMo: Primary 1-1.5B コホート) 確証的再現
│   ├── run_v2_v3_full_pipeline.sh     # V2 & V3 統合パイプライン実行シェルスクリプト
│   │
│   ├── # --- 気分一致因果実験 (Mood Congruency) ---
│   ├── run_mood_congruency_experiment.py # 内部情動ステアリング下での他者感情認識バイアス測定
│   ├── analyze_mood_congruency.py     # 気分一致効果の統計解析・可視化
│   │
│   ├── # --- 因果局在化スイープ (Causal Localization Sweep) ---
│   ├── run_causal_localization_sweep.py # 全層 (MLP/Attn/Resid) デコード能 vs 因果影響の解離
│   ├── run_generation_time_causal_sweep.py # 生成時マルチレイヤー因果スイープ
│   ├── run_focused_39pairs_sweep.py   # 絞り込みペアに対する詳細スイープ
│   │
│   ├── # --- 多層・多様体整列パッチング ---
│   ├── run_multilayer_aligned_patching.py # 多層同時整列パッチング
│   ├── run_aligned_cross_model_patching.py # モデル横断 最適輸送 (OT) 多様体整列パッチング
│   ├── run_probe_aligned_necessity_sweep.py # プローブ整列部分空間の除去による必要性検証
│   ├── run_focused_necessity_n100.py  # 厳密 N=100 ペアでの必要性検証
│   │
│   ├── # --- ベースライン・統制検証 ---
│   ├── run_within_model_positive_control.py # モデル内ポジティブコントロール
│   ├── run_qwen_recognition_baseline.py # Qwen 感情認識ベースライン評価
│   ├── run_dual_outcome_behavior.py   # 二重アウトカム行動評価
│   ├── run_ridge_alpha_sweep.py       # Ridge 正則化パラメータ α スイープ
│   ├── expand_strict_dataset.py       # 厳密データセットの拡張
│   ├── analyze_alignment_fidelity_and_manifold.py # 整列忠実度・多様体幾何解析
│   │
│   └── # --- 論文用図版・付録図版・統計要約 ---
│       ├── plot_main_figure1.py       # 論文 Main Figure 1 生成
│       ├── plot_paper_figures.py      # 本文掲載図版の一括生成
│       ├── plot_appendix_figures.py   # 付録図版の一括生成
│       ├── compute_bootstrap_ci.py    # ブートストラップ 95% 信頼区間計算
│       ├── print_all_cis.py           # 全信頼区間のコンソール出力
│       └── summarize_results.py       # 全実験結果の要約テーブル出力
│
├── data/                              # v3 前処理済み刺激データ
└── results/
    ├── raw/                           # 介入生ログ・尤度分布キャッシュ
    └── derived/                       # 仲介効果、OT回復率、気分一致効果等の集計表
```

---

## 3. 評価プロトコルと主要評価指標

### 3.1 2次元最適輸送回復率 (Joint 2D Optimal Transport Recovery)
離散化された81候補の確率分布 $P$ に対し、Wasserstein-1 距離（Earth Mover's Distance）に基づく回復率を定義します：
$$\text{Recovery}_{\text{OT}}(P_{\text{patch}}) = \frac{\text{OT}(P_{\text{clean}}, P_{\text{target}}) - \text{OT}(P_{\text{patch}}, P_{\text{target}})}{\text{OT}(P_{\text{clean}}, P_{\text{target}})}$$
- `clean`: 介入前の基準状態（例: Neutral）
- `target`: 到達目標の情動状態（例: Peak / Clinical）
- `patch`: 活性化パッチング適用後の分布

### 3.2 意味的生成アンカー (Semantic Generation Stages)
JSONフォーマット `{"valence": V, "arousal": A}` の自己報告生成時において、以下の6段階のトークン位置を動的に同定し、段階特異的な因果効果を測定します：
1. `response_start`: JSON生成開始直後
2. `pre_V`: Valence キー出力直後（値の直前トークン）
3. `V_value`: Valence 数値トークン位置
4. `pre_A`: Arousal キー出力直後（値の直前トークン）
5. `A_value`: Arousal 数値トークン位置
6. `response_end`: JSON終了トークン

### 3.3 気分一致効果指標 (Mood Congruency Index)
内部感情状態を正/負にステアリングした状態で客観的感情認識（他者予測）を行わせ、注入強度 $\alpha$ に対する推定値の回帰勾配 $\beta_{\text{mood}}$ および Pearson 相関係数 $r$ を算出します。

---

## 4. 実行手順 (How to Run)

### 4.1 環境準備
共通の仮想環境を有効化し、プロジェクトルートから実行します。

```bash
cd /mnt/nas/home/hiromi/src/emo
source .venv/bin/activate
```

### 4.2 統合パイプライン (V2 & V3 Full Pipeline) の実行
`run_v2_v3_full_pipeline.sh` を用いることで、V2の解析からV3の全実験までを順次自動実行できます。

```bash
# 全ステップ (Step 1〜7) の実行
bash v3/scripts/run_v2_v3_full_pipeline.sh --device cuda

# V3 のステップのみを実行する場合 (例: Step 5〜7)
bash v3/scripts/run_v2_v3_full_pipeline.sh --device cuda --from-step 5

# 動作確認 (ドライランモード)
bash v3/scripts/run_v2_v3_full_pipeline.sh --dry-run
```

### 4.3 主要な個別実験スクリプトの実行

#### ① V3-RQ1: 内部情動状態の誘導と Go/No-Go ゲート
```bash
python v3/scripts/run_v3_state_induction.py --device cuda --layer 14
```

#### ② V3-RQ2 & RQ3: 時空間全探索とパス仲介分析
```bash
# 全層 × 生成ステージの時空間マップ探索
python v3/scripts/run_v3_spatiotemporal_maps.py --device cuda

# パス仲介分析 (Discovery / Confirmation)
python v3/scripts/run_v3_path_mediation.py --device cuda
```

#### ③ 気分一致因果実験 (Mood Congruency)
```bash
# 気分一致実験の実行 (内部ステアリング下での認識評価)
python v3/scripts/run_mood_congruency_experiment.py

# 結果の統計分析とプロット
python v3/scripts/analyze_mood_congruency.py
```

#### ④ 因果局在化スイープ (Decodability vs Causal Influence)
```bash
# MLP / Attention / Residual stream の全層因果スイープ
python v3/scripts/run_causal_localization_sweep.py --device cuda

# 生成時マルチレイヤー因果スイープ
python v3/scripts/run_generation_time_causal_sweep.py --device cuda
```

#### ⑤ 多層多様体整列パッチング
```bash
python v3/scripts/run_multilayer_aligned_patching.py --device cuda
python v3/scripts/run_aligned_cross_model_patching.py --device cuda
```

### 4.4 論文用図版・結果要約の生成
```bash
# 論文 Main Figure 1 の生成
python v3/scripts/plot_main_figure1.py

# 論文本文掲載図版の一括生成
python v3/scripts/plot_paper_figures.py

# 付録用図版の生成
python v3/scripts/plot_appendix_figures.py

# 全実験の統計的信頼区間出力と結果要約
python v3/scripts/print_all_cis.py
python v3/scripts/summarize_results.py
```
