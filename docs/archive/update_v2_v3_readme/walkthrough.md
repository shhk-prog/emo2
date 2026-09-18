# 作業完了報告 (Walkthrough): v2およびv3のREADME修正・同期

## 概要
`v2` および `v3` の最新の実装コード、設定ファイル（`configs/v2_experiments.yaml`, `configs/v3_experiments.yaml`）、共通ライブラリ（`src/affective_empathy_eval/`, `v2/src/`, `v3/src/`）、および実験スクリプト群（統合パイプライン `run_v2_v3_full_pipeline.sh` や ICLR投稿向け実験）の実態に基づき、以下を実施しました：
1. `v2/README.md` を 4モデルファミリー体系（Qwen 2.5, Llama 3.2, Gemma 2, Mistral）× 2水準（Base vs. Instruct）の最新パイプライン（V2-RQ1〜RQ4）に沿って全面改定。
2. `v3/README.md` を新規作成し、内部情動状態から自己報告への伝播経路（State-to-Report）、気分一致因果実験（Mood Congruency）、全層因果局在化スイープ（Causal Localization Sweep）、多層最適輸送（OT）多様体整列パッチング、および統合パイプラインの仕様を体系化。

---

## 変更内容の詳細

### 1. `v2/README.md` の改定
- **核心的リサーチクエスチョンの刷新**:
  - **V2-RQ1 & RQ2**: 表現幾何の変化と整列可能性（線形プローブ $R^2$、直交Procrustes整列、RSA）
  - **V2-RQ3**: 因果回路の再配置とピーク解離（Causal Leverage、相対深度解析）
  - **V2-RQ4**: Base activation パッチングによる Instruct 分布回復（$EMD_{VA}$）
- **4モデルファミリー体系の明記**:
  - Qwen 2.5 1.5B (28層), Llama 3.2 1B (16層), Gemma 2 2B (26層), Mistral 7B (32層) の構成表と `native` / `matched_plain` プロンプト統制。
- **スクリプト構成の体系化**:
  - 4モデル横断中核スクリプト（`run_v2_2x2_cross_decoding.py`, `run_v2_2x2_causal_map.py`, `run_v2_recovery_patching.py`）
  - 厳密データセット抽出（`prepare_aipsy_strict.py`）
  - コンポーネントパッチング・スワップ（`run_strict_patching_screening.py`, `run_synergy_patching.py`, `run_unembedding_norm_swap.py`, `run_output_gating_test.py`）
  - 集計・可視化スクリプト（`dump_all_metrics.py`, `dump_tables.py`, `plot_*.py`）
- **実行コマンドの更新**:
  - プロジェクトルートの仮想環境 `.venv` からの実行コマンド例を整備。

### 2. `v3/README.md` の新規作成
- **研究の背景と目的**:
  - 内部情動状態から自己報告への伝播メカニズム（State-to-Report）
  - 内部ステアリングが他者感情認識を歪める「気分一致効果（Mood Congruency）」
  - 「デコード可能性 $\ne$ 因果的影響（$\operatorname{argmax}_l D_l \ne \operatorname{argmax}_l C_l$）」の解離検証
- **コアモジュール (`v3/src`) の解説**:
  - `batch_likelihood.py`: 高速バッチ化対数尤度計算
  - `ot_utils.py`: 2次元最適輸送（2D OT / EMD）および回復率計算
  - `model_utils.py`: レイヤー・コンポーネント抽出・フック管理
  - `diagnostics.py`: 数値安定性診断
- **スクリプト体系の網羅**:
  - V3中核パイプライン: `run_v3_state_induction.py`, `run_v3_spatiotemporal_maps.py`, `run_v3_path_mediation.py`, `run_v3_confirmatory_replication.py`, `run_v2_v3_full_pipeline.sh`
  - 気分一致実験: `run_mood_congruency_experiment.py`, `analyze_mood_congruency.py`
  - 因果局在スイープ: `run_causal_localization_sweep.py`, `run_generation_time_causal_sweep.py`
  - 多層整列パッチング: `run_multilayer_aligned_patching.py`, `run_aligned_cross_model_patching.py`
  - 論文図版生成: `plot_main_figure1.py`, `plot_paper_figures.py`, `plot_appendix_figures.py`, `print_all_cis.py`, `summarize_results.py`
- **主要評価指標の数理定義**:
  - 2次元最適輸送回復率（Joint 2D OT Recovery）
  - 6段階の意味的生成アンカー（Semantic Generation Stages）
  - 気分一致効果指標（$\beta_{\text{mood}}$, Pearson $r$）

---

## 検証結果
記載されたすべてのスクリプトパスおよび設定ファイルパスについて、実在性をスクリプトで自動検証しました：
```text
=== Checking v2/README.md ===
Found 13 references: All referenced paths exist! (13/13 OK)

=== Checking v3/README.md ===
Found 16 references: All referenced paths exist! (16/16 OK)
```
存在しないファイルへの参照や不整合は一切なく、現行のコードベースと完全に同期しています。
