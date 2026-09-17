# 実装計画: v2およびv3のREADMEを現在のコードベースに沿って全面改定・新規作成

## 目的
`v2` および `v3` のディレクトリに存在する実際のコード、設定ファイル（`configs/v2_experiments.yaml`, `configs/v3_experiments.yaml`）、共通ライブラリ（`src/affective_empathy_eval/`, `v2/src/`, `v3/src/`）、および最新の実験スクリプト群（`run_v2_v3_full_pipeline.sh` や ICLR向け因果介入実験等）の仕様に厳密に準拠した形で、`v2/README.md` を改定し、`v3/README.md` を新規作成する。

---

## 調査結果の要約

### 1. `v2` の現状と改定点
- **研究焦点**: 事後学習（Post-training / Instruction tuning）が情動自己報告に及ぼす影響の内部メカニズム解明（Base vs. Instruct）。
- **中核スクリプト体系 (V2-RQ1〜RQ4)**:
  - `run_v2_2x2_cross_decoding.py`: 4モデルファミリー（Qwen 2.5, Llama 3.2, Gemma 2, Mistral）× 2水準（Base, Instruct）の表現幾何変化（Cross-decoding, RSA, Procrustes整列）。
  - `run_v2_2x2_causal_map.py`: 4モデルファミリー × 4条件（Base/Inst × Reader/Self）の因果回路再配置・ピーク解離解析。
  - `run_v2_recovery_patching.py`: Base activation パッチングによる Instruct 分布回復（$EMD_{VA}$）実験。
- **補完スクリプト群**:
  - `prepare_aipsy_strict.py`: 厳密マッチングトリプレット抽出。
  - コンポーネントパッチング (`run_strict_patching_screening.py`, `run_synergy_patching.py`)、出力層スワップ (`run_unembedding_norm_swap.py`)、因果スクラビング (`run_strict_causal_scrubbing.py`, `run_strict_path_patching.py`)、ステアリング・用量反応 (`run_steering_and_likelihood.py`, `run_lambda_dose_response.py`)。
- **改定方針**: 単一モデル中心だった記述を「4モデルファミリー × 2水準（Base/Instruct）」の最新2x2体系に刷新し、RQ1〜RQ4の論理構造と最新実行コマンドを整理。

### 2. `v3` の現状と新規作成内容
- **現状**: `v3/README.md` が未作成（存在しない）。
- **研究焦点**: 内部情動状態から自己報告への因果経路解明（State-to-Report）、感情認識と自己報告の解離・相互作用（Mood Congruency）、全層因果局在化スイープ（Causal Localization Sweep）、多層多様体整列パッチング（Multilayer Aligned Patching）。
- **中核スクリプト体系 (V3-RQ1〜RQ3 & Confirmatory Replication)**:
  - `run_v3_state_induction.py`: 内部情動状態誘導と Go/No-Go ゲート判定（5基準: Sufficiency, Necessity, Specificity, Dose-response, Selectivity）。
  - `run_v3_spatiotemporal_maps.py`: 時空間全探索（28層 × 6意味的生成ステージ）。
  - `run_v3_path_mediation.py`: パス仲介分析（Discovery / Confirmation 分割）。
  - `run_v3_confirmatory_replication.py`: Llama, Gemma, Mistral への確証的再現。
  - `run_v2_v3_full_pipeline.sh`: V2 & V3 の統合パイプライン実行スクリプト（Step 1〜7）。
- **高度な発展的因果実験群**:
  - 気分一致因果実験: `run_mood_congruency_experiment.py`, `analyze_mood_congruency.py`
  - 全層因果局在スイープ（MLP, Attention, Residual）: `run_causal_localization_sweep.py`, `run_generation_time_causal_sweep.py`
  - 最適輸送（OT）多様体整列パッチング: `run_multilayer_aligned_patching.py`, `run_aligned_cross_model_patching.py`
  - 部分空間除去による必要性検証: `run_probe_aligned_necessity_sweep.py`, `run_focused_necessity_n100.py`
- **モジュール構造 (`v3/src`)**:
  - `batch_likelihood.py`: 高速バッチ対数尤度計算
  - `ot_utils.py`: 2次元最適輸送（2D OT / EMD）および回復率計算
  - `model_utils.py`: レイヤー/コンポーネント抽出・フック管理
  - `diagnostics.py`: 数値安定性診断

---

## 提案する変更内容

### 1. [MODIFY] `v2/README.md`
- **第1章**: 研究の背景とリサーチクエスチョン（Base vs. Instruct、事後学習による中立化のメカニズム）。
- **第2章**: V2の4大リサーチクエスチョン（V2-RQ1 & RQ2: 表現幾何と整列可能性、V2-RQ3: 因果回路の再配置とピーク解離、V2-RQ4: 分布回復パッチング）。
- **第3章**: スクリプト構成と役割（最新の `run_v2_2x2_*.py` 系を中核に据え、4モデルファミリー対応、および既存のコンポーネント解析・スワップ実験を体系化）。
- **第4章**: データセットおよび評価指標（`test_strict.csv`, 81状態Sequence-Likelihood, $WD_V$, 2D EMD, JSD）。
- **第5章**: 実行手順（個別スクリプトおよび統合パイプラインでの実行コマンド）。

### 2. [NEW] `v3/README.md`
- **第1章**: 研究の背景と目的（内部情動状態から自己報告への因果経路解明、感情認識との相互作用、デコード能と因果影響の解離）。
- **第2章**: 実験フレームワークとパイプライン（V3-RQ1: State Induction & Go/No-Go Gate、V3-RQ2 & RQ3: 時空間全探索とPath Mediation、Confirmatory Replication）。
- **第3章**: 発展的・確証的実験群（Mood Congruency実験、全層因果局在化スイープ、多層OT多様体整列パッチング、部分空間除去スイープ）。
- **第4章**: ディレクトリおよびモジュール構成（`v3/src` の各モジュール詳細、`v3/scripts` の機能一覧）。
- **第5章**: 実行手順（`run_v2_v3_full_pipeline.sh` による一括実行、および各主要スクリプトの実行コマンド例）。

---

## 検証手順
1. `v2/README.md` および `v3/README.md` に記載されたスクリプトパス、引数、設定ファイル名が実ファイルと100%一致しているかを自動チェック。
2. Markdownの構文（見出し、コードブロック、数式、リンク）の整合性を確認。
3. `walkthrough.md` に改定内容を記録。
