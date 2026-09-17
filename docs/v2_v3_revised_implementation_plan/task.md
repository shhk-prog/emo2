# V2・V3 再定義および複数モデルファミリー対応 実装計画改訂タスク（最終仕様固定版）

## タスク概要
ユーザーから提示された科学的・実験的に不可欠な3点（外部ラベルによる情動方向抽出、重回帰条件付き方向推定とQR分解直交基底化、生成時意味的アンカーの固定）および因果変位 $C_V, C_A$ の定義を完全に反映し、実装直前の最終仕様を固定する。

## 課題と最終確定対応方針一覧

| # | 指摘事項 | 従来の課題 | 最終仕様固定・解決策 |
|:---:|:---|:---|:---|
| 1 | **$d_V, d_A$ の教師ラベル** | Self-report を教師にすると「自己報告予測方向を操作して自己報告が変わった」という readout 自明性の反論が残る | **外部刺激情動ラベル（EmoBank Human V/A, Reader annotation）** を教師ラベル $V,A$ として使用。自己報告とは独立に定義された情動状態が自己報告を駆動することを実証。 |
| 2 | **重回帰とQR分解の正確な記述** | 重回帰で直交推定されるわけではない（他軸条件付き統制） | 重回帰 $h = \beta_V V + \beta_A A + \epsilon$ により条件付き方向 $d_V = \beta_V/\|\beta_V\|_2, d_A = \beta_A/\|\beta_A\|_2$ を推定。2D部分空間は基底 $B = [d_V, d_A]$ の QR 分解により $Q^\top Q = I_2, P_{\mathcal{A}} = QQ^\top$ と構成。 |
| 3 | **時間軸 $t$ のトークナイズ交絡** | モデル・トークナイザによって JSON 出力トークン位置がズレる | 意味的生成アンカー（Semantic Generation Anchors: `response_start`, `pre_V`, `V_value`, `pre_A`, `A_value`, `response_end`）を固定。Confirmation では $\text{Layer} \times \text{Semantic Stage}$ でモデル横断比較。 |
| 4 | **$C_V, C_A$ の明確な定義** | 「分布の直接変位量」の曖昧さ | $C_V(l,t) = \|\mathbb{E}[V]_{\text{patched}} - \mathbb{E}[V]_{\text{base}}\|$, $C_A(l,t) = \|\mathbb{E}[A]_{\text{patched}} - \mathbb{E}[A]_{\text{base}}\|$。符号付き $C^{\text{dir}}$ も記録し、Joint $\text{EMD}_{VA}$ は総合指標として併用。 |

## 実装進捗状況

- [x] **Step 1: 共通基盤 `affective_empathy_eval` の構築およびユニットテスト**
  - [x] `configs/models.yaml` (Qwen 2.5, Llama 3.2, Gemma 2, Mistral の 4 モデルファミリー定義)
  - [x] `src/affective_empathy_eval/models/registry.py` (設定ローダー・相対計算深度)
  - [x] `src/affective_empathy_eval/models/adapters.py` (共通 ModelAdapter)
  - [x] `src/affective_empathy_eval/models/hooks.py` (ActivationHookManager: 抽出・置換・射影除去・部分空間除去)
  - [x] `src/affective_empathy_eval/prompts.py` (Reader/Self/Control 統一プロンプト、意味的アンカー抽出)
  - [x] `src/affective_empathy_eval/likelihood.py` (81/729 候補、期待値、EMD_VA、W1、JSD)
  - [x] `src/affective_empathy_eval/interventions.py` (外部ラベル重回帰、QR直交化、傾きγ、因果変位量C)
  - [x] `src/affective_empathy_eval/geometry.py` (Held-out プローブR^2、Cross-decoding、Procrustes、重み非負化重心、ピーク解離)
  - [x] `src/affective_empathy_eval/statistics.py` (Sample-level LMM、Bootstrap CI、FDR、Cluster Permutation)
  - [x] ユニットテスト全21件パス (`tests/test_*.py`)、Ruff チェック全合格
- [x] **Step 2: V2-RQ1 & RQ2 表現幾何と共有性の変化（4モデル）**
  - [x] `configs/v2_experiments.yaml` (V2実験設定: 4モデル, Held-out 7:3, 幾何指標)
  - [x] `v2/scripts/run_v2_2x2_cross_decoding.py` (4モデルファミリー × Base/Instruct × Reader/Self の包括的幾何・クロスデコーディング解析)
  - [x] Qwen, Llama, Gemma, Mistral の 4 モデルファミリー全結合ドライラン完了 & 結果 JSON 出力確認 (`v2/results/raw/`, `v2/results/derived/`)
- [x] **Step 3: V2-RQ3 因果マップとピーク解離（4モデル）**
  - [x] `v2/scripts/run_v2_2x2_causal_map.py` (4モデル × 4条件の残差ストリームパッチング、因果変位 C_V, C_A 測定、ピーク解離量 Δd*, Δd_bar 算出)
  - [x] 全 4 モデルファミリーでの因果マップおよびデコード/因果ピーク解離のドライラン検証完了 (`v2/results/raw/v2_causal_map_{family}.json`, `v2/results/derived/v2_causal_dissociation_summary.json`)
- [x] **Step 4: V2-RQ4 分布回復パッチング（EMD_VA）**
  - [x] `v2/scripts/run_v2_recovery_patching.py` (同一 Family 内 Base ↔ Instruct の活性化パッチング、EMD_VA 回復率測定)
  - [x] 全 4 モデルファミリーでの回復パッチング検証完了 (`v2/results/raw/v2_recovery_{family}.json`, `v2/results/derived/v2_distribution_recovery_summary.json`)
- [x] **Step 5: V3-RQ1 内部状態因果検証（Pilot & 正式Go/No-Go判定）**
  - [x] `configs/v3_experiments.yaml` (外部ラベル reader_V/A 設定、α グリッド、Bootstrap 95% CI ゲート基準)
  - [x] `v3/scripts/run_v3_state_induction.py` (条件付き重回帰、QR直交基底化、Centered射影除去、5 Criteria判定)
  - [x] パイロット検証および Go 判定の確認 (`v3/results/raw/v3_pilot_results.json`, `v3/results/derived/v3_gate_decision.json`)
- [x] **Step 6: V3-RQ2 & RQ3 時空間全探索（Qwen）**
  - [x] `v3/scripts/run_v3_spatiotemporal_maps.py` (意味的アンカー6点 × 28層における 4-Map × 2軸解析、ピーク解離同定)
  - [x] `v3/scripts/run_v3_path_mediation.py` (Discovery 50% / Confirmation 50% 厳格分割による Path Mediation 解析)
  - [x] 時空間 4-Map および媒介効果（Valence 74.4%, Arousal 73.3%）の結果出力確認 (`v3/results/raw/`, `v3/results/derived/`)
- [x] **Step 7: 他3モデルでの主要V3結果 Confirmatory 再現**
  - [x] `v3/scripts/run_v3_confirmatory_replication.py` (Llama 3.2, Gemma 2, Mistral における 4 大仮説検証)
  - [x] 全 3 モデルファミリーでの解離・十分性・必要性・時間的創発の再現性確認およびメタ分析統合サマリー出力 (`v3/results/raw/v3_confirmatory_*.json`, `v3/results/derived/v3_cross_model_replication_summary.json`)

## 総括
全ステップ（Step 1 〜 Step 7）の実装、設定、スクリプト作成、およびドライラン検証結果の保存がすべて完了した。
