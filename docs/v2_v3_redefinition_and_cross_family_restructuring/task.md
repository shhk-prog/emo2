# タスク管理: V2・V3の再定義および複数モデルファミリー対応（改訂版）

## 概要
- **目的**:
  - V2を「中立化の解明」から「Base vs Instruct × Reader vs Self (2×2) による事後学習の内部計算再編（$\Delta\Delta_{\text{post}}$）」へ再定義する。
  - V3を「内部表現から自己報告への計算プロセス（5つの因果判定基準、Train/Test完全分離、射影除去Necessity、統制方向、オープンなTask Selectivity、時空間結合マップ）」へ再定義する。
  - Qwen 2.5 (1.5B), Llama 3.2 (1B), Gemma 2 (2B), Mistral (7B) の4モデルファミリーに対し、正規化深度 $d \in [0, 1]$ と統一 `ModelAdapter` を通じた比較基盤を構築する。
  - 計算リソース配分: V2は4モデル網羅、V3はQwen先行＋他3モデルは主要Confirmatory再現。
- **保存ディレクトリ**: `docs/v2_v3_redefinition_and_cross_family_restructuring/`

---

## タスクリスト

- [ ] **Step 1: 共通基盤モジュールの構築 (`src/affective_empathy_eval/` & `configs/`)**
  - [ ] `configs/models.yaml`: 4ファミリー（Qwen, Llama, Gemma, Mistral）定義
  - [ ] `configs/v2_experiments.yaml` & `configs/v3_experiments.yaml`: 実験設定の分離
  - [ ] `models/adapters.py` & `models/hooks.py`: 正規化深度 $d \in [0, 1]$、概念名フック（`pre_attn_resid`, `post_mlp_resid`, `mlp_out`等）、共通Adapter
  - [ ] `prompts.py`: Self, Reader, 非感情Control（Topic/Domain分類等）の標準プロンプト
  - [ ] `interventions.py`: Train/Test分離下の状態注入（Sufficiency）、直交射影除去（Necessity）、Norm一致ランダム・直交統制方向生成
  - [ ] `geometry.py`: 正規化深度対応 Procrustes, Ridge, RSA, Cross-decoding
  - [ ] `statistics.py`: 差の差（$\Delta\Delta_{\text{post}}$）、混合効果モデル（$Y \sim \text{Model} \times \text{Task} \times \text{Site}$）検定

- [ ] **Step 2: V2-RQ1 & RQ2 (幾何変化・共有性, 4モデル対応)**
  - [ ] `v2/README.md` の改訂（2×2要因配置、中立化ストーリーの完全撤去）
  - [ ] `v2/scripts/run_v2_2x2_cross_decoding.py` の実装
    - Base R ↔ Inst R, Base S ↔ Inst S
    - $\Delta_{\text{post}}^R(d)$ vs $\Delta_{\text{post}}^S(d)$
    - $\text{CrossDecode}(R \rightarrow S)$ Base vs Instruct

- [ ] **Step 3: V2-RQ3 (因果回路マップ, 4モデル対応)**
  - [ ] `v2/scripts/run_v2_2x2_causal_map.py` の実装
    - 4条件（B-R, B-S, I-R, I-S）での正規化深度因果マップ
    - 初期層 vs 後期層の特異化交互作用検定

- [ ] **Step 4: V2-RQ4 (分布差の介入的回復・縮小)**
  - [ ] `v3/scripts/run_aligned_cross_model_patching.py` を `v2/scripts/` へ移管・統合
  - [ ] `v2/scripts/run_v2_recovery_patching.py` の実装（Component, Path, Unembedding/Norm swap）

- [ ] **Step 5: V3-RQ1 (内部状態参照の因果検証 — Go/No-Go判定)**
  - [ ] `v3/README.md` の改訂（State-to-Reportフレームワーク、5つの判定基準）
  - [ ] `v3/scripts/run_v3_state_induction.py` の実装
    - Train/Test分離下の内部状態注入（Sufficiency: $h + \alpha d$）
    - 射影除去（Necessity: $h - (h^\top \hat{d})\hat{d}$）
    - 対照群（Norm一致ランダム方向 $d_{\text{random}}$、直交方向 $d_{\perp}$）
    - タスク特異性検証（Self vs Reader vs Control: Shared / Self-specific / Amplified のオープン識別）
    - Dose response ($\alpha$ スイープ) & held-out 一般化

- [ ] **Step 6: V3-RQ2 & RQ3 (時空間マッピング & パス検証, Qwen先行)**
  - [ ] `v3/scripts/run_v3_path_mediation.py`: Discovery split（$l_D^*, l_C^*$）と Confirmatory split での Path Mediation
  - [ ] `v3/scripts/run_v3_spatiotemporal_maps.py`: 刺激交絡除去下の $D(l,t), \beta(l,t), \gamma(l,t), C(l,t)$ マップ

- [ ] **Step 7: 動作検証・テスト・論文ドラフト整合 (Verification & Documentation)**
  - [ ] 全スクリプトの `--dry-run` / `--max-samples 2` によるCPU動作確認
  - [ ] 論文ドラフト（`v3/docs/paper_v1.md` 等）の論理構成改訂案の提示
  - [ ] `walkthrough.md` の作成と完了報告
