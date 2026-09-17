# タスクリスト: Post-Trainingによる情動的自己報告のReadout乖離の解明

- [ ] フェーズ1: Preliminary Stage (Contrastive Projection と IBC)
  - [x] pair_id単位のGroup Splitの実装と検証
  - [x] Base / Instruct 双方での 81候補尤度分布の取得（$E[V], E[A]$, entropy 等の計算）
  - [x] 隠れ状態（hidden states）の保存と抽出バッチ実行 (`scripts/run_extract_all.sh`)
  - [x] カテゴリ・強度条件を用いた contrastive direction ($d_V, d_A$) の構築（train split）
  - [x] Projection score $z_{V,l}, z_{A,l}$ の層別抽出
  - [x] Internal–Behavior Coupling (IBC) の算出 ($z_{V,l}$ と $E[V_{self}]$ 等) と $\Delta\mathrm{IBC}$ の比較
  - [x] Neutral / Moderate / Peak に対する内部・自己報告のresponse curve（dose-response）取得
  - [x] メタデータ（revision, prompt hash, tokenizer等）の記録機構の実装
- [x] フェーズ1.5: Confirmatory Stage (人手アノテーション取得後)
  - [x] 人手VA評定 ($V_H, A_H$) の取得
  - [x] Continuous VA Probe (Ridge) の学習とTest set評価
  - [x] HIC / HBC の正式算出と Dissociation Gap の検証
- [x] フェーズ3A: 前提検証（フェーズ1と並行実施）
  - [x] Cross-model decoding (Direct / Orthogonal Procrustes / Regularized linear transfer) の実施
  - [x] Steering Slope検証（2〜4層での品質維持領域 $\mathcal{A}_{safe}$ の特定と $\alpha$ 規格化による傾き算出）
- [x] フェーズ3B: 包括的ストレステスト
  - [x] 温度 $\tau$・token正規化手法の影響検証
  - [x] 外部データへの転移、複数モデルサイズでの一般化検証スクリプト作成

### フェーズ 4: Dissociation Circuit Discovery
- [x] ボトルネックモジュール（Attention Head / MLP）の特定と可視化 (`plot_module_probing.py`)
- [x] **Activation Patching**: 因果的回路検証の実装 (`run_circuit_patching.py`)
- [x] **三空間RSAと統制分析**: 空間構造と表層交絡の排除 (`run_rsa_and_controlled_coupling.py`)
- [x] **Output-Gating検証**: JSONフォーマットバイアスの検証 (`run_output_gating_test.py`)

### 最終検証とドキュメント整備
- [x] 全フェーズを通じた統合テストとデバッグ
- [x] Walkthrough (成果物とプロット) の作成・更新
