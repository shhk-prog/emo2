# 実装計画書: v1草稿におけるLinear ProbeとAblationの関係および全層検証の実態

## 概要
ユーザーからの「v1について」という再確認に基づき、v1の草稿（`v1/docs/paper_draft.md`）、設定ファイル（`v1/configs/main_experiment.yaml`）、および実行スクリプト（`v1/scripts/run_causal_intervention.py`, `run_probing_aipsy.py`）を厳密に照合し、v1段階における実態を解明・整理します。

## 調査の焦点
1. **v1草稿で「0, 4, 8, 12, 16, 20, 24, 27層」と書かれていた理由**:
   - `main_experiment.yaml` の初期設定 `layers_to_extract: [0, 4, 8, 12, 16, 20, 24, 27]` が草稿の第3章・手法の記述にそのまま残っていた。
2. **v1においてLinear Probeは全層やっていたのか？**:
   - 草稿「5.3 RQ2: Internal Encoding (全層線形プロービング)」およびスクリプトでは、実際には全28層でプロービングを実行し、「Layer 14–25でROC-AUC > 97.5%」という結論を導いている。
3. **v1においてAblationも全層やった方が良いのではないか？**:
   - ユーザーの指摘通り、Ablationも全層で行うべきであり、**実際の実装（`run_causal_intervention.py`）では全28層すべてに対してMean Ablationを実行していた**（草稿行283-285にその結果が記載されている）。
4. **v1におけるLinear ProbeとAblationの関係**:
   - **プローブ（内的符号化・情報の保持）**: Layer 14〜25で最大（情報は消えていない）。
   - **Ablation（因果的必要性・機能的役割）**: Layer 4で最大42.26%、Layer 12–14で中程度の低減（単一層では完全消去できず、分散している）。
   - 「プローブで最もよく読める層」と「Ablationで最も影響が出る層」が一致しないため、全層プロファイルによる対比が不可欠であること。
