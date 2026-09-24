# タスクリスト: V2 H3 LMM 4-Family カバレッジ検証および H4 Recovery 調査

## 背景
ICLR 2027論文の結果監査において、Behavioral、V1、V3、V2のPresentation / Confirmatory Note等はすべてfreeze可能と判定された。
残る確認・対応事項は以下の2点（+ 1点クリーンアップ）に絞られている。

## タスク項目

- [x] **1. `v2_causal_pair_level.csv` の family 内訳確認と RQ3 状況の特定**
  - 今朝 03:07 に Llama も含めて全4ファミリー（Qwen, Llama, Gemma, OLMo）の 516,000 件が揃い、Primary LMM (N=344,000) もフィッティング完了していることを確認。
- [x] **2. H4 Recovery のデータ実態調査**
  - Gemma, OLMo は完了済みキャッシュあり。Qwen は現在実行中（Reader完了、Self実行中）。Llama は Qwen 完了後に自動実行されることを確認。
- [x] **3. Behavioral RQ4 Note の科学的表現修正（凍結対応）**
  - output-level behavioral covariation である旨を明記し、内部ダイナミクスや事後学習への過大解釈を抑制する文面に改修。
- [x] **4. V3 Confirmatory Matrix Note の科学的表現修正（凍結対応）**
  - 「棄却」を「支持されなかった」に改め、事前 Gate NO_GO に基づき全体結論を正確に記載する文面に改修。
- [ ] **5. V2 パイプライン完了待ちおよびテーブル最終再生成**
  - Qwen/Llama の RQ4 完了（06:30頃見込み）後、最新の 4-family 完全データから最終 table を再生成。
