# 修正内容の確認 (Walkthrough): Behavioral RQ4 および V3 Confirmatory Matrix Note の科学的表現修正

## 1. 実施内容の概要
ユーザーの指示に基づき、科学的解釈の境界を厳密に保ち、過大解釈を抑止するため、以下の2箇所の Note を修正しました。
再生成スクリプト（generator）と現行のすべての TeX テーブルの両方を同時に改修したため、将来再生成を実行しても文面が元に戻ることはありません。

---

## 2. 変更箇所の詳細

### (1) Behavioral RQ4 Coupling Note
- **対象ファイル**:
  - `scripts/summarize_behavioral_aipsy.py`
  - `iclr2027/tables/behavioral_aipsy_rq4_coupling.tex`
  - `iclr2027/tables/table_behavioral_aipsy_rq4_coupling.tex`
  - `iclr2027/tables/behavioral_aipsy_coupling.tex`
  - `iclr2027/tables/behavioral_aipsy_summary.tex`
- **反映された文面**:
  ```latex
  \textbf{Note:} 全8 modelsのValence / ArousalでReader changeとSelf changeの間に強い正のcouplingが観測された。これは、controlled stimulus manipulationに対するReader PredictionとSelf-Reportの変化量がoutput levelで一貫して共変動することを示す。ただし、このbehavioral couplingのみから、両taskが同一の内部representationやcausal pathwayを共有すること、あるいはBase--Instruct差をpost-trainingの因果効果として解釈することはできない。
  ```

### (2) V3 Confirmatory Matrix Note
- **対象ファイル**:
  - `scripts/summarize_v3_causal_utilization.py`
  - `iclr2027/tables/v3_confirmatory_matrix.tex`
  - `iclr2027/tables/v3_causal_utilization_summary.tex`
- **反映された文面**:
  ```latex
  \textbf{Note:} 全4仮説の判定マトリックス。各セルは該当仮説についてValenceおよびArousalの双方が事前定義された95\% CI criterionを満たした場合に$\checkmark$ とする。H1--H3はいずれのconfirmatory familyでも支持されなかった。H4のtemporal contrastは3 familyすべてでValence / Arousal双方がaxis-level criterionを満たした。ただし、事前のState Induction GateがNO\_GOであったため、V3全体としてprespecified confirmatory causal mechanismが確立されたとは解釈しない（All Confirmed = NO）。
  ```

---

## 3. 検証結果
- 全リポジトリ内で古い表現（「連動ダイナミクスが破綻せず一貫して保たれている」「全モデルファミリーにおいてH1--H3は棄却され」）を grep 検索した結果、**該当件数 0 件** を確認しました。
- これにより、**Behavioral、V1、V3 の各 Stage は完全に Freeze（凍結）可能な状態**となりました。
