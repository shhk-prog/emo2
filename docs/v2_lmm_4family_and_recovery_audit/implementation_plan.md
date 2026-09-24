# 実装計画: Behavioral RQ4 および V3 Confirmatory Matrix Note の科学的表現修正

## 1. 目的
ICLR 2027論文に向けた結果フォルダ最終凍結監査に基づき、Behavioral Stage（RQ4 Coupling）および V3 Stage（Confirmatory Matrix）の表Noteにおける解釈文を、科学的測定境界および事前 Gate 設計（NO_GO）に厳密に準拠した表現へと修正する。

## 2. 修正対象ファイル
### ① Generator スクリプト
- `scripts/summarize_behavioral_aipsy.py`
- `scripts/summarize_v3_causal_utilization.py`

### ② 生成済み LaTeX テーブル
- `iclr2027/tables/behavioral_aipsy_rq4_coupling.tex`
- `iclr2027/tables/table_behavioral_aipsy_rq4_coupling.tex`
- `iclr2027/tables/behavioral_aipsy_coupling.tex`
- `iclr2027/tables/behavioral_aipsy_summary.tex`
- `iclr2027/tables/v3_confirmatory_matrix.tex`
- `iclr2027/tables/v3_causal_utilization_summary.tex`

## 3. 修正内容
### (1) Behavioral RQ4 Coupling Note
- **旧表現**: 「全条件で $q < 10^{-15}$ の強固な結合を示し、事後学習を経てもモデル内部における感情認識と自己報告の連動ダイナミクスが破綻せず一貫して保たれていることが実証される。」
- **新表現**: 「全8 modelsのValence / ArousalでReader changeとSelf changeの間に強い正のcouplingが観測された。これは、controlled stimulus manipulationに対するReader PredictionとSelf-Reportの変化量がoutput levelで一貫して共変動することを示す。ただし、このbehavioral couplingのみから、両taskが同一の内部representationやcausal pathwayを共有すること、あるいはBase--Instruct差をpost-trainingの因果効果として解釈することはできない。」
- **理由**: $\mathrm{Corr}(\Delta R, \Delta S)$ は output レベルの行動共変動指標であり、内部機構の同一性や事後学習への直接帰属は示せないため。

### (2) V3 Confirmatory Matrix Note
- **旧表現**: 「全モデルファミリーにおいてH1--H3は棄却され、H4のみが支持されたため、全体結論として情動表現の機能的・因果的活用仮説は支持されなかった（All Confirmed = NO）。」
- **新表現**: 「全4仮説の判定マトリックス。各セルは該当仮説についてValenceおよびArousalの双方が事前定義された95\% CI criterionを満たした場合に$\checkmark$ とする。H1--H3はいずれのconfirmatory familyでも支持されなかった。H4のtemporal contrastは3 familyすべてでValence / Arousal双方がaxis-level criterionを満たした。ただし、事前のState Induction GateがNO\_GOであったため、V3全体としてprespecified confirmatory causal mechanismが確立されたとは解釈しない（All Confirmed = NO）。」
- **理由**: 事前基準を満たさなかった結果は「棄却」ではなく「支持されなかった」と表現するのが科学的に正確であり、また事前 Gate が NO_GO であったことを明記して H4 単独での過剰解釈を防ぐため。
