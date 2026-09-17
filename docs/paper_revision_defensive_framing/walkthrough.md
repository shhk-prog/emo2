# ウォークスルー: 査読耐性向上のための論文草稿改訂

## 実施した変更の概要

ユーザーから提示された洗練された構成案に基づき、`/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` を更新しました。

査読者から突っ込まれやすいポイント（過度な一般化、因果的証拠と相関的証拠の混同、不整合なサンプル数表記）を先回りして保守的に抑え込むことで、論文の中心命題である **「Decodability Without Causal Substitutability」** の強度が大幅に高まりました。

---

## 修正の主要ポイント

### 1. 中心命題（Core Proposition）の固定とタイトル調整
- **タイトル**: `Decodability Without Causal Substitutability: Post-Training Decouples Affect-Relevant Representations from Constrained First-Person Reports in Language Models`
- **中心命題**:
  > *A representation may remain linearly decodable, become predictively recoverable across models after alignment, and fall within the target activation distribution, yet still fail to be causally substitutable for a downstream policy-constrained reporting distribution.*
- N2（Predictive–causal dissociation）を中心的な貢献として位置づけ、主張をブレさせない構成に固定しました。

### 2. N3の主張の抑制（No evidence for a localized bottleneck）
- **旧記述**: 「distributed remapping を実証」「局所的な単一・少数のボトルネック仮説を反証」
- **新記述**: 
  - `No evidence for a localized bottleneck within the tested intervention family.`
  - 検証した中間MLP介入範囲において、単一層または少数の連続層（最大8連続層）に局在したボトルネックのみで自己報告変化が説明される仮説を支持する証拠は得られなかったという表現に修正。
  - Attention、residual stream、late readoutなど未検証経路が残ることを明記し、「distributed remapping is consistent with the evidence」にとどめ、「proven」とは主張しない境界線を引きました。

### 3. N4の解釈の厳密化（Layer-dependent causal effects）
- **旧記述**: 「feature sharing を確立」「mood congruency circuit の実証」
- **新記述**:
  - `Layer-dependent causal effects on third-person affect recognition.`
  - Layer 16においてrandom-direction controlにも強い効果（$\beta = +0.0444$）が認められた事実を正面から記載。
  - 単純な「回路の特定」や「純粋なmood congruencyの証明」ではなく、「後半層（L20）においてaffect-related方向が三人称認識タスクの出力へ因果的影響を及ぼし得る証拠」として保守的に解釈しました。

### 4. サンプル数表記の整合
- **旧記述**: 「192完全ペア・422 samples」（ペア数とサンプル数の整合が疑われる表記）
- **新記述**: 
  - `192 pair_id groups / 422 samples`
  - 「pair_id」は解析上のグループ単位を意味し、すべてのグループが同数の条件を持つことを意味しない旨を明記。
  - 分割の内訳（Train: 76 groups / 169 samples, Alignment-dev: 58 groups / 124 samples, Held-out test: 58 groups / 129 samples）を明確化。

---

## 保存されたファイル
- 対象ファイル:
  - [`v3/docs/paper.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper.md)
- ドキュメント保存先 (`docs/paper_revision_defensive_framing/`):
  - `task.md`
  - `implementation_plan.md`
  - `walkthrough.md`
