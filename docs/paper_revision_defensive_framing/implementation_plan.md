# 実装計画: 査読耐性向上のための論文草稿改訂

## 概要
ユーザーから提示された方針に基づき、`/mnt/nas/home/hiromi/src/emo/v3/docs/paper.md` の記述を改訂します。
過度な主張（overclaiming）を抑え、査読者からの批判（「distributed remappingを実証したと言えるのか」「Layer 16のrandom control効果が大きいのにvalence-specificと言えるのか」「192完全ペアで422サンプルという数値の不整合」）を先回りして防ぎます。

## 主な変更方針
1. **N2を中心貢献（Core Proposition）に一本化**:
   - `Decodability Without Causal Substitutability: Post-Training Decouples Affect-Relevant Representations from Constrained First-Person Reports in Language Models`
   - 「線形デコード可能・アライメントで予測回復・活性化分布内（OODでない）であっても、下流の制約付き報告分布に対する因果的代替性を保証しない」という一点を堅固にする。
2. **N3の表現の抑制（No evidence for a localized bottleneck）**:
   - 「distributed remappingを実証」という強い表現を退け、検証したMLP介入範囲において「局所的なボトルネック仮説を支持する証拠は得られなかった（No evidence for a localized bottleneck within the tested intervention family）」と保守的に記述。
   - distributed remappingは「可能性として整合的だが、証明されたわけではない（attentionやresidual streamなど未検証経路が残る）」と議論で線引き。
3. **N4の解釈の厳密化（Layer-dependent causal effects on third-person affect recognition）**:
   - 「feature sharingを確立」「mood congruency circuitの実証」といった飛躍を排除。
   - Layer 16でrandom controlにも有意な効果が認められた事実を正面から記載し、純粋なvalence-specific mechanismではなく「後半層においてaffect-related方向が三人称認識出力へ因果的影響を及ぼし得る証拠」として位置づける。
4. **サンプル数表記の整合**:
   - 「192完全ペア・422 samples」から「192 pair_id groups / 422 samples」へ修正。pair_idは解析グループ単位であり全グループが同数条件を持つわけではない旨を明記。

## 変更対象ファイル
- [MODIFY] `v3/docs/paper.md`
- [NEW] `docs/paper_revision_defensive_framing/task.md`
- [NEW] `docs/paper_revision_defensive_framing/implementation_plan.md`
- [NEW] `docs/paper_revision_defensive_framing/walkthrough.md`

## 検証手順
- `v3/docs/paper.md` の更新内容がユーザー提示テキストと完全一致し、整合しているか確認。
- 保存ルール（日本語回答、`docs/` 配下への履歴保存）の遵守確認。
