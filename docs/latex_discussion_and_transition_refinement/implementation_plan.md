# 実装計画書: LaTeX考察・接続の徹底検証と重複修正（第2パス）

## 1. 目的とスコープ
`iclr2027/iclr2027_conference2.tex` を対象とし、以下の作業を行う。
1. **最新結果データ（4ファミリー完全版）との厳密な整合**:
   - V2 H3 LMM: ValenceのAlignment $\times$ Task（$q=0.044$）が支持され、層深度の再配置（Alignment $\times$ Depth等）は支持されなかった最新の確定Noteと一致させる。
   - V2 H4: 全4ファミリー（Qwen, Llama, Gemma, OLMo）の実測値、および正しい介入方向（Instruct表現をBase表現で置換・整列）の反映。
2. **AIコピペ残骸の完全除去**:
   - Line 8471: `:chatgpt-content-reference{index="3"}` の削除。
   - Line 8555: `:chatgpt-content-reference{index="4"}` の削除。
3. **無駄な重複の削減と読みやすさ（可読性・レイアウト）の改善**:
   - 1行の数値や単語（`\[ 0.189 \]`, `\[ layer \] \times \[ stage \]` 等）を無駄に何行も別行立て数式にしている箇所を、自然な文章・インライン数式に整理。
   - 重複しているセクションヘッダーコメント（Line 6306-6311）の整理。
   - 全体考察における各ステージの単なる重複再掲を整理し、総合的な計算像・三重の境界線・ガードレールへの含意として論理を凝縮。

---

## 2. 修正対象箇所と具体的な変更案

### 2.1 V2考察 (Line 6094〜6305)
- **H1b (Peak Shift)**:
  - $\Delta d_D^*$ の推定値（Valence Reader $-0.119$, Valence Self $0.007$, Arousal Reader $0.200$, Arousal Self $0.017$）を1つの見やすい数式またはインラインに集約。
- **H2 (Sharing Reorganization)**:
  - $\Delta\text{Sharing}$ の推定値（Valence $-0.553$, Arousal $-0.278$）と95% CIを整理。
- **H3 (LMM Causal Reorganization)**:
  - 「Post-training $\times$ Task interactionが、$\beta = 1.63\times10^{-4}, q=0.044$ となり、Primary interactionの中で唯一FDR補正後にも支持された。一方、Post-training $\times$ Depthおよび三者interactionは支持されなかった」という最新確定結果を明記。
- **H4 (Distribution Recovery)**:
  - Qwen (Matched AUC: Reader 0.189, Self 0.181; Max Recovery ~0.42-0.44)、Llama (Reader 0.005, Self -0.006)、Gemma (Reader -0.052, Self 0.001)、OLMo (Reader -0.170, Self -0.160) を文章中に整理し、Primary H4（Self--Reader AUC差 $\Delta AUC = 0.011$, 95% CI $[-0.010, 0.038]$）が0を跨ぎ支持されなかった論理を簡潔に展開。
- **レイアウトスリム化**:
  - `\[ 0.189 \] for Reader, \[ 0.181 \] for Self` などの無駄な改行数式をインラインに統合。

### 2.2 V2からV3への接続 (Line 6312〜6382)
- 重複しているコメントヘッダー（Line 6306-6311）を1つに統合。
- `\[ \text{layer} \] \times \[ \text{generation stage} \]` などの過剰な数式ブロックを整理し、論理の流れを明確化。

### 2.3 全体の考察 (Line 8351〜8575)
- `:chatgpt-content-reference{index="3"}` (Line 8471) および `:chatgpt-content-reference{index="4"}` (Line 8555) を完全削除。
- 各ステージの個別結果の単なる重複記述を凝縮し、4段階の証拠階層の統合、三重の境界線、心理的ガードレールへの実践的含意を明瞭に提示。
