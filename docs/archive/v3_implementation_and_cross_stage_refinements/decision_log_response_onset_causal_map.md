# Decision Log: Response-Onset Causal Map & Intervention Specification

## 1. 決定の背景と位置づけ
Llama 3.2 1B Base の V1 Phase C 因果パッチング実験において、差分ベクトルの抽出位置と注入位置を精査した結果、以下の介入構造となっていることが確認された。

- **Source（差分抽出位置）**: Prompt 末尾トークン位置における affective–neutral 差分ベクトル
  $$\Delta h_T^{(l)} = h_{T,\mathrm{aff}}^{(l,\mathrm{prompt\ end})} - h_{T,\mathrm{neu}}^{(l,\mathrm{prompt\ end})}$$
- **Target（介入注入位置）**: 回答生成開始時の最初の candidate トークン位置（response-onset hidden state）
  $$\tilde{h}_{T,\mathrm{response\ onset}}^{(l)} = h_{T,\mathrm{response\ onset}}^{(l)} + \Delta h_T^{(l)}$$

当初想定していた「prompt-end $\rightarrow$ prompt-end」の同位置介入ではなく、
**「prompt-derived affect difference $\rightarrow$ response-onset intervention」**
となっている。

---

## 2. 729候補の先頭トークン検証結果（実験的根拠）
本実験を再実行せず正式結果として採択・再利用できる根拠として、全729個の VAD 候補文字列の先頭トークンを `meta-llama/Llama-3.2-1B` の tokenizer で厳密に検証した。

- **実行スクリプト**: [`scratch/check_first_token.py`](file:///mnt/nas/home/hiromi/src/emo/scratch/check_first_token.py)
- **検証結果**: [`scratch/first_token_result.json`](file:///mnt/nas/home/hiromi/src/emo/scratch/first_token_result.json)
  ```json
  {
    "unique_first_token_ids": [5018],
    "num_unique_first_tokens": 1,
    "decoded": {
      "5018": "'{\\\"'"
    }
  }
  ```
- **理論的帰結**:
  1. 全729候補の先頭トークンは例外なく完全同一（Token ID: 5018, `'{"'`）である。
  2. Causal LM において、candidate の第1トークンの確率はその直前（prompt 末尾位置）の logits で決まるため、全候補で共通項（softmax の定数倍）となる。
  3. したがって、response-onset（第1トークン位置）へのパッチングは、主に第2トークン以降の生成確率（VAD 数値の選択）に対して作用する。
  4. Sequence Likelihood 計算において Boundary mismatches は 0 であり、尤度評価の数理的整合性は完全に保たれている。

---

## 3. 実験の再定義と名称改訂

### 3.1 E3: Response-Onset Causal Map（旧称: Shared Causal Map）
- **定義**: Reader/Self の刺激処理（prompt 末尾）から抽出した情動差分ベクトル $\Delta h_T^{(l)}$ を、同一層の回答生成開始位置（response-onset state）へ注入し、自己報告 VAD 分布への因果効果を測定する。
- **数式**:
  $$\Delta h_T^{(l)} = h_{T,\mathrm{aff}}^{(l,\mathrm{prompt\ end})} - h_{T,\mathrm{neu}}^{(l,\mathrm{prompt\ end})}$$
  $$\tilde{h}_{T,\mathrm{response\ onset}}^{(l)} = h_{T,\mathrm{response\ onset}}^{(l)} + \Delta h_T^{(l)}$$

### 3.2 E4: Cross-Perspective Interchangeability
- **定義**: Reader の刺激処理から得られた情動表現（Reader prompt-derived affect vector）を、Self の回答生成開始位置（Self response-onset state）へ移植し、回答生成に及ぼす因果効果を検証する。
- **認知的・機械論的意義**:
  「他者感情認識（Reader）で形成された情動表現が、自己報告（Self）の生成段階で利用可能か」という問いになり、研究全体のストーリー（特に V3 の spatiotemporal dynamics や mediation 解析）と極めて整合的である。

---

## 4. 論文および報告書の記述規程

### 4.1 削除すべき表現
- 「Self Neutral 状態の同じ末尾 token 位置に注入した」
- 「prompt 末尾層の状態を直接操作した」

### 4.2 正式採用する記述
- **英語**:
  > "For each matched affective–neutral pair, we extracted the affective difference vector from the final prompt representation and injected it at the response-onset position of the same layer. We then measured the induced shift in the likelihood-based VAD report distribution."
- **日本語**:
  > "各affective–neutral最小対について、刺激処理終了時のhidden stateから情動差分ベクトルを抽出し、同一層の回答生成開始位置へ注入した。介入前後のVAD候補分布をSequence Likelihoodにより比較し、回答分布への因果効果を評価した。"

---

## 5. 他モデルおよび Phase C 全体の統一仕様
- 本仕様（`prompt-derived affect difference -> response-onset intervention`）を V1 Phase C の正式プロトコルとする。
- これにより、Llama 3.2 1B Base の計算成果（約12時間分）を完全に保持し、今後実行する他モデル（Llama Instruct, Qwen, Gemma, Mistral 等）とも同一の介入定義で厳密に比較可能とする。
- 結果ディレクトリには [`intervention_metadata.json`](file:///mnt/nas/home/hiromi/src/emo/v1/results/derived/v1_phase_c/llama3.2_1b_base/intervention_metadata.json) を付与し、介入位置メタデータを永続記録する。
