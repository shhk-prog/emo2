# 修正内容の確認 (Walkthrough): 論文結果表・生成ロジックの監査と修正

## 実施した変更の概要

ユーザーによる20項目の詳細監査フィードバックに基づき、プレゼンテーション層（LaTeXおよびテーブル生成スクリプト）の修正を完了しました。原データおよび raw results は一切改変せず、実データに忠実な集計・表示を行っています。

---

## 1. LaTeX 環境・構文修正
- **`\usepackage{multirow}` の追加**:
  - `iclr2027/iclr2027_conference2.tex` の preamble（Line 12）に `multirow` を追加し、`Undefined control sequence: \multirow` による停止を解消。
- **数式外 `\text{}` の修正**:
  - Line 706 および Line 7456 付近（V3 RQ3 解釈上の境界）で、数式外で呼ばれていた `\text{mediated attenuation}` や `\text{endogenous relevance}` を `\textit{...}` に置換し、LaTeX構文エラーを防止。

---

## 2. V2 の修正（Placeholder全廃・数式矛盾解消・LMM修正）
- **Placeholder `dec_peak = 0.5` の全廃**:
  - `v2/scripts/build_paper_summary.py` 内の仮置き値 `0.5` を `np.nan` に置き換え。
  - recovery のデフォルト値（0.72, 0.15 等）も全廃。
- **Causal Relocation の数式矛盾解消**:
  - `scripts/summarize_v2_reorganization.py` の `generate_causal_relocation_table` において、事後学習に伴う因果再配置変位 $\Delta d_C^*$ を以下のように定義・計算：
    $$\Delta d_C^* = d_{C, \text{Instruct}}^* - d_{C, \text{Base}}^*$$
  - Base Peak = 1.000, Instruct Peak = 1.000 の場合、$\Delta d_C^* = 0.000$ となり、数式通り正確に表示。
- **V2 LMM の exact mapping**:
  - `table_v2_3c_lmm.csv` の読み込みで `TERM_MAP` による完全一致を導入。
  - 以前生じていた `Intercept = 0.625, CI = [-0.472, -0.416]` のようなズレを解消し、実データ通り `Intercept: -0.444 [-0.472, -0.416]`, `Post-training: 0.181 [0.145, 0.217]` を出力。
- **`v2_confirmatory_summary.tex` の実データ動的生成**:
  - hard-coded 配列 `conf_items = [...]` を完全撤廃し、`table_v2_confirmatory.csv` から動的にパース・生成。

---

## 3. V1 表 Note の実数値整合・学術的厳密化
- **E1 (`v1_peak_decodability.tex`)**:
  - 「全ファミリーで微小（<0.15）」を修正し、一部モデル・軸（Llama Base Valence: 0.20, Gemma Instruct Arousal: 0.36 等）で相対深度0.2以上の乖離が存在することを明記。
- **E2 (`v1_shared_geometry.tex`)**:
  - 全条件で正の RSA を示す一方、direct cross-decoding は大きく負となる条件が多く、Procrustes 整列後に改善することから、「同一の表現幾何」ではなく「線形変換によって部分的に整列可能」と正確に表現。
- **E3 (`v1_causal_profile.tex`)**:
  - 順位相関の強さはモデル間で異なり（Llama Base: 0.582）、余弦類似度がほぼ0となる条件も存在するため、「高度に重複した回路を証明」を撤回し、部分共有に修正。
- **E4 (`v1_interchangeability.tex`)**:
  - Reader 由来の差分ベクトルを移植した際の Specificity は、全条件で FDR 補正後に非有意（$q > 0.3$, CI が 0 を跨ぐ）であるため、「因果交換可能」から「cross-task causal interchangeability を支持する robust evidence は得られなかった」へ結論を反転。
- **E6 (`v1_specialization.tex`)**:
  - 全ファミリー共通ではなく一部モデル（Qwen Base/Inst, Llama Base/Inst, OLMo Base）のみで有意であることを明記。さらに表に **FDR $q$** 列を新たに追加。
- **Phase B (`v1_semantic_controls.tex`)**:
  - Word Shuffle に対する一貫したドロップは BoW shortcut 否定を支持するが、Paraphrase / Reversal は nonfallback sample 数が小さいため補助的エビデンスと位置づけ。

---

## 4. Behavioral 表 Note の修正
- **EmoBank (`behavioral_emobank_3way_vad.tex`)**:
  - Base/Instruct 間で prompt format も異なるため、これを事後学習の因果効果とは解釈しない旨を明記。
- **AIPsy RQ1 (`behavioral_aipsy_summary.tex`)**:
  - OLMo Instruct Valence ($d_z = 0.32\sim0.36$) の実数を反映。
- **AIPsy RQ3 (`behavioral_aipsy_summary.tex`)**:
  - 特異性は一様ではなくモデル・タスク依存であった旨を正確に記述。

---

## 5. V3 表の修正（Gate 4条件化・両軸化・family-specific CI・厳密判定）
- **Gate 表 (`v3_gate_decision.tex`)**:
  - `attenuation_ci_low`, `attenuation_pass` を追加し、4条件化（Sufficiency, Specificity, Endogenous Relevance, Topic TVD）。
  - 注記に事前登録閾値（0.10, 0.05, 0.05, 0.15）を明記。
- **Confirmatory Details 表 (`v3_confirmatory_details.tex`)**:
  - Valence と Arousal の両軸を全モデル・全仮説で出力（計24行）。
  - cross-family CI の使い回しを廃止し、各ファミリー固有の 95% CI を表示。
  - 合否判定（Pass/Fail）を point estimate から事前登録された CI criterion（H2: $\text{CI}_{\text{low}} > 0.10$ 等）による厳密判定へ変更。
- **Mediated Attenuation 表 (`v3_mediated_attenuation.tex`)**:
  - ランダム統制 $M_{\text{rand}}$ と正味媒介減衰量 $M_{\text{net}} = M - M_{\text{rand}}$ 列を追加。
- **Confirmatory Matrix 表 (`v3_confirmatory_matrix.tex`)**:
  - Valence pass AND Arousal pass の両軸達成を基準として評価（All Confirmed: **NO**）。

---

## 6. 旧・重複テーブルの整理
`iclr2027/tables/archive/` ディレクトリを作成し、本文で使用されていない旧 alias や不要ファイルを移動・退避しました。現在 `iclr2027/tables/` には本文から `\input{...}` されている canonical な 19 ファイルのみが配置されています。

```text
iclr2027/tables/
├── behavioral_emobank_3way_vad.tex
├── behavioral_aipsy_summary.tex
├── v1_peak_decodability.tex
├── v1_shared_geometry.tex
├── v1_semantic_controls.tex
├── v1_causal_profile.tex
├── v1_interchangeability.tex
├── v1_specialization.tex
├── v2_h1_h2_reorganization.tex
├── v2_causal_relocation.tex
├── v2_causal_controls.tex
├── v2_h3_causal_lmm.tex
├── v2_distribution_recovery.tex
├── v2_confirmatory_summary.tex
├── v3_gate_decision.tex
├── v3_spatiotemporal_dynamics.tex
├── v3_mediated_attenuation.tex
├── v3_confirmatory_details.tex
├── v3_confirmatory_matrix.tex
└── archive/
```

---

## 7. 生成検証
マスター生成コマンド：
```bash
python scripts/generate_paper_results_tables.py --repo-root . --out-dir iclr2027/tables
```
により、エラーおよび警告なく全ファイルが実データから決定論的に再生成されることを確認しました。
これで、論文の考察（Discussion）執筆に安心して進むことができる確固たるデータ・表基盤が完成しました。
