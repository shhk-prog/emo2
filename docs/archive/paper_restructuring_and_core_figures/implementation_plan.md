# 論文再構成と主図表（4図＋3表）整備の実装計画

## 背景と目的
現在の原稿（`v3/docs/paper2.md`）には、全28層の探索的スクリーニング、多変量OOD診断、Base→Instructアライメント、多層同時介入、LLaMA追試など膨大な実験結果が含まれており、論文本文にすべてを均等に記述すると読者が遭難（情報の森に埋没）してしまいます。

そこで、ユーザーからのご提案に基づき、本文の中心メッセージである **「Decodability peak と causal leverage peak が空間的・時間的に解離する（Decodability does not localize causal leverage）」** を一目で読者に納得させる構成へと再編成します。

本文を **4つの図（Figures 1–4）＋ 3つの表（Tables 1–3）** を軸とした明快なストーリーに集約し、全層探索結果や頑健性検証、探索的実験は **体系的なAppendix（再現性の保管庫）** へ移送します。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> 1. **本文構成のコンパクト化**:
>    - 本文は「RQ1: Greedy collapse vs Distributional sensitivity (Fig 4)」「RQ2: Decodability vs Prompt-time causal recovery & necessity (Fig 2 A-C, Table 2)」「RQ3: Generation-time causal emergence & Peak-site dissociation (Fig 2 D, Fig 3, Table 2)」「Claims Summary (Table 3)」を中心に論理を展開します。
>    - Cross-model alignment、Multi-layer saturation、LLaMA追試は、本文では要約段落（1〜2パラグラフ）にとどめ、詳細な図表はAppendixへ配置します。
> 2. **原本保全の徹底**:
>    - 既存の `v3/docs/paper2.md` は上書きせずそのまま保持し、再構成後の完成版は `v3/docs/paper_restructured.md` として新規作成します。

---

## 提案する変更内容 (Proposed Changes)

### 1. 視覚的エビデンス（図）の生成スクリプト作成と実行

#### [NEW] `v3/scripts/plot_paper_figures.py`
以下の4つの本文用図（高解像度PNG/PDF）を生成するスクリプトを作成します：

1. **Figure 1: Overall Experimental Framework (概念図)**
   - 感情文/中立文入力 → Qwen2.5-1.5B-Instruct
   - Prompt-time hidden states: Linear probe ($D_\ell$), Matched substitution ($S_\ell$), Probe ablation ($N_\ell$)
   - Generation-time hidden states: Matched substitution ($G_{\ell,t}$)
   - 中心命題: $\text{Where information is decodable} \neq \text{where/when it becomes causally effective}$

2. **Figure 2: Four-Panel Representational–Causal Profile (主図)**
   - **Panel A**: Layerwise Decodability ($D_\ell$: MLP, Attention, Residual). L15 MLP ($R^2=0.561$), L18 Attention, L14 Residual をマーカーで強調。
   - **Panel B**: Prompt-Time Causal Recovery ($S_\ell$: MLP, Attention, Residual; L15 MLP $\approx 0.5\%$). Panel Aとの同一横軸対比。
   - **Panel C**: Probe-Aligned Necessity ($N_\ell$: $R_{\mathrm{neut}}$ または $Z_\perp$). 0ライン破線、84条件がほぼゼロ近傍であることを可視化。
   - **Panel D**: Generation-Time Causal Recovery ($G_{\ell,t}$). 後段Residualの急激な立ち上がり（L15 MLP = -0.06% vs L24 Residual = 53.24%）。

3. **Figure 3: Direct Peak-Site Dissociation (直接対比・統計的確証)**
   - L15 MLP ($D=0.561, S=0.51\%, G=-0.06\%$) vs L24 Residual ($D=0.147, S=-0.26\%, G=53.24\%$)
   - 39ペアごとの paired slope / scatter 可視化、$\Delta G = 53.30\%$, 95% CI [45.34, 61.16] の表示。

4. **Figure 4: Greedy Collapse vs Distributional Sensitivity**
   - 左: Base (12.4%) vs Instruct (98.6%) の (5,5) 集中率
   - 右: Human Valence vs Expected $E[V]$ の散布図（Base $r=0.365$ vs Instruct $r=0.629$）

5. **Appendix図用スクリプト（`v3/scripts/plot_appendix_figures.py`）**
   - Fig A1〜A10（全層網羅データ、アブレーション詳細、Ridgeスイープ、LLaMA追試等）

---

### 2. 本文用の表（Tables 1–3）の整備

- **Table 1: Dataset and Experimental Design**
  - モデル、レイヤー数、AIPsy-Affect Strict Expanded構成（Train 76 groups/169, Dev 58/124, Test 58/129, Complete test pairs 39）、81候補列、評価指標。
- **Table 2: Main Representative-Site Results**
  - 代表6サイト（L14 Resid, L15 MLP, L18 Attn, L18 Resid, L20 Resid, L24 Resid）の $D_\ell$, $S_\ell$, $G_{\ell,t}$, 95% CI。
- **Table 3: Main Claims and Evidence**
  - 研究の問い（Question）、得られた証拠（Evidence）、結論（Result）を明快に対照させた総括表。

---

### 3. 本文原稿の再構成

#### [NEW] `v3/docs/paper_restructured.md`
- 読者が迷わず、中心的主張「Decodability does not localize causal leverage」に集中できる4図＋3表主体の構成。
- 探索的実験・感度分析・詳細数値はAppendixへ整理。

---

## 検証計画 (Verification Plan)

### 自動テスト / 実行確認
1. `v3/scripts/plot_paper_figures.py` を実行し、Figure 1〜4（PNG/PDF）がエラーなく生成され、数値が元の結果CSVと厳密に一致していることを確認する。
2. 表（Table 1, 2, 3）の数値が既存の `v3/results/` 下のCSV（`focused_causal_sweep_39pairs.csv`, `causal_localization_sweep_joint_ot.csv`, `probe_aligned_necessity_sweep.csv` 等）と100%合致しているかクロスチェックする。
3. Markdown原稿内の数式、図表参照、相互リンクをチェック。
