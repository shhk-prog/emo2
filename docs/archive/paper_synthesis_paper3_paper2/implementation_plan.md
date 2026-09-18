# paper3.mdを基本としたpaper2.md統合・拡充の実装計画

## 背景と目的
現在、`v3/docs/` には以下の2つの主要な論文原稿が存在します：
- **`paper3.md`**: 査読耐性に優れ、抑制的・客観的かつ明晰な学術的文体で書かれた最新原稿。
- **`paper2.md`**: 実験プロトコルの詳細、数式定義、中核図表（Figure 1–4、Table 1–3）、全28層の完全な数値データ表（Table A1–A8）、および補助的・探索的実験の全記録を含む包括的な原稿。

ユーザーからのご指示「**基本はpaper3.mdの方として，足りない分の情報をpaper2からおぎなって詳しい論文にして**」に基づき、`paper3.md` の端正な構成・文体を主軸としながら、`paper2.md` に含まれる豊富な詳細情報・厳密な数式・中核図表・全数データ表を余すところなく統合し、1本の完全無欠な決定版論文原稿として仕上げます。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> 1. **主軸の維持**:
>    - 基本構成・論理展開・文体は `paper3.md` のアカデミックで抑制的なスタイルを厳格に維持します。過度な誇張や感情の帰属を避け、操作的定義と測定されたエビデンスに立脚した記述を徹底します。
> 2. **統合・補完の範囲**:
>    - 本文には、中核となる **4つの図（Figure 1–4）のアスキーアート／詳細解説** および **3つの表（Table 1–3）** を美しく統合します。
>    - 理論的・数学的背景（2D Optimal Transport、系列尤度拘束正規化、直交射影消去、帰無分布検定、マハラノビス距離等）を厳密に定式化します。
>    - Appendixには、`paper2.md` に収録されている **全28層×3コンポーネント（84サイト）の完全な数値データ表（Table A1, A2, A3, A5, A6, A7, A8）** および探索的実験（二重アウトカム、気分一致性、多層介入、LLaMA追試）を完全網羅します。
> 3. **ファイル更新方針**:
>    - 統合された決定版論文は、基本となる [`paper3.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md) に上書き反映し、完成させます（`paper2.md` は変更せず保持）。

---

## 提案する章構成と統合内容 (Proposed Structure)

### 本文 (Main Text)
1. **要旨 (Abstract)**
   - 問題意識、手法、中核エビデンス（L15 MLP $R^2=0.561 \to S=0.51\%$, L24 Resid $R^2=0.147 \to G=53.24\%$, $\Delta G = +53.30\%$, プローブ方向非特異性）、結論の厳密な要約。
2. **第1章 はじめに (Introduction)**
   - Linear probing の普及と representation–use gap の理論的課題。
   - **Figure 1 (Overall Experimental Framework)** のアスキーアート・中心仮説の提示。
   - 本研究の4つの測定軸（アクセス可能性、Prompt回復、プローブ必要性、Generation回復）と主たる貢献。
3. **第2章 タスク・データ・測定手法 (Task, Dataset, and Evaluation Design)**
   - 概念的範囲（3つの操作的性質の分離）。
   - **Table 1 (Dataset and Experimental Design Specifications)** の配置。
   - AIPsy-Affect Strict Expandedの統制基準（文長 $118 \pm 9$語、ドメイン整合、感情語排除）。
   - 81候補列拘束シーケンス尤度評価、長さ正規化スコア $s^{\mathrm{norm}}$、期待値 $E[V], E[A]$。
   - 2D Optimal Transport (Wasserstein-2, Manhattan cost) と OT分母セーフガード（$<0.05$）の数学的定義。
4. **第3章 実験結果 (Results)**
   - 3.1: Greedy中立化と分布的感度の解離（**Figure 4** の配置、Base $r=0.365$ vs Instruct $r=0.629$、ダイナミックレンジ圧縮）。
   - 3.2: 中間層デコーダビリティ（**Figure 2a**、MLP L15 $R^2=0.561$, Attn L18 $R^2=0.550$, Resid L14 $R^2=0.502$）。
   - 3.3: Prompt時因果回復の欠如（**Figure 2b**、全層相関 $\rho \le 0.296$、L15 MLP $0.51\%$）。
   - 3.4: 生成時因果回復の後段Residualシフトと直接対比（**Figure 2d**、**Figure 3 (Direct Peak-Site Dissociation)**、**Table 2 (Main Representative-Site Results)**、$\Delta G = +53.30\%$, 95% CI $[+45.34\%, +61.16\%]$, $p < 10^{-12}$）。
   - 3.5: プローブ方向の非特異性・局所必要性の欠如（**Figure 2c**、84サイト検定、FDR $q > 0.85$、N=100追試）。
   - 3.6: 境界条件と補助的結果（多層Residual飽和約55%、Base-Instructアライメントと多様体崩壊、LLaMA追試）。
5. **第4章 考察と理論的統合 (Discussion)**
   - 4.1: アクセス可能性は局所的因果レバレッジではない。
   - 4.2: 本研究が支持すること／支持しないこと。
   - 4.3: プロービング研究への方法論的含意。
   - 4.4: 限界事項（局所介入の範囲、タスク拘束性、モデル規模、感情の非主観性）。
   - **Table 3 (Synthesis of Core Claims and Empirical Evidence)** の配置。
6. **第5章 結論 (Conclusion)**
   - 結論の明確な提示。

---

### 付録 (Comprehensive Archival Appendices)
- **Appendix A**: データセット構築・ペアマッチング・分割完全性プロトコル
- **Appendix B**: 81候補列尤度測定・正規化・プロンプト全文
- **Appendix C**: 活性化抽出フック・プローブ推定詳細・全28層決定係数
- **Appendix D**: Matched-Substitution プロトコル（Prompt vs Generation）
- **Appendix E**: 全28層 Prompt-Time スクリーニング完全数値表 (**Table A1**) および Marginal OT 頑健性
- **Appendix F**: 全28層 Generation-Time スクリーニング完全数値表 (**Table A2**) および 代表18サイト39ペア詳細統計 (**Table A5**)
- **Appendix G**: プローブ整合型局所必要性検定（84サイト完全数値表 **Table A3**、N=100高解像度追試 **Table A4**）
- **Appendix H**: 多層Residual同時介入の飽和特性 (**Table A6**)
- **Appendix I**: Base-to-Instructアライメント・Ridge正則化スイープ・マハラノビスOOD診断 (**Table A7**)
- **Appendix J**: 代替アウトカムと探索実験（二重アウトカム行動評価、気分一致性ステアリング）
- **Appendix K**: LLaMA-3.2-1B-Instruct クロスファミリー追試プロファイル
- **Appendix L**: 再現スクリプト・成果物ハッシュ・マニフェスト対応表 (**Table A8**)

---

## 検証計画 (Verification Plan)
1. `paper3.md` の更新内容が、既存の `paper3.md` の学術的トーン・厳密性を完全に継承しているか確認する。
2. `paper2.md` にあったすべての主要な数値（$R^2$, 回復率, CI, $p$値, $q$値, マハラノビス距離等）が正確に統合され、矛盾がないことをクロスチェックする。
3. 全28層の数値表（Table A1〜A8）および図版参照（Figure 1〜4、Figure A1〜A10）がすべて過不足なく記述されていることを確認する。
