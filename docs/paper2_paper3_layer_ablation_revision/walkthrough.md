# 改訂確認レポート (Walkthrough): paper2.md および paper3.md の層別プロービング・全層Ablation記述

## 1. 改訂の背景と目的
v1初期草稿に存在した「Layer 0, 4, 8, 12, 16, 20, 24, 27 など複数層（代表層）」という記述に対し、以下の3つの論点を最新の論文草稿（[`v3/docs/paper3.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md) および [`v3/docs/paper2.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md)）に明確に反映しました。

1. **「Linear Probeは全層やっているのか」**:
   - 初期プロトタイプ設定（`main_experiment.yaml`）でストライド間隔（4層おき）が指定されていた経緯を明記した上で、本研究では全28層すべて（かつ全84コンポーネント）で網羅的プロービングを実施していることを明記。
2. **「Ablationも全層やった方が良いのではないか」**:
   - 代表層のみの介入では「中間層での情報集中（Decodability最大）」と「浅層（Layer 4）や深層（Layer 24）での因果的効果」のズレを見落とすため、**Ablation / Substitution も全28層・全84サイト網羅的に行わなければならなかった方法論的必然性**を明記。
3. **Linear Probe と Ablation の関係（Representation–Use Distinction）の論述強化**:
   - 「外部からの読み出し可能性（Decodability / 相関）」と「内部計算における必要性（Ablation Necessity / 因果）」の解離を対比させ、本論文の核心命題（*Decodability does not localize causal leverage*）の説得力を補強。

---

## 2. 具体的な変更内容と箇所

### 2.1 [`v3/docs/paper3.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md) の改訂
1. **第1章（はじめに / Introduction & Motivation）**:
   - 初期の代表層サンプリング（0, 4, 8...）の限界（空間的ズレの見落とし）を指摘し、全28層 × 3コンポーネント（計84サイト）を網羅的にプロービングおよび介入することの必然性を加筆。
2. **第3章（実験設計・介入手法 / Projection Ablation）**:
   - Linear Probe（相関的受動観測）と Ablation（能動的必要性検証）の認識論的・方法論的違いを定式化し、全層網羅的スイープが必須である理論的理由（残差接続によるバイパスや分散的補償の存在）を加筆。
3. **Appendix C（再現性・先行実験 C.3, C.4）**:
   - C.3: 初期設定ファイル（`v1/configs/main_experiment.yaml`）で4層間隔のストライド抽出が定義されていた経緯と、全28層網羅プロービングへの拡張を明記。
   - C.4: プロービングと同様、Ablationも一部層の介入にとどめず、全28層（Layer 0〜27）網羅的にMean Ablationを実行したこと、および「プローブ精度最大層（L14〜25）」と「Ablation低減最大層（L4）」が空間的に乖離していた事実が本研究の中心命題へ直結した経緯を明記。

### 2.2 [`v3/docs/paper2.md`](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md) の改訂
- `paper3.md` と完全に一致する整合性をもって、以下のセクションを改訂：
  1. **Section 1 (Introduction)**: 全層プロービングおよび全層介入の必要性の加筆。
  2. **Section 19.2 (反論1: Probe direction ablation)**: Linear Probe（相関）とAblation（必要性）の対比、および全84サイト網羅的介入の意義を加筆。
  3. **Appendix C (C.3 & C.4)**: v1プロトタイプストライド抽出の背景と、全28層プロービングおよび全28層Mean Ablationの実行記録を明記。

---

## 3. 検証結果
- 両ファイルにおいて、数式表記・Markdown構文・用語の統一性が保たれており、FDR補正結果（84サイト中0箇所が有意）やJoint OT指標との矛盾がないことを確認しました。
