# 実装計画書: paper2.md および paper3.md における全層プロービングと全層Ablationの整合的改訂

v1草稿における「代表層（0, 4, 8, 12, 16, 20, 24, 27層）の抽出・介入」という初期記述に対し、ユーザーから提起された**「Linear Probeは全層やっているのか」「Ablationも全層やった方が良いのではないか」「Linear ProbeとAblationの関係」**を、`v3/docs/paper3.md` および `v3/docs/paper2.md` の本文・手法・Appendixに明確かつ整合的に反映します。

## ユーザー確認事項
- v1初期草稿において存在した「4層おき（0, 4, 8, 12, 16, 20, 24, 27層）の代表層サンプリング」という初期設定の経緯を明記しつつ、現行の実験体系では**「Linear ProbeもAblationも、なぜ全28層（および全84コンポーネント）すべてで行う必然性があったのか」**を論文の核心的論理（Decodability does not localize causal leverage）として強調・補強します。

## 主な改訂内容

### 1. `v3/docs/paper3.md` の修正
1. **第1章（はじめに / Introduction & Motivation）**:
   - 代表層のみの間引きサンプリング（例: 4層間隔）では、「情報が最も濃く読み取れる層（中間層）」と「因果的に出力を左右する層（浅層や深層）」の局所性のズレを見落とす危険性があることを指摘。
   - したがって、**Linear Probe（情報のアクセス可能性）と Ablation / Substitution（因果的必要性・十分性）の双対関係を検証するためには、全28層すべて、かつ Residual / MLP / Attention の全84コンポーネントを網羅的にスイープすることが方法論的に不可欠であること**を明記。
2. **第3章（実験設計・介入手法 / Interventions）**:
   - Linear Probe（外部からの復元性）と Probe-aligned Projection Ablation（内部計算からの方向特異的除去）の数理的関係を対比。
   - なぜ「Ablationも全層やらなければならないのか」（特定層の部分的介入ではバイパス経路や非局所性を過小評価する）を理論的に明記。
3. **Appendix C（再現性・先行実験 C.3, C.4）**:
   - 初期プロトタイプ設定（`v1/configs/main_experiment.yaml`）でストライド抽出（0, 4, 8...）が用いられていた経緯を記載。
   - 実際の実装（`run_causal_intervention.py`）において全28層の Mean Ablation が実行され、Layer 4 で最大42.26%低減、中間層で分散的低減が観測された事実と、それが現行の全84サイト射影Ablationへと発展した接続関係を明瞭化。

### 2. `v3/docs/paper2.md` の修正
- `paper3.md` と同一の理論的整合性をもって、第1章、手法、およびAppendix Cの該当箇所を改訂。

---

## 検証手順
1. `paper2.md` および `paper3.md` の修正前後におけるdiffを確認し、誤字脱字、数式整合性、および論理展開の滑らかさをチェック。
2. 他の記述（全層84サイトのFDR補正結果やOptimal Transport指標）との矛盾が生じていないことを確認。
3. 修正内容をまとめた `walkthrough.md` を作成。
