# 実装計画: 実験実測値（Exp A/B/C）を統合した論文草稿の改訂

## 概要
ユーザーによって実行・検証された実験A（Within-model positive control）、実験B（Full-state reconstruction）、実験C（Empirical manifold diagnostics）の実測データを全て論文草稿（`v3/docs/paper.md`）に統合します。
査読者の懸念（「Within-modelでも動かないならpost-trainingのせいとは言えない」「Ridgeは低次元しか復元していない」「$D_M$ の理論値との乖離」）を完全に消化し、論文の主張を「プロービング可能性と因果的十分性の構造的解離」という強固な命題へと昇華させます。

## 改訂方針

### 1. タイトルとコア命題の昇華
- **タイトル**: `Decodability Without Causal Substitutability: Why Predictively Aligned Representations Fail Downstream Policy-Constrained Reports in Language Models`
- **中心命題**:
  > 線形プローブで高精度に感情がデコードでき、Base/Instruct間で高次元活性化全体の分散の約50%・CKA 0.82・検索精度75%で予測復元可能であり、かつ多様体距離が劇的に縮小（$D_M=256 \rightarrow 4.98$）していても、下流の制約付き報告分布に対する因果的代替性は生じない（0.11%）。
  > さらに決定的な発見として、**そのプローブが感情を読み取っている最終トークン活性化自体が、同一モデル内ですら下流報告分布に対して因果的十分性を持たない（Within-model recovery 0.00%）**。このことは、解釈可能性研究における「プローブ成功部位＝因果的メカニズム」という暗黙の仮定に対する構造的な反証となる。

### 2. 数値と表の完全アップデート
- **実験B（高次元再構築度）の明記**:
  - $R^2_{\text{activation}} = 0.4966$ (median: $0.5028$)
  - Linear CKA = $0.8236$
  - Pair Retrieval Top-1 = $75.61\%$
- **実験C（経験的多様体適合検定）の明記**:
  - Natural Instruct $D_M$ 経験的パーセンタイル（5th: 34.20, 50th: 39.66, 95th: 51.43）と理論値 $\sqrt{1536} \approx 39.19$ の完全一致。
  - Aligned Base $D_M = 4.98$（中心部への過剰凝縮）と Raw Base $D_M = 256.47$ の比較。
  - Two-sample Linear Classifier AUC = $0.6386$。
  - コサイン対照群: Matched ($0.9950$) > Unmatched ($0.9830$) > Natural Pair ($0.9785$)。
- **実験A（Within-model positive control）の明記**:
  - L15 MLP (0.00%), L15 Resid (0.00%), L13-16 Resid (0.00%)。
  - 最終トークンパッチングの因果的不十分性（Causal Insufficiency）を論理の中核に位置づける。

### 3. ドキュメント保存先
- `docs/empirical_resolution_and_paper_update/`
  - `task.md`
  - `implementation_plan.md`
  - `walkthrough.md`
