# 実装計画: paper3.md への Related Work（関連研究）の統合

## 概要
ユーザーから提供された Related Work（4つの関連研究領域と5つの新規性）を、`/mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md` の「1. はじめに」の直後に「2. 関連研究 (Related Work)」として追加し、全体の論文セクション構成を整合させる。

## 構成変更案

### セクション番号の更新
- **1. はじめに** (維持)
- **2. 関連研究 (Related Work)** [新規挿入]
  - 2.1 Linear Probing とその方法論的限界
  - 2.2 Activation Patching と Causal Tracing
  - 2.3 因果的位置の時間的側面
  - 2.4 線形表現仮説とその反証
  - 2.5 感情・感情極性のLLM内部表現に関する研究
  - 2.6 本研究の新規性と位置づけ
- **3. 操作的定義** (旧 2.)
- **4. モデルとデータ** (旧 3.)
- **5. 制約付きSequence-Likelihood Evaluation** (旧 4.)
- **6. Joint Optimal Transport による因果回復率** (旧 5.)
- **7. Statistical Analysis** (旧 6.)
- **8. Results** (旧 7.)
- **9. Secondary and Boundary Analyses** (旧 8.)
- **10. Discussion** (旧 9.)
- **11. Limitations** (旧 10.)
- **12. Conclusion** (旧 11.)

### テキスト整形の詳細
- 検索由来の断片トークン（`arxiv+2`, `proceedings.mlr+2`, `openreview+2`, `alphaxiv+1`, `aclanthology+1` など）を削除し、学術論文にふさわしい清書スタイルにする。
- 論文リンク（`[asmadotgh.github](...)`, `[aclanthology](...)` など）は適切な文献タイトルまたはURLリンクとして整形する。
- 既存の `paper3.md` のトーン・書式（見出し階層、数式、区切り線 `⸻`）と完全に調和させる。

## 検証計画
- `paper3.md` の構文、見出し番号の連続性、リンク切れの有無を確認。
- 各セクション見出しが正しく階層化されていることを `grep_search` 等で検証。
