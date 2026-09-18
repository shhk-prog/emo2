# 修正内容の確認 (Walkthrough): paper3.md への Related Work 統合

## 実施した作業概要
ユーザーから提供された「Related Work（関連研究）」の内容（4つの関連研究領域および5つの新規性の位置づけ）を、`/mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md` の学術論文構成に準拠した形で統合しました。

## 反映内容の詳細

### 1. 「2. 関連研究 (Related Work)」の新規挿入
「1. はじめに」の直後に新セクションとして挿入し、以下の6つの小節で構成しました：
- **2.1 Linear Probing とその方法論的限界**
  - Hewitt & Liang (2019) の control tasks と probe の選択性（selectivity）
  - Belinkov (2021, 2022) のサーベイと representation–use distinction / utilization gap
  - Ravfogel et al. (2020, 2022) の concept erasure 研究および Kumar et al. (2022) による probe 方向の因果的不十分性の指摘
- **2.2 Activation Patching と Causal Tracing**
  - clean / corrupted / patched の反実仮想的3段階比較
  - Meng et al. (2022) ROME による mid-layer MLP 局在・late-layer attention 伝達との時間的分離、および本研究の「高decodabilityかつ極低causal recovery」という逆方向の解離の対比
  - Hase et al. (2024) による局在部位と編集有効性の乖離知見との整合
  - Zhang & Nanda (2024) のベストプラクティスと、本研究における 2次元 Joint Optimal Transport（81候補離散VA分布, Manhattan ground cost）への拡張
- **2.3 因果的位置の時間的側面**
  - causal tracing における early site / late site の分離
  - Prompt 処理内だけでなく自己回帰生成過程全体を跨いだ時間的比較（Prompt-time vs Generation-time 直前）
  - Inference-Time Intervention (ITI; Li et al., 2024) との対比と、matched-substitution recovery の非単調な乖離
- **2.4 線形表現仮説とその反証**
  - Representation Engineering (RepE; Zou et al., 2023) と Linear Artificial Tomography (LAT)
  - 線形表現仮説の限界に関する理論的・力学系的反証研究
  - 本研究の probe-aligned direction ablation と random orthogonal direction の非有意な差異
- **2.5 感情・感情極性のLLM内部表現に関する研究**
  - Di Palma et al. (2025; ACL 2025) の感情極性 mid-layer encoding と本研究の decodability peak の整合
  - 既存研究が probing 精度・表現の存在に留まるのに対し、本研究が因果的介入（matched activation substitution, ablation）を行う点での差別化
- **2.6 本研究の新規性と位置づけ**
  1. Accessibility、Local Causal Leverage、Directional Necessity の体系的対比
  2. 生成過程を跨ぐ「時間的動員（Temporal Recruitment）」の実証
  3. 離散二次元分布に対する 2D Joint Optimal Transport の導入
  4. Random Direction Null Control と FDR 補正による方向特異性の厳格検証
  5. Cross-Family Partial Replication によるモデル依存性の同定

### 2. 全体セクション番号の整合性更新
「2. 関連研究」の新設に伴い、後続セクション番号を順次繰り上げ、論文全体の番号体系を完全に統一しました：
- 1. はじめに
- **2. 関連研究 (Related Work)** [NEW]
- 3. 操作的定義 (旧 2.)
- 4. モデルとデータ (旧 3.)
- 5. 制約付きSequence-Likelihood Evaluation (旧 4.)
- 6. Joint Optimal Transport による因果回復率 (旧 5.)
- 7. Statistical Analysis (旧 6.)
- 8. Results (旧 7.)
- 9. Secondary and Boundary Analyses (旧 8.)
- 10. Discussion (旧 9.)
- 11. Limitations (旧 10.)
- 12. Conclusion (旧 11.)

## 検証結果
- `grep_search` による全セクション・サブセクションの見出し一覧の検査を行い、1から12までの番号および各小節（2.1–2.6, 3.1–3.3, 4.1–4.2, 5.1–5.2, 8.1–8.6, 9.1–9.3, 10.1–10.4）が重複や欠番なく整合していることを確認しました。
