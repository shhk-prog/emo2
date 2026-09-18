# Walkthrough: クロスファミリー感情認識評価 & 論文ナラティブ改訂

## 1. 実施概要
本作業では、主要4大オープンモデルファミリー（Qwen, Llama, Mistral, Gemma）× 2条件（Base / Instruct）の計8モデルにおいて、EmoBank（$N=321$）および AIPsy-Affect（$N=144$）での感情認識能（Recognition）と自己報告（Self-Report）の網羅的ベースライン評価を実施し、得られた実測データに基づいて `v3/docs/paper2.md` および `paper3.md` の学術ナラティブを全面的に改訂した。

---

## 2. 実施内容と成果物

### 2.1 評価パイプラインの実装と実行
- **設定ファイル**: `v1/configs/eval_cohort.yaml`（4ファミリー8モデルの統一定義）
- **汎用評価スクリプト**: `v1/scripts/run_cross_family_recognition.py`
  - 81通りの $(V, A) \in [1..9]^2$ 候補プロンプトに対する連続尤度期待値 $E[V], E[A]$ および離散 Greedy 出力を算出。
  - トークン打ち切り耐性を持つ正規表現フォールバックを完備。
- **集計スクリプト**: `v1/scripts/summarize_cross_family_recognition.py`
  - EmoBank ベンチマーク表と AIPsy-Affect コホート表を独立かつ綺麗に分離出力するよう改訂。

### 2.2 論文（paper2.md / paper3.md）の改訂
- **導入部・要約・概念図の更新**:
  - 旧記述（Qwen単体で83.33%中立化するというパースアーティファクトに基づく一律抑制仮説）を完全に排除。
  - 8モデルのマルチモデル検証に基づき、「感情認識能は複数ファミリーで人間同等に獲得されているが、事後学習（Post-training）が自己報告に与える影響はモデルファミリーに依存する（Post-training effect is model-family dependent）」という客観的事実を提示。
  - 「Suppression（一律抑制）」から **「Representation-to-report Remapping（内部情動表現から表層報告へのマッピング再編成）」** へと研究のコアを昇華。
  - 更新された概念図：
    $$\text{刺激文} \;\longrightarrow\; \underbrace{\text{Recognition (普遍的認識能: $r_V \approx 0.83$)}}_{\text{認知的理解}} \;\longrightarrow\; \underbrace{\text{Internal Probe (幾何学的復元: $R^2=0.561$)}}_{\text{中間層符号化}} \;\longrightarrow\; \underbrace{\text{Remapping (表現-報告再編成)}}_{\text{ファミリー依存の表層マッピング}} \;\longrightarrow\; \underbrace{\text{Intervention (因果レバレッジ)}}_{\text{Decodability} \neq \text{Causal Leverage}}$$
- **リサーチクエスチョン（RQ）の再定式化**:
  - RQ1: 候補出力尤度分布における刺激依存的な感情構造の保存。
  - RQ5 (V3核心): 「高い Decodability を示す部位が、対応する出力に対して大きな Local Causal Leverage を持つか（$\text{Decodability} \neq \text{Causal Leverage}$ の時空間的解離検証）」。自己報告中立化の有無に依存しない普遍的解釈可能性の問いへ確立。
- **付録 C.1b（ベースライン評価）の刷新**:
  - 8モデル × 2データセットの完全な実測結果表（表C.1a および 表C.1b）を両ペーパーに反映。

---

## 3. 検証結果
- [paper2.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper2.md) および [paper3.md](file:///mnt/nas/home/hiromi/src/emo/v3/docs/paper3.md) の両ファイルにおいて、古いパースバグ由来の「83.33% 中立化」という記述がゼロになり、最新の8モデル実測結果および再定義されたナラティブとの完全な整合性を確認した。
- 全実測データは [evaluation_report.md](file:///mnt/nas/home/hiromi/src/emo/docs/cross_family_emobank_recognition/evaluation_report.md) および `v1/results/recognition_baseline/` に永続保存されている。
