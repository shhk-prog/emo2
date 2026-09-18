# ウォークスルー: 査読指摘への即応実装

## 実施した実装の完了報告

査読者視点の厳しい指摘（Weak Reject / Borderline: 4-5/10）を受け、論文ドラフトの論理防御の改訂と、指摘された穴を塞ぐための追加実験用スクリプト群を実装しました。

---

## 1. 論文ドラフト (`v3/docs/paper.md`) の改訂内容
- **C1の位置づけ変更**:
  - 新規手法としての過大主張を退け、Martorell (2024) 等の先行知見に準拠した「表層のgreedy collapseを回避する標準的測定プロトコル」として整理。主要貢献リストを再編成しました。
- **C4（三人称ステアリング）の主結果からの除外**:
  - Layer 16でrandom control（$\beta = +0.0444$）がvalence（$\beta = -0.0169$）より2倍以上強く動く「generic disruption」の懸念を認め、主結果から除外して「補助的・探索的知見」へと後退させました。
- **Singh et al. (Can LLMs Introspect? 2026) への防御注記**:
  - 「first-person report」は出力の文法および課題形式を指すものであり、モデルの内省能力や主観的体験の証拠ではないことを明記しました。
- **Post-training因果帰属の限定**:
  - 単一チェックポイントペア比較であることから、「post-trainingが結合を変化させた」から「Base/Instruct比較において観察される結合差異（consistent with post-training-associated changes）」へと厳密化しました。
- **Limitationsの充実**:
  - Within-model positive controlの確立、Ridge alignmentの高次元完全性（$R^2_{\text{activation}}$）、経験的参照分布との比較、他モデルファミリー検証の必要性を「今後の検証課題」として先回りして誠実に記載しました。

---

## 2. 新規実装した実験スクリプト

### ① `v3/scripts/run_within_model_positive_control.py`（最優先: 実験A）
同一Instructモデル内において、`Peak` 刺激の活性化を `Neutral` 刺激実行時へパッチした際に報告分布が動くか（Positive Control）を検証するスクリプトです。
- **比較プロトコル**:
  1. `mlp_last_token_L15`: 既存のLayer 15 MLP出力・最終トークン
  2. `resid_last_token_L15`: Layer 15 残差ストリーム・最終トークン
  3. `resid_all_tokens_L15`: Layer 15 残差ストリーム・全プロンプトトークン
  4. `multi_resid_last_L13_16`: Layer 13-16 残差ストリーム・最終トークン
  5. `multi_resid_all_L13_16`: Layer 13-16 残差ストリーム・全プロンプトトークン
- **出力指標**: 2D EMD Recovery、Expected Valence Shift

### ② `v3/scripts/analyze_alignment_fidelity_and_manifold.py`（実験B & C）
Ridge alignmentが低次元属性だけでなく高次元活性化全体を復元できているか、またAligned活性化が自然なInstruct活性化と統計的に区別不能かを多角的に診断するスクリプトです。
- **実験B（Full-State Reconstruction）**:
  - $R^2_{\text{activation}}$（全1536次元平均および中央値）
  - Linear CKA（Centered Kernel Alignment）
  - Pair Retrieval Top-1 Accuracy（最近傍探索による正解ペア当て率）
- **実験C（Empirical Manifold Test）**:
  - 自然なInstruct活性化の $D_M$ 経験的パーセンタイル（5%, 25%, 50%, 75%, 95%）とAligned Baseのパーセンタイル順位
  - Two-sample Linear Classifier（Natural vs Aligned）による5分割交差検証AUC（識別不能 $\approx 0.50$ かの検証）
  - Cosine類似度対照群（Matched vs Unmatched vs Natural-Natural）

---

## 3. スクリプトの検証とバグ修正
- **データパスの修正**:
  - 正しいデータセット名 `v3/data/aipsy_strict_expanded.csv` にデフォルト値を更新。
- **構文エラーの解消**:
  - `analyze_alignment_fidelity_and_manifold.py` におけるimport構文エラーを修正し、`format_prompt` 関数を明示定義。
  - `transformers` の非推奨警告（`torch_dtype` -> `dtype`）に対応。
- **構文コンパイル検証**:
  - `.venv/bin/python -m py_compile` により、両スクリプトが正常にコンパイルされることを確認済み。

---

## 4. 実行コマンド
ユーザーがターミナルで実行するコマンド：
```bash
# 仮想環境の有効化
source .venv/bin/activate

# 実験A: Within-model positive control
python v3/scripts/run_within_model_positive_control.py

# 実験B & C: Alignment Fidelity & Manifold Diagnostics
python v3/scripts/analyze_alignment_fidelity_and_manifold.py
```
