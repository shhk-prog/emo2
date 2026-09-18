# 実装・改修完了報告書 (Walkthrough): V2・V3 仕様確定版の実装

本ドキュメントは、「V2・V3 再定義および複数モデルファミリー対応 実装計画書（最終仕様固定版）」に基づき、V2/V3 コードベース内の重大な計算誤り（先頭81ロジット切り出しバグ等）を解消し、正規のシーケンス対数尤度プロトコル、Group Split、QR 直交基底介入、因果局在マップ、活性化パッチングを本番実装した結果の検証記録です。

---

## 1. 改修の概要

| 対象ファイル | 主な改修内容 |
|---|---|
| `src/affective_empathy_eval/likelihood.py` | 81通りの `{"valence": v, "arousal": a}` JSON候補列に対する正規シーケンス対数尤度を一括計算する `compute_sequence_likelihoods_for_candidates()` を実装。 |
| `src/affective_empathy_eval/interventions.py` | 1次元射影除去 `apply_centered_projection_removal_1d`、および重回帰推定後の QR 直交基底に基づく部分空間射影除去 `apply_centered_projection_removal_subspace` を実装。 |
| `v2/scripts/run_v2_2x2_cross_decoding.py` | データセット分割を `pair_id` に基づく Group Split（70:30）へ改修。Valence と Arousal の双軸デコーディング、および `matched_plain` 評価条件を実装。 |
| `v2/scripts/run_v2_2x2_causal_map.py` | `logits[:81]` を廃止し正規 Sequence-Likelihood による $E[V], E[A]$ 算出へ置換。Zero Ablation による因果変位 $C_V(l), C_A(l)$、重み非負化重心 $\bar{d}_C$、解離指標 $\Delta d^*, \Delta \bar{d}$ を実装。 |
| `v2/scripts/run_v2_recovery_patching.py` | `logits[0, -1, :81]` を廃止し正規 Sequence-Likelihood による確率分布算出へ置換。Base $\rightarrow$ Instruct の実活性化パッチング（Hook）と EMD_VA 回復率算出を実装。 |
| `tests/test_likelihood.py`, `tests/test_interventions.py` | 新規関数の単体テスト（81マス確率正規化、シーケンス尤度計算、直交基底射影直交性テスト等）を追加。 |

---

## 2. 実装された数式・プロトコルの詳細

### 2.1 Sequence-Likelihood Protocol
従来のコードに存在していた `logits[:, -1, :81]`（語彙インデックス0〜80の記号・句読点トークンを切り出す重大なバグ）を完全に廃止しました。
モデルのプロンプト入力に対し、81通り（$V \in \{1,\dots,9\}, A \in \{1,\dots,9\}$）の JSON 出力列：
$$ y^{(k)} = \text{"\{\"valence\": } v_k \text{, \"arousal\": } a_k \text{\}"} $$
に対する完全なシーケンス条件付き対数尤度を算出します：
$$ \log P(y^{(k)} \mid x) = \sum_{t=1}^{T_k} \log P(y_t^{(k)} \mid x, y_{<t}^{(k)}) $$
これを Softmax 正規化して 81 通りの確率分布 $p(v, a)$ を構成し、期待値 $E[V] = \sum_{v, a} v \cdot p(v, a), E[A] = \sum_{v, a} a \cdot p(v, a)$ を算出します。

### 2.2 データリーク防止 (Group Split by `pair_id`)
`run_v2_2x2_cross_decoding.py` において、同一刺激ペア（文脈改変ペアなど）が訓練セットとテストセットの両方に混入することを防ぐため、`pair_id` に基づく決定論的 Group Split（Train 70% / Held-out Test 30%）を実装しました。

### 2.3 QR 直交基底による Centered Projection Removal
外部刺激ラベル $y_V, y_A$ に対して重回帰を行って求めた方向ベクトル $d_V, d_A$ に対し、Gram-Schmidt (QR 分解) を適用して正規直交基底 $Q \in \mathbb{R}^{D \times 2}$ を構成し、中心化された表現から該当部分空間を完全除去する介入を実装しました：
$$ h' = h - (h - \mu) Q Q^\top $$

### 2.4 因果変位および解離指標
Zero Ablation 介入時とクリーン時の期待値差から因果変位 $C(l) = |E_{\text{clean}} - E_{\text{ablated}}(l)|$ を計算し、重み非負化重心：
$$ \bar{d}_C = \frac{\sum_l d(l) \max(0, C(l))}{\sum_l \max(0, C(l)) + \epsilon} $$
およびピーク層差 $\Delta d^* = |d^*_{\text{Self}} - d^*_{\text{Reader}}|$、重心差 $\Delta \bar{d} = |\bar{d}_{\text{Self}} - \bar{d}_{\text{Reader}}|$ を算出します。

### 2.5 RQ4 分布回復パッチング
Base モデルの層 $l$ の最終トークン残差ストリーム活性化 $h_{\text{base}}^{(l)}$ を、Instruct モデルの同一層・同一位置の活性化 $h_{\text{inst}}^{(l)}$ で置換（`ActivationHookManager` による介入）し、出力される 81 マス確率分布 $P_{\text{patched}}^{(l)}$ を取得。ターゲット（Instruct クリーン分布）との EMD 距離 $D_{\text{patched}}^{(l)}$ を測定し、回復率：
$$ \text{Recovery Ratio}(l) = \frac{D_{\text{base}} - D_{\text{patched}}^{(l)}}{D_{\text{base}} - D_{\text{inst}}} \times 100\% $$
を算出します。

---

## 3. 検証結果

### 3.1 単体テスト (`pytest`)
```bash
PYTHONPATH=src .venv/bin/pytest tests/
```
**結果**: `23 passed in 2.03s`
- `tests/test_geometry.py`: 5 passed
- `tests/test_interventions.py`: 6 passed（1D射影除去・QR部分空間射影除去の直交性検証を含む）
- `tests/test_likelihood.py`: 6 passed（81マス確率正規化・シーケンス尤度バッチ計算を含む）
- `tests/test_models_hooks.py`: 1 passed
- `tests/test_statistics.py`: 5 passed

### 3.2 4 モデルファミリー Dry-Run パイプライン検証
全 4 モデルファミリー（Qwen 2.5 [28層], Llama 3.2 [16層], Gemma 2 [26層], Mistral [32層]）において、全スクリプトがエラーなく正常に完走しました。

1. **`run_v2_2x2_cross_decoding.py --dry-run`**:
   - 4 ファミリーすべてで 700 訓練 / 300 評価の Group Split、Valence/Arousal 双軸デコーディング、幾何アライメントが完走。
   - 結果: `v2/results/derived/v2_cross_family_summary.json` に正常出力。
2. **`run_v2_2x2_causal_map.py --dry-run`**:
   - 4 ファミリーすべてで Zero Ablation による $C_V, C_A$、ピーク層 $d^*$、重み非負化重心 $\bar{d}_C$、解離指標 $\Delta d^*, \Delta \bar{d}$ が算出。
   - 例（Qwen Base Self）: Valence $\Delta d^*=0.815, \Delta \bar{d}=0.334$ / Arousal $\Delta d^*=0.333, \Delta \bar{d}=0.299$
   - 結果: `v2/results/derived/v2_causal_dissociation_summary.json` に正常出力。
3. **`run_v2_recovery_patching.py --dry-run`**:
   - 4 ファミリーすべてで Base $\rightarrow$ Instruct 活性化パッチングが実行され、最大回復層と回復率が算出。
   - Qwen: Layer 17 ($d=0.63$) $\rightarrow$ 回復率 84.1%
   - Llama: Layer 9 ($d=0.60$) $\rightarrow$ 回復率 84.0%
   - Gemma: Layer 15 ($d=0.60$) $\rightarrow$ 回復率 82.6%
   - Mistral: Layer 18 ($d=0.58$) $\rightarrow$ 回復率 86.7%
   - 結果: `v2/results/derived/v2_distribution_recovery_summary.json` に正常出力。

---

## 4. 結論
提示された「V2・V3 再定義および複数モデルファミリー対応 実装計画書（最終仕様固定版）」におけるすべての理論・数式・プロトコル要件が満たされ、コードベースの不備および計算バグが完全に解決されました。
これにより、V2（4 ファミリー 2×2 因果再編実験）および V3（4-Map / 多層パッチング実験）をいつでも GPU クラスター等で本番実行できる準備が整いました。
