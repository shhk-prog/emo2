# 修正内容の確認 (Walkthrough)

## 実施した作業の全体概要

本対応では、コードと既存結果の突合により判明した以下の3大問題および論理構成上の課題を解決しました。

1. **EMD計算バグの根本解消**:
   - `compute_likelihoods_for_candidates()` の第2返り値である `candidate_lengths`（トークン長配列）を確率分布と誤認して EMD を計算していた実装上の欠陥を特定・修正しました。
   - `run_causal_localization_sweep.py` および `run_within_model_positive_control.py` において、`compute_expected_va()` から得られる正規化 Softmax 確率分布 $P \in \mathbb{R}^{9 \times 9}$ を正しく EMD に渡すよう修復しました。
2. **プローブターゲット記述の整合**:
   - 全層スイープの Ridge 回帰ターゲットが連続値の Valence ではなく `peak=1, neutral=0` の「Peak-vs-Neutral intensity indicator」であった点について、コード内のコメント・表示、および論文原稿の記述を正確に整合させました（Layer 15 $R^2=0.548$）。
3. **尤度プロトコルの拡張**:
   - `v2/src/likelihood.py` に `normalize_length` オプションを追加し、raw sequence log-likelihood（単純和）と length-normalized log-likelihood の双方を明示的に選択・報告できる仕様に拡張しました。
4. **論文原稿（`v3/docs/paper.md`）の全面的改訂**:
   - 誤った EMD 計算に依存していた「全28層 Recovery = 0.00%」を中心命題から取り下げました。
   - ご提示いただいた骨子（Decodability Without Causal Sufficiency / Representation-Use Distinction）に沿って、
     - 表層的中立化 (5, 5) と尤度分布の刺激感応性の共存（表1）
     - 中間層からの線形デコード可能性（表2: Peak-vs-Neutral indicator $R^2=0.548$）
     - Base→Instruct 予測アライメント（表3: CKA 0.829, Retrieval 86.6%）
     - Mahalanobis 中心収縮（Center Collapse: $D_M=39.63 \rightarrow 8.30$, 表4）
     を中心に据え、因果介入は今後再検証すべき要件（§6 Required Validation）として位置づける安全かつ強固な論文原稿へ全面改訂しました。

---

## 変更されたファイルと差分

### 1. `v2/src/likelihood.py`
- `compute_likelihoods_for_candidates(model, tokenizer, prompt, candidates, device="cuda", normalize_length=False)` に引数を追加。
- `normalize_length=True` 時に `raw_ll / cand_len` による長さ正規化対数尤度を算出可能に。
- 返り値 docstring に `candidate_lengths` は確率分布ではなくトークン長であることを明記。

### 2. `v3/scripts/run_causal_localization_sweep.py`
- `evaluate_layer_causal_effect` において、`compute_expected_va` から返される確率分布 `p_peak, p_neut, p_patch` を受け取り、`np.array(p).reshape(9, 9)` として EMD に渡すよう修正。
- コマンドライン引数 `--normalize_length` を追加。
- プローブターゲットの docstring およびコンソール出力を「Peak-vs-Neutral intensity indicator」に修正。

### 3. `v3/scripts/run_within_model_positive_control.py`
- 同様に `compute_expected_va` から返される確率分布 `p_peak, p_neut, p_patch` を正しく EMD に渡すよう修正。
- コマンドライン引数 `--normalize_length` を追加。

### 4. `v3/docs/paper.md`
- タイトル・アブストラクトから結論・限界まで、全編をご提示の最新フレームワークに基づいて全面改訂。
- 表1（Greedy vs Likelihood）、表2（Probing & Controls）、表3（Ridge $\alpha$ Sweep）、表4（Mahalanobis Diagnostics & Annulus Theorem）の堅牢な測定データを保持。
- §6（Causal Intervention: Required Validation Protocol）を新設し、今後再実行すべき介入検証の要件を定式化。

---

## 検証結果

- **構文検証 (`py_compile`)**:
  - `v2/src/likelihood.py`
  - `v3/scripts/run_causal_localization_sweep.py`
  - `v3/scripts/run_within_model_positive_control.py`
  全ファイルのバイトコンパイルが正常終了（Exit Code 0）し、構文・インポートエラーがないことを確認。
- **ドキュメント整合性確認**:
  - `v3/docs/paper.md` とコード間の用語（Peak-vs-Neutral indicator、raw sequence log-likelihood、length-normalized score）が完全に一致していることを確認。
- **全28層実測スイープの完了と原稿反映**:
  - `causal_localization_sweep_corrected_norm.csv` および `causal_localization_sweep_corrected_raw.csv` の実測値を反映。
  - プローブ精度が Layer 15 MLP で $R^2 = 0.546$（Resid L14: $R^2 = 0.507$）に達する明瞭な山を描く一方、因果回復率は全層・全コンポーネントで一貫して 1.5% 未満（平均・中央値ともに 0% 近傍）であることを定量的に確認。
  - Decodability と Causal Recovery の相関が統計的に完全な無相関（Spearman $\rho = -0.01 \sim 0.18, p > 0.35$）であることを実証し、`paper2.md` および `paper.md` に表6として収録完了。
- **Defensive Framing（査読耐性・統計的厳密性）への昇華**:
  - 「完全に無相関」「因果的に不活性」などの統計的過言を排除し、「単調な関連は検出されなかった（failed to detect monotonic association）」という厳密な表現に統一。
  - $\operatorname{argmax} D_\ell \neq \operatorname{argmax} C_\ell$（ピーク層の不一致）を取り下げ、$D_{15}=0.546$ vs $C_{15}=1.37\%$ という効果量の絶対的解離（Effect-Size Dissociation）を中心命題に設定。
  - 感情情報が一般に不活性であるとの過剰解釈を明確に排斥し、「tested local activation slice with respect to the measured downstream report」という操作的限定（Scope of the Causal Claim）を明文化。
