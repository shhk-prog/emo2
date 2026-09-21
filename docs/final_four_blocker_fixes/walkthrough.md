# 本番実行前 4大必須修正・検証完了報告 (Walkthrough)

本番全実行を安全かつ学術的厳密性を担保して開始するために指定された **4大必須修正**、**プロトコル固定**、および **ドキュメント整合** の全作業が完了しました。

---

## 1. 実施した修正の詳細

### ① 【必須】V1 E2 PCA-Procrustes & Alignable Geometry 判定の修正
- **問題点**: $N_{\text{train}} \approx 800 < D$ (1000〜4000) の高次元条件下でフル次元直交Procrustesの回転がnull space側で同定不能であったこと、失敗時にDirect Cross-Decodingへサイレントフォールバックしていたこと、およびRSA失敗時に0.0を返していたこと。
- **実施した修正**:
  - `configs/v1_experiments.yaml`: `phase_a.procrustes_pca_dim: 64` を追加。
  - `v1/primary/run_phase_a.py` の `evaluate_cross_decoding_and_geometry()`:
    1. 各foldで $k = \min(\text{procrustes\_pca\_dim}, N_{\text{train}}-1, D)$ を求め、$\begin{bmatrix} H_R^{\text{train}} \\ H_S^{\text{train}} \end{bmatrix}$ に合同PCAを適合。
    2. PCA空間上で直交Procrustes行列 $Q = U V^\top$（SVD）を同定。
    3. 同一PCA空間上で Ridge 分類器を学習し、`r2_pca_direct` と `r2_aligned_transfer` を算出。
    4. `alignment_gain = r2_aligned_transfer - r2_pca_direct` を算出・保存。
    5. サイレントフォールバック（`except Exception: pred_aligned = pred_direct`）を完全撤廃。
    6. RSA失敗時は `0.0` ではなく `float("nan")` を返却。
    7. Geometry パターン判定を論文定義に整合:
       - `direct_transfer >= 0.3` $\rightarrow$ `"Operational: Shared Geometry"`
       - `r2_aligned_transfer >= 0.3 and alignment_gain > 0` $\rightarrow$ `"Operational: Alignable Geometry"`
       - それ以外 $\rightarrow$ `"Operational: Task-Divergent Geometry"`

---

### ② 【必須】V3 Confirmatory H1 の Joint Bootstrap 化
- **問題点**: $D(l)$ を全サンプル固定とし、$C(l)$ の標本変動のみをブートストラップしていたため、$\Delta d = d_C - d_D$ のCIが $D(l)$ の不確実性を無視していたこと。
- **実施した修正**:
  - `v3/primary/run_confirmatory_replication.py`:
    1. 各層の全サンプル OOF 予測 `oof_preds_v_by_layer[l]`, `oof_preds_a_by_layer[l]` を記録。
    2. 全体192 pairの $D(l)$ は `d_profile_full_n` として descriptive secondary に保存。
    3. Confirmatory H1 については、$C(l)$ を評価した介入サンプル集合 `h1_idx = np.asarray(intervention_sample_indices, dtype=int)` と同一の held-out pair 集合上で matched-support $D(l)$ を算出。
    4. ブートストラップループ内で同一のサンプルインデックス `pos = rng.integers(0, n_h1, size=n_h1)` をリサンプリングし、$D(l)$ と $C(l)$ を同時に再計算。
    5. `delta_d_peak` と `delta_d_center` の 95% CI を $D(l)$ と $C(l)$ の同時標本変動を含む形で算出。

---

### ③ 【必須QA】Behavioral EmoBank summary の空入力時例外化
- **問題点**: upstream が結果を出力しなかった場合でも `print` して exit code 0 で終了し、統合ランナーから「成功」と誤認されるサイレント不全リスクがあったこと。
- **実施した修正**:
  - `behavioral/analysis/summarize_behavioral_emobank.py`:
    `if not files:` の分岐で `print` ではなく `raise FileNotFoundError("No Behavioral EmoBank result CSVs found in ...")` を送出。
  - `tests/test_pre_production_fixes.py` に `test_emobank_summary_fails_without_input()` を追加（空入力時に非ゼロ終了コードとなることを自動検証）。

---

### ④ 【必須QA】V1 Phase C summary の空入力時例外化
- **問題点**: Phase C のモデル出力が空の場合でも exit code 0 で正常終了していたこと。
- **実施した修正**:
  - `v1/primary/phase_c/summarize_phase_c.py`:
    `if not summary_rows:` の分岐で `raise FileNotFoundError("No Phase C model outputs found in ...")` を送出。
  - `tests/test_pre_production_fixes.py` に `test_phase_c_summary_fails_without_input()` を追加。

---

### ⑤ Sequence-Likelihood 固定プロトコル検証アサーション
- **実施した修正**:
  - `affective_empathy_eval/likelihood.py` に `validate_sequence_likelihood_protocol()` を追加。
  - `compute_sequence_likelihoods_for_candidates()` の開始時に、`normalize_length=True` かつ `temperature=1.0` を強制検証し、それ以外の値が渡された場合は `ValueError` を送出。
  - これにより、コードベース全体（40箇所以上の呼び出し）が自動的に固定測定プロトコル（Token mean, $T=1.0$）に統一され、意図しない設定変更やドリフトを恒久的に防止。

---

### ⑥ ドキュメントおよび README の整合
- `v3/README.md`: H3 Endogenous Relevance の判定基準を「absolute mediated attenuation の bootstrap 95% CI 下限が閾値を超え、かつ matched-rank random 2D subspace control に対する Net attenuation の 95% CI 下限が 0 を超える」に更新。
- `behavioral/README.md`: FDR family に `Specificity` を追加（Sensitivity, Dose-Response, Specificity, Coupling）。
- `v1/docs/paper_draft.md`: 729 VAD 候補空間、EmoBank Primary 1,000 刺激、revision 固定、精度設定（bfloat16/float16）などの現行プロトコルと完全に整合させ刷新。

---

## 2. 検証結果

- **単体テスト追加**:
  - `tests/test_pre_production_fixes.py`:
    - `test_emobank_summary_fails_without_input` (空入力検知)
    - `test_phase_c_summary_fails_without_input` (空入力検知)
    - `test_v1_e2_procrustes_pca_n_less_than_d` ($N=40 < D=128$ の高次元条件下で有限の held-out 評価値・gain を算出することを検証)
  - `tests/test_confirmatory_pipeline.py`:
    - `test_v3_confirmatory_h1_joint_bootstrap` (D(l)とC(l)の同時標本変動による95% CI算出を検証)
- **結果ディレクトリのクリーン化**:
  - 既存の結果は `archive/results_<timestamp>/` 配下へ退避済みであり、本番実行のための準備が完全に整いました。
