# タスクリスト: 本番前最終4大必須修正 (Final Four Blockers Before Production)

## 状況サマリー
前回の修正により、実行不能バグはほぼ全滅し、pytest (115 passed) および全ステージの dry-run は完走。
本番前に修正すべき項目は、論文妥当性2件およびサイレント失敗防止2件の計4件に厳格に絞られた。

---

## タスク一覧

### 1. 【必須】V1 E2 Procrustes の PCA→Procrustes 化 & Geometry 判定修正
- [x] `configs/v1_experiments.yaml` に `procrustes_pca_dim: 64` を追加
- [x] `v1/primary/run_phase_a.py`:
  - [x] 各 fold で $k = \min(\text{procrustes\_pca\_dim}, N_{\text{train}}-1, D)$ で $HR, HS$ の共通 PCA を fit
  - [x] PCA 空間上で $ZR_{\text{tr}}, ZS_{\text{tr}}, ZS_{\text{te}}$ を取得し、直交 Procrustes $Q = U V^T$ を学習
  - [x] Ridge 回帰を $ZR_{\text{tr}}$ 上で学習し、`pred_pca_direct` と `pred_aligned_s_to_r` を算出
  - [x] `r2_pca_direct`, `r2_aligned_transfer`, `alignment_gain = r2_aligned_transfer - r2_pca_direct` を保存
  - [x] `except Exception` による direct cross-decoding へのサイレント fallback を廃止（例外送出）
  - [x] RSA 失敗時は 0.0 ではなく `float("nan")` を保存
  - [x] Geometry パターン判定を `r2_aligned_transfer >= 0.3 and alignment_gain > 0` に修正（RSA 依存を排除）

### 2. 【必須】V3 Confirmatory H1 の D(l) & C(l) Joint Bootstrap 化
- [x] `v3/primary/run_confirmatory_replication.py`:
  - [x] 全層の held-out 予測 `oof_preds_v_by_layer`, `oof_preds_a_by_layer` を保存
  - [x] C(l) を計測した `intervention_sample_indices` (`h1_idx`) と同一の matched support 上で D(l) を算出
  - [x] bootstrap の各反復において同一の pair index を resample し、D(l) と C(l) の両方を同時に再計算して $\Delta d_{\text{peak}}, \Delta d_{\text{center}}$ を算出
  - [x] 全192 pair の D(l) は `d_profile_full_n` として descriptive secondary に保存

### 3. 【必須QA】Behavioral EmoBank summary の入力ゼロ時 FileNotFoundError 化
- [x] `behavioral/analysis/summarize_behavioral_emobank.py`:
  - [x] `if not files:` で `FileNotFoundError` を送出
- [x] `tests/test_pre_production_fixes.py`: 入力ゼロ時に非ゼロ終了コードになるテストを追加

### 4. 【必須QA】V1 Phase C summary の入力ゼロ時 FileNotFoundError 化
- [x] `v1/primary/phase_c/summarize_phase_c.py`:
  - [x] `if not summary_rows:` で `FileNotFoundError` を送出
- [x] `tests/test_pre_production_fixes.py`: 入力ゼロ時に非ゼロ終了コードになるテストを追加

### 5. 整合性・ドキュメント更新
- [x] Sequence-Likelihood 設定の固定プロトコル検証（起動時に `normalize_length=True`, `temperature=1.0` をアサート）
- [x] `v3/README.md`: H3 判定基準に Net attenuation CI下限 > 0 を反映
- [x] `behavioral/README.md`: FDR family に Specificity を明記
- [x] `v1/docs/paper_draft.md`: 729 VAD, 1000 EmoBank等の現行仕様に整合

### 6. テスト・検証 & 結果退避
- [x] 新規テスト追加 (`test_v1_e2_procrustes_pca_n_less_than_d`, `test_v3_confirmatory_h1_joint_bootstrap`, `test_emobank_summary_fails_without_input`, `test_phase_c_summary_fails_without_input`)
- [x] `bash scripts/clear_stage_results.sh` で全結果を安全退避（`archive/results_...`）
