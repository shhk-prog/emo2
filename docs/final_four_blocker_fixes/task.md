# タスクリスト: 本番前最終4大必須修正 (Final Four Blockers Before Production)

## 状況サマリー
前回の修正により、実行不能バグはほぼ全滅し、pytest (115 passed) および全ステージの dry-run は完走。
本番前に修正すべき項目は、論文妥当性2件およびサイレント失敗防止2件の計4件に厳格に絞られた。

---

## タスク一覧

### 1. 【必須】V1 E2 Procrustes の PCA→Procrustes 化 & Geometry 判定修正
- [ ] `configs/v1_experiments.yaml` に `procrustes_pca_dim: 64` を追加
- [ ] `v1/primary/run_phase_a.py`:
  - [ ] 各 fold で $k = \min(\text{procrustes\_pca\_dim}, N_{\text{train}}-1, D)$ で $HR, HS$ の共通 PCA を fit
  - [ ] PCA 空間上で $ZR_{\text{tr}}, ZS_{\text{tr}}, ZS_{\text{te}}$ を取得し、直交 Procrustes $Q = U V^T$ を学習
  - [ ] Ridge 回帰を $ZR_{\text{tr}}$ 上で学習し、`pred_pca_direct` と `pred_aligned_s_to_r` を算出
  - [ ] `r2_pca_direct`, `r2_aligned_transfer`, `alignment_gain = r2_aligned_transfer - r2_pca_direct` を保存
  - [ ] `except Exception` による direct cross-decoding へのサイレント fallback を廃止（例外送出）
  - [ ] RSA 失敗時は 0.0 ではなく `float("nan")` を保存
  - [ ] Geometry パターン判定を `r2_aligned_transfer >= 0.3 and alignment_gain > 0` に修正（RSA 依存を排除）

### 2. 【必須】V3 Confirmatory H1 の D(l) & C(l) Joint Bootstrap 化
- [ ] `v3/primary/run_confirmatory_replication.py`:
  - [ ] 全層の held-out 予測 `oof_preds_v_by_layer`, `oof_preds_a_by_layer` を保存
  - [ ] C(l) を計測した `intervention_sample_indices` (`h1_idx`) と同一の matched support 上で D(l) を算出
  - [ ] bootstrap の各反復において同一の pair index を resample し、D(l) と C(l) の両方を同時に再計算して $\Delta d_{\text{peak}}, \Delta d_{\text{center}}$ を算出
  - [ ] 全192 pair の D(l) は `d_profile_full_n` として descriptive secondary に保存

### 3. 【必須QA】Behavioral EmoBank summary の入力ゼロ時 FileNotFoundError 化
- [ ] `behavioral/analysis/summarize_behavioral_emobank.py`:
  - [ ] `if not files:` で `FileNotFoundError` を送出
- [ ] `tests/test_behavioral_pipeline.py`: 入力ゼロ時に非ゼロ終了コードになるテストを追加

### 4. 【必須QA】V1 Phase C summary の入力ゼロ時 FileNotFoundError 化
- [ ] `v1/primary/phase_c/summarize_phase_c.py`:
  - [ ] `if not summary_rows:` で `FileNotFoundError` を送出
- [ ] `tests/test_v1_pipeline.py`: 入力ゼロ時に非ゼロ終了コードになるテストを追加

### 5. 整合性・ドキュメント更新
- [ ] Sequence-Likelihood 設定の固定プロトコル検証（起動時に `normalize_length=True`, `temperature=1.0` をアサート）
- [ ] `v3/README.md`: H3 判定基準に Net attenuation CI下限 > 0 を反映
- [ ] `behavioral/README.md`: FDR family に Specificity を明記
- [ ] `v1/docs/paper_draft.md`: 729 VAD, 1000 EmoBank等の現行仕様に整合

### 6. テスト・検証 & 結果退避
- [ ] `pytest -q` (115+ passed)
- [ ] `bash scripts/clear_stage_results.sh` で全結果を安全退避
- [ ] 全ステージ dry-run 完走確認
