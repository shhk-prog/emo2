# 実装計画: 本番再実行前の最終厳密化 (Final Pre-Rerun Rigorous Alignment)

本計画は、論文の科学的ストーリー（**Behavioral Covariation $\rightarrow$ V1 Partial Representation/Causal Overlap $\rightarrow$ V2 Post-training-Associated Reorganization $\rightarrow$ V3 Localized Causal Leverage**）を完全に担保した上で、本番実験再実行前に解消すべき残る15項目の実装課題を包括的に解決するためのものです。

---

## ユーザー確認・承認事項

> [!IMPORTANT]
> - 本修正では、論文ストーリーの変更や新しいRQの追加は一切行いません。
> - 各 Stage の定義を最後まで一貫させ、推定の厳密性と本番データの安全性を確保します。
> - 修正完了後は、V1 Phase C (E3/E4, E6, summary)、V2 RQ3、V2 Confirmatory、V3 RQ1/RQ2/RQ3、V3 Confirmatory を再実行可能な状態にします（Behavioral、V1 Phase A/B、V2 RQ1/RQ2 は再実行不要）。

---

## 変更内容詳細

### 1. 【P0】V1 Phase C の layer indexing 1層ずれ修正
- **対象**: `v1/primary/run_phase_c.py`
- **問題**: Hugging Face の `outputs.hidden_states` は長さ $L+1$ で、`[0]` が embedding output、`[1]` が Transformer block 0、`[L]` が Transformer block $L-1$ です。現在、`outputs.hidden_states[l]` の差分を `layer_idx=l` の Transformer block へ介入しているため、厳密に1層ずれていました。
- **修正**:
  - `get_block_hidden_state(outputs.hidden_states, block_idx)` または `outputs.hidden_states[block_idx + 1]` を使用し、各 Transformer block（$0 \dots L-1$）の出力を正しく `all_layer_vecs[block_idx]` に格納。
- **テスト追加**: `tests/test_refinement_suite.py`（または新規テスト）に `test_phase_c_block_index_mapping()` を追加し、抽出された各層ベクトルが Transformer block 出力と一致することをアサート。

### 2. 【P0】V3 Confirmatory H2 を matched-neutral への注入に修正
- **対象**: `v3/primary/run_confirmatory_replication.py`
- **問題**: RQ1 では Sufficiency を「中立文への方向注入」と再定義したのに対し、Confirmatory H2 では依然として affective stimulus 自体に注入していました。
- **修正**:
  - `neu_text = resolve_matched_neutral_text(row, df)` により対応する中立文を取得。
  - `prompt_neu_self` を構築し、未介入の `clean_neu_ev, clean_neu_ea` を基準として、注入後の変位 `shift_v = injected_ev - clean_neu_ev`, `shift_a = injected_ea - clean_neu_ea` を測定。
  - H3（必然性 / Endogenous relevance）は、`prompt_aff_self` からの 2D 情動部分空間除去による減衰測定のまま維持。

### 3. 【P0】V3 Confirmatory H2 の介入層を RQ1 と統一
- **対象**: `v3/primary/run_confirmatory_replication.py`, `configs/v3_experiments.yaml`
- **問題**: RQ1 では relative depth $\approx 0.5$ を prespecified site として十分性を検証しているのに対し、Confirmatory では `int(num_layers * 0.65)` のような別層を使っていました。
- **修正**:
  - `configs/v3_experiments.yaml` の `confirmatory` セクションに `sufficiency_relative_depth: 0.5` を明記。
  - コード側で `sufficiency_layer = round(v3_cfg.get("confirmatory", {}).get("sufficiency_relative_depth", 0.5) * (num_layers - 1))` を算出して H2 に適用。
  - 変数名も `sufficiency_layer`, `temporal_map_layer`, `mediation_layer` に目的別に分離。

### 4. 【P0】V2 RQ3 に matched-plain causal map を追加
- **対象**: `v2/primary/run_rq3_causal_map.py`
- **問題**: 現在 Causal map は Base plain vs Instruct native-chat のみ比較しており、因果的差異が post-training のものか chat template の差異かが分離できていませんでした。
- **修正**:
  - 条件を 3 条件に拡張:
    1. `Base plain` (`alignment="base"`, `format_condition="plain"`, `format_type="plain"`)
    2. `Instruct matched-plain` (`alignment="inst"`, `format_condition="matched_plain"`, `format_type="plain"`)
    3. `Instruct native-chat` (`alignment="inst"`, `format_condition="native_chat"`, `format_type="chat"`)
  - pair-level CSV に `format_condition` 列を追加保存。

### 5. 【P0】V1 dry-run の出力先ディレクトリ完全分離
- **対象**: `v1/primary/run_phase_a.py`, `run_phase_b.py`, `run_phase_c.py`, `run_e6_steering.py`, `phase_c/summarize_phase_c.py`, `src/affective_empathy_eval/run.py`
- **問題**: V1 のスクリプトで `--dry-run` を指定した場合でも本番結果ディレクトリ `v1/results/derived/...` に書き込まれていました。
- **修正**:
  - `args.dry_run` 時に `out_dir` を `v1/results/derived/dry_run/...` または `v1/results/dry_run/...` に自動分岐。
  - `summarize_phase_c.py` にも `--dry-run` 引数を追加し、`run.py` から連携。

### 6. 【P1】V2 Confirmatory H3 を matched-plain Primary にする
- **対象**: `v2/primary/run_confirmatory_analysis.py`
- **修正**:
  - `v2_causal_pair_level.csv` から `(alignment=="base" & format_condition=="plain")` と `(alignment=="inst" & format_condition=="matched_plain")` を抽出して Primary LMM を fit。
  - native-chat 条件は Secondary 解析として別途集約。

### 7. 【P1】V2 Confirmatory H4 も matched-plain recovery AUC を Primary にする
- **対象**: `v2/primary/run_confirmatory_analysis.py`
- **修正**:
  - Primary: `auc_recovery_matched_plain` の Self vs Reader 差分および 95% Bootstrap CI。
  - Secondary: `auc_recovery` (native-chat)。
  - Mechanistic control: `auc_recovery_aligned`。

### 8. 【P1】V3 RQ2 の $C(l,t)$ を $\alpha=1.0$ (reference alpha) に固定
- **対象**: `v3/primary/run_rq2_spatiotemporal_maps.py`, `configs/v3_experiments.yaml`
- **修正**:
  - config に `causal_reference_alpha: 1.0` を明記。
  - $\alpha$ sweep の末尾ではなく、$\alpha=1.0$ のインデックスから $C(l,t)$ を算出。$\gamma(l,t)$ は全 sweep の slope。
  - JSON に `"causal_reference_alpha": 1.0` を保存。

### 9. 【P1】V3 RQ1 の attenuation ratio で natural shift < 0.05 を 0 とせず除外
- **対象**: `v3/primary/run_rq1_state_induction.py`
- **修正**:
  - `MIN_NATURAL_SHIFT = 0.05`。natural shift がこれ以下のサンプルは ratio 計算対象外とし、`n_total, n_valid_ratio_v, n_valid_ratio_a` を保存。
  - Primary は absolute attenuation、Secondary は valid sample における attenuation ratio とする。

### 10. 【P1】valid ratio sample が 0件のとき effect=0 ではなく NaN にする
- **対象**: `v3/primary/run_rq3_path_mediation.py`, `v3/primary/run_confirmatory_replication.py`
- **修正**:
  - 分母が有効なサンプルが 0 件の場合は `ratio = np.nan`, `status = "insufficient_valid_samples"` とする。

### 11. 【P1】V2 H1a geometry で matched が無い場合 native へ fallback せず RuntimeError
- **対象**: `v2/primary/run_confirmatory_analysis.py`
- **修正**:
  - `reader_distortion_matched`, `self_distortion_matched`, `rsa_reader_matched`, `rsa_self_matched` が欠損している場合は `RuntimeError` を発生させ、厳格に matched-plain のみを Primary とする。

### 12. 【P1】RSA は similarity として名称・解釈を揃える
- **対象**: `v2/primary/run_confirmatory_analysis.py`, `v2/primary/run_rq1_rq2_cross_decoding.py`
- **修正**:
  - 相関係数は値が大きいほど幾何構造が保持されているため、レポートおよび出力辞書で `rsa_similarity` と明記。

### 13. 【P1】manifest / cache validation の厳格化
- **対象**: `src/affective_empathy_eval/manifests.py`, 各 primary スクリプト
- **修正**:
  - 実験に影響する config 全体（辞書）を `create_run_manifest` に渡し、`config_hash` を厳密に生成・照合。

### 14. 【P2】`tests/run_all_v3_dryruns.py` を production runner 呼び出し形式に改修
- **対象**: `tests/run_all_v3_dryruns.py`
- **修正**:
  - 古いモジュール名の直接 import を廃止し、`python -m affective_empathy_eval.run --stage v3 --family qwen --dry-run --device cpu` を実行して終了コードを検証する方式に改修。

### 15. 【P2】V3 $\beta$-map の emotion covariate 重複防止
- **対象**: `src/affective_empathy_eval/data.py`
- **修正**:
  - `target_emotion` があればそれを使い、なければ `emotion` を使うように優先度をつけて単一の感情列のみを採用。

---

## 検証計画

### 1. ユニットテストおよび構文検証
```bash
python -m py_compile v1/primary/run_phase_c.py v2/primary/run_rq3_causal_map.py v2/primary/run_confirmatory_analysis.py v3/primary/run_rq1_state_induction.py v3/primary/run_rq2_spatiotemporal_maps.py v3/primary/run_rq3_path_mediation.py v3/primary/run_confirmatory_replication.py
pytest tests/ -q
```
- 新設テスト `test_phase_c_block_index_mapping` を含む全テストが通過することを確認。

### 2. 統合ランナー dry-run フル実行
```bash
python -m affective_empathy_eval.run --stage all --dry-run --max-samples 2 --model-set primary_small --force-after-no-go
```
- Behavioral, V1, V2, V3 の全ステージがエラーなく通過することを確認。
- 本番成果物ディレクトリ（`results/` 配下）に一切の未追跡ファイルや上書きが発生せず、すべて `dry_run/` 配下に出力されていることを確認。
