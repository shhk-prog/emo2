# 一本化論文に向けた厳密な測定不一致解消と実行停止バグ修正の実装計画

本計画は、論文の単一ストーリー（$\text{Covariation} \rightarrow \text{Representation} \rightarrow \text{Reorganization} \rightarrow \text{Causal Utilization}$）を堅持しつつ、本番実行時に停止する3系統の致命的バグ（P0）を即時修正し、ステージ間（Behavioral / V1 / V2 / V3）の測定不一致・仮説検証の乖離（P1）を完全に解消することを目的とします。

## User Review Required

> [!IMPORTANT]
> **V3 Confirmatory (H4) の方向推定プロトコルの統一**
> 旧コードでは、Discovery RQ2 が各 layer $\times$ generation stage の局所活性化から方向 $d_{l,t}$ を推定していたのに対し、Confirmatory では prompt-end で推定した固定方向を後続 stage へ transport していました。
> 本計画ではユーザー指示に基づき、**Confirmatory でも train fold 内で layer $\times$ stage ごとに direction を推定し、held-out test fold の同一 layer $\times$ stage へ介入する局所プロトコルへ統一**します。これにより Discovery と完全に対をなす真の再現検証となります。

> [!IMPORTANT]
> **論文 Section 番号および中心的主張の確定**
> - **Section 構成**:
>   - §3. Unified Experimental Framework
>   - §4. Behavioral Characterization of Reader–Self Covariation
>   - §5. Internal Representation and Causal Sharing（*in Base Models* は削除）
>   - §6. Post-training-Associated Reorganization
>   - §7. From Representation to Causal Utilization
>   - §8. Discussion and Limitations
> - **中心主張文**:
>   > *"LLM self-reports cannot be characterized by output-level covariation alone, nor as a uniform direct readout of all decodable affect-relevant information."*
>   > *"LLM self-reports are systematically related to affect-relevant internal representations, but this relationship is partial and task-dependent, is reorganized in association with post-training, and becomes causally consequential only at particular stages of computation."*
> - **仮説図**:
>   `Stimulus` $\rightarrow$ `Shared affect representation` のような結論前提の直線的図を廃止し、刺激から独立に並列する情報処理経路と、その内部での部分的共有度を問う図へ改訂します。

## Open Questions
現時点で未解決の疑問点はありません。提示された全15項目について、コードベースとの整合性を完全に確認済みです。

---

## Proposed Changes

### Component 1: Behavioral Primary & Analysis

#### [MODIFY] [run_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py)
- `from typing import Optional` をインポートし、`checkpoint_path: Optional[str] = None` による `NameError` を解消。
- 独自実装 `build_vad_candidates()`（スペース入りフォーマット）を削除し、`src/affective_empathy_eval/likelihood.py::build_vad_candidates()`（compact JSON, `separators=(',', ':')`）に一本化。

#### [MODIFY] [run_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
- `from typing import Optional` をインポートし、`--help` 起動時の `NameError` を解消。
- 独自 VAD candidate 文字列生成を廃止し、`src/affective_empathy_eval/likelihood.py::build_vad_candidates()` に一本化。

#### [MODIFY] [summarize_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
- matched neutral が存在しない場合の `else 5.0` 固定フォールバックを削除し、欠損（NaN / エラー）として明示的に処理。

---

### Component 2: V1 Representation & Causal Sharing

#### [MODIFY] [run_phase_a.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
- `args.device == "cuda"` を `str(args.device).startswith("cuda")`（または共通判定）に修正し、`cuda:0` 等の指定時でも正常に GPU および float16 が適用されるように修正。
- forward pass の tokenization に `add_special_tokens=False` を明示指定。
- probe の group split において、グループ数不足時に通常の `KFold` / `StratifiedKFold` へフォールバックすることを禁止。`n_splits = min(cv, n_unique_groups)` とし、2未満の場合は NA / スキップとしてデータリークを防止。

#### [MODIFY] [run_phase_b.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
- デバイス判定を `str(device).startswith("cuda")` に統一。
- tokenization に `add_special_tokens=False` を明示指定。

#### [MODIFY] [run_phase_c.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py)
- `args.device == "cuda"` を `str(args.device).startswith("cuda")` に修正。

#### [MODIFY] [run_e6_specialization.py](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
- `args.device == "cuda"` を `str(args.device).startswith("cuda")` に修正。

---

### Component 3: V2 Post-training Reorganization

#### [MODIFY] [run_rq1_rq2_cross_decoding.py](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py)
- キャッシュ判定において `is_manifest_matching()` に `expected_config_hash`, `expected_dataset_hash`, `expected_code_version`, `expected_prompt_version` などを渡し、厳格なキャッシュ検証を実施。

---

### Component 4: V3 Causal Utilization & Confirmatory Replication

#### [MODIFY] [run_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
- `d_profile_a = []` の未初期化バグを修正（`d_profile_v = []` と同位置で初期化）。
- H4 (Temporal Emergence) において、prompt-end 方向の transport ではなく、Discovery RQ2 と同様に各 layer $\times$ generation stage の局所表現から学習した方向 $d_{l,s}$ を推定し、held-out test サンプルの同一 stage に介入する局所プロトコルへ完全統一。

#### [MODIFY] [run_rq2_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
- D-map と C-map の trajectory 不一致について、`pre_V` などの candidate-independent stage を Primary とし、それ以降は canonical neutral teacher-forced trajectory であることを明記・整理。

---

### Component 5: Configuration & Legacy Cleanup

#### [DELETE] [eval_cohort.yaml](file:///mnt/nas/home/hiromi/src/emo2/v1/configs/eval_cohort.yaml)
- 古い設定ファイル `v1/configs/eval_cohort.yaml` を `v1/configs/legacy/eval_cohort.yaml` へ移動。`configs/models.yaml` を唯一のモデル定義として確定。

---

### Component 6: Documentation & Outline Synchronization

#### [MODIFY] [paper_outline.md](file:///mnt/nas/home/hiromi/src/emo2/docs/v3_prerun_five_fixes/paper_outline.md)
- 「Shared representation を前提にした図」を、刺激から分岐する並列計算から共有度を検証する論理図に改訂。
- Section 番号を統一（§3 Unified Framework, §4 Behavioral, §5 V1 Representation Sharing, §6 V2 Reorganization, §7 V3 Causal Utilization, §8 Discussion）。
- 中心主張文の更新、および 729 VAD / 81 VA の感度分析の補足と絶対値比較不可の明記。

#### [MODIFY] [README.md](file:///mnt/nas/home/hiromi/src/emo2/README.md)
- Section 番号を paper_outline と統一（§3〜§8）。
- V1 の見出しから `in Base Models` を削除。
- 論文の中心的主張文を更新。

---

### Component 7: Unit Tests & Smoke Tests

#### [NEW] [test_v1_token_and_probe_alignment.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_v1_token_and_probe_alignment.py)
- Phase A, Phase B, Phase C で同一 prompt に対する `prompt_end` トークン位置が完全に一致することを検証するテスト。
- Phase A の probe において、グループ数不足時に通常 KFold への fallback が発生せずリークが防がれていることを検証するテスト。

---

## Verification Plan

### Automated Tests
1. **構文チェック (py_compile)**:
   ```bash
   .venv/bin/python -m py_compile \
     behavioral/primary/run_behavioral_emobank.py \
     behavioral/primary/run_behavioral_aipsy.py \
     behavioral/analysis/summarize_behavioral_aipsy.py \
     v1/primary/run_phase_a.py \
     v1/primary/run_phase_b.py \
     v1/primary/run_phase_c.py \
     v1/primary/phase_c/run_e6_specialization.py \
     v2/primary/run_rq1_rq2_cross_decoding.py \
     v3/primary/run_rq2_spatiotemporal_maps.py \
     v3/primary/run_confirmatory_replication.py
   ```
2. **全エントリーポイントの `--help` 動作検証**:
   - `python behavioral/primary/run_behavioral_emobank.py --help`
   - `python behavioral/primary/run_behavioral_aipsy.py --help`
   - `python v1/primary/run_phase_a.py --help`
   - `python v1/primary/run_phase_c.py --help`
   - `python v1/primary/phase_c/run_e6_specialization.py --help`
   - `python v3/primary/run_confirmatory_replication.py --help`
   （NameError や未定義変数が解消されたことを確認）
3. **pytest スイート**:
   ```bash
   .venv/bin/python -m pytest tests/test_v1_token_and_probe_alignment.py tests/test_refinement_suite.py tests/test_confirmatory_pipeline.py tests/test_production_entrypoints.py -v
   ```
4. **統合 dry-run 検証**:
   ```bash
   .venv/bin/python tests/run_all_v3_dryruns.py
   ```

### Manual Verification
- `docs/v3_prerun_five_fixes/paper_outline.md` および `README.md` の Section 番号と主張文が整合していることを目視確認。
- `v1/configs/legacy/eval_cohort.yaml` への移動と `configs/models.yaml` の唯一性を確認。
