# 一本化論文に向けた厳密な測定不一致解消と実行停止バグ修正の確認 (Walkthrough)

本作業では、論文の単一ストーリー（$\text{Covariation} \rightarrow \text{Representation} \rightarrow \text{Reorganization} \rightarrow \text{Causal Utilization}$）を堅持しつつ、本番実行時に停止する3系統の致命的バグ（P0）を即時修正し、ステージ間（Behavioral / V1 / V2 / V3）の測定条件の不一致・仮説検証の乖離（P1）を完全に解消しました。

---

## 1. 実施した修正内容の詳細

### 1.1 【P0】実モデル実行停止バグ（3系統）の解消
1. **Behavioral スクリプトの `Optional` 未インポート修正**:
   - 対象: [`behavioral/primary/run_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py), [`behavioral/primary/run_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
   - `from typing import Optional` を追加し、`checkpoint_path: Optional[str] = None` に起因する `--help` 起動時の `NameError` を解消。
2. **V1 の `cuda:0` デバイス判定および dtype 不一致修正**:
   - 対象: [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py), [`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py), [`v1/primary/run_phase_c.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_c.py), [`v1/primary/phase_c/run_e6_specialization.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/phase_c/run_e6_specialization.py)
   - `args.device == "cuda"` の厳格一致から `str(device).startswith("cuda") and torch.cuda.is_available()` 判定へ統一。
   - `cuda:0` 等のデバイス指定時にモデルが CPU 扱いとなり float32 に落ちる問題を解消し、適切に GPU および float16/bfloat16 がロードされるように修正。
3. **V3 Confirmatory の `d_profile_a` 未初期化バグ修正**:
   - 対象: [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
   - `d_profile_v = []` と同位置で `d_profile_a = []` を初期化。実モデル実行時に最初の layer で `NameError` が発生する問題を解消。

---

### 1.2 【P1】Behavioral & V1 測定条件・候補の統一
1. **VAD 候補 JSON の空白有無不一致の解消**:
   - 対象: [`behavioral/primary/run_behavioral_emobank.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_emobank.py), [`behavioral/primary/run_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/primary/run_behavioral_aipsy.py)
   - Behavioral 独自のスペース入り JSON 文字列生成（`{"valence": 1, ...}`）を廃止し、共通モジュール [`src/affective_empathy_eval/likelihood.py::build_vad_candidates()`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/likelihood.py)（compact JSON: `separators=(',', ':')`）に一本化。
   - Sequence Likelihood において空白トークンが確率分布に影響を与えるリスクを完全に排除。
2. **AIPsy summary の固定 5.0 fallback 排除**:
   - 対象: [`behavioral/analysis/summarize_behavioral_aipsy.py`](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
   - neutral baseline が存在しない場合の `else 5.0` fallback を廃止し、`neut_baseline = np.nan` として欠損（NA）を明示化。V3 の固定 5.0 fallback 禁止プロトコルと完全に統一。
3. **V1 Phase A/B/C/E6 の tokenization 条件統一**:
   - 対象: [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py), [`v1/primary/run_phase_b.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_b.py)
   - Phase C と同様に、全 forward pass の tokenization に `add_special_tokens=False` を明示指定。同一プロンプトに対するトークン末尾位置（`prompt_end`）の不一致を防止。
4. **V1 probe の group leakage fallback 禁止**:
   - 対象: [`v1/primary/run_phase_a.py`](file:///mnt/nas/home/hiromi/src/emo2/v1/primary/run_phase_a.py)
   - グループ数が不足した場合に通常の `KFold` / `StratifiedKFold` へフォールバックすることを禁止。`n_splits = min(cv, n_unique_groups)` とし、2未満の場合は NaN（NA）を返してペア間データリークを完全に遮断。

---

### 1.3 【P1】V3 Confirmatory 方向推定プロトコルの Discovery RQ2 準拠化
1. **各 generation stage 局所表現からの方向推定と介入への統一**:
   - 対象: [`v3/primary/run_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_confirmatory_replication.py)
   - 旧実装では prompt-end で学習した固定方向 $d$ を各 generation stage へ transport していたのに対し、**Discovery RQ2 と同様に、各 generation stage 自身の表現から train fold で局所方向 $d_{l,s}$ を推定し、test fold の同一 stage に介入するプロトコルに完全統一**。
   - これにより、Discovery RQ2（「その時点に存在する affect direction はどれだけ因果力を持つか」）と Confirmatory H4 の仮説検証が完全に対をなす真の再現実験となりました。
2. **D-map / C-map の trajectory 差異の明確化**:
   - 対象: [`v3/primary/run_rq2_spatiotemporal_maps.py`](file:///mnt/nas/home/hiromi/src/emo2/v3/primary/run_rq2_spatiotemporal_maps.py)
   - 全候補で trajectory が一致する candidate-independent なステージ（`pre_V`）を Primary とし、`pre_A` 以降は正準中立 teacher-forced trajectory（`{"valence":5,"arousal":5}`）を prefix とする表現であることを docstring および出力メタデータ（`primary_stage`, `subsequent_stages_note`）に明記。

---

### 1.4 【P1】モデル管理・キャッシュ・設定の整理
1. **Model Registry の一本化**:
   - 対象: [`v1/configs/eval_cohort.yaml`](file:///mnt/nas/home/hiromi/src/emo2/v1/configs/eval_cohort.yaml) $\rightarrow$ [`v1/configs/legacy/eval_cohort.yaml`](file:///mnt/nas/home/hiromi/src/emo2/v1/configs/legacy/eval_cohort.yaml)
   - 古いモデルコホート設定ファイルを `legacy/` へ移動し、元ファイルには `configs/models.yaml` を唯一の正本とする deprecation 案内を記載。
2. **V2 RQ1/RQ2 のキャッシュ manifest 検証強化**:
   - 対象: [`v2/primary/run_rq1_rq2_cross_decoding.py`](file:///mnt/nas/home/hiromi/src/emo2/v2/primary/run_rq1_rq2_cross_decoding.py)
   - `is_manifest_matching()` において、`expected_config_hash`, `expected_dataset_hash`, `expected_code_version` 等を厳格に照合。設定やデータセットの変更時に古いキャッシュが誤って再利用されるリスクを排除。

---

### 1.5 【P1】論文構成・図・表記・主張の統一
1. **前提図の改訂**:
   - 対象: [`docs/v3_prerun_five_fixes/paper_outline.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/v3_prerun_five_fixes/paper_outline.md)
   - V1 の結論を先取りしていた直線的図（`Stimulus` $\rightarrow$ `Shared affect representation`）を廃止し、刺激から並列する情報処理経路と、その内部での部分的共有度を問う図に改訂:
     ```text
     Stimulus
       ↓
     Affect-relevant internal states
       ├→ Reader computation
       └→ Self-report computation
     ```
2. **論文 Section 番号およびタイトルの統一**:
   - 対象: [`docs/v3_prerun_five_fixes/paper_outline.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/v3_prerun_five_fixes/paper_outline.md), [`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md)
   - §3. Unified Experimental Framework
   - §4. Behavioral Characterization of Reader–Self Covariation
   - §5. Internal Representation and Causal Sharing（*in Base Models* を削除し、モデル内共有の解明に特化）
   - §6. Post-training-Associated Reorganization
   - §7. From Representation to Causal Utilization
   - §8. Discussion and Limitations
3. **論文の中心的主張文の更新**:
   > *"LLM self-reports cannot be characterized by output-level covariation alone, nor as a uniform direct readout of all decodable affect-relevant information."*  
   > *"LLM self-reports are systematically related to affect-relevant internal representations, but this relationship is partial and task-dependent, is reorganized in association with post-training, and becomes causally consequential only at particular stages of computation."*
4. **729 VAD / 81 VA の比較に関する規約と感度分析の明記**:
   - Stage 間の絶対的な $E[V], E[A]$ の直接比較を禁止。同一刺激サブセットにおいて 729 候補と 81 候補で主要傾向（相関・変位）が保たれる感度分析（Sensitivity Analysis）を明記。

---

## 2. 新規テストの追加

- **[`tests/test_v1_token_and_probe_alignment.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_v1_token_and_probe_alignment.py)**:
  1. `test_v1_prompt_end_invariance_across_phases`: Phase A, Phase B, Phase C, E6 の全 forward pass において、同一 prompt に対する `prompt_end` 位置が `add_special_tokens=False` により完全一致することを決定論的トークナイザーで検証。
  2. `test_v1_probe_group_leakage_fallback_banned`: グループ数不足時（`unique_groups < 2`）に、回帰・分類・Cross-task 一般化 probe が通常 KFold へフォールバックせず正しく NaN を返すことを検証。

---

## 3. 検証手順（ターミナルでの実行推奨）

プロジェクトの仮想環境（`.venv`）を有効化した上で、以下のコマンドを実行してください。

### 3.1 構文チェック
```bash
source .venv/bin/activate
python -m py_compile \
  behavioral/primary/run_behavioral_emobank.py \
  behavioral/primary/run_behavioral_aipsy.py \
  behavioral/analysis/summarize_behavioral_aipsy.py \
  v1/primary/run_phase_a.py \
  v1/primary/run_phase_b.py \
  v1/primary/run_phase_c.py \
  v1/primary/phase_c/run_e6_specialization.py \
  v2/primary/run_rq1_rq2_cross_decoding.py \
  v3/primary/run_rq2_spatiotemporal_maps.py \
  v3/primary/run_confirmatory_replication.py \
  tests/test_v1_token_and_probe_alignment.py
```

### 3.2 エントリーポイントの `--help` 動作確認
（`Optional` の NameError や未定義変数が解消されたことを確認）
```bash
python behavioral/primary/run_behavioral_emobank.py --help
python behavioral/primary/run_behavioral_aipsy.py --help
python v1/primary/run_phase_a.py --help
python v1/primary/run_phase_c.py --help
python v1/primary/phase_c/run_e6_specialization.py --help
python v3/primary/run_confirmatory_replication.py --help
```

### 3.3 単体テストの実行
```bash
python -m pytest tests/test_v1_token_and_probe_alignment.py tests/test_refinement_suite.py tests/test_confirmatory_pipeline.py tests/test_production_entrypoints.py -v
```

### 3.4 統合 dry-run の実行
```bash
python tests/run_all_v3_dryruns.py
```
