# Implementation Plan: V3 and Cross-Stage Final Refinements

ユーザーからの最終レビューに基づき、**V3 (Topic control, Confirmatory実測化, Group split, Necessity基準, 図表固定値排除)** を最優先とし、**V1 (E6 Discovery拡大, CLI自然化, Cosine区別)**、**V2 (Legacy隔離, RQ4 native/matched分離, Descriptive位置づけ)**、**Behavioral & 全体テスト (警告解消, 4ブロック構成)** を包括的に修正します。

---

## 1. User Review Required

> [!IMPORTANT]
> - **V3 Confirmatory Replication**: 以前のコードにあった人工的指数関数 $C(l) = |\mathrm{shift}|\exp(\cdots)$ や固定値 `attenuated_shift = natural_shift * 0.45` を完全に削除し、真の実モデル forward + 介入パイプラインに刷新します。GPU がない環境（または `--dry-run`）では、明確に `is_simulation: true` のタグを付与し、論文 Primary 結果への混入を防ぎます。
> - **V3 Necessity 基準**: 自然変位を $|E[V] - 5|$（中立からの乖離）ではなく、matched-neutral baseline との差 $|E[V]_{\mathrm{aff}} - E[V]_{\mathrm{neutral}}|$ を Primary に切り替えます。
> - **V1 E6**: Discovery サンプル数を 10 pair から 50 pair（または split 全体）に拡大し、CLI 引数を `--no-split-eval` に自然化します。

---

## 2. Proposed Changes

### V3: 因果解析・統計・プロットの抜本的修正 (最優先)

#### [MODIFY] [run_v3_state_induction.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_state_induction.py)
1. **Topic control の clean forward バグ修正**:
   - `with patch hook:` 内で `out_base_c` と `out_patch_c` の両方を計算していたバグを修正。
   - `out_base_c` を hook の**外**で計算し、`out_patch_c` だけを hook 内で計算することで、純粋な介入差分を測定。
2. **Train/Test を pair-group split 化**:
   - `pair_id` が存在する場合、ペアが Train と Test に跨がらないよう `unique_pair_ids` に基づく 50/50 Group split（または `GroupShuffleSplit`）に変更。
3. **Necessity 基準の改訂**:
   - 自然変位を $|E[V] - 5|$ ではなく、刺激の対となる matched neutral baseline との差 $|E[V]_{\mathrm{aff}} - E[V]_{\mathrm{neutral}}|$ を Primary 指標として計算。

#### [MODIFY] [run_v3_spatiotemporal_maps.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_spatiotemporal_maps.py)
1. **$D(l,t)$ の held-out 評価の厳密化**:
   - 線形プローブの予測性能 $D(l,t)$ が train/test 分割（5-fold cross-validation または 70/30 held-out split）で計算されていることを確認・保証。
2. **$\beta(l,t)$ の偏回帰係数算出**:
   - 刺激長・感情強度・基本共変量を統制した重回帰において、`internal score -> report` の偏回帰係数として $\beta(l,t)$ を厳密に計算。

#### [MODIFY] [run_v3_confirmatory_replication.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_confirmatory_replication.py)
1. **人工関数・固定値の完全排除**:
   - `c_profile_v` を人工指数関数から生成するロジック、および `attenuated_shift = natural_shift * 0.45` などの固定乗算を完全削除。
   - H1 (全層 D と C の実測)、H2 (実測 alpha sweep)、H3 (実測 projection removal による消去)、H4 (実測 generation-stage intervention) の実モデル推論コードに置き換え。
   - `--dry-run` 実行時は明確に `is_simulation: true` および `note: "MOCK SIMULATION - NOT FOR PAPER PRIMARY RESULTS"` を出力。

#### [MODIFY] [plot_paper_figures.py](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/plot_paper_figures.py)
1. **固定値・乱数の完全排除**:
   - 行357の `collapse_rates = [12.4, 98.6]` などの固定配列を削除。
   - すべてのプロット関数について、実測 JSON / CSV（`v3/results/derived/` 等）が存在する場合にのみ実測値から描画し、ファイルが存在しない場合はダミーを描かずスキップまたは警告を出す設計に変更。

---

### V1: E6 規模拡大・CLI 改善・表記統一

#### [MODIFY] [run_v1_phase_c_targeted_ablation.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py)
1. **Discovery split のサンプル数拡大**:
   - `sub_idx = disc_indices[:min(10, len(disc_indices))]` を、デフォルトで 50 pair（または `--discovery-sample-size` 指定、利用可能全数）に拡大。
2. **CLI 引数の自然化**:
   - `--split-eval` (default=True) を廃止し、`--no-split-eval` (`action="store_false"`, `dest="split_eval"`) に変更。
3. **論文表記の統一**:
   - "Double Dissociation" という過剰な表現をコード内のログ・コメント・出力サマリーから排除し、**"Task-specific causal specialization"** に統一。
4. **本番実行 `--limit` チェック**:
   - 実験実行時に `--limit` が指定されている場合は pilot である旨を明記し、本番集計時は `--limit 0` (全件) を標準とする。

#### [MODIFY] [generate_phase_c_report.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_c_report.py)
1. **E3 Cosine の区別**:
   - Mean vector cosine と Pairwise cosine をレポート上で明確に分離して出力。

---

### V2: Legacy スクリプトの完全隔離と RQ4 出力整理

#### [MODIFY] [v2/scripts/run_annotation_proxy.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_annotation_proxy.py) $\to$ [MOVE] `v2/scripts/legacy/`
- 乱数疑似ラベル生成スクリプトを `legacy/` に移動し、Primary パイプラインから除外。

#### [MODIFY] [v2/scripts/run_rsa_and_controlled_coupling.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_rsa_and_controlled_coupling.py) $\to$ [MOVE] `v2/scripts/legacy/`
- ランダム射影のプレースホルダー方向が含まれているため、Primary パイプライン（`run_mixed_effects_coupling.py`）と分離し `legacy/` に移動。

#### [MODIFY] [run_v2_recovery_patching.py](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_recovery_patching.py)
1. **Native と Matched-plain の明確な分離**:
   - 最終 JSON および集計結果において、`recovery_ratios_native` と `recovery_ratios_matched_plain` を明確に独立したブロックとして出力。
2. **Cross-family 記述的集計**:
   - 4ファミリーの集計は $n=4$ の記述統計（Descriptive mean/std）であることを明記し、強い一般化結論や過剰な p値解釈を行わないよう注記。

---

### Behavioral & 共通テスト: 警告解消とドキュメント整理

#### [MODIFY] [v1/src/affective_empathy_eval/metrics.py](file:///mnt/nas/home/hiromi/src/emo/v1/src/affective_empathy_eval/metrics.py)
1. **定数配列に対する相関計算の安全化**:
   - ブートストラップサンプルで標準偏差が 0 になった場合、`pearsonr` / `spearmanr` を呼ぶ前に `0.0` または `np.nan` を返すようにガードを入れ、`ConstantInputWarning` を抑止。

#### [MODIFY] [summarize_3way_vad.py](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/summarize_3way_vad.py)
1. **最終レポートの4ブロック固定化**:
   - (1) Human grounding, (2) Affective sensitivity, (3) Dose-response / complexity control, (4) Reader–Self coupling の4セクション構成に統一。

---

## 3. Verification Plan

### Automated Tests
- ルートから仮想環境内のテストを実行：
  ```bash
  .venv/bin/pytest -q
  ```
  - 警告が解消され、全テストが 100% パスすることを確認。
- 各修正スクリプトのドライラン・構文確認：
  ```bash
  .venv/bin/python v3/scripts/run_v3_state_induction.py --dry-run
  .venv/bin/python v3/scripts/run_v3_spatiotemporal_maps.py --dry-run
  .venv/bin/python v3/scripts/run_v3_path_mediation.py --dry-run
  .venv/bin/python v3/scripts/run_v3_confirmatory_replication.py --dry-run
  .venv/bin/python v1/scripts/run_v1_phase_c_targeted_ablation.py --help
  ```

### Manual Verification
- `scratch/first_token_result.json` および `v1/results/derived/v1_phase_c/llama3.2_1b_base/intervention_metadata.json` の妥当性確認。
- `v2/scripts/legacy/` への旧スクリプト完全隔離と、Primary パイプライン（RQ1〜RQ4）の健全性確認。
