# Walkthrough: V3 and Cross-Stage Final Refinements

本ドキュメントでは、包括的レビューに基づき実施した **V3 (Topic control, Confirmatory 実測化, Group split, Necessity 基準, 図表固定値排除)**、**V1 (E6 拡大, CLI 自然化, 表記改訂, Cosine 区別)**、**V2 (Legacy 完全隔離, RQ4 native/matched 分離)**、**Behavioral & テスト (4本柱構成, ConstantInputWarning 解消)** の改訂内容および検証結果を報告します。

---

## 1. 主要な修正内容と効果

### 1.1 V3: 因果解析・統計・プロットの抜本的修正 (最重要)
1. **Topic Control の clean forward バグ修正**:
   - [`run_v3_state_induction.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_state_induction.py):
     `with ActivationHookManager(adapter) as hook_mgr:` 内で `out_base_c` と `out_patch_c` の両方を計算していたバグを解消。
     `out_base_c`（クリーン forward）を hook の**外**で計算し、`out_patch_c` だけを hook 内で計算することで、Topic control における介入前後の純粋な差分ノルムが正確に測定されるよう修正しました。
2. **Train/Test を pair_id Group Split 化**:
   - 単純インデックス分割を廃止し、データセットに `pair_id` が存在する場合は `unique_pairs` に基づく 50/50 Group split を実施。同一ペアのデータが Train と Test に跨がるデータリークを厳密に防止しました。
3. **Necessity 基準を matched-neutral baseline 差へ統一**:
   - 自然変位の基準を固定値 $|E[V] - 5|$ ではなく、matched neutral baseline との差 $|E[V]_{\mathrm{aff}} - E[V]_{\mathrm{neutral}}|$ を Primary 指標に改訂。Behavioral の中立化脱却方針と完全に整合させました。
4. **Confirmatory Replication の完全実測化**:
   - [`run_v3_confirmatory_replication.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/run_v3_confirmatory_replication.py):
     `run_real_model_confirmatory()` 内に存在していた人工指数関数 $C(l) = |\mathrm{shift}|\exp(\cdots)$ や固定減衰率 `attenuated_shift = natural_shift * 0.45`、定数倍 temporal emergence を完全削除。
     全層 $C(l)$ の実測、実測 2D 射影消去（Necessity）、実測各ステージ介入（Temporal emergence）の真の実推論パイプラインに刷新しました。
5. **論文図表の固定値・乱数の完全排除**:
   - [`plot_paper_figures.py`](file:///mnt/nas/home/hiromi/src/emo/v3/scripts/plot_paper_figures.py):
     行357の `collapse_rates = [12.4, 98.6]` などの固定配列を削除し、実測 JSON / CSV（`v3_cross_model_replication_summary.json` 等）から動的に読み込む設計に変更。データが存在しない場合はスキップまたは警告を出す安全な実装に改訂しました。
   - 旧「greedy collapse 98.6%」ストーリーを Primary 論文導線から分離しました。

### 1.2 V1: E6 規模拡大・CLI 自然化・表記統一
1. **Discovery split のサンプル数拡大**:
   - [`run_v1_phase_c_targeted_ablation.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/run_v1_phase_c_targeted_ablation.py):
     Auto-discovery スクリーニングのペア数を従来の最大 10 pair からデフォルト 50 pair（`--discovery-sample-size` で指定可能）に拡大し、サイト選定の統計的安定性を向上させました。
2. **CLI 引数の自然化**:
   - `--split-eval` (default=True) を廃止し、`--no-split-eval` (`action="store_false"`, `dest="split_eval"`) に変更しました。
3. **論文表記の統一**:
   - 過剰な解釈を避けるため、ログや出力サマリーの表記を "Double Dissociation" から **"Task-specific causal specialization"** に改訂しました。
4. **E3 Cosine の区別**:
   - [`generate_phase_c_report.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/generate_phase_c_report.py):
     Mean vector cosine と Pairwise cosine をレポート上で明確に分離して併記出力するよう整形しました。

### 1.3 V2: Legacy スクリプトの完全隔離と RQ4 出力整理
1. **旧・疑似スクリプトの完全隔離**:
   - 乱数疑似ラベル生成スクリプト [`run_annotation_proxy.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/legacy/run_annotation_proxy.py) およびプレースホルダー方向を含む [`run_rsa_and_controlled_coupling.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/legacy/run_rsa_and_controlled_coupling.py) を `v2/scripts/legacy/` に移動・隔離。
   - [`v2/scripts/legacy/README.md`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/legacy/README.md) に対応関係を明記しました。
2. **RQ4 native vs matched-plain の明確な分離**:
   - [`run_v2_recovery_patching.py`](file:///mnt/nas/home/hiromi/src/emo/v2/scripts/run_v2_recovery_patching.py):
     出力 JSON に `summary_by_control_type`（`native` と `matched_plain` の各最大回復率および層別回復率）を追加し、統制条件の差を一目で比較可能にしました。
3. **Cross-family 記述的集計の明確化**:
   - 4ファミリーの集計は $n=4$ の記述統計（descriptive）として位置づけ、過剰な p値解釈を行わないよう注記しました。

### 1.4 Behavioral & テスト: 警告解消と構成固定化
1. **ブートストラップ ConstantInputWarning の解消**:
   - [`metrics.py`](file:///mnt/nas/home/hiromi/src/emo/v1/src/affective_empathy_eval/metrics.py) および [`tests/test_v1_refinements.py`](file:///mnt/nas/home/hiromi/src/emo/tests/test_v1_refinements.py):
     リサンプリングで配列が定数（標準偏差ゼロ）になった場合に `np.nan` または `0.0` を返すゼロ分散ガードを追加し、`scipy.stats` の `ConstantInputWarning` を完全に解消しました。
2. **Behavioral 最終レポートの 4 本柱固定化**:
   - [`summarize_3way_vad.py`](file:///mnt/nas/home/hiromi/src/emo/v1/scripts/summarize_3way_vad.py):
     冒頭サマリーとして **Four Methodological Pillars** (1. Human grounding, 2. Affective sensitivity, 3. Dose-response / complexity control, 4. Reader–Self coupling) を明記するブロックを追加しました。

---

## 2. 変更ファイル一覧

| コンポーネント | ファイルパス | 変更内容 |
|---|---|---|
| **V3** | `v3/scripts/run_v3_state_induction.py` | Topic control クリーン forward バグ修正、pair_id group split、matched neutral necessity 基準 |
| | `v3/scripts/run_v3_confirmatory_replication.py` | 人工関数・固定値 (0.45) 排除、H1〜H4 真の実推論パイプライン化 |
| | `v3/scripts/plot_paper_figures.py` | 行357の `collapse_rates = [12.4, 98.6]` 等の固定値排除、実測ファイル読み込み化 |
| **V1** | `v1/scripts/run_v1_phase_c_targeted_ablation.py` | Discovery サンプル数 50 拡大、`--no-split-eval`、Task-specific causal specialization 表記 |
| | `v1/scripts/generate_phase_c_report.py` | E3 レポートで Mean vector cosine と Pairwise cosine を分離・併記 |
| **V2** | `v2/scripts/run_v2_recovery_patching.py` | `summary_by_control_type` 追加（native と matched-plain の明確分離） |
| | `v2/scripts/legacy/run_annotation_proxy.py` | `legacy/` へ移動・隔離 |
| | `v2/scripts/legacy/run_rsa_and_controlled_coupling.py` | `legacy/` へ移動・隔離 |
| | `v2/scripts/legacy/README.md` | Primary vs Legacy の対応表更新 |
| **Behavioral / 共通** | `v1/src/affective_empathy_eval/metrics.py` | 相関計算のゼロ分散ガード追加（ConstantInputWarning 解消） |
| | `tests/test_v1_refinements.py` | `corr_stat` のゼロ分散ガード追加（ConstantInputWarning 解消） |
| | `v1/scripts/summarize_3way_vad.py` | Four Methodological Pillars 冒頭サマリーブロック追加 |
| **ドキュメント** | `docs/v3_and_cross_stage_final_refinements/*` | `task.md`, `implementation_plan.md`, `walkthrough.md` |

---

## 3. 結論と次のステップ

今回の改訂により、Behavioral, V1, V2, V3 の全ステージにおいて、論理的欠陥（Topic control バグ）、人工的シミュレーション値（Confirmatory の固定乗算・指数関数）、データリーク（index split）、旧ストーリーの残骸（98.6% collapse 固定値）が完全に一掃されました。
すべてが実測データ駆動の強固な因果・統計パイプラインとして完成しています。
