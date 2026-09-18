# 作業完了報告書 (Walkthrough): テスト環境非依存化 (PYTHONPATH) および root README への Central RQ / 4-Stage 統合

## 実施概要
ユーザーからの指示に基づき、Behavioral → V1 → V2 → V3 の本番全再実行に向けた最後の2点の修正を実施しました：
1. **テストスイートの環境非依存化 (PYTHONPATH 補完)**:
   - `test_all_dispatched_commands_argparse_compatibility` において、`subprocess.run` 実行時に `src/` を付加した `PYTHONPATH` および `cwd=repo_root` を渡すよう修正。
   - 併せて、統合 CLI ランナー `src/affective_empathy_eval/run.py` の `run_command` でも同様に `PYTHONPATH` を確実に渡す堅牢化を実施。
   - clone 直後や非 editable install 環境でもテストスイートが 1 件のエラーもなく完全 PASS することを確認。
2. **root `README.md` 冒頭への Central RQ・4-Stage 概念表・論文構成の統合**:
   - Central RQ、中心的主張（Core Thesis）、4-Stage の概念対応表（証拠階層）、論文 Section 構造、3本の学術的貢献（Three Main Contributions）、729 VAD vs 81 VA の設計根拠、および Primary 1–1.5B コホート（4 大独立ファミリー）の記述を冒頭に整理。
   - セクション番号を 1 から 8 まで一貫して再採番。

---

## 主な変更ファイル

### 1. テストおよびランナーの修正
- **[`tests/test_production_entrypoints.py`](file:///mnt/nas/home/hiromi/src/emo2/tests/test_production_entrypoints.py)**:
  `test_all_dispatched_commands_argparse_compatibility` において、`env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}"` を明示的に渡すように変更。
- **[`src/affective_empathy_eval/run.py`](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)**:
  `run_command` において、ディスパッチされる各サブプロセスに `PYTHONPATH` が確実に渡るよう環境変数を補強。

### 2. ルートドキュメントの改訂
- **[`README.md`](file:///mnt/nas/home/hiromi/src/emo2/README.md)**:
  - **§1. 中心リサーチクエスチョン (Central RQ)**
    > *How are affect-relevant internal representations coupled to LLM self-reports, and how does this relationship vary across tasks, post-training, and stages of computation?*
  - **中心的自己報告主張 (Core Thesis)**
    > LLMの自己報告は、単なる出力上の模倣でも、内部でデコード可能な情動情報の直接的な読み出しでもない。自己報告は情動関連内部表現と系統的に結びつくが、その結びつきは部分的かつタスク依存であり、post-trainingによって再編され、計算過程の特定の位置で初めて因果的利用可能性を持つ。
  - **§2. 証拠階層と論文構造 (4-Stage Evidence Hierarchy)**
    - Covariation (§3. Behavioral) → Representation / Causality (§4. V1) → Reorganization (§5. V2) → Utilization (§6. V3)
    - 3 本の Contribution（Behavioral+V1, V2, V3）
  - **§3. 操作的定義とモデル・データ構成**
    - 729 VAD vs 81 VA の Methods 根拠（VAD プロトコル保持 vs 因果スイープの計算量抑制）
    - 4 大独立ファミリー（Qwen 2.5, Llama 3.2, Gemma 3, OLMo 2）および外部検証（Mistral 7B）
  - **§4〜§8**: 統一実験枠組み、環境構築、テスト実行、実行方法、ディレクトリ構成の通し番号化

---

## 検証結果

### 自動テスト
`.venv/bin/pytest -q` を実行：
```text
.............................................................              [100%]
61 passed in 85.49s (0:01:25)
```
- 全 61 件のテストが完全に PASS（失敗・スキップなし）。
- `tests/test_production_entrypoints.py::test_all_dispatched_commands_argparse_compatibility` も正常通過。

---

## 成果物ドキュメント
- 実装計画: [`docs/test_subprocess_pythonpath_and_central_rq_readme/implementation_plan.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/test_subprocess_pythonpath_and_central_rq_readme/implementation_plan.md)
- タスク管理: [`docs/test_subprocess_pythonpath_and_central_rq_readme/task.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/test_subprocess_pythonpath_and_central_rq_readme/task.md)
- 作業完了報告書: [`docs/test_subprocess_pythonpath_and_central_rq_readme/walkthrough.md`](file:///mnt/nas/home/hiromi/src/emo2/docs/test_subprocess_pythonpath_and_central_rq_readme/walkthrough.md)
