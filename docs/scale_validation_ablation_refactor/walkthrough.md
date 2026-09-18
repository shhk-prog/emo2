# Scale Validation (Mistral 7B) の Ablation 独立分離 完了報告 (Walkthrough)

## 概要
研究の Primary 解析（1〜1.5B コホート：Qwen, Llama, Gemma, OLMo）の純粋性を担保し、論文上で「7B の外部スケール検証はあくまで主結果の頑健性を補足確認するアブレーション（Ablation / Supplementary）」として明確に位置づけるため、`scale_validation` を専用の設定ファイル・独立した実行スクリプト・専用の出力ディレクトリに分離しました。

---

## 主な改修点と成果物

### 1. 専用設定ファイルの新設 (`configs/scale_validation.yaml`)
- **Mistral 7B の独立定義**:
  - `base`: `mistralai/Mistral-7B-v0.3`
  - `instruct`: `mistralai/Mistral-7B-Instruct-v0.3`
  - `role`: `scale_validation`
- **専用出力ディレクトリ**:
  - `results/ablation/scale_validation/raw/`
  - `results/ablation/scale_validation/derived/`
  - Primary 実験の `v2/results/` や `v3/results/` と完全に分離。

### 2. 独立実行スクリプトの新設 (`scripts/run_scale_validation.py`)
- Primary パイプラインを介さず、Mistral 7B 単体で以下の 3 大検証を一気通貫で実行・出力：
  1. **表現幾何・交差デコード（RQ1 & RQ2 相当）**: `scale_validation_geometry_mistral.json`
  2. **因果マッピング・ピーク解離（RQ3 相当）**: `scale_validation_causal_mistral.json`
  3. **分布回復パッチング（RQ4 相当）**: `scale_validation_recovery_mistral.json`
  4. **統合サマリーレポート**: `scale_validation_summary.json`
- `--dry-run`, `--device`, `--max-samples` に対応。

### 3. 統合ランナー CLI の拡張 (`src/affective_empathy_eval/run.py`)
- `--stage scale_validation` を追加し、ワンコマンドでアブレーションを実行可能に：
  ```bash
  # 統合ランナーから実行
  python -m affective_empathy_eval.run --stage scale_validation [--dry-run]

  # 専用スクリプトから直接実行
  python scripts/run_scale_validation.py [--dry-run]
  ```

---

## 検証結果

### 1. 専用スクリプトのドライラン検証
```bash
.venv/bin/python scripts/run_scale_validation.py --dry-run
```
- HuggingFace から `Mistral-7B-Instruct-v0.3` の層数（32層）および隠れ層次元（4096）を動的解決。
- Step 1〜3 の全検証が正常に完了し、`results/ablation/scale_validation/` 配下への出力を確認 (Code 0)。

### 2. 統合ランナーからの呼び出し検証
```bash
.venv/bin/python -m affective_empathy_eval.run --stage scale_validation --dry-run
```
- 統合 CLI から正常にディスパッチされ、完走を確認 (Code 0)。

### 3. テストスイート回帰テスト
```bash
.venv/bin/pytest -q
```
- **45 passed in 6.25s**（全件合格）。

---

## 結論
これにより、Primary 1〜1.5B コホート（主実験）のデータ・設定・ログを一切汚すことなく、査読者対応や補足分析として「7B モデルへのスケール頑健性」を独立したアブレーション実験枠として即座に実行・提示できる体制が整いました。
