# Scale Validation (Mistral 7B) の Ablation 独立分離 実装計画

## 概要
研究の Primary 解析（1〜1.5B コホート：Qwen, Llama, Gemma, OLMo）をクリーンに保ち、論文上で「7B の外部スケール検証はあくまで主結果の頑健性を補足確認するアブレーション（Ablation / Supplementary）」として明確に位置づけるため、`scale_validation` を専用の設定ファイル・独立した実行スクリプト・専用の出力ディレクトリに分離します。

---

## ユーザーレビュー必須項目

> [!IMPORTANT]
> **Ablation 扱いの独立実行体系**:
> 1. **専用設定ファイル**: `configs/scale_validation.yaml` を新設し、Mistral 7B のモデル定義およびアブレーション用パラメータ・出力先を完全に集約。
> 2. **専用実行スクリプト**: `scripts/run_scale_validation.py` を新設し、Primary パイプラインを介さずに Mistral 7B 単体で幾何・因果・分布回復の検証を実行可能に。
> 3. **統合ランナー拡張**:
>    - `python -m affective_empathy_eval.run --stage scale_validation`（または `python scripts/run_scale_validation.py`）により、Primary 実験と完全に切り離してワンコマンド実行可能に。
> 4. **成果物保存先**: `results/ablation/scale_validation/` に隔離し、Primary の成果物（`v2/results/`, `v3/results/`）と混在しない構造にします。

---

## 提案される変更点

### 1. 設定層 (`configs/`)

#### [NEW] [configs/scale_validation.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/scale_validation.yaml)
- Mistral 7B（Base: `mistralai/Mistral-7B-v0.3`, Instruct: `mistralai/Mistral-7B-Instruct-v0.3`）の定義。
- アブレーション専用の出力パス（`results/ablation/scale_validation/`）。

#### [MODIFY] [configs/models.yaml](file:///mnt/nas/home/hiromi/src/emo2/configs/models.yaml)
- `primary_small` を主役としつつ、`scale_validation:`（Ablation 参照用）を明確に区分。

---

### 2. 専用スクリプトおよび統合ランナー

#### [NEW] [scripts/run_scale_validation.py](file:///mnt/nas/home/hiromi/src/emo2/scripts/run_scale_validation.py)
- Mistral 7B 専用のアブレーション実行スクリプト：
  - 表現幾何・交差デコード（RQ1 & RQ2 相当）
  - 因果マッピング（RQ3 相当）
  - 分布回復パッチング（RQ4 相当）
- `--dry-run`, `--device`, `--max-samples` をサポート。

#### [MODIFY] [src/affective_empathy_eval/run.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/run.py)
- `--stage` の選択肢に `scale_validation` を追加：
  ```bash
  python -m affective_empathy_eval.run --stage scale_validation [--dry-run]
  ```

---

## 検証計画

### 1. スモークテスト
- 専用スクリプトのドライラン確認:
  ```bash
  .venv/bin/python scripts/run_scale_validation.py --dry-run
  ```
- 統合ランナーからの呼び出し確認:
  ```bash
  .venv/bin/python -m affective_empathy_eval.run --stage scale_validation --dry-run
  ```

### 2. 回帰テスト
- 既存テストスイート全件合格確認:
  ```bash
  .venv/bin/pytest -q
  ```
