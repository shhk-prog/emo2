# Behavioral 二重集計防止 & コードバージョン更新 実装計画

本番全実行を完全に安全に行うため、指示された2系統の必須修正（Behavioral EmoBank 二重計上防止 & 旧キャッシュ無効化）を実施します。

---

## ユーザー確認事項
- 修正内容はご指示通りの2系統に厳密に限定し、既存のモデル構成やパイプライン設計の改変は一切行いません。
- 旧結果の読み込みが必要な場合のために `--allow-legacy-fallback` オプションを設け、デフォルトでは暗黙のフォールバックを禁止して `FileNotFoundError` を出します。

---

## 提案する変更内容

### 1. Behavioral EmoBank 集計スクリプト
#### [MODIFY] [summarize_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_emobank.py)
- **ファイル検索の厳格化**:
  - デフォルトでは正式ファイルパターン `behavioral_emobank_*_3way_vad.csv` のみを検索。
  - `--allow-legacy-fallback` 引数を新設。本フラグが明示された場合のみ、互換ファイル（`*_3way_vad.csv`）や旧ディレクトリ（`behavioral/results/emobank_3way` など）へのフォールバックを許可。
  - ファイルが存在しない場合は直ちに `FileNotFoundError` を送出。
- **モデル名抽出の正規化**:
  - `csv_path` からモデル名を抽出する際、`behavioral_emobank_` プレフィックスおよび `_3way_vad` サフィックスを除去し、`qwen_base` 等の正規モデル名を取得。
  - `analyze_file()` 内の `model` 列および `p555["model"]` の両方に適用。

---

### 2. Behavioral AIPsy 集計スクリプト
#### [MODIFY] [summarize_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py)
- **自動フォールバックの禁止**:
  - `--allow-legacy-fallback` 引数を新設。
  - デフォルトでは `args.input_dir` が空の場合に別ディレクトリへフォールバックせず、直ちに `FileNotFoundError` を送出。

---

### 3. マニフェスト・コードバージョン更新
#### [MODIFY] [manifests.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py)
- `DEFAULT_CODE_VERSION = "2.3.0"` に更新。
- これにより、`is_manifest_matching()` で以前のバージョン（`2.2.0` など）で生成されたキャッシュが自動的に不一致となり、旧結果の誤用を確実に防止。

---

### 4. 単体テストの追加
#### [MODIFY] [test_pre_production_fixes.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_pre_production_fixes.py)
- **`test_behavioral_emobank_summary_dedup_and_model_name()`**:
  - 正式CSV (`behavioral_emobank_qwen_base_3way_vad.csv`) と互換CSV (`qwen_base_3way_vad.csv`) が同一ディレクトリに存在する場合、集計対象が1モデルのみであり、かつモデル名が `"qwen_base"` となることを検証。
- **`test_manifest_code_version_cache_invalidation()`**:
  - `code_version="2.2.0"` の manifest に対し、現在の `DEFAULT_CODE_VERSION="2.3.0"` 下で `is_manifest_matching()` が `False` を返すことを検証。

---

## 検証手順

1. 単体テストの実行:
   ```bash
   PYTHONPATH=src:. pytest -q -k "behavioral_emobank_summary or manifest_code_version"
   ```
2. 全単体テストの実行:
   ```bash
   PYTHONPATH=src:. pytest -q
   ```
3. Behavioral dry-run の再確認:
   ```bash
   bash scripts/run_production_behavioral.sh cpu --dry-run
   ```
   集計結果でモデル数が重複せず、モデル名がクリーンに出力されていることを確認。
