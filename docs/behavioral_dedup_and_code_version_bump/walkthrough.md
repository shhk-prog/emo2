# Behavioral 二重集計防止 & コードバージョン更新 完了報告 (Walkthrough)

本番全実行を安全に開始するため、指摘された2系統の必須修正（Behavioral EmoBank 集計の二重計上防止 & 旧キャッシュ無効化）をすべて完了しました。

---

## 1. 実施した修正の詳細

### ① Behavioral EmoBank 集計の二重計上防止 & モデル名抽出修正
- **問題点**: 
  - `behavioral/primary/run_behavioral_emobank.py` が保存する正式ファイル（`behavioral_emobank_*_3way_vad.csv`）と互換ファイル（`*_3way_vad.csv`）の両方が `*_3way_vad.csv` にマッチし、同一モデルが2回集計されていた。
  - `analyze_file()` のモデル名抽出が `behavioral_emobank_qwen_base` のようにプレフィックス付きで抽出されていた。
  - ファイルが見つからない場合に別ディレクトリへ自動フォールバックしていた。
- **実施内容**:
  - [summarize_behavioral_emobank.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_emobank.py):
    - `extract_model_name()` を追加し、`behavioral_emobank_` プレフィックスおよび `_3way_vad` サフィックスを確実に除去して `qwen_base` 等の正規モデル名のみを抽出。
    - ファイル検索において、まず正式ファイル `behavioral_emobank_*_3way_vad.csv` のみを対象とするよう変更。
    - `--allow-legacy-fallback` 引数を新設し、明示的に指定された場合のみ互換ファイル名や旧ディレクトリへのフォールバックを許可。
    - デフォルトでは入力ディレクトリが空の場合に直ちに `FileNotFoundError` を送出。

---

### ② Behavioral AIPsy 集計の自動フォールバック禁止
- **問題点**:
  - `summarize_behavioral_aipsy.py` も入力ファイルがない場合に過去の `v1/results` 等の旧ディレクトリへ自動フォールバックしていた。
- **実施内容**:
  - [summarize_behavioral_aipsy.py](file:///mnt/nas/home/hiromi/src/emo2/behavioral/analysis/summarize_behavioral_aipsy.py):
    - `--allow-legacy-fallback` 引数を追加。
    - デフォルトでは自動フォールバックを禁止し、対象が存在しない場合は直ちに `FileNotFoundError` を送出。

---

### ③ コードバージョン更新による旧キャッシュ無効化
- **問題点**:
  - `DEFAULT_CODE_VERSION` が `"2.2.0"` のままであったため、コード修正前の旧実験結果が `is_manifest_matching()` でキャッシュヒットとしてスキップされるリスクがあった。
- **実施内容**:
  - [manifests.py](file:///mnt/nas/home/hiromi/src/emo2/src/affective_empathy_eval/manifests.py):
    - `DEFAULT_CODE_VERSION = "2.3.0"` に更新。
    - これにより、`code_version="2.2.0"` 以下の旧成果物・キャッシュは自動的に不一致となり、完全な再計算が担保される。

---

### ④ 単体テストの追加
- **追加内容 ([test_pre_production_fixes.py](file:///mnt/nas/home/hiromi/src/emo2/tests/test_pre_production_fixes.py))**:
  - `test_behavioral_emobank_summary_dedup_and_model_name()`:
    - `extract_model_name()` の単体動作確認。
    - 同一ディレクトリに正式CSVと互換CSVが同居していても、集計結果が1件のみ（モデル名 `"qwen_base"`）になることを実証。
  - `test_manifest_code_version_cache_invalidation()`:
    - 旧コードバージョン `"2.2.0"` の manifest に対し、現在の `DEFAULT_CODE_VERSION="2.3.0"` 下で `is_manifest_matching()` が `False` を返すことを検証。

---

## 2. ターミナルでの確認手順

```bash
source .venv/bin/activate

# 1. 新規単体テストの実行
PYTHONPATH=src:. pytest -q -k "behavioral_emobank_summary or manifest_code_version"

# 2. 全単体テストの実行
PYTHONPATH=src:. pytest -q

# 3. Behavioral dry-run の再実行（モデルが重複せず正しく集計されるか確認）
bash scripts/run_production_behavioral.sh cpu --dry-run
```
