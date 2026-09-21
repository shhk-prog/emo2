# タスクリスト: Behavioral二重集計防止 & コードバージョン更新 (Behavioral Dedup & Code Version Bump)

## 状況サマリー
本番フル実行直前に、以下の2系統の必須修正を実施する。
1. Behavioral EmoBank の集計における正式CSVと互換CSVの二重計上防止、モデル名パース修正、自動ディレクトリフォールバックの制限（AIPsy側も統一）
2. `DEFAULT_CODE_VERSION` を `2.3.0` に引き上げ、過去のキャッシュ（2.2.0等）の意図せぬ再利用を完全に防止する

---

## タスク一覧

### 1. Behavioral EmoBank 集計の二重計上防止 & モデル名抽出修正
- [x] `behavioral/analysis/summarize_behavioral_emobank.py`:
  - [x] デフォルトで正式ファイル `behavioral_emobank_*_3way_vad.csv` を優先検索し、存在しない場合かつ `--allow-legacy-fallback` 指定時のみ `*_3way_vad.csv` を対象とするよう修正
  - [x] `analyze_file()` および集計ループでのモデル名抽出を、`behavioral_emobank_` プレフィックスを除去して `qwen_base` 等のクリーンな名称にするよう修正
  - [x] 別ディレクトリ（旧パス）への自動フォールバックを廃止し、対象ファイルがなければ `FileNotFoundError` を送出。`--allow-legacy-fallback` 指定時のみ旧ディレクトリを探索する仕様に変更

### 2. Behavioral AIPsy 集計のフォールバック仕様統一
- [x] `behavioral/analysis/summarize_behavioral_aipsy.py`:
  - [x] 別ディレクトリへの自動フォールバックを廃止し、デフォルトでは指定ディレクトリが空なら直ちに `FileNotFoundError` を送出。`--allow-legacy-fallback` 指定時のみ探索する仕様に統一

### 3. コードバージョン更新による旧キャッシュ無効化
- [x] `src/affective_empathy_eval/manifests.py`:
  - [x] `DEFAULT_CODE_VERSION` を `"2.2.0"` から `"2.3.0"` に更新

### 4. 単体テストの追加
- [x] `tests/test_pre_production_fixes.py`:
  - [x] 正式CSV（`behavioral_emobank_qwen_base_3way_vad.csv`）と互換CSV（`qwen_base_3way_vad.csv`）が同居する場合に集計されるモデルが1つだけ（`model == "qwen_base"`）になることを検証するテスト
  - [x] 旧 manifest（`code_version="2.2.0"`）に対して、現在の `DEFAULT_CODE_VERSION`（`"2.3.0"`）下で `is_manifest_matching` が `False` を返すことを検証するテスト

### 5. 検証とドキュメント作成
- [x] `PYTHONPATH=src:. pytest -q` (122 passed, 1 deselected)
- [x] `bash scripts/run_production_behavioral.sh cpu --dry-run` 完走 & モデル重複解消確認
- [x] `docs/behavioral_dedup_and_code_version_bump/walkthrough.md` の作成
