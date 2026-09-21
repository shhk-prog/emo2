# 本番全実行前 最終4点ピンポイント修正 実装計画 (Implementation Plan)

本番全実行を安全に開始するため、指摘された4点の問題（Chat template fallback、EmoBank JSON上書き、EmoBank CVテキスト重複リーク、V2 cross-family両軸集約）を修正します。不要なリファクタや設計変更は行わず、指示された4点に厳格に限定して実施します。

## ユーザーレビュー事項 (User Review Required)
- 本計画はユーザーから指摘された4点の修正のみを対象とし、既存のインターフェースやファイル出力の後方互換性を完全に維持します。
- V1のStandardScaler適用 vs V2/V3のnative activation座標保持、およびV2のN=4 family bootstrapについては、大改造を行わず方法論上の前提として取り扱います。

## 変更計画の詳細

### 1. V1 Phase B / Phase C の chat-template fallback 修正 & テスト追加
- **対象ファイル**:
  - `v1/primary/run_phase_b.py` (92–104行付近)
  - `v1/primary/run_phase_c.py` (223–235行付近)
- **変更内容**:
  `except Exception:` 内で元の `messages` を再度呼び出している実装ミスを解消し、`[{"role": "user", "content": user_content}]` のみにフォールバックするよう修正。
  ```python
  except Exception:
      return tokenizer.apply_chat_template(
          [{"role": "user", "content": user_content}],
          tokenize=False,
          add_generation_prompt=True,
      )
  ```
- **テスト追加**:
  `tests/test_pre_production_fixes.py` に `test_chat_template_system_role_fallback()` を追加：
  - system role を受容する tokenizer では通常の 2-turn messages で呼び出されること
  - system role で例外を送出する（Gemma風）tokenizer では fallback して 1-turn (user only) で正常にテンプレートが適用されること

### 2. V1 Phase A の modular JSON が EmoBank 結果を上書きする問題の修正
- **対象ファイル**:
  - `v1/primary/run_phase_a.py` (1072–1080行付近, 1294–1306行付近, dry-run 800行付近)
- **変更内容**:
  `--dataset both` 実行時に、先に保存された `v1_e1_decodability_{model_prefix}.json` の EmoBank 結果が AIPsy の結果で上書きされて消失する問題を解消。
  `e1_payload = {}` を関数スコープで保持し、EmoBank と AIPsy の完了時にそれぞれキーを追加して、Phase A の最後に1回だけ `save_experiment_result()` を実行する。
  dry-run も本番と同一のスキーマ（`{"emobank": ..., "aipsy_primary": ..., ...}`）に統一する。

### 3. V1 EmoBank の CV における同一テキスト leakage 修正
- **対象ファイル**:
  - `v1/primary/run_phase_a.py` (1000行付近)
- **変更内容**:
  現在 `group_ids = df_emobank["id"].values` となっており、同一テキスト（"Dear Name:", "Sincerely,"）が異なるIDとして別foldに分配される微小リーク（4/1000件）を防ぐため：
  ```python
  content_groups = (
      df_emobank["text"]
      .astype(str)
      .str.normalize("NFKC")
      .str.strip()
  )
  group_ids = content_groups.values
  ```
  として、同一テキスト内容が必ず同一foldに割り当てられるように修正する。

### 4. V2 cross-family summary の Valence / Arousal 両軸集約化
- **対象ファイル**:
  - `v2/primary/run_rq1_rq2_cross_decoding.py` (380行付近, 660行付近)
- **変更内容**:
  - `results["summary_metrics"]` を `by_axis: {"valence": {...}, "arousal": {...}}` に拡張し、Valence と Arousal の両軸の重心・ピーク・sharing delta を算出。
  - 後方互換性のため、ルートレベルの既存キー（`peak_depth_base_cross`, `peak_depth_inst_cross`, `mean_delta_sharing` 等）は Valence alias として維持。
  - `v2_cross_family_summary.json` においても、`by_axis` 配下で Valence と Arousal を個別に cross-family bootstrap 集約し、ルートレベルにも後方互換用の Valence 集約を維持。

---

## 検証手順 (Verification Plan)

### 自動テスト
1. `pytest -q` によるテスト全件実行（新規追加テスト含む）
2. `ruff check src behavioral v1/primary v2/primary v3/primary tests`

### 各ステージ Dry-Run 確認
1. `bash scripts/run_production_behavioral.sh cpu --dry-run`
2. `bash scripts/run_production_v1.sh cpu --dry-run`
3. `bash scripts/run_production_v2.sh cpu --dry-run`
4. `bash scripts/run_production_v3.sh cpu --dry-run`
