# タスクリスト: 本番前最終4点ピンポイント修正 (Final 4 Pre-Production Fixes)

## 状況サマリー
全体のパイプラインおよび全ステージの dry-run、テストは実質全通過している。
本番フル実行前に直すべき実質的な問題は以下の4点に限定される。

---

## タスク一覧

### 1. V1 Phase B / Phase C の chat-template fallback 修正 & テスト追加
- [x] `v1/primary/run_phase_b.py`: `except Exception:` 内の引数を `[{"role": "user", "content": user_content}]` に修正
- [x] `v1/primary/run_phase_c.py`: `except Exception:` 内の引数を `[{"role": "user", "content": user_content}]` に修正
- [x] `tests/test_pre_production_fixes.py`: chat template fallback（system role例外時にuser-onlyで成功すること）の単体テストを追加

### 2. V1 Phase A の modular JSON が EmoBank 結果を上書きする問題の修正
- [x] `v1/primary/run_phase_a.py`:
  - [x] `e1_payload = {}` を先に初期化し、`if emobank: e1_payload["emobank"] = ...`、`if aipsy: e1_payload["aipsy_primary"] = ...` で集約
  - [x] Phase A の最後に1回だけ `v1_e1_decodability_{model_prefix}.json` を保存
  - [x] dry-run モードでも同一の JSON schema を出力するように統一

### 3. V1 EmoBank の CV における同一テキスト leakage 修正
- [x] `v1/primary/run_phase_a.py`:
  - [x] 行1000付近の `group_ids = df_emobank["id"].values` を `content_groups = (df_emobank["text"].astype(str).str.normalize("NFKC").str.strip()).values` に修正し、同一テキストを同一foldに固定

### 4. V2 cross-family summary の Valence / Arousal 両軸集約化
- [x] `v2/primary/run_rq1_rq2_cross_decoding.py`:
  - [x] `summary_metrics` を `by_axis: {"valence": {...}, "arousal": {...}}` 構造に拡張（旧rootキーはValence aliasとして後方互換維持）
  - [x] cross-family summary (`v2_cross_family_summary.json`) でも `valence` と `arousal` を別々に集約し、既存キーは後方互換として保持

### 5. 検証
- [ ] `pytest -q`
- [ ] `ruff check src behavioral v1/primary v2/primary v3/primary tests`
- [ ] 各ステージ dry-run 完走確認
- [x] `docs/final_four_pre_production_fixes/walkthrough.md` 作成

